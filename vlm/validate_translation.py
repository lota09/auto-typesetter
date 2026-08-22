#!/usr/bin/env python3
"""번역 결과를 결정론적으로 검사하고, 걸린 박스만 다시 요청한다.

사람이 승인하는 단계는 두지 않는다. 대신 **기계로 확실히 판정되는 것만** 잡는다.
애매한 품질 판단(자연스러운가, 뉘앙스가 맞나)은 여기서 다루지 않는다 — 그건
규칙으로 잡으려 하면 오탐만 늘고, 어차피 사람이 안 보는 파이프라인이다.

잡는 것:
  residual_script    한국어 결과에 한자·가나가 남았다 (`영주起床 후送餐`)
  not_translated     원문과 결과가 같다
  empty_target       원문이 있는데 결과가 비었다
  repetition         같은 조각이 계속 반복된다 (생성 붕괴)
  length_anomaly     원문 대비 길이가 비정상이다 (설명을 덧붙이는 실패 모드)
  glossary_violation 용어집에 있는 낱말인데 지정 역어를 쓰지 않았다
  register_mismatch  시트가 정한 말투와 실제 종결어미가 다르다
  leak               지시문·JSON 조각이 결과에 섞였다

말투 검사가 핵심이다. 이 프로젝트의 목표가 "반말했다 존댓말했다 하지 않는 것"인데,
한국어 종결어미는 유한하게 열거되므로 모델을 믿지 않고 직접 확인할 수 있다.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import client_for  # noqa: E402
from translate_chapter import render_styleguide, ask, TRANSLATION_SCHEMA  # noqa: E402

import requests  # noqa: E402

HAN_KANA = re.compile(r"[㐀-䶿一-鿿぀-ヿ]")
HANGUL = re.compile(r"[가-힣]")
LEAK = re.compile(r"```|\"key\"|\"target\"|\bEMPTY\b|^\s*\{", re.M)

# 종결어미로 말투를 판정한다. 문장부호·말줄임표·따옴표를 먼저 털어낸다.
TRAIL = re.compile(r"[\s.,!?~…·\"'’”」』\)\]]+$")
# `합니다`·`감사합니다` 는 `습니다` 도 `ㅂ니다`(자모) 도 아니다. 합성된 음절로 오므로
# `니다$` 로 봐야 잡힌다. 이걸 놓쳐서 정중한 문장을 반말로 오판했었다.
FORMAL = re.compile(r"(니다|십시오|십시다)$")
POLITE = re.compile(r"(요|죠|쇼)$")
# 연결어미로 끝나면 종결 말투가 없다. 만화는 한 문장이 여러 말풍선에 걸쳐 끊기므로
# 이 형태가 흔하고, 반말로 세면 오탐이 쏟아진다.
CONNECTIVE = re.compile(r"(고|며|면서|지만|도록|어서|아서|여서|는데|은데|려고"
                        r"|거나|든지|면|니까|자마자)$")
# 종결어미가 없는 감탄사·효과음·명사구는 말투가 없다. 2음절까지만 건너뛴다 —
# 3음절까지 빼면 `도와줘`(반말)·`갑니까`(격식체)처럼 말투가 분명한 것도 놓친다.
# 대신 `돼` 같은 1음절 반말은 포기한다. 오탐을 내지 않는 쪽이 낫다.
NO_ENDING = re.compile(r"^[^가-힣]*$|^[가-힣]{1,2}$")
LETTERISH = re.compile(r"[가-힣㐀-䶿一-鿿぀-ヿA-Za-z]")


def _jongseong(ch):
    """한글 음절의 종성 인덱스. 음절이 아니면 -1."""
    code = ord(ch) - 0xAC00
    return code % 28 if 0 <= code < 11172 else -1


def _is_formal_question(body):
    """`갑니까`·`합니까` 는 격식체, `괜찮으니까` 는 연결어미다.

    둘 다 `니까` 로 끝나므로 앞 음절의 종성으로 가른다 (ㅂ=17, 또는 `습`).
    """
    if not body.endswith("니까") or len(body) < 3:
        return False
    prev = body[-3]
    return prev == "습" or _jongseong(prev) == 17

# 조사로 끝나면 문장 조각이다. 만화는 한 문장을 말풍선 여러 개로 쪼개므로 흔하다.
PARTICLE_END = re.compile(r"(은|는|이|가|을|를|도|에|에서|와|과|의|로|으로|만|부터|까지)$")
# 호격 — 이름을 부르는 말은 동사가 없어 말투를 담지 않는다 (`클로에 씨`, `로라 님`).
# `아`/`야` 는 넣지 않는다. `데려온 거야` 같은 흔한 반말 종결어미와 충돌해서,
# 넣으면 진짜 말투 붕괴를 못 잡는다. `릴리야` 류 호격을 놓치는 편이 낫다.
VOCATIVE = re.compile(r"(씨|님|짱|쨩)$")

STYLE_TO_CLASS = {"formal": "formal", "polite": "polite", "casual": "plain",
                  "rough": "plain", "childish": "plain"}
CLASS_LABEL = {"formal": "합니다체", "polite": "해요체", "plain": "반말"}
# 존댓말 안에서 합니다체/해요체가 섞이는 것은 한국어에서 정상이다. 이 프로젝트가
# 막아야 하는 붕괴는 존댓말↔반말이므로, 위반 판정은 이 층위에서만 한다.
HONORIFIC = {"formal": "honorific", "polite": "honorific", "plain": "plain"}


def classify_register(text):
    """한국어 문장의 말투를 formal/polite/plain 중 하나로. 판정 불가면 None."""
    if not text:
        return None
    body = TRAIL.sub("", text.strip())
    if not body or NO_ENDING.match(body):
        return None
    # 한글만 세어 2자 이하면 종결어미가 없다. `네… 네` 처럼 문장부호가 섞인 짧은
    # 응답까지 걸러야 하므로 문장부호를 털고 센다. 3자부터는 `도와줘`(반말) 처럼
    # 말투가 분명하므로 남긴다.
    tokens = [w for w in re.split(r"[\s,·]+", body) if w]
    if len(re.sub(r"[^가-힣]", "", body)) <= 2:
        return None
    # 호격은 두 낱말 이하일 때만 인정한다. 긴 문장의 `~잡아주대요` 를 호격으로
    # 오인하면 진짜 위반을 놓친다.
    if len(tokens) <= 2 and VOCATIVE.search(body):
        return None
    if FORMAL.search(body) or _is_formal_question(body):
        return "formal"
    if POLITE.search(body):
        return "polite"
    # 연결어미·조사 판정은 반말보다 뒤에 둘 수 없다. `싫지만`·`여러분도` 를
    # 반말로 세면 안 된다.
    if CONNECTIVE.search(body) or PARTICLE_END.search(body):
        return None
    if HANGUL.search(body):
        return "plain"
    return None


def parse_expected(ending):
    """시트의 koreanEnding 문자열을 말투 클래스로. 못 읽으면 None.

    모델이 여기에 일본어 어미(`よ/わ`)를 적어 놓는 일이 실제로 있었다. 못 읽는
    값은 조용히 건너뛴다 — 시트가 이상한 것을 번역 오류로 보고하면 안 된다.
    """
    if not ending:
        return None
    if any(k in ending for k in ("합니다", "습니다", "하십시오", "격식")):
        return "formal"
    if "해요" in ending:
        return "polite"
    # 해라체·한다체는 권위적 문어체지만 경어가 아니다 → 반말 층위.
    if any(k in ending for k in ("반말", "해체", "해라", "한다체")):
        return "plain"
    return None


def expected_register(sg_index, speaker, addressee):
    """(화자, 수신자) 에 대해 시트가 기대하는 말투. relationTo 가 default 를 덮는다."""
    prof = sg_index.get((speaker or "").lower())
    if not prof:
        return None, None
    for r in prof.get("registerTo") or []:
        toward = (r.get("toward") or "").lower()
        if addressee and toward and (toward in addressee.lower()
                                     or addressee.lower() in toward):
            cls = parse_expected(r.get("koreanEnding"))
            if cls:
                return cls, f"{prof['displayName']}→{r['toward']}"
    return STYLE_TO_CLASS.get(prof.get("speechStyle")), \
        f"{prof['displayName']} 기본({prof.get('speechStyle')})"


def build_index(sg):
    """화자 이름과 시각 서술자를 모두 키로 하는 인물 색인.

    1단이 돌려주는 speaker 는 아직 시각 서술자("silver-haired maid")인데
    시트의 인물명은 실제 이름("Chloe")이다. 양쪽으로 찾을 수 있어야 한다.
    """
    idx = {}
    for c in sg.get("characters", []):
        for key in [c.get("displayName"), c.get("targetName"),
                    *(c.get("sourceNames") or []), *(c.get("visualDescriptors") or [])]:
            if key:
                idx[key.lower()] = c
    return idx


def has_repetition(text, min_unit=2, times=4):
    """같은 조각이 times 번 이상 연달아 반복되는지."""
    if not text:
        return False
    for size in range(min_unit, max(min_unit + 1, len(text) // times + 1)):
        for i in range(len(text) - size * times + 1):
            unit = text[i:i + size]
            if unit.strip() and text[i:i + size * times] == unit * times:
                return True
    return False


def check(t, sg_index, glossary, min_ratio, max_ratio):
    src = (t.get("ocr") or "").strip()
    tgt = (t.get("target") or "").strip()
    kind = t.get("kind")
    issues = []

    if src and not tgt:
        issues.append(("empty_target", "원문이 있는데 결과가 비었다"))
        return issues
    if not tgt:
        return issues

    if HAN_KANA.search(tgt):
        left = "".join(HAN_KANA.findall(tgt))
        issues.append(("residual_script", f"한자/가나 잔류: {left}"))
    # 원문이 말줄임표·문장부호뿐이면 결과가 같은 것이 정상이다.
    if (src and LETTERISH.search(src)
            and re.sub(r"\s+", "", src) == re.sub(r"\s+", "", tgt)):
        issues.append(("not_translated", "원문과 동일"))
    if LEAK.search(tgt):
        issues.append(("leak", "지시문/JSON 조각 혼입"))
    if has_repetition(tgt):
        issues.append(("repetition", "반복 붕괴"))

    # 짧은 원문은 길이비가 요동치므로 일정 길이 이상만 본다.
    if src and len(src) >= 6:
        ratio = len(tgt) / len(src)
        if ratio < min_ratio or ratio > max_ratio:
            issues.append(("length_anomaly", f"길이비 {ratio:.2f} (원문 {len(src)}자 → {len(tgt)}자)"))

    for g in glossary:
        s, tg = (g.get("source") or "").strip(), (g.get("target") or "").strip()
        if s and tg and s in src and tg not in tgt:
            issues.append(("glossary_violation", f"'{s}' → '{tg}' 미적용"))
            break

    # 효과음·나레이션, 그리고 **속마음**은 화자 말투 규칙의 대상이 아니다.
    # 독백은 수신자가 없어서 경어를 쓸 상대가 없고, 정중한 인물이 속으로
    # 하다체를 쓰는 것은 한국어에서 정상이다.
    if kind == "dialogue":
        got = classify_register(tgt)
        want, why = expected_register(sg_index, t.get("speaker_name"), t.get("addressee"))
        if got and want and HONORIFIC[got] != HONORIFIC[want]:
            issues.append(("register_mismatch",
                           f"{why}: {CLASS_LABEL[want]} 기대인데 {CLASS_LABEL[got]}"))
    return issues


REPAIR_PROMPT = """These Korean translations from a comic chapter failed automated checks.
Produce a corrected translation for each key.

STYLE GUIDE — the registers below are fixed and must not drift:

{styleguide}

Repair rules:
- Output Korean only. No Chinese characters, no kana, no leftover source text.
- Keep it as short as spoken comic dialogue. Never add explanation.
- Apply the speaker's register exactly as the style guide and the stated problem require.
- Use the glossary's Korean targets.

Failed items — each line is:
  <key> [kind] speaker -> addressee : source
    previous: <previous Korean output>
    problem: <what the automated check found>

{items}
{extra}"""

# 위반 코드별 강한 지시. 일반적인 "한국어로 쓰라"는 말이 안 통한 실패가 있었으므로
# (`できたよー` → `出来了~` 가 2회 재요청을 버텼다) 코드별로 못을 박는다.
CODE_DIRECTIVES = {
    "residual_script": ("The source text is NOT Korean. Do not copy any source "
                        "characters. Every character you output must be a Hangul "
                        "syllable, a space, or punctuation."),
    "register_mismatch": ("Rewrite the sentence ending so the register matches. "
                          "합니다체 ends in -습니다/-ㅂ니다; 해요체 ends in -요; "
                          "반말 ends in -아/-어/-야/-지/-대/-래 and never in -요."),
    "not_translated": "Produce an actual Korean translation, not a copy of the source.",
    "length_anomaly": "Match the length of spoken comic dialogue. Do not explain.",
    "repetition": "Write it once. Do not repeat any phrase.",
    "glossary_violation": "Use the glossary's Korean target verbatim.",
    "leak": "Output only the translated line itself.",
    "empty_target": "Produce a Korean translation. Never leave it blank.",
    "judge": "Fix exactly the cited requirement. Change nothing else.",
}

# 회차마다 실제로 다른 요청이 되도록 지시를 바꾼다. temperature 0 은 그리디라
# 같은 프롬프트에 같은 출력이 나온다 — 회차를 반복해도 결과가 그대로였던 이유다.
ROUND_NUDGES = [
    "",
    "\nYour previous attempt was rejected. Choose a different wording this time, "
    "not a variation of the same phrasing.",
    "\nTwo attempts were rejected. Produce the simplest possible Korean line that "
    "satisfies the constraint, even if it loses nuance.",
]


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "requirement": {"type": "string"},
            "problem": {"type": "string"},
        },
        "required": ["key", "requirement", "problem"]}}},
    "required": ["findings"],
}

JUDGE_PROMPT = """You are reviewing someone else's Korean translation of a comic
chapter. You did not write it. Judge only whether it obeys the requirements below.

REQUIREMENTS:

{styleguide}

Additional requirements:
- Each speaker's Korean register must match the style guide, including the
  relational overrides toward specific addressees.
- Glossary terms must use the Korean target given above.
- Lines must read as spoken comic dialogue: short, no added explanation.
- Korean only. No source-language characters left in the output.

For each line below you are given the source and the Korean translation.

Report ONLY violations you can point at. For every finding you must fill
"requirement" by quoting the specific requirement it breaks — a character's
register, a glossary entry, or one of the additional requirements above. If you
cannot name the requirement, do not report the line. Do not report matters of
taste, and do not report a line merely because you would have worded it
differently. An empty findings list is the correct answer for a clean chapter.

Lines:
{items}"""


def judge(client, sheet, boxes, max_tokens, thinking):
    """새 컨텍스트에서 번역을 판정한다.

    같은 세션에서 "네가 쓴 걸 검토해라" 하는 것과 다르다. 이 호출은 이전 대화도,
    직전 시도도 보지 않는다 — 만든 기억이 없으므로 자기 출력을 방어할 동기가
    없고, 오판을 낳은 추론 사슬도 컨텍스트에 없다. 사실상 제3자 검토다.

    한계는 남는다. 모델의 한국어 감각 자체가 기울어 있으면 새 컨텍스트도 같은
    오류를 승인한다. 그래서 기계로 판정되는 것은 여전히 정규식이 맡고, 이쪽은
    --harsh 로 선택할 때만 돈다.

    근거 인용(requirement)을 강제하는 이유: "철저히 지켰는지 보라"고 물으면
    판정자는 존재를 정당화하려고 문제를 만들어 낸다. 어떤 요구사항을 어겼는지
    적게 하면 반증 가능해지고, 못 적은 지적은 버릴 수 있다.
    """
    items = []
    for key, t in boxes.items():
        tgt = (t.get("target") or "").strip()
        if not tgt:
            continue
        items.append(f"{key} [{t.get('kind')}] {t.get('speaker_name')} -> "
                     f"{t.get('addressee') or '-'}\n"
                     f"    source: {t.get('ocr')}\n"
                     f"    korean: {tgt}")
    res = ask(client, JUDGE_PROMPT.format(styleguide=sheet, items="\n".join(items)),
              JUDGE_SCHEMA, "findings", max_tokens, thinking)
    out = {}
    for f in res.get("findings") or []:
        key, req, prob = f.get("key"), (f.get("requirement") or "").strip(), \
            (f.get("problem") or "").strip()
        # 인용 없는 지적은 노이즈로 버린다.
        if key in boxes and req and prob:
            out.setdefault(key, []).append(("judge", f"{req} — {prob}"))
    return out


def main():
    p = argparse.ArgumentParser(description="번역 결과 검사 + 실패 박스 자동 재요청")
    p.add_argument("--translated-json", required=True)
    p.add_argument("--styleguide", required=True)
    p.add_argument("--out", help="수정 결과를 쓸 경로 (없으면 검사만 한다)")
    p.add_argument("--repair", action="store_true", help="걸린 박스를 다시 요청한다")
    p.add_argument("--thinking", action="store_true",
                   help="판정자에서 추론을 켠다. 판별 과제라 도움이 될 수 있다")
    p.add_argument("--harsh", action="store_true",
                   help="결정론적 검사를 통과한 뒤, 새 컨텍스트 판정자로 한 번 더 훑는다. "
                        "판정자 자체도 틀릴 수 있으니 기본은 끔")
    p.add_argument("--max-rounds", type=int, default=3, help="재요청 반복 상한")
    p.add_argument("--repair-temperature", type=float, default=0.25,
                   help="회차당 올릴 온도 (1회차는 0). 0 이면 그리디라 회차를 "
                        "반복해도 같은 출력이 나온다")
    p.add_argument("--config")
    p.add_argument("--model", help="stages 설정을 무시하고 이 모델을 쓴다")
    p.add_argument("--min-ratio", type=float, default=0.25)
    p.add_argument("--max-ratio", type=float, default=4.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args()

    doc = json.load(open(args.translated_json, encoding="utf-8"))
    sg = json.load(open(args.styleguide, encoding="utf-8"))
    sg_index = build_index(sg)
    glossary = sg.get("glossary") or []

    boxes = {}
    for pg in doc["pages"]:
        for t in pg["texts"]:
            boxes[f"p{pg['index']+1}_t{t['id']}"] = t

    def run_checks():
        found = {}
        for key, t in boxes.items():
            iss = check(t, sg_index, glossary, args.min_ratio, args.max_ratio)
            if iss:
                found[key] = iss
        return found

    issues = run_checks()
    total_boxes = sum(1 for t in boxes.values() if (t.get("target") or "").strip())
    print(f"박스 {total_boxes}개 검사 → 위반 {len(issues)}개")
    counts = {}
    for iss in issues.values():
        for code, _ in iss:
            counts[code] = counts.get(code, 0) + 1
    for code, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {code:20s} {n}")
    for key, iss in sorted(issues.items()):
        t = boxes[key]
        print(f"\n  {key} [{t.get('kind')}] {t.get('speaker_name')}")
        print(f"    원문: {t.get('ocr')}")
        print(f"    결과: {t.get('target')}")
        for code, msg in iss:
            print(f"    ✗ {code}: {msg}")

    sheet = render_styleguide(sg)
    repair_client = client_for("repair", args.config, args.model)
    judge_client = client_for("judge", args.config, args.model)

    if args.repair and issues:
        for rnd in range(1, args.max_rounds + 1):
            items = []
            for key, iss in sorted(issues.items()):
                t = boxes[key]
                codes = [c for c, _ in iss]
                directives = " ".join(CODE_DIRECTIVES[c] for c in codes
                                      if c in CODE_DIRECTIVES)
                items.append(
                    f"{key} [{t.get('kind')}] {t.get('speaker_name')} -> "
                    f"{t.get('addressee') or '-'} : {t.get('ocr')}\n"
                    f"    previous: {t.get('target')}\n"
                    f"    problem: {'; '.join(m for _, m in iss)}\n"
                    f"    required: {directives}")
            # 온도도 함께 올린다. 지시만 바꿔도 그리디는 같은 봉우리에 머무를 수 있다.
            temp = args.repair_temperature * (rnd - 1)
            nudge = ROUND_NUDGES[min(rnd - 1, len(ROUND_NUDGES) - 1)]
            print(f"\n[재요청 {rnd}회차] {len(items)}개 (temperature {temp:.2f})")
            try:
                res = ask(repair_client,
                          REPAIR_PROMPT.format(styleguide=sheet, items="\n".join(items),
                                               extra=nudge),
                          TRANSLATION_SCHEMA, "translations",
                          args.max_tokens, False, temperature=temp)
            except Exception as e:
                print(f"  재요청 실패: {e}")
                break
            fixed = reverted = 0
            for row in res["translations"]:
                key, tgt = row.get("key"), (row.get("target") or "").strip()
                if key not in boxes or not tgt:
                    continue
                t = boxes[key]
                prev = t.get("target")
                before = {c for c, _ in check(t, sg_index, glossary,
                                              args.min_ratio, args.max_ratio)}
                t["target"] = tgt
                now = {c for c, _ in check(t, sg_index, glossary,
                                           args.min_ratio, args.max_ratio)}
                # 개선이 없으면 원본을 남긴다. 없던 위반이 생기는 경우(`다 됐어~`
                # → `出来了よ~`)뿐 아니라, 같은 위반이 그대로 남는 경우도 되돌린다.
                # 나아지지 않은 교체를 받아들이면 멀쩡했던 표현만 잃는다.
                if now - before or now >= before:
                    t["target"] = prev
                    reverted += 1
                else:
                    t["target_before_repair"] = prev
                    fixed += 1
            after = run_checks()
            print(f"  {fixed}개 갱신, {reverted}개 되돌림 → 위반 {len(issues)} → {len(after)}")
            if not after or len(after) >= len(issues):
                # 줄지 않으면 반복해도 나아지지 않는다. 남은 것은 그대로 통과시킨다.
                issues = after
                break
            issues = after
        for key in issues:
            boxes[key]["validation_failed"] = [c for c, _ in issues[key]]
        print(f"\n최종 잔여 위반 {len(issues)}개 — 그대로 통과시킨다"
              f"{': ' + ', '.join(sorted(issues)) if issues else ''}")

    # ── --harsh: 결정론적 검사가 끝난 뒤 새 컨텍스트 판정자로 한 번 더 ──────
    # 결정론적 층 다음에 두는 이유: 정규식으로 잡히는 것을 판정자에게 물으면
    # 비싸고 덜 정확하다. 판정자는 규칙으로 표현할 수 없는 것만 맡는다.
    if args.harsh:
        print("\n[--harsh] 새 컨텍스트 판정자 실행")
        try:
            found = judge(judge_client, sheet, boxes, args.max_tokens, args.thinking)
        except Exception as e:
            print(f"  판정 실패, 건너뜀: {e}")
            found = {}
        print(f"  근거를 댄 지적 {len(found)}건")
        for key, iss in sorted(found.items()):
            print(f"  {key}: {boxes[key].get('target')}")
            for _, msg in iss:
                print(f"    ⚠ {msg}")
        if found and args.repair:
            items = []
            for key, iss in sorted(found.items()):
                t = boxes[key]
                items.append(
                    f"{key} [{t.get('kind')}] {t.get('speaker_name')} -> "
                    f"{t.get('addressee') or '-'} : {t.get('ocr')}\n"
                    f"    previous: {t.get('target')}\n"
                    f"    problem: {'; '.join(m for _, m in iss)}\n"
                    f"    required: {CODE_DIRECTIVES['judge']}")
            try:
                res = ask(repair_client,
                          REPAIR_PROMPT.format(styleguide=sheet,
                                               items="\n".join(items), extra=""),
                          TRANSLATION_SCHEMA, "translations",
                          args.max_tokens, False)
            except Exception as e:
                print(f"  판정 후 재요청 실패: {e}")
                res = {"translations": []}
            applied = rejected = 0
            for row in res.get("translations") or []:
                key, tgt = row.get("key"), (row.get("target") or "").strip()
                if key not in boxes or not tgt:
                    continue
                t = boxes[key]
                prev = t.get("target")
                t["target"] = tgt
                # 판정자 지적을 고치려다 결정론적 검사를 깨면 되돌린다.
                # 확실한 규칙이 불확실한 판정보다 우선한다.
                if check(t, sg_index, glossary, args.min_ratio, args.max_ratio):
                    t["target"] = prev
                    rejected += 1
                else:
                    t["target_before_harsh"] = prev
                    applied += 1
            print(f"  {applied}개 반영, {rejected}개 되돌림 "
                  f"(결정론적 검사를 깨는 수정은 받지 않는다)")
            issues = run_checks()
            print(f"  결정론적 위반 최종 {len(issues)}개")
        for key, iss in found.items():
            boxes[key]["judge_findings"] = [m for _, m in iss]

    if args.out:
        doc["validation"] = {"remaining": {k: [c for c, _ in v] for k, v in issues.items()}}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
