#!/usr/bin/env python3
"""2단 — 챕터 전체를 한 컨텍스트에 넣어 말투를 확정하고 일괄 번역한다.

왜 페이지 단위로 번역하지 않는가:
  반말/존댓말이 오락가락하는 것을 막으려면 "매 페이지 잘 부탁한다"로는 안 된다.
  번역을 시작하기 **전에** 인물별 말투를 한 번 확정하고, 그 시트를 전 페이지에
  똑같이 적용해야 한다. 그래서 이 단계는 두 번 호출한다.
    1) 시트 만들기 — 시각 서술자를 실제 이름으로 해소하고 말투를 결정
    2) 그 시트를 물려 전체 번역

이미지가 아니라 텍스트로 컨텍스트를 채우는 이유:
  페이지 이미지는 최소 1024 image token, 실질 1~2K 다. 10장이면 벌써 10~20K 를
  먹는다. 반면 화자 라벨이 붙은 챕터 전사는 66개 박스가 1.5K 토큰이다. 일관성에
  필요한 것은 그림이 아니라 "누가 무슨 말을 했는지의 전체 목록"이므로, 텍스트로
  채우는 쪽이 같은 컨텍스트로 훨씬 멀리 간다.

시트는 챕터가 아니라 **작품** 단위로 저장한다 (--styleguide). 다음 권에서 같은
파일을 물려주면 말투와 용어가 이어진다. 스키마는 CarrotMangaTranslator 의
workContextTypes.ts 를 따랐고, 세 가지를 더했다:
  - 박스 단위 화자 (Carrot 은 페이지↔인물만 잇는다)
  - 관계형 말투 registerTo (메이드가 주인에겐 극존대, 동료에겐 반말인 구조)
  - 페이지별 원문 언어 (소재가 중/일 혼재다)
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from backend import client_for  # noqa: E402
from register_cues import (contradictions, evidence_by_speaker,  # noqa: E402
                            render_evidence)

SPEECH_STYLES = ["neutral", "polite", "casual", "rough", "childish",
                 "elderly", "formal", "custom"]

STYLEGUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "displayName": {"type": "string"},
                "sourceNames": {"type": "array", "items": {"type": "string"}},
                "targetName": {"type": "string"},
                "visualDescriptors": {"type": "array", "items": {"type": "string"}},
                "speechStyle": {"type": "string", "enum": SPEECH_STYLES},
                "customSpeechStyle": {"type": "string"},
                "registerTo": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"toward": {"type": "string"},
                                   "koreanEnding": {"type": "string"},
                                   "reason": {"type": "string"}},
                    "required": ["toward", "koreanEnding"]}},
                "note": {"type": "string"},
            },
            "required": ["displayName", "targetName", "visualDescriptors",
                         "speechStyle", "registerTo"]}},
        "glossary": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "target": {"type": "string"},
                "category": {"type": "string",
                             "enum": ["character", "alias", "place", "term",
                                      "honorific", "other"]},
                "note": {"type": "string"}},
            "required": ["source", "target", "category"]}},
        "rules": {
            "type": "object",
            "properties": {
                "honorifics": {"type": "string", "enum": ["preserve", "adapt", "drop"]},
                "sfxMode": {"type": "string", "enum": ["preserve", "translate", "note"]},
                "defaultTone": {"type": "string", "enum": ["natural_korean", "literal"]}},
            "required": ["honorifics", "sfxMode", "defaultTone"]},
        "chapterSummary": {"type": "string"},
    },
    "required": ["characters", "glossary", "rules", "chapterSummary"],
}

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "target": {"type": "string"},
                       "speaker": {"type": "string"}, "note": {"type": "string"}},
        "required": ["key", "target"]}}},
    "required": ["translations"],
}

STYLEGUIDE_PROMPT = """You are preparing to translate a comic chapter into Korean.

MEASURED REGISTER EVIDENCE — computed directly from the source text by counting
honorific and plain markers. This is observation, not inference. It OVERRIDES any
social reasoning about age, rank or closeness. If a character is listed as using
honorific speech, you must not assign them plain speech toward anyone who appears
in these lines, no matter how peer-like the relationship looks.

{evidence}


Below is the full chapter transcript. Each line is:
  <key> [lang/kind] speaker -> addressee : source text

{cast_block}The speakers are often visual descriptions ("silver-haired maid") because names
were not yet known. Your job is to build a style guide BEFORE any translation.

Field meanings — get these the right way round:
  "displayName"        the character's name as it appears in the SOURCE
                       (e.g. "クロエ"). Never Korean.
  "sourceNames"        other spellings of that source name, including honorific
                       forms found in the text (e.g. "クロエさん").
  "targetName"         the KOREAN name to use in the translation (e.g. "클로에").
                       Always Hangul. Never the source script.
  "visualDescriptors"  the transcript's placeholder descriptions for this person
                       (e.g. "silver-haired maid").

1. Resolve identities. Characters address each other by name in the dialogue —
   use that to map each visual descriptor to a real name. Put every descriptor
   that refers to the same person into that character's "visualDescriptors".
2. Decide each character's Korean speech register. This is the point of the
   exercise: the register must be fixed once and never drift.
   - "speechStyle" is their default.
   - "registerTo" holds relational overrides: a servant may use the highest
     deference toward a master and plain speech toward peers. For each, give
     "koreanEnding" as the concrete Korean ending to use (e.g. "합니다체",
     "해요체", "해체(반말)") and a short "reason".
   - Base this on evidence in the source when the source marks register
     (Japanese keigo, humble forms, sentence-final particles). When the source
     does NOT mark register (Chinese, English), infer it from the relationships
     and situation, and say so in "reason".
3. Extract a glossary of names, places and recurring terms with Korean targets.
4. Set the work-level rules.
5. Write a two or three sentence chapterSummary for continuity.

Transcript:
{transcript}"""

SHEET_FIX_PROMPT = """The style guide you produced contradicts the measured register
evidence from the source text. The evidence is authoritative — it was counted from
the source, not inferred.

MEASURED EVIDENCE:
{evidence}

CONTRADICTIONS FOUND:
{contradictions}

Produce the corrected style guide. Keep everything that was not contradicted.
A character who uses honorific speech in the source must be given an honorific
Korean ending (합니다체 / 해요체) toward the characters in those lines — never 반말.

Transcript:
{transcript}"""

TRANSLATE_PROMPT = """Translate this comic chapter into natural Korean.
{story}
STYLE GUIDE — follow it exactly. The registers below were fixed in advance so
that they never drift across the chapter. Do not re-decide them per line.

{styleguide}

Rules:
- Translate EVERY key listed in the transcript. Reply with the same keys.
- Apply the speaker's register from the style guide. If a "registerTo" entry
  matches the addressee, that overrides the default speechStyle.
- Use the glossary's Korean targets for every name and term.
- Comic dialogue: keep it short and spoken. Do not pad or explain.
- kind=sfx: render as a Korean onomatopoeia. kind=narration: plain narration
  register, not dialogue. kind=thought: the speaker's own inner voice.
- Put anything you were unsure about in "note", and leave "target" as your best
  attempt anyway.

Transcript:
{transcript}"""


def load_lines(doc, include_empty=False):
    """챕터 전사를 한 줄씩, 안정적인 key 와 함께 만든다.

    key 는 페이지·박스 번호에서 만든다. 모델이 만든 이름이 아니라 우리 좌표에
    묶인 식별자여야 번역 결과를 원래 박스로 되돌릴 수 있다.
    """
    lines, keys = [], []
    for pg in doc["pages"]:
        for t in pg["texts"]:
            text = (t.get("ocr") or "").strip()
            if not text and not include_empty:
                continue
            key = f"p{pg['index']+1}_t{t['id']}"
            lang = t.get("lang") or pg.get("source_lang") or "?"
            kind = t.get("kind") or "?"
            spk = t.get("speaker_name") or "unknown"
            to = t.get("addressee") or "-"
            lines.append(f"{key} [{lang}/{kind}] {spk} -> {to} : {text}")
            keys.append(key)
    return lines, keys


def ask(client, prompt, schema, name, max_tokens, thinking, temperature=0.0):
    """단계별 모델 호출. HTTP 조립과 추론 제어는 backend.Client 가 맡는다.

    라우터 모드에서는 요청에 model 이 없으면 400 이 나므로 직접 post 하면 안 된다.
    """
    return client.chat(prompt, schema=schema, schema_name=name,
                       thinking=thinking, max_tokens=max_tokens,
                       temperature=temperature)


def render_styleguide(sg):
    """모델에 다시 먹일 시트를 사람이 읽을 수 있는 형태로 압축한다.

    JSON 을 그대로 넣지 않는 이유: 지시문 안에서는 산문 쪽이 지켜질 확률이 높고,
    토큰도 덜 먹는다.
    """
    out = ["[등장인물]"]
    for c in sg["characters"]:
        desc = ", ".join(c.get("visualDescriptors") or []) or "-"
        line = (f"- {c['displayName']} → 한국어 표기 '{c['targetName']}' "
                f"| 기본 말투 {c['speechStyle']}"
                f"{': ' + c['customSpeechStyle'] if c.get('customSpeechStyle') else ''}"
                f" | 시각 서술자: {desc}")
        out.append(line)
        for r in c.get("registerTo") or []:
            out.append(f"    · {r['toward']} 에게는 {r['koreanEnding']}"
                       f"{'  (' + r['reason'] + ')' if r.get('reason') else ''}")
        if c.get("note"):
            out.append(f"    · 비고: {c['note']}")
    out.append("\n[용어집]")
    for g in sg["glossary"]:
        out.append(f"- {g['source']} → {g['target']} ({g['category']})"
                   f"{'  ' + g['note'] if g.get('note') else ''}")
    r = sg["rules"]
    out.append(f"\n[규칙] 호칭 {r['honorifics']} / 효과음 {r['sfxMode']} / 어조 {r['defaultTone']}")
    out.append(f"\n[줄거리] {sg['chapterSummary']}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="챕터 단위 말투 확정 + 일괄 번역")
    p.add_argument("--page-json", required=True, help="read_page.py 출력")
    p.add_argument("--out", required=True, help="번역을 채운 JSON 경로")
    p.add_argument("--styleguide", required=True,
                   help="작품 단위 스타일 시트 JSON. 없으면 만들고, 있으면 물려 쓴다")
    p.add_argument("--rebuild-styleguide", action="store_true",
                   help="시트가 이미 있어도 다시 만든다")
    p.add_argument("--config")
    p.add_argument("--sheet-model", help="시트 단계 모델 강제")
    p.add_argument("--translate-model", help="번역 단계 모델 강제")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--thinking", action="store_true",
                   help="번역 패스에서 추론을 켠다")
    p.add_argument("--no-sheet-thinking", action="store_true",
                   help="시트 생성에서 추론을 끈다 (기본은 켬 — 추론 과제다)")
    p.add_argument("--sheet-max-tokens", type=int, default=24576,
                   help="시트 호출 전용 토큰 예산. 추론을 켜면 많이 먹는다")
    p.add_argument("--sheet-fix-rounds", type=int, default=2,
                   help="증거와 모순될 때 시트를 다시 만들 횟수")
    args = p.parse_args()

    doc = json.load(open(args.page_json, encoding="utf-8"))
    lines, keys = load_lines(doc)
    if not lines:
        print("전사된 텍스트가 없습니다. 먼저 read_page.py 를 돌리세요", file=sys.stderr)
        return 2
    transcript = "\n".join(lines)

    sheet_client = client_for("styleguide", args.config, args.sheet_model)
    tr_client = client_for("translate", args.config, args.translate_model)
    if not sheet_client.health():
        print(f"백엔드에 연결할 수 없습니다: {sheet_client.base_url}", file=sys.stderr)
        return 3
    print(f"시트: {sheet_client.name} | 번역: {tr_client.name}")

    print(f"박스 {len(lines)}개, 전사 {len(transcript)}자")

    # ── 1) 스타일 시트 ────────────────────────────────────────────────────
    if os.path.exists(args.styleguide) and not args.rebuild_styleguide:
        sg = json.load(open(args.styleguide, encoding="utf-8"))
        print(f"기존 시트 재사용: {args.styleguide} (인물 {len(sg['characters'])}명)")
    else:
        t = time.time()
        # 원문에서 경어 표지를 세어 증거로 준다. 모델이 사회적 추론으로 이걸
        # 뒤집는 일이 실제로 있었다 ("동료이고 나이가 비슷하니 반말" → 원문은
        # 극존대). 증거를 주면 추론할 여지가 없어진다.
        # build_cast.py 가 이미 인물을 확정했으면 시트가 다시 정하게 두지 않는다.
        # 두 단계가 독립적으로 신원을 해소하면 결과가 어긋난다 — 실제로 한쪽은
        # `cat` 을 Pirate Girl 로, 다른 쪽은 Kumin 으로 합친 적이 있다.
        settled = (doc.get("cast") or {}).get("characters") or []
        cast_block = ""
        if settled:
            lines = []
            for c in settled:
                al = ", ".join(a for a in (c.get("aliases") or [])
                               if a.lower() != c["id"].lower())
                lines.append(f"  {c['id']}" + (f"  (also appears as: {al})" if al else ""))
            cast_block = ("SETTLED CAST — these identities were already resolved from the\n"
                          "full chapter. Use exactly these as displayName. Do not merge them,\n"
                          "split them, or invent new characters.\n\n"
                          + "\n".join(lines) + "\n\n")
            print(f"확정된 인물 {len(settled)}명을 시트에 물려줍니다")

        acc = evidence_by_speaker(doc)
        evidence = render_evidence(acc)
        print("측정된 말투 증거:\n" + evidence)

        # 시트 생성은 관계를 저울질하는 추론 과제다. 전사와 달리 추론을 켜는 것이
        # 이득이고, 챕터당 한 번뿐이라 비용도 무의미하다.
        # 추론을 켜면 토큰을 많이 먹는다. 예산이 모자라면 본문이 비어서 돌아오므로
        # (finish_reason=length) 시트 호출은 별도 예산을 쓰고, 그래도 모자라면
        # 추론을 끄고 한 번 더 시도한다. 실패로 끝내는 것보다 낫다.
        try:
            sg = ask(sheet_client,
                     STYLEGUIDE_PROMPT.format(transcript=transcript, evidence=evidence,
                                          cast_block=cast_block),
                     STYLEGUIDE_SCHEMA, "styleguide",
                     args.sheet_max_tokens, not args.no_sheet_thinking)
        except RuntimeError as e:
            if "본문이 비었" not in str(e) or args.no_sheet_thinking:
                raise
            print(f"  시트 생성이 예산을 소진했습니다 ({e}). 추론을 끄고 재시도합니다")
            sg = ask(sheet_client,
                     STYLEGUIDE_PROMPT.format(transcript=transcript, evidence=evidence,
                                          cast_block=cast_block),
                     STYLEGUIDE_SCHEMA, "styleguide",
                     args.sheet_max_tokens, False)

        # 자기 검증 — 다만 "알아서 확인해라"가 아니라 계산된 모순을 들이민다.
        # 막연한 자기 비판은 같은 근거로 같은 오답을 반복한다.
        for attempt in range(1, args.sheet_fix_rounds + 1):
            bad = contradictions(sg, acc)
            if not bad:
                break
            print(f"  [시트 교정 {attempt}회차] 증거와 모순 {len(bad)}건")
            for name, msg in bad:
                print(f"    ✗ {name}: {msg}")
            try:
                sg = ask(sheet_client, SHEET_FIX_PROMPT.format(
                             evidence=evidence, transcript=transcript,
                             contradictions="\n".join(f"- {n}: {m}" for n, m in bad)),
                         STYLEGUIDE_SCHEMA, "styleguide",
                         args.sheet_max_tokens, not args.no_sheet_thinking)
            except Exception as e:
                print(f"    교정 실패, 그대로 진행: {e}")
                break
        left = contradictions(sg, acc)
        if left:
            print(f"  잔여 모순 {len(left)}건 — 그대로 진행한다")
        sg["schemaVersion"] = 1
        sg["registerEvidence"] = {k: {kk: vv for kk, vv in v.items()}
                                  for k, v in acc.items()}
        os.makedirs(os.path.dirname(os.path.abspath(args.styleguide)), exist_ok=True)
        with open(args.styleguide, "w", encoding="utf-8") as fh:
            json.dump(sg, fh, ensure_ascii=False, indent=1)
        print(f"시트 생성 {time.time()-t:.1f}초 → 인물 {len(sg['characters'])}명, "
              f"용어 {len(sg['glossary'])}개 → {args.styleguide}")

    sheet = render_styleguide(sg)
    print("\n" + sheet + "\n")

    # ── 2) 일괄 번역 ──────────────────────────────────────────────────────
    t = time.time()
    # build_cast.py 가 뽑아둔 줄거리를 실어준다. 개별 대사만 보면 알 수 없는
    # 관계와 의도가 여기서 들어온다 — 짧은 대사의 말투를 정하는 근거가 된다.
    story = doc.get("story")
    story_block = f"\nCHAPTER CONTEXT — what happens in this chapter:\n{story}\n" if story else ""
    if story:
        print(f"줄거리를 번역 맥락으로 사용합니다 ({len(story)}자)")
    res = ask(tr_client,
              TRANSLATE_PROMPT.format(styleguide=sheet, transcript=transcript,
                                      story=story_block),
              TRANSLATION_SCHEMA, "translations", args.max_tokens, args.thinking)
    by_key = {r["key"]: r for r in res["translations"] if r.get("key")}
    print(f"번역 {time.time()-t:.1f}초, {len(by_key)}/{len(keys)}개 응답")

    missing = []
    for pg in doc["pages"]:
        for txt in pg["texts"]:
            key = f"p{pg['index']+1}_t{txt['id']}"
            if key not in keys:
                continue
            row = by_key.get(key)
            if row is None:
                missing.append(key)
                continue
            txt["target"] = row.get("target")
            if row.get("note"):
                txt["target_note"] = row["note"]

    doc["styleguide_path"] = os.path.abspath(args.styleguide)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    if missing:
        print(f"⚠️  번역 누락 {len(missing)}개: {missing[:10]}")
    print(f"→ {args.out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
