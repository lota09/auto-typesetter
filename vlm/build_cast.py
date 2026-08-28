#!/usr/bin/env python3
"""인물·스토리 확정 패스 — 챕터 전사를 한 번에 보고 화자 식별자를 정규화한다.

왜 필요한가:
  페이지 판독은 페이지를 **한 장씩 서로 독립적으로** 본다. 그래서 같은 인물에
  매번 다른 라벨이 붙는다. 실측에서 한 작품의 화자 목록이 이렇게 흩어졌다 —
  Yun / male / female / boy / girl with horns / unknown. 말투 시트가 화자
  식별자를 키로 잡으므로, 한 인물이 셋으로 쪼개지면 서로 다른 말투 세 개가
  배정되고 그게 곧 말투 붕괴다.

왜 이미지가 아니라 텍스트로 하는가 (실측 근거):
  "여러 페이지를 한꺼번에 보여주고 인물을 확정한다"를 먼저 재봤다. 이 모델은
  이미지가 늘면 장당 토큰 예산을 나눠 쓰고(1장 1077 → 3장 711 → 10장 530),
  **4장을 넘으면 위치 지목이 무너진다** (페이지에 번호를 찍어 보내도 5장에서
  틀렸다). 10페이지를 한 번에 볼 수 없으므로 배치로 쪼개야 하고, 그러면 배치
  간 인물을 다시 맞추는 원래 문제가 규모만 커진 채 돌아온다.

  반면 전사 텍스트는 챕터 전체가 1.5K 토큰이다. 그리고 인물 확정에 필요한
  신호가 실제로 텍스트에 있다 — 호칭(`だんちょ`, `クロエさん`), 순서 교대,
  경어 전환. 이미지 지목 문제가 존재하지 않는다.

이 패스는 페이지를 순차로 읽는 도중이 아니라 **전사가 다 모인 뒤** 판단하므로,
초기 오류가 뒤로 전파되지 않는다. 누적 명부 방식이 실패한 지점이 그것이었다.

출력 크기는 챕터 길이에 비례하면 안 된다:
  입력(전사)은 길어도 된다 — 77페이지가 32K자, ctx 65536 안에 넉넉히 들어간다.
  망가지는 것은 **출력**이다. 처음에는 "speaker 를 바꿔야 하는 모든 key"를
  reassign 에 나열하게 했는데, 그러면 항목 수가 대사 줄 수를 따라간다.
  692줄짜리 챕터에서 26,228자에서 잘려 죽었다 (max_tokens 16384).

  그런데 그 목록은 **없어도 된다.** 아래 `alias_to_id` 가 별칭 → 확정 id 사전을
  만들어 모든 줄을 기계적으로 덮는다. 모델이 할 일은 별칭을 묶는 것(인물 수만큼)
  이지 줄마다 답을 쓰는 것이 아니다. 그래서 reassign 은 **기계적 치환이 틀리는
  줄만** 받는다. 이제 출력 크기는 O(인물 수)다.

  같은 이유로 "pages"(페이지마다 등장 인물 목록)도 뺐다. 페이지 수에 비례해
  커지는데 읽는 곳이 어디에도 없었다.

스토리 요약도 같은 응답에서 받는다. 번역 단계에 실어주면 맥락이 붙고, 작품
단위 파일에 누적하면 다음 권으로 이어진다.
"""

import argparse
import json
import os
import re
import sys

# 모델이 notPeople 에 박스 키를 섞어 넣는다. 라벨과 키를 형태로 가른다.
KEY_RE = re.compile(r"^p\d+_t\d+$")
# kind 에 따라 정해진 예약 라벨. 사람이 아니지만 unknown 도 아니다.
RESERVED_BY_KIND = {"narration": "narration", "sfx": "sfx"}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progress as PROG  # noqa: E402
from backend import bound_schema, client_for, usage_line  # noqa: E402

CAST_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "maxItems": 40, "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "sourceNames": {"type": "array", "maxItems": 8,
                                "items": {"type": "string"}},
                "aliases": {"type": "array", "maxItems": 20,
                            "items": {"type": "string"}},
                "evidence": {"type": "string", "maxLength": 400},
            },
            "required": ["id", "aliases", "evidence"]}},
        "notPeople": {"type": "array", "maxItems": 60, "items": {"type": "string"}},
        "reassign": {"type": "array", "maxItems": 120, "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "speaker": {"type": "string"}},
            "required": ["key", "speaker"]}},
        "story": {"type": "string", "maxLength": 1500},
    },
    "required": ["characters", "notPeople", "reassign", "story"],
}
# 남은 문자열·배열에 기본 상한을 채운다. 명시한 evidence(400)·story(1500)는 보존된다.
bound_schema(CAST_SCHEMA)

PROMPT = """You are consolidating the cast of a comic chapter before translation.

Below is the full chapter transcript, in reading order. Each line is:
  <key> [lang/kind] provisional_speaker -> addressee : source text

The provisional speaker labels were assigned one page at a time by a reader that
could not see the other pages. They are therefore **unreliable and inconsistent**:
the same person may appear as "Yun" on one page and "boy" or "male" on another,
and a label may even be an ordinary word mistaken for a name.

Produce:

1. "characters" — the real cast. For each person give:
     "id"           a single canonical identifier, used everywhere from now on.
                    Prefer a name characters actually use to address them.
                    Only if no name exists, use a short stable description.
     "sourceNames"  names as spelled in the source text, if any
     "aliases"      **every provisional label from the transcript that refers to
                    this person** — this is how the labels get merged
     "evidence"     the line(s) that justify this identity, e.g. someone is
                    addressed by that name, or answers to it
2. "notPeople" — provisional labels that are not characters at all: ordinary
   words misread as names, sound effects, narration markers.
3. "reassign" — **exceptions only.** Every provisional label you listed under
   "aliases" is already remapped mechanically, line by line, by the caller.
   So do NOT restate those. List a key here only when that mechanical remap
   would be *wrong* for that one line — for example the reader attributed the
   line to the wrong person entirely, or the line is narration that was given a
   speaker. Use "narration" for narration and "unknown" only when genuinely
   undecidable. Most chapters need very few entries here; an empty list is a
   perfectly good answer.
4. "story" — a summary of what happens in this chapter, in reading order,
   naming characters by their canonical ids. Two to five sentences. This is
   carried into translation as context, so include who wants what and how
   relationships stand.

Rules for merging:
- Turn-taking is strong evidence. A question and its answer are different people.
- A term of address tells you who the *other* party is, not the speaker.
- Do not merge two people just because their labels look similar. If the
  transcript shows them speaking to each other, they are distinct.
- Do not invent characters who never speak.

Transcript:
{transcript}"""


def build_transcript(doc):
    lines, keys = [], {}
    for pg in doc["pages"]:
        for t in pg["texts"]:
            text = (t.get("ocr") or "").strip()
            if not text:
                continue
            key = f"p{pg['index']+1}_t{t['id']}"
            keys[key] = t
            lines.append(f"{key} [{t.get('lang') or '?'}/{t.get('kind') or '?'}] "
                         f"{t.get('speaker_name') or 'unknown'} -> "
                         f"{t.get('addressee') or '-'} : {text}")
    return "\n".join(lines), keys


def main():
    p = argparse.ArgumentParser(description="인물 정규화 + 스토리 추출")
    p.add_argument("--read-json", required=True, help="read_page/merge 출력")
    p.add_argument("--out", required=True)
    p.add_argument("--log", help="이 경로에 전체 로그를 덧붙인다 "
                   "(터미널은 짧게, 파일은 빠짐없이)")
    p.add_argument("--config")
    p.add_argument("--model", help="stages 설정을 무시하고 이 모델을 쓴다")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--no-thinking", action="store_true",
                   help="추론을 끈다. 이 단계는 관계를 저울질하는 추론 과제다")
    args = p.parse_args()

    PROG.open_log(getattr(args, 'log', None))
    doc = json.load(open(args.read_json, encoding="utf-8"))
    transcript, keys = build_transcript(doc)
    if not transcript:
        print("전사가 비어 있습니다", file=sys.stderr)
        return 2

    client = client_for("styleguide", args.config, args.model)
    if not client.health():
        print(f"백엔드에 연결할 수 없습니다: {client.base_url}", file=sys.stderr)
        return 3
    print(f"모델: {client.name} | 대사 {len(keys)}줄, {len(transcript)}자")

    PROG.prompt_block("④ 인물·스토리", PROMPT.format(transcript="<챕터 전사 전문>"))
    res = client.chat(PROMPT.format(transcript=transcript), schema=CAST_SCHEMA,
                      schema_name="cast", thinking=not args.no_thinking,
                      max_tokens=args.max_tokens)

    # 별칭 → 확정 id 사전. reassign 이 빠뜨린 줄도 이걸로 메운다.
    alias_to_id = {}
    for c in res["characters"]:
        for a in [c["id"], *(c.get("aliases") or []), *(c.get("sourceNames") or [])]:
            if a:
                alias_to_id[a.strip().lower()] = c["id"]

    # 별칭이 실제로 **화자로 등장한 적 있는지** 확인한다.
    # `cat` 을 Pirate Girl 로 합친 적이 있는데, 근거로 제시된 줄에서 `cat` 은
    # 화자가 아니라 **수신자**였다. 호칭은 상대를 알려주는 것이지 화자를 알려주는
    # 것이 아니다. 전사에 화자로 한 번도 안 나온 별칭은 병합 근거가 될 수 없다.
    seen_speakers = {(t.get("speaker_name") or "").strip().lower()
                     for t in keys.values()}
    seen_speakers.discard("")
    for c in res["characters"]:
        bogus = [a for a in (c.get("aliases") or [])
                 if a.strip().lower() not in seen_speakers
                 and a.strip().lower() != c["id"].strip().lower()]
        if bogus:
            c["aliases"] = [a for a in (c.get("aliases") or []) if a not in bogus]
            c.setdefault("rejected_aliases", []).extend(bogus)
            print(f"  ⚠ {c['id']}: 화자로 등장한 적 없는 별칭 제거 {bogus}")

    # notPeople 을 그대로 믿으면 안 된다. 실측에서 17개 중 8개가 **방금 인물로
    # 병합한 별칭**이었고 6개는 박스 키(p1_t4)였다. 모델이 모순된 목록을 낸다.
    # 별칭으로 쓰인 것과 키 형태를 기계적으로 걸러낸다.
    not_people = set()
    for n in (res.get("notPeople") or []):
        n = (n or "").strip()
        if not n or KEY_RE.match(n) or n.lower() in alias_to_id:
            continue
        not_people.add(n.lower())

    explicit = {r["key"]: r["speaker"] for r in (res.get("reassign") or [])}

    changed = dropped = kept = 0
    for key, t in keys.items():
        old = (t.get("speaker_name") or "").strip()
        kind = t.get("kind")
        new = explicit.get(key) or alias_to_id.get(old.lower())

        # 나레이션·효과음은 "사람이 아님"이 맞지만 올바른 라벨은 kind 에 맞는
        # 예약어다. unknown 으로 떨어뜨리면 정보를 잃는다 — 실제로 나레이션
        # 6줄이 그렇게 망가졌다.
        if kind in RESERVED_BY_KIND:
            new = RESERVED_BY_KIND[kind]
        elif new is None and old.lower() in not_people:
            new = "unknown"
            dropped += 1

        if new is None or new == old:
            kept += 1
            continue
        t["speaker_provisional"] = old
        t["speaker_name"] = new
        changed += 1

    doc["cast"] = {"characters": res["characters"], "notPeople": res.get("notPeople")}
    doc["story"] = res["story"]

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    print(f"\n[확정 인물 {len(res['characters'])}명]")
    for c in res["characters"]:
        al = ", ".join(a for a in (c.get("aliases") or []) if a.lower() != c["id"].lower())
        print(f"  {c['id']}" + (f"  ← {al}" if al else ""))
        if c.get("evidence"):
            print(f"      근거: {c['evidence'][:110]}")
    if res.get("notPeople"):
        print(f"[인물 아님] {', '.join(res['notPeople'])}")
    print(f"\n[스토리]\n{res['story']}")
    print(f"\n라벨 변경 {changed}줄 / 유지 {kept}줄 / 인물 아님 처리 {dropped}줄 → {args.out}")
    print(usage_line(client))
    return 0


if __name__ == "__main__":
    sys.exit(main())
