#!/usr/bin/env python3
"""1단 — 페이지 하나를 통째로 VLM 에 주고 박스별 원문·화자·언어를 한 번에 받는다.

크롭을 하나씩 보내는 read_texts.py 와의 차이:
  전사 정확도는 크롭이 유리하다 (글자가 크게 보인다). 하지만 **화자**는 크롭만
  봐서는 알 수 없다. 누가 말하는지는 말풍선 꼬리가 누구를 가리키는지, 대사 내용이
  누구를 부르는지, 앞뒤 대사가 어떻게 이어지는지를 봐야 결정된다. 그래서 화자
  판정은 페이지 전체를 보는 이 패스가 맡는다.

Magi 대신 VLM 이 화자를 정하는 이유:
  Magi 는 기하학과 외형만 본다. 게다가 클러스터 ID 가 페이지 로컬이고, 입력을
  흑백으로 바꿔서 (이 소재의 인물 구분 단서인) 머리색을 버린다. VLM 은 대사
  내용까지 읽으므로 "クロエさん" 이라고 부르는 대사 다음 줄이 클로에의 응답이라는
  것을 안다. 그리고 페이지 로컬 번호가 아니라 **실제 이름**을 돌려준다.

박스 번호를 그려 넣는 이유:
  VLM 의 좌표 출력(그라운딩)은 몇 px 씩 틀리고, 우리는 지우고 다시 얹을 영역이라
  픽셀 단위여야 한다. 그래서 좌표는 Magi 것을 쓰고, VLM 에는 "몇 번 박스" 로만
  참조하게 한다. 의미 판단을 정확한 기하학에 묶는 방법이다.

입력: magi_worker.py 가 낸 JSON
출력: 같은 구조 + texts[].ocr / speaker_name / is_dialogue, pages[].source_lang
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progress as PROG  # noqa: E402
from backend import bound_schema, client_for  # noqa: E402

INSTRUCTION = """You are transcribing a comic page for translation.

The page image has numbered text regions outlined in red, each labelled with its
id. {n} regions: ids {ids}.

For EVERY id, report:
  "id"          the region id (integer)
  "text"        verbatim transcription. Vertical text: read top-to-bottom, and
                with multiple columns read the RIGHTMOST column first. Do not
                translate. Never convert between Chinese, Japanese and Korean
                characters. Use "" if the region has no legible text.
  "lang"        script of this region: "ja", "zh", "en", or "other"
  "speaker"     who utters it. Prefer a name that appears in the page's dialogue
                (e.g. a character addressed as "X" elsewhere). If no name is
                known, use a short stable visual description such as
                "silver-haired maid". Use "narration" for caption boxes that are
                not spoken by anyone, and "unknown" only if truly undecidable.
  "kind"        "dialogue"   spoken aloud by a character
                "thought"    a character's inner voice
                "narration"  caption boxes that tell the story
                "sfx"        onomatopoeia
                "incidental" text that exists inside the story world but nobody
                             utters: shop signs, street signs, electronic
                             displays, product labels, book covers, letters
                "credits"    text that is NOT part of the story at all: author
                             or artist names, the work's title, chapter number,
                             page number, website URLs, licence notices, lists
                             of donors or supporters, publisher marks
                For "incidental" and "credits", set "speaker" to "none".
  "addressee"   who it is said to, same naming rules, or "" if unclear

Use the speech-bubble tails, who is facing whom, and the flow of the conversation
to decide the speaker. Keep each speaker's name spelled identically everywhere.

Be strict about "credits": a page corner carrying the artist's name, a URL, a
licence line or a donor list is not something a character said, and it must not
be given a speaker.

ALSO look for text the red boxes MISSED. An automatic detector drew those boxes and
it does miss things — whole caption panels, small bubbles, text laid over artwork.
Report each missed region in "missed" with:
  "text"  what it says, transcribed the same way as above
  "kind"  same categories as above
  "box"   [x1, y1, x2, y2] as PER-MILLE of the page (0-1000), left-to-right and
          top-to-bottom, tight around the text
Only report a region that is NOT already inside one of the numbered boxes. If every
piece of text is already boxed, return an empty list. Do not invent text.

Reply as {{"regions": [ ... ], "missed": [ ... ]}}, regions in ascending id order."""

# 응답 형식을 프롬프트로 부탁하지 않고 스키마로 강제한다.
# 부탁했을 때 8페이지 중 5페이지가 파싱에 실패했다 — 배열 괄호를 빼먹거나,
# 콤마를 누락하거나, 뒤에 잡소리를 붙이는 방식이 매번 달랐다. 정규식으로
# 뒤쫓는 것은 끝이 없다. llama.cpp 가 문법 제약을 걸어주므로 그걸 쓴다.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string", "maxLength": 600},
                    "lang": {"type": "string", "enum": ["ja", "zh", "en", "other"]},
                    "speaker": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["dialogue", "thought", "narration", "sfx",
                                      "incidental", "credits"]},
                    "addressee": {"type": "string"},
                },
                "required": ["id", "text", "lang", "speaker", "kind"],
            },
        },
        "missed": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 600},
                    "kind": {"type": "string",
                             "enum": ["dialogue", "thought", "narration", "sfx",
                                      "incidental", "credits"]},
                    "box": {"type": "array", "minItems": 4, "maxItems": 4,
                            "items": {"type": "integer"}},
                },
                "required": ["text", "box"],
            },
        },
    },
    "required": ["regions"],
}
# 한 페이지의 박스 수만큼 나온다. 실측 최대 20개 남짓, 넉넉히 잡는다.
RESPONSE_SCHEMA["properties"]["regions"]["maxItems"] = 200
bound_schema(RESPONSE_SCHEMA)


CAST_HINT = """
KNOWN CAST SO FAR — pages are read in order and this list carries over. Reuse an
existing label **exactly** whenever the same person appears again. Add a new label
only for someone genuinely not listed.

{cast}

A label is an identifier, not a translation: what matters is that the same person
keeps the same label across the whole chapter. Prefer a name that characters
actually use to address someone. If a candidate label looks like an ordinary word
lifted from the dialogue rather than a name, use a visual description instead.
"""

REASK_HINT = """
GEOMETRY CROSS-CHECK — a separate detector may have seen more speakers than you did.

It grouped the drawn figures on this page by appearance and anchored each text
region to one of them, using speech-bubble tails and body positions. It cannot
read the dialogue. Its grouping was:
{grouping}

That grouping put the regions into {magi_n} groups, while you assigned {vlm_n}
distinct speakers.

**Treat this as a question, not a correction.** The detector splits and merges
figures by how they are drawn, so it can be wrong in either direction. Use it only
to ask yourself whether you collapsed two different people into one label: a
stretch of lines that alternates between asking and answering, or that contains a
term of address, cannot all come from one speaker.

If, after re-reading, you still think your original assignment was right, keep it.
Produce the full answer again for every id."""


class Cast:
    """페이지를 넘어 유지되는 등장인물 명부.

    필요한 이유: VLM 이 페이지를 한 장씩 **서로 독립적으로** 읽으면 같은 인물에
    매번 새 라벨을 붙인다. 실측에서 한 작품의 화자 목록이 이렇게 흩어졌다 —
    Yun / male / female / boy / girl with horns / unknown. 말투 시트가 화자
    식별자를 키로 잡으므로, 한 인물이 셋으로 쪼개지면 서로 다른 말투 세 개가
    배정되고 그게 곧 말투 붕괴다.

    중요한 것은 라벨의 내용이 아니라 **챕터 전체에서 안정적인가**다. 이름이든
    시각 서술자든, 같은 사람에게 같은 라벨이 붙으면 목적을 달성한다.
    """

    # 사람이 아닌 라벨. 명부에 올리면 인물처럼 취급되어 오염된다.
    NON_PERSON = {"narration", "unknown", "", "sfx", "none",
                  "incidental", "credits"}

    def __init__(self):
        self.entries = {}   # 라벨 → {"pages": [...], "count": n}

    def observe(self, page_no, rows):
        for row in rows:
            label = (row.get("speaker") or "").strip()
            if label.lower() in self.NON_PERSON:
                continue
            e = self.entries.setdefault(label, {"pages": [], "count": 0})
            e["count"] += 1
            if page_no not in e["pages"]:
                e["pages"].append(page_no)

    def render(self, max_entries=20):
        if not self.entries:
            return None
        # 많이 나온 인물을 앞에 둔다. 목록이 길어지면 뒤쪽은 잘라낸다 —
        # 한 번 스쳐간 라벨까지 계속 들고 다니면 오히려 재사용을 방해한다.
        ranked = sorted(self.entries.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
        lines = [f"  {label} — {e['count']}개 대사, 페이지 {', '.join(map(str, e['pages']))}"
                 for label, e in ranked[:max_entries]]
        return "\n".join(lines)


def magi_grouping(texts, only_ids=None):
    """Magi 의 박스→인물 클러스터 배정을 사람이 읽을 형태로.

    클러스터 ID 는 페이지 로컬이라 이름으로는 못 쓴다. 하지만 한 페이지 안에서
    "몇 명인가"와 "어느 박스가 같은 사람에 걸렸나"는 기하학에서 나온 정보라
    VLM 의 의미 판단과 독립적이고, 그래서 교차 검증에 쓸 수 있다.

    only_ids 로 비교 대상을 맞춘다. Magi 는 대사와 나레이션을 구분하지 못하므로
    전체 박스를 세면, 나레이션 한 칸만 있는 페이지가 "Magi 1 vs VLM 0" 으로
    허위 불일치를 낸다. 양쪽 모두 **VLM 이 대사로 분류한 박스**만 센다.
    """
    groups = {}
    for t in texts:
        if only_ids is not None and t["id"] not in only_ids:
            continue
        c = t.get("speaker_cluster")
        if c is not None:
            groups.setdefault(c, []).append(t["id"])
    if not groups:
        return None, 0
    lines = [f"  group {chr(65 + i)}: regions {', '.join(map(str, sorted(ids)))}"
             for i, (_, ids) in enumerate(sorted(groups.items()))]
    return "\n".join(lines), len(groups)


def label_page(image, texts, outline_width, font_size):
    """박스를 빨간 테두리로 그리고 번호를 붙인다.

    번호는 박스 **바깥** 왼쪽 위에 둔다. 안에 그리면 글자를 가려서 전사가 망가진다.
    자리가 없으면(페이지 가장자리) 안쪽으로 접어 넣는다.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for t in texts:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=outline_width)
        tag = str(t["id"])
        tw = draw.textlength(tag, font=font)
        bx, by = x1, y1 - font_size - 4
        if by < 0:                       # 위에 자리가 없으면 박스 안쪽 위로
            by = y1 + 2
        if bx + tw + 6 > canvas.width:   # 오른쪽으로 넘치면 왼쪽으로 당긴다
            bx = canvas.width - tw - 6
        draw.rectangle([bx, by, bx + tw + 6, by + font_size + 4], fill=(220, 30, 30))
        draw.text((bx + 3, by + 2), tag, fill=(255, 255, 255), font=font)
    return canvas


def to_data_url(image, max_side):
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((round(image.width * s), round(image.height * s)),
                             Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), image.size


def ask_page(client, data_url, instruction, max_tokens, thinking):
    """스키마를 강제해 regions 배열을 그대로 받는다.

    HTTP 조립과 '추론 끄기' 규약은 backend.Client 가 맡는다. 라우터 모드에서는
    요청에 model 이 없으면 400 이 나므로, 직접 post 하지 않고 반드시 Client 를
    거쳐야 한다.
    """
    res = client.chat(
        [{"type": "image_url", "image_url": {"url": data_url}},
         {"type": "text", "text": instruction}],
        schema=RESPONSE_SCHEMA, schema_name="regions",
        thinking=thinking, max_tokens=max_tokens, temperature=0.0,
    )
    return res["regions"], res.get("missed") or []


def adopt_missed(pg, missed, size, existing_ids):
    """VLM 이 "빨간 박스가 놓쳤다" 고 한 영역을 박스로 받아들인다.

    필요한 이유: Magi 가 통째로 놓치는 영역이 있다. 실측(maid2 12쪽)에서 세로
    캡션 `幫忙早上整理打扮` 은 두 번의 실행 모두 검출되지 않았고, `主人` 은 한
    실행에서는 잡히고 다른 실행에서는 빠졌다 — 검출이 불안정하다. 안 잡히면
    지워지지도 번역되지도 않으므로 **원문이 그대로 남는다.**

    좌표는 거칠어도 된다. 뒤에서 말풍선 확장이 실제 말풍선에 맞춰 넓히고,
    마스크는 박스 안의 획을 찾아 지운다.
    """
    W, H = size
    out, nid = [], max(existing_ids, default=-1) + 1
    boxes = [t["bbox"] for t in pg["texts"]]
    for m in missed:
        if not (m.get("text") or "").strip():
            continue
        try:
            a, b, c, d = [int(v) for v in m["box"]]
        except (KeyError, TypeError, ValueError):
            continue
        x1, y1 = max(0, min(a, c)) * W // 1000, max(0, min(b, d)) * H // 1000
        x2, y2 = min(1000, max(a, c)) * W // 1000, min(1000, max(b, d)) * H // 1000
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        # 페이지의 4분의 1을 넘는 "글자 영역" 은 없다. 헛짚은 것이다.
        if (x2 - x1) * (y2 - y1) > 0.25 * W * H:
            continue
        # 이미 있는 박스와 절반 넘게 겹치면 중복이다.
        dup = False
        for bx1, by1, bx2, by2 in boxes:
            ox = max(0, min(x2, bx2) - max(x1, bx1))
            oy = max(0, min(y2, by2) - max(y1, by1))
            if ox * oy > 0.5 * min((x2 - x1) * (y2 - y1),
                                   max(1, (bx2 - bx1) * (by2 - by1))):
                dup = True
                break
        if dup:
            continue
        out.append({"id": nid, "bbox": [x1, y1, x2, y2], "ocr": None,
                    "essential": True, "speaker": None, "speaker_cluster": None,
                    "has_tail": False, "found_by": "vlm"})
        boxes.append([x1, y1, x2, y2])
        nid += 1
    return out



def index_rows(rs):
    out = {}
    for row in rs:
        try:
            out[int(row["id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def spoken_ids(idx):
    return {i for i, r in idx.items()
            if (r.get("kind") or "") in ("dialogue", "thought")}


def speaker_count(idx):
    return len({(r.get("speaker") or "").strip() for i, r in idx.items()
                if i in spoken_ids(idx) and (r.get("speaker") or "").strip()})


def main():
    p = argparse.ArgumentParser(description="페이지 단위 전사 + 화자 판정")
    p.add_argument("--magi-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", help="이 경로에 전체 로그를 덧붙인다 "
                   "(터미널은 짧게, 파일은 빠짐없이)")
    p.add_argument("--config", help="config.json 경로 (기본: config/config.json)")
    p.add_argument("--model", help="stages 설정을 무시하고 이 모델을 쓴다")
    p.add_argument("--pages", type=int, nargs="+", help="이 페이지(1-base)만 처리")
    p.add_argument("--max-side", type=int, default=1536,
                   help="페이지를 보낼 때 긴 변 상한. 작으면 글자가 뭉개진다")
    p.add_argument("--outline-width", type=int, default=4)
    p.add_argument("--font-size", type=int, default=44)
    p.add_argument("--dump-labelled", help="번호를 그린 페이지를 저장할 디렉터리")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="기존 출력이 있어도 처음부터 다시 판독한다")
    p.add_argument("--no-cast-memory", action="store_true",
                   help="페이지를 넘는 등장인물 명부를 쓰지 않는다 (A/B 비교용)")
    p.add_argument("--reask-on-disagreement", action="store_true", default=True,
                   help="Magi 와 화자 수가 어긋나면 Magi 그룹핑을 힌트로 다시 묻는다")
    p.add_argument("--no-reask", dest="reask_on_disagreement", action="store_false",
                   help="교차 검증 재질의를 끈다")
    p.add_argument("--no-adopt-missed", action="store_true",
                   help="Magi 가 놓친 영역을 판독이 보고해도 박스로 만들지 않는다")
    p.add_argument("--workers", type=int, default=4,
                   help="동시에 던질 페이지 수. **--no-cast-memory 일 때만** 적용된다 "
                        "— 명부를 물려주면 페이지 N 이 1..N-1 에 의존하므로 순서를 "
                        "지켜야 한다. 실측 4에서 1.5배(56.1→36.9초)")
    p.add_argument("--thinking", action="store_true",
                   help="추론을 켠다. 화자 판정에는 도움이 될 수 있으나 느리다")
    args = p.parse_args()

    PROG.open_log(getattr(args, 'log', None))
    doc = json.load(open(args.magi_json, encoding="utf-8"))
    client = client_for("read_page", args.config, args.model)
    if not client.health():
        print(f"백엔드에 연결할 수 없습니다: {client.base_url}", file=sys.stderr)
        return 3
    print(f"모델: {client.name} (model={client.model})")

    if args.dump_labelled:
        os.makedirs(args.dump_labelled, exist_ok=True)

    done_pages = set()
    if args.resume and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            by_idx = {pg["index"]: pg for pg in prev["pages"]}
            for pg in doc["pages"]:
                old = by_idx.get(pg["index"])
                # 판독이 실제로 채워진 페이지만 완료로 본다. read_error 가 있거나
                # ocr 이 비었으면 다시 시도한다.
                # read_skipped 는 두 번 시도하고 포기한 페이지다. 다시 시도하면
                # 또 막히므로 완료로 취급한다. --no-resume 이면 처음부터 다시 한다.
                if old and (old.get("read_skipped") or
                            (not old.get("read_error") and
                             any((t.get("ocr") or "").strip() for t in old["texts"]))):
                    pg.update(old)
                    done_pages.add(pg["index"])
        except Exception as e:
            print(f"기존 결과를 못 읽어 처음부터 갑니다: {e}", file=sys.stderr)
    if done_pages:
        print(f"이미 판독된 페이지 {len(done_pages)}장은 건너뜁니다")

    failed = []
    targets = [pg for pg in doc["pages"]
               if pg["texts"] and pg["index"] not in done_pages
               and (not args.pages or (pg["index"] + 1) in args.pages)]
    if not targets:
        # "읽을 게 없다"에는 두 경우가 있고 결과가 반대다.
        #   ① 이미 다 읽었다      → 재개 성공이다. 0 을 줘야 파이프라인이 이어간다.
        #   ② 애초에 아무것도 없다 → 입력이 잘못됐다. 그건 실패다.
        # 예전에는 둘 다 2 를 줬다. 그래서 뒷단계가 죽어 다시 돌리면, 판독이
        # 멀쩡히 끝나 있는데도 파이프라인이 "② 실패"로 멈췄다.
        if done_pages and not args.pages:
            print(f"모든 페이지가 이미 판독돼 있습니다 ({len(done_pages)}장) — 건너뜁니다")
            return 0
        print("처리할 페이지가 없습니다", file=sys.stderr)
        return 2

    t0 = time.time()
    done = missing_total = adopted_total = 0
    cast = Cast()
    out_lock = threading.Lock()

    # 페이지 판독을 두 국면으로 나눈다.
    #
    #   ① 모델에 묻는 부분  — 페이지끼리 독립이라 **동시에** 던질 수 있다
    #   ② 상태를 바꾸는 부분 — 명부 누적·카운터·필드 대입. **페이지 순서대로** 한다
    #
    # 이렇게 갈라야 동시성이 결과를 바꾸지 않는다. cast.observe 를 스레드에서
    # 부르면 명부에 페이지가 들어가는 순서가 실행마다 달라진다.
    #
    # ①을 동시에 던져도 되는 것은 파이프라인이 --no-cast-memory 로 부르기
    # 때문이다. 그 플래그가 없으면 페이지 N 이 1..N-1 의 명부를 받으므로
    # 순차 의존이 생긴다 — 그때는 아래에서 workers 를 1 로 되돌린다.
    def query_page(pg):
        """모델에 묻기만 한다. 공유 상태를 건드리지 않는다."""
        n = len(pg["texts"])
        ids = [t["id"] for t in pg["texts"]]
        page_img = Image.open(pg["file"])
        labelled = label_page(page_img, pg["texts"], args.outline_width, args.font_size)
        if args.dump_labelled:
            labelled.save(os.path.join(args.dump_labelled, f"p{pg['index']+1:03d}.png"))
        data_url, sent_size = to_data_url(labelled, args.max_side)

        instruction = INSTRUCTION.format(n=n, ids=", ".join(str(i) for i in ids))
        roster = None if args.no_cast_memory else cast.render()
        if roster:
            instruction += CAST_HINT.format(cast=roster)
        t = time.time()
        rows = None
        for attempt, budget in enumerate((args.max_tokens, args.max_tokens * 2), 1):
            try:
                rows, missed = ask_page(client, data_url, instruction, budget,
                                        args.thinking)
                break
            except Exception as e:
                # 대부분 응답이 토큰 한도에서 잘려 JSON 이 깨진 경우다. 예산을
                # 두 배로 주고 한 번 더 해본다.
                if attempt == 1:
                    with out_lock:
                        print(f"p{pg['index']+1} 파싱 실패, 예산 2배로 재시도: {e}",
                              flush=True)
                    continue
                # 여기까지 오면 두 번 다 실패한 것이다. 대사가 거의 없는 페이지에서
                # 모델이 응답을 조기에 끊는 일이 있었는데(1 개 박스, 52 자에서 절단)
                # 예산을 늘려도 같았다. 영구 실패로 보고 **글자 없는 페이지로 확정**
                # 한다. 그러지 않으면 재실행마다 같은 페이지에서 막힌다.
                pg["read_error"] = f"{type(e).__name__}: {e}"
                pg["read_skipped"] = True
                for t2 in pg["texts"]:
                    t2.setdefault("ocr", None)
                    t2.setdefault("speaker_name", None)
                    t2.setdefault("kind", None)
                with out_lock:
                    print(f"p{pg['index']+1} 두 번 실패 — 글자 없는 페이지로 넘깁니다: {e}",
                          flush=True)
                return None
        if rows is None:
            return None

        by_id = index_rows(rows)

        # 기하학 교차 검증 — Magi 가 센 화자 수와 VLM 이 센 화자 수를 비교한다.
        # 실측에서 양방향으로 어긋났고 어긋난 곳이 곧 오류였다: 두 사람이 묻고
        # 답하는 13개 박스를 VLM 이 한 사람으로 몰아버린 페이지에서 Magi 는 2명을
        # 셌고, 반대로 Magi 가 남녀를 한 인물로 병합한 페이지에서는 VLM 이 맞았다.
        # 어느 쪽도 권위는 없지만 **수의 불일치는 이 페이지를 믿을 수 없다는 신호**다.
        spoken = spoken_ids(by_id)
        grouping, magi_n = magi_grouping(pg["texts"], spoken)
        vlm_n = speaker_count(by_id)
        # 대사 박스가 2개 미만이면 화자 수로는 아무것도 알 수 없다. 나레이션·효과음
        # 전용 페이지에서 허위 불일치가 나던 원인이다.
        # **비대칭이다.** Magi 가 VLM 보다 *많이* 셀 때만 다시 묻는다.
        #
        # 근거: maid2 실측에서 두 값이 어긋난 20쪽 중 12쪽이 "Magi 가 더 적게" 였고,
        # 그 방향은 대개 Magi 가 **닮은꼴 인물을 한 클러스터로 합친** 것이다
        # (같은 제복을 입은 인물이 여럿인 작품에서 특히). 그걸 근거로 VLM 에게
        # 화자를 합치라고 지시하면 인물 과병합을 우리가 만들어 내게 된다.
        # 반대 방향(Magi 가 더 많음)은 "VLM 이 둘을 하나로 봤다" 는 분리 증거이고,
        # 분리는 말풍선 꼬리·신체 위치에서 나오므로 그나마 믿을 만하다.
        comparable = len(spoken) >= 2 and magi_n > 0
        agreed = (not comparable) or magi_n <= vlm_n
        if not agreed and args.reask_on_disagreement and grouping:
            with out_lock:
                print(f"p{pg['index']+1} 화자 수 (Magi {magi_n} > VLM {vlm_n}) "
                      f"→ 합쳤는지 재확인", flush=True)
            try:
                rows2, _ = ask_page(
                    client, data_url,
                    instruction + REASK_HINT.format(magi_n=magi_n, vlm_n=vlm_n,
                                                    grouping=grouping),
                    args.max_tokens, args.thinking)
                by2 = index_rows(rows2)
                n2 = speaker_count(by2)
                # 재질의가 불일치를 줄였을 때만 받는다. 늘리거나 그대로면 원본을
                # 남긴다 — 힌트가 오히려 화자를 쪼개게 만들 수도 있다.
                if by2 and abs(n2 - magi_n) < abs(vlm_n - magi_n):
                    by_id, vlm_n = by2, n2
                    msg = f"  → 재질의 채택 (VLM {n2}명)"
                else:
                    msg = f"  → 재질의 기각 (VLM {n2}명, 개선 없음)"
            except Exception as e:
                msg = f"  → 재질의 실패, 원본 유지: {e}"
            with out_lock:
                print(msg, flush=True)
        return {"by_id": by_id, "magi_n": magi_n, "vlm_n": vlm_n,
                "spoken": spoken, "comparable": comparable, "missed": missed,
                "sent_size": sent_size, "seconds": time.time() - t, "n": n}

    # 이 단계가 모든 페이지 요청에 공통으로 넣는 것을 한 번만 남긴다. 박스 수·id
    # 목록만 페이지마다 달라지므로 표본으로 하나를 채워 보여준다.
    PROG.prompt_block(
        "② 페이지 판독 (화자·언어)",
        INSTRUCTION.format(n="<박스 수>", ids="<박스 id 목록>"),
        {"페이지마다 덧붙는 인물 명부(CAST_HINT)": CAST_HINT.format(cast="<지금까지의 명부>"),
         "화자 수가 어긋날 때만 덧붙는 재질의(REASK_HINT)":
             REASK_HINT.format(magi_n="<기하 검출 수>", vlm_n="<판독 수>",
                               grouping="<박스→인물 배정>")})

    # ① 묻기. --no-cast-memory 일 때만 동시에 던진다.
    workers = args.workers if args.no_cast_memory else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(query_page, targets))
    else:
        results = [query_page(pg) for pg in targets]

    # ② 반영. 페이지 순서대로 — 여기서만 공유 상태를 바꾼다.
    for pg, r in zip(targets, results):
        if r is None:
            failed.append(pg["index"] + 1)
            continue
        by_id, n = r["by_id"], r["n"]
        pg["speaker_agreement"] = {"magi": r["magi_n"], "vlm": r["vlm_n"],
                                   "spoken_boxes": len(r["spoken"]),
                                   "agreed": (r["magi_n"] == r["vlm_n"])
                                             if r["comparable"] else None}

        cast.observe(pg["index"] + 1, list(by_id.values()))

        langs = {}
        for txt in pg["texts"]:
            row = by_id.get(txt["id"])
            if row is None:
                txt["ocr"] = None
                txt["speaker_name"] = None
                missing_total += 1
                continue
            txt["ocr"] = (row.get("text") or "").strip() or None
            txt["speaker_name"] = (row.get("speaker") or "").strip() or None
            txt["addressee"] = (row.get("addressee") or "").strip() or None
            txt["kind"] = (row.get("kind") or "").strip() or None
            txt["lang"] = (row.get("lang") or "").strip() or None
            if txt["lang"]:
                langs[txt["lang"]] = langs.get(txt["lang"], 0) + 1

        # 페이지 언어는 박스 다수결로 정한다. 효과음 한 칸이 페이지 언어를
        # 뒤집지 않게 하려는 것이다.
        pg["source_lang"] = max(langs, key=langs.get) if langs else None

        # Magi 가 놓친 영역을 VLM 이 본 대로 받아들인다. 전사는 비워 두고 —
        # 다음 단계(크롭 전사)가 채운다. 좌표가 거칠어도 말풍선 확장이 맞춘다.
        if not args.no_adopt_missed and r.get("missed"):
            extra = adopt_missed(pg, r["missed"],
                                 (pg["size"][0], pg["size"][1]),
                                 [t["id"] for t in pg["texts"]])
            for e, m in zip(extra, [x for x in r["missed"]][:len(extra)]):
                e_txt = (m.get("text") or "").strip()
                pg["texts"].append(dict(e, ocr=e_txt or None, speaker_name=None,
                                        addressee=None,
                                        kind=(m.get("kind") or "dialogue"),
                                        lang=pg.get("source_lang")))
            if extra:
                adopted_total += len(extra)
                PROG.log(f"  p{pg['index']+1} Magi 가 놓친 영역 {len(extra)}개를 "
                         f"판독에서 받아들였습니다")
        miss = [t["id"] for t in pg["texts"] if t["id"] not in by_id]
        done += 1
        PROG.step(f"  [{done}/{len(doc['pages'])}] p{pg['index']+1} 박스 {n}개 · "
                  f"화자 {len({t['speaker_name'] for t in pg['texts'] if t.get('speaker_name')})}명"
                  f"{f' · 누락 {miss}' if miss else ''} "
                  f"({r['seconds']:.1f}초, {r['sent_size'][0]}x{r['sent_size'][1]})")

    PROG.done()

    doc["read_page_pass"] = {"model": client.name, "max_side": args.max_side,
                             "cast_memory": not args.no_cast_memory}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    if cast.render():
        PROG.log("\n[등장인물 명부]\n" + cast.render())
    doc["cast"] = cast.entries
    doc["read_failures"] = failed
    PROG.log(f"\n페이지 {done}/{len(targets)} 처리, 박스 누락 {missing_total}개  "
             f"({time.time()-t0:.1f}초) → {args.out}")
    if failed:
        print(f"건너뛴 페이지 {len(failed)}장: {failed}  "
              f"— 다시 실행하면 이 페이지만 재시도합니다")
    # 페이지 실패로 단계를 실패시키지 않는다. 산출물을 썼으면 성공이다.
    # 재개로 실패 페이지 하나만 남았을 때 done==0 이 되어 파이프라인이 멈추는
    # 문제가 있었다 — 영구 실패 페이지 하나 때문에 72 장이 막혔다.
    return 0


if __name__ == "__main__":
    sys.exit(main())
