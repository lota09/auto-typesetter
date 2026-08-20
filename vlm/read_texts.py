#!/usr/bin/env python3
"""읽기 패스 — Magi 가 잡은 텍스트 박스를 VLM 으로 전사한다.

전용 OCR 모델을 쓰지 않는 이유:
  OCR 이 줄 수 있는 고유한 값은 글자 좌표인데, 이 파이프라인에서 좌표는 이미
  Magi 가 준다. 지우기용 마스크도 OCR 이 아니라 세그멘테이션이 만든다. 남는
  건 전사뿐이고, 어차피 번역이 LLM 을 거치므로 같은 VLM 에 맡기면 프레임워크
  가 하나 줄어든다. 세로쓰기 번체는 작은 OCR 모델이 특히 약한 영역이라
  이쪽이 유리하기도 하다.

박스를 하나씩 잘라 보내는 이유:
  페이지를 통째로 주고 글자를 받으면 "어느 글자가 어느 박스냐"를 다시 풀어야
  한다. 우리가 직접 자르면 그 매핑이 공짜다. 대신 크롭만 보면 맥락이 없어
  환각이 늘기 때문에, --pad 로 주변을 조금 붙여 보낸다.

번역은 여기서 하지 않는다. 챕터 전체 번역은 모든 원문이 모인 뒤에야 시작할 수
있으므로, 읽기와 번역은 반드시 별개의 패스다.

입력: magi_worker.py 가 낸 JSON
출력: 같은 구조에 text[].ocr 를 채운 JSON
"""

import argparse
import base64
import io
import json
import os
import sys
import time

import requests
from PIL import Image

# 프롬프트를 영어로 쓴다. 한국어로 지시했더니 출력이 한국어 문자로 끌려갔다
# (あ → 아, 的 → の). 지시문의 언어가 전사 결과의 문자 체계를 오염시킨다.
#
# 한 줄로 이어 달라고 하지 않고 **열별로 한 줄씩** 받는 이유:
# 세로쓰기 여러 열을 한 줄로 달라고 하면 모델이 우→좌 순서를 자주 틀린다
# (`我得做好` + `女僕的工作` → `女僕的工作我得做好`). 추론 로그를 보면 행으로
# 읽을지 열로 읽을지 자체를 헷갈린다. 열을 따로 받아 이어붙이면 순서 결정을
# 모델에게 맡기지 않게 되고, 원본 열 정보도 남아 나중에 대조에 쓸 수 있다.
PROMPT_TEMPLATE = (
    "Transcribe the text in this cropped comic image. {script_hint}\n"
    "- Output ONE COLUMN PER LINE, starting with the RIGHTMOST column and "
    "moving left. Read each column top to bottom.\n"
    "- If the text is horizontal instead, output one line per line of text, "
    "top to bottom.\n"
    "- Do NOT translate. Output the original characters only.\n"
    "- Preserve the original script exactly. Never convert between Chinese, "
    "Japanese and Korean characters.\n"
    "- Keep the original punctuation.\n"
    "- If there is no legible text, reply with exactly: EMPTY\n"
    "- Output only the transcription: no explanation, no quotes, no preamble."
)

# 특정 작품에 맞춘 기본값을 두지 않는다. maid2 로 실험할 때 "번체 중국어 세로쓰기"
# 로 못박아 뒀는데, 소재가 바뀌면 그 힌트가 오히려 오답을 유도한다. 언어를 지정
# 하지 말고 세로쓰기 가능성만 알려주는 것이 여러 작품에 안전하다.
DEFAULT_SCRIPT_HINT = ("The source may be Japanese, Chinese, Korean or English, "
                       "and may be written vertically. Detect it from the image; "
                       "do not assume a language.")


def crop(image, bbox, pad, min_side, max_side):
    """박스를 자른다. pad 는 박스 짧은 변에 대한 비율이다.

    고정 픽셀이 아니라 비율로 넓히는 이유: 페이지 해상도와 말풍선 크기가
    제각각이라 고정값은 작은 박스에선 과하고 큰 박스에선 무의미해진다.

    min_side 로 작은 크롭을 **확대**하는 것이 핵심이다. 말풍선 하나는 2040×2880
    페이지에서도 200px 대에 불과한데, Qwen-VL 은 이미지 토큰 수가 적으면 정확도가
    무너진다 (llama.cpp 가 로드 시 최소 1024 토큰을 권고한다). 축소만 하고
    확대를 안 했을 때 큰 세로쓰기 블록이 통째로 EMPTY 로 돌아왔다.
    """
    x1, y1, x2, y2 = bbox
    m = int(pad * min(x2 - x1, y2 - y1))
    box = (max(0, x1 - m), max(0, y1 - m),
           min(image.width, x2 + m), min(image.height, y2 + m))
    out = image.crop(box)

    scale = 1.0
    if min_side and max(out.size) < min_side:
        scale = min_side / max(out.size)
    if max_side and max(out.size) * scale > max_side:
        scale = max_side / max(out.size)
    if scale != 1.0:
        out = out.resize((max(1, round(out.width * scale)), max(1, round(out.height * scale))),
                         Image.LANCZOS)
    return out


def to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def read_one(session, url, image, prompt, timeout, temperature, max_tokens, thinking):
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": to_data_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # 전사는 창의성이 필요 없다. 샘플링을 최대한 죽인다.
        "top_p": 1.0,
        "stream": False,
    }
    if not thinking:
        # 추론형 모델이라 기본값으로는 토큰 예산을 전부 사고에 쓰고 본문을 못 내는
        # 일이 생긴다 (finish_reason=length, content=''). 전사에 추론은 불필요하고,
        # 끄고 비교해도 결과가 같았다.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    r = session.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    if not content and choice.get("finish_reason") == "length":
        raise RuntimeError("토큰 예산 소진 전에 본문이 나오지 않았습니다 "
                           "(--max-tokens 를 올리거나 --thinking 을 끄세요)")
    return content


def join_columns(text):
    """열별 응답을 원문 한 줄로 이어붙인다.

    세로쓰기 CJK 는 열 사이에 공백이 들어가지 않으므로 구분자 없이 붙인다.
    열 목록도 함께 돌려주어 나중에 대조·재조판에 쓸 수 있게 남긴다.
    """
    cols = [c.strip() for c in text.splitlines() if c.strip()]
    return "".join(cols), cols


def main():
    p = argparse.ArgumentParser(description="Magi 박스를 VLM 으로 전사")
    p.add_argument("--magi-json", required=True, help="magi_worker.py 출력")
    p.add_argument("--out", required=True, help="ocr 을 채운 JSON 경로")
    p.add_argument("--server", default="http://127.0.0.1:8081",
                   help="llama-server 주소 (기본 %(default)s)")
    p.add_argument("--pad", type=float, default=0.15,
                   help="박스 짧은 변 대비 여백 비율 (기본 %(default)s)")
    p.add_argument("--min-side", type=int, default=896,
                   help="크롭 긴 변 하한. 작은 말풍선을 여기까지 확대한다 "
                        "(Qwen-VL 은 이미지 토큰이 적으면 못 읽는다)")
    p.add_argument("--max-side", type=int, default=1536,
                   help="크롭 긴 변 상한. 크게 두면 느려진다")
    p.add_argument("--script-hint", default=DEFAULT_SCRIPT_HINT,
                   help="원문 문자 체계 힌트. 소재가 바뀌면 이걸 바꾼다")
    p.add_argument("--limit", type=int, help="앞에서 N 개만 처리 (검증용)")
    p.add_argument("--pages", type=int, nargs="+", help="이 페이지 번호(1-base)만 처리")
    p.add_argument("--dump-crops", help="보낸 크롭을 저장할 디렉터리 (검증용)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--thinking", action="store_true",
                   help="모델의 추론을 켠다. 전사 품질은 같고 훨씬 느려진다")
    p.add_argument("--timeout", type=float, default=180)
    args = p.parse_args()

    doc = json.load(open(args.magi_json, encoding="utf-8"))
    url = args.server.rstrip("/") + "/v1/chat/completions"

    session = requests.Session()
    try:
        h = session.get(args.server.rstrip("/") + "/health", timeout=10)
        if h.status_code != 200:
            print(f"서버가 준비되지 않았습니다 ({h.status_code}): {h.text[:200]}", file=sys.stderr)
            return 3
    except requests.RequestException as e:
        print(f"서버에 연결할 수 없습니다: {e}", file=sys.stderr)
        return 3

    if args.dump_crops:
        os.makedirs(args.dump_crops, exist_ok=True)

    jobs = []  # (page_record, text_record) 순서대로
    for page in doc["pages"]:
        if args.pages and (page["index"] + 1) not in args.pages:
            continue
        for t in page["texts"]:
            jobs.append((page, t))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print("처리할 텍스트가 없습니다", file=sys.stderr)
        return 2

    prompt = PROMPT_TEMPLATE.format(script_hint=args.script_hint)
    print(f"텍스트 {len(jobs)}개 전사 시작 "
          f"(pad={args.pad}, 크롭 긴 변 {args.min_side}~{args.max_side}px)")
    images = {}
    t0 = time.time()
    ok = empty = fail = 0
    for i, (page, t) in enumerate(jobs, 1):
        path = page["file"]
        if path not in images:
            images[path] = Image.open(path).convert("RGB")
        piece = crop(images[path], t["bbox"], args.pad, args.min_side, args.max_side)

        if args.dump_crops:
            piece.save(os.path.join(args.dump_crops,
                                    f"p{page['index']+1:03d}_t{t['id']:03d}.png"))
        try:
            raw = read_one(session, url, piece, prompt, args.timeout,
                           args.temperature, args.max_tokens, args.thinking)
        except Exception as e:
            t["ocr"] = None
            t["ocr_error"] = f"{type(e).__name__}: {e}"
            fail += 1
            print(f"  [{i}/{len(jobs)}] p{page['index']+1} t{t['id']} 실패: {e}", flush=True)
            continue

        if raw == "EMPTY" or not raw:
            t["ocr"], t["ocr_columns"] = None, []
            empty += 1
            shown = "(빈 박스)"
        else:
            t["ocr"], t["ocr_columns"] = join_columns(raw)
            ok += 1
            shown = t["ocr"]
        print(f"  [{i}/{len(jobs)}] p{page['index']+1} t{t['id']} "
              f"{'대사' if t['essential'] else '기타'} | {shown[:60]}", flush=True)

    doc["read_pass"] = {
        "server": args.server, "pad": args.pad,
        "min_side": args.min_side, "max_side": args.max_side,
        "script_hint": args.script_hint,
        "prompt_sha": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:12],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    dt = time.time() - t0
    print(f"\n전사 {ok}개 / 빈 박스 {empty}개 / 실패 {fail}개  "
          f"({dt:.1f}초, {dt/len(jobs):.1f}초/개) → {args.out}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
