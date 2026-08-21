#!/usr/bin/env python3
"""manga-ocr 로 크롭을 전사한다 — VLM 크롭 패스와 같은 조건에서 비교하려고.

왜 별도 환경인가:
  manga-ocr 은 transformers >= 4.45 를 요구하는데 Magiv2 는 4.44.1 에서만 뜬다.
  같은 venv 에 넣었더니 Magi 가 즉시 깨졌다 (AutoModel requires PyTorch). 그래서
  ocr/.venv 로 격리했다. torch 는 --system-site-packages 로 공유하므로 디스크
  비용은 작다.

왜 재보는가:
  전용 OCR 을 **측정 없이 배제했었다.** 근거는 "좌표는 Magi 가 주니 OCR 의 고유
  가치가 없다" 였는데, 그건 검출에 대한 얘기지 인식에 대한 얘기가 아니었다.
  manga-ocr 은 일본 만화 텍스트에 파인튜닝된 모델이라 범용 VLM 보다 나을 수 있다.

입출력은 read_texts.py 와 같은 스키마다. 그래야 같은 채점기로 잰다.
"""

import argparse
import json
import os
import sys
import time

from PIL import Image


def crop(image, bbox, pad, min_side):
    """read_texts.py 와 같은 규칙으로 자른다. 조건을 맞춰야 비교가 성립한다."""
    x1, y1, x2, y2 = bbox
    m = int(pad * min(x2 - x1, y2 - y1))
    box = (max(0, x1 - m), max(0, y1 - m),
           min(image.width, x2 + m), min(image.height, y2 + m))
    out = image.crop(box)
    if min_side and max(out.size) < min_side:
        s = min_side / max(out.size)
        out = out.resize((max(1, round(out.width * s)), max(1, round(out.height * s))),
                         Image.LANCZOS)
    return out


def main():
    p = argparse.ArgumentParser(description="manga-ocr 크롭 전사")
    p.add_argument("--magi-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--min-side", type=int, default=896)
    p.add_argument("--pages", type=int, nargs="+")
    args = p.parse_args()

    from manga_ocr import MangaOcr

    doc = json.load(open(args.magi_json, encoding="utf-8"))
    t0 = time.time()
    mocr = MangaOcr()
    print(f"모델 로드 {time.time()-t0:.1f}초", flush=True)

    images, n, t = {}, 0, time.time()
    for pg in doc["pages"]:
        if args.pages and (pg["index"] + 1) not in args.pages:
            continue
        path = pg["file"]
        if path not in images:
            images[path] = Image.open(path).convert("RGB")
        for box in pg["texts"]:
            piece = crop(images[path], box["bbox"], args.pad, args.min_side)
            try:
                box["ocr"] = (mocr(piece) or "").strip() or None
            except Exception as e:
                box["ocr"] = None
                box["ocr_error"] = f"{type(e).__name__}: {e}"
            n += 1

    doc["read_pass"] = {"engine": "manga-ocr", "pad": args.pad,
                        "min_side": args.min_side}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    dt = time.time() - t
    print(f"전사 {n}개 ({dt:.1f}초, {dt/max(n,1):.2f}초/개) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
