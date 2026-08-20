#!/usr/bin/env python3
"""⑧⑨ — 글자를 지우고 번역문을 얹어 완성 페이지를 만든다.

파이프라인에서의 자리:
  ① Magi 가 준 박스 좌표를 여기서 **세 번째로** 쓴다. 읽으려고 자를 때(②),
  지울 마스크를 만들 때(⑦), 그리고 지금 번역문을 놓을 때. 같은 좌표를 쓰므로
  글자가 원래 있던 자리에 정확히 들어간다.

인페인팅에 신경망을 쓰지 않는 이유 (당장은):
  만화 대사는 대부분 흰 말풍선 안에 있어서 주변 색으로 메우는 고전적 방법으로도
  충분히 깨끗하다. 신경망(LaMa 등)이 값을 하는 곳은 **그림 위에 얹힌 효과음**
  처럼 복원할 무늬가 있는 경우다. 먼저 전체가 도는 것을 보고, 필요한 곳에만
  올리는 편이 낫다.

조판은 최소한만 한다:
  이 프로젝트에서 타이포그래피는 우선순위가 낮다고 정했다. 박스에 맞게 줄바꿈
  하고 크기를 맞추는 것까지만 하고, 원본 폰트 재현이나 효과음 변형은 하지 않는다.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mask import mask_for_box  # noqa: E402

DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def wrap_to_box(draw, text, font, max_w):
    """한국어는 공백 단위로 끊되, 한 낱말이 폭을 넘으면 글자 단위로 쪼갠다."""
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        while draw.textlength(ln, font=font) > max_w and len(ln) > 1:
            cut = len(ln)
            while cut > 1 and draw.textlength(ln[:cut], font=font) > max_w:
                cut -= 1
            out.append(ln[:cut])
            ln = ln[cut:]
        if ln:
            out.append(ln)
    return out


def fit_text(draw, text, box_w, box_h, font_path, max_size, min_size, line_gap):
    """박스에 들어가는 가장 큰 글자 크기를 이분 탐색한다."""
    best = None
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        lines = wrap_to_box(draw, text, font, box_w)
        lh = int(mid * line_gap)
        if len(lines) * lh <= box_h:
            best = (font, lines, lh)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = ImageFont.truetype(font_path, min_size)
        best = (font, wrap_to_box(draw, text, font, box_w), int(min_size * line_gap))
    return best


def main():
    p = argparse.ArgumentParser(description="지우기 + 번역문 조판")
    p.add_argument("--translated-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pages", type=int, nargs="+")
    p.add_argument("--font", default=DEFAULT_FONT)
    p.add_argument("--inpaint", choices=["telea", "ns", "none"], default="telea",
                   help="고전 인페인팅 방식. none 이면 지우기만 하고 메우지 않는다")
    p.add_argument("--inpaint-radius", type=int, default=5)
    p.add_argument("--max-font", type=int, default=64)
    p.add_argument("--min-font", type=int, default=11)
    p.add_argument("--line-gap", type=float, default=1.18)
    p.add_argument("--margin", type=float, default=0.06,
                   help="박스 안쪽 여백 비율. 글자가 말풍선 선에 닿지 않게 한다")
    p.add_argument("--skip-kinds", nargs="*", default=["sfx"],
                   help="이 종류는 지우지도 얹지도 않는다 (효과음은 조판이 따로 필요)")
    args = p.parse_args()

    doc = json.load(open(args.translated_json, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    inpaint_flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}.get(args.inpaint)

    for pg in doc["pages"]:
        pno = pg["index"] + 1
        if args.pages and pno not in args.pages:
            continue
        img = cv2.imread(pg["file"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"  p{pno}: 이미지를 못 읽음", file=sys.stderr)
            continue
        H, W = img.shape[:2]

        targets = [t for t in pg["texts"]
                   if (t.get("target") or "").strip() and t.get("kind") not in args.skip_kinds]
        if not targets:
            print(f"  p{pno}: 얹을 번역이 없음")
            continue

        # ⑦ 마스크 — 지울 대상만 모은다
        mask = np.zeros((H, W), np.uint8)
        for t in targets:
            m, box = mask_for_box(img, t["bbox"], 0.12, 12, 0.35, 3, 5)
            if m is None:
                continue
            cx1, cy1, cx2, cy2 = box
            mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(mask[cy1:cy2, cx1:cx2], m)

        # ⑧ 인페인팅
        if inpaint_flag is not None:
            cleaned = cv2.inpaint(img, mask, args.inpaint_radius, inpaint_flag)
        else:
            cleaned = img.copy()
            cleaned[mask > 0] = (255, 255, 255)

        # ⑨ 조판 — PIL 로 넘겨 한글을 그린다
        pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        placed = 0
        for t in targets:
            x1, y1, x2, y2 = (int(round(v)) for v in t["bbox"])
            bw, bh = x2 - x1, y2 - y1
            mx = int(bw * args.margin)
            my = int(bh * args.margin)
            iw, ih = max(8, bw - 2 * mx), max(8, bh - 2 * my)

            font, lines, lh = fit_text(draw, t["target"].strip(), iw, ih,
                                       args.font, args.max_font, args.min_font,
                                       args.line_gap)
            total_h = len(lines) * lh
            cy = y1 + my + max(0, (ih - total_h) // 2)
            for ln in lines:
                tw = draw.textlength(ln, font=font)
                cx = x1 + mx + max(0, (iw - tw) / 2)
                draw.text((cx, cy), ln, font=font, fill=(0, 0, 0))
                cy += lh
            placed += 1

        stem = os.path.splitext(os.path.basename(pg["file"]))[0]
        dest = os.path.join(args.out_dir, f"{stem}_ko.jpg")
        pil.convert("RGB").save(dest, quality=92)
        print(f"  p{pno:02d} 번역 {placed}개 얹음 | 마스크 {100.0*(mask>0).sum()/(H*W):.2f}% → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
