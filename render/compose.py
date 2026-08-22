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
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bubble import typeset_rect_multi  # noqa: E402
from glyph_size import measure_from_mask  # noqa: E402
from make_mask import mask_for_box  # noqa: E402

DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# 줄 앞에 혼자 올 수 없는 글자 (닫는 부호·문장부호). 앞줄 끝에 붙여야 한다.
NO_LINE_START = "…‥。、．，!?！？)]}』」〉》”’·~〜ー・"
# 줄 끝에 혼자 올 수 없는 글자 (여는 부호).
NO_LINE_END = "([{『「〈《“‘"


def _fix_orphans(lines):
    """문장부호가 줄 머리에 혼자 오는 것을 앞줄로 당긴다.

    `네 … 네.` 가 `네` / `…` / `네.` 로 쪼개져 나오는 것을 막는다. 조판에서
    금칙 처리라 부르는 규칙이고, 없으면 부호 한 글자가 줄을 통째로 차지한다.
    """
    out = []
    for ln in lines:
        if out and ln and ln[0] in NO_LINE_START:
            # 앞줄로 옮길 수 있는 만큼 당긴다.
            k = 0
            while k < len(ln) and ln[k] in NO_LINE_START:
                k += 1
            out[-1] += ln[:k]
            ln = ln[k:].lstrip()
            if not ln:
                continue
        while ln and out and out[-1] and out[-1][-1] in NO_LINE_END:
            out[-1] = out[-1][:-1]
            ln = out[-1][-1:] + ln if False else ln
            break
        out.append(ln)
    return [l for l in out if l]


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
    return _fix_orphans(out)


def measured_glyph_px(img, t, mask_fn):
    """원문 글자 크기를 **측정**한다. 실패하면 None.

    말풍선 크기와 완전히 독립이다. 획 마스크의 투영 프로파일에서 글줄 방향의
    봉우리 폭을 재므로, 박스 여백이나 OCR 글자 수 오류에 영향받지 않는다.

    실측 비교 (maid2 p03): 측정 54~56px 로 거의 균일했는데, 면적 역산은 63~77px
    로 흩어지며 20~40% 과대했다. 같은 작품의 대사는 원래 크기가 일정하다.
    """
    m, _ = mask_fn(img, t["bbox"])
    if m is None:
        return None
    g, _dirn, _why = measure_from_mask(m)
    return g


def source_glyph_px(t):
    """원문 글자 하나의 크기를 **면적에서** 역산한다.

    아무도 글자 크기를 알려주지 않는다 — Magi 는 좌표만 주고 VLM 은 글자만 읽는다.
    하지만 박스 넓이를 원문 글자 수로 나누면 글자 하나가 차지한 면적이 나오고,
    그 제곱근이 글자 크기다. CJK 는 정사각형에 가까워 이 근사가 잘 맞는다.

        g ≈ sqrt(박스 넓이 / 글자 수)

    처음에는 `박스 폭 ÷ 열 수` 로 구했는데, 병합 단계가 크롭 패스의 ocr_columns
    를 넘겨주지 않아 열 수가 늘 1 이 되었고 **박스 폭 전체를 글자 하나로** 봤다.
    추정값이 133px 로 부풀어 모든 대사가 말풍선을 넘쳤다. 면적 기반은 그런 부속
    정보에 기대지 않는다. 세로쓰기·가로쓰기를 가릴 필요도 없다.
    """
    x1, y1, x2, y2 = t["bbox"]
    area = max(1.0, (x2 - x1) * (y2 - y1))
    src = (t.get("ocr") or "").strip()
    n = len(re.sub(r"\s+", "", src))
    if n < 1:
        return None
    return (area / n) ** 0.5


def measured_base(img_by_page, pages, mask_fn, cv_reject=0.5):
    """측정값의 중앙값을 기준 크기로. 이상치는 버린다.

    측정이 늘 되는 것은 아니다. 획이 잡음처럼 흩어진 박스에서는 봉우리 폭이
    제각각이라 엉뚱한 값이 나온다 (실측에서 10px 이 한 번 나왔다). 중앙값에서
    크게 벗어난 값은 빼고 센다.
    """
    import statistics
    vals = []
    for pg in pages:
        img = img_by_page.get(pg["index"])
        if img is None:
            continue
        for t in pg["texts"]:
            if not (t.get("target") or "").strip() or t.get("kind") == "sfx":
                continue
            g = measured_glyph_px(img, t, mask_fn)
            if g:
                vals.append(g)
    if not vals:
        return None
    med = statistics.median(vals)
    kept = [v for v in vals if abs(v - med) <= med * cv_reject]
    return statistics.median(kept) if kept else med


def base_glyph_px(pages, scope):
    """기준 글자 크기 = 추정값의 중앙값.

    최대한 크게 채우는 방식은 글자 수와 말풍선 크기의 우연한 조합이 크기를
    정한다. 실측에서 한 페이지 안에 34~64pt, 1.9 배 차이가 났고 같은 5 자짜리
    대사가 34pt 와 64pt 로 갈렸다. 사람이 조판하면 그렇게 하지 않는다 — 한 작품의
    대사 크기는 대체로 일정하고 말풍선이 거기 맞춰 그려진다.

    챕터 단위가 더 일관되지만, 회상처럼 의도적으로 작게 쓴 페이지를 뭉갤 수 있어
    scope 로 고를 수 있게 둔다.
    """
    import statistics
    vals = [g for pg in pages for t in pg["texts"]
            if (t.get("target") or "").strip() and t.get("kind") != "sfx"
            for g in [source_glyph_px(t)] if g]
    return statistics.median(vals) if vals else None


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


def _overlap_ratio(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    ov = max(0, x2 - x1) * max(0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ov / small if small > 0 else 0.0


def split_shared_rects(placements, thresh=0.5):
    """같은 말풍선을 배정받은 대사들을 위아래로 나눠 겹치지 않게 한다.

    세로쓰기 원문에서는 Magi 가 한 말풍선의 열을 **별개 박스**로 잡는 일이 있다.
    그런데 말풍선 검출은 박스마다 따로 도니, 두 박스가 각각 같은 말풍선을 찾아
    거의 같은 자리를 받는다 (실측 겹침 92%, 99%). 각자 그 한가운데 그리면 글자가
    포개져 판독 불가가 된다.

    가로쓰기 서양 레이아웃(ep11)에서는 말풍선당 박스가 하나라 드러나지 않았다.
    말풍선 확장 기능이 만든 회귀다.

    나누는 방향은 위아래다. 한국어는 가로쓰기라 폭을 줄이면 줄바꿈이 망가지지만
    높이는 나눠도 읽는 순서가 유지된다. 몫은 글자 수에 비례해 준다.
    """
    used = [False] * len(placements)
    for i in range(len(placements)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(placements)):
            if used[j]:
                continue
            if _overlap_ratio(placements[i][0], placements[j][0]) >= thresh:
                group.append(j)
        if len(group) == 1:
            continue
        for k in group:
            used[k] = True
        # 원문 읽는 순서(박스 id)대로 위에서 아래로 배치한다.
        group.sort(key=lambda k: placements[k][1])
        x1 = min(placements[k][0][0] for k in group)
        y1 = min(placements[k][0][1] for k in group)
        x2 = max(placements[k][0][2] for k in group)
        y2 = max(placements[k][0][3] for k in group)
        weights = [max(1, placements[k][2]) for k in group]
        total = sum(weights)
        top = y1
        for k, wgt in zip(group, weights):
            share = int((y2 - y1) * wgt / total)
            placements[k] = ((x1, top, x2, min(y2, top + share)),
                             placements[k][1], placements[k][2])
            top += share
    return placements


def area_cap(box_w, box_h, text, fill=0.85):
    """이 자리에 이 글자 수를 넣을 때 가능한 최대 글자 크기.

    원문 크기를 추정할 때 쓴 것과 **같은 공식의 대칭**이다. 원문은 넓이를 글자
    수로 나눠 크기를 역산했고, 여기서는 번역문이 필요로 하는 넓이로 상한을 낸다.

        cap = sqrt(자리 넓이 / 글자 수) × fill

    fill 은 여백·줄간격·줄바꿈 낭비를 감안한 것이다. 완벽하게 채울 수는 없다.

    이 상한이 있어야 하는 이유: 기준 크기만 고집하면 좁은 말풍선에서 한 줄에 한두
    글자만 들어가 세로로 쌓인다. 크기가 균일해도 그렇게 되면 못 읽는다. 자리의
    실제 크기가 상한을 정해야 한다.
    """
    n = max(1, len(re.sub(r"\s+", "", text)))
    return max(1.0, ((box_w * box_h) / n) ** 0.5 * fill)


def fit_at_base(draw, text, box_w, box_h, font_path, base_size, fill,
                line_gap, max_size, min_size):
    """기준 크기와 자리 상한 중 작은 쪽에서 시작해, 실제로 들어갈 때까지 줄인다.

    인위적인 바닥은 두지 않는다. 상한이 자리에서 나오므로 억지로 크게 유지할
    이유가 없고, 그렇게 하면 글자가 세로로 쌓인다.
    """
    top = int(min(max_size, base_size, area_cap(box_w, box_h, text, fill)))
    top = max(min_size, top)
    for size in range(top, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_to_box(draw, text, font, box_w)
        if len(lines) * int(size * line_gap) <= box_h:
            return font, lines, int(size * line_gap), False
    font = ImageFont.truetype(font_path, min_size)
    lines = wrap_to_box(draw, text, font, box_w)
    return font, lines, int(min_size * line_gap), True


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
    p.add_argument("--size-scope", choices=["page", "chapter", "off"], default="chapter",
                   help="기준 글자 크기를 어디서 낼지. off 면 예전처럼 최대한 크게")
    p.add_argument("--size-scale", type=float, default=1.0,
                   help="추정한 원문 크기에 곱할 배수. 한글이 원문보다 크게 보이면 낮춘다")
    p.add_argument("--size-fill", type=float, default=0.85,
                   help="자리 넓이를 글자가 채우는 비율. 낮추면 여유롭게 조판된다")
    p.add_argument("--margin", type=float, default=0.06,
                   help="박스 안쪽 여백 비율. 글자가 말풍선 선에 닿지 않게 한다")
    p.add_argument("--no-bubble", action="store_true",
                   help="말풍선 검출을 끄고 텍스트 박스에 그대로 조판한다")
    p.add_argument("--bubble-tol", type=int, default=18,
                   help="flood fill 색 허용치. 크면 말풍선 밖으로 샌다")
    p.add_argument("--skip-kinds", nargs="*", default=["sfx"],
                   help="이 종류는 지우지도 얹지도 않는다 (효과음은 조판이 따로 필요)")
    args = p.parse_args()

    doc = json.load(open(args.translated_json, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    inpaint_flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}.get(args.inpaint)

    mask_fn = lambda im, bb: mask_for_box(im, bb, 0.12, 12, 0.35, 3, 5)
    chapter_base = None
    if args.size_scope == "chapter":
        imgs = {}
        for pg in doc["pages"]:
            if args.pages and (pg["index"] + 1) not in args.pages:
                continue
            im = cv2.imread(pg["file"], cv2.IMREAD_COLOR)
            if im is not None:
                imgs[pg["index"]] = im
        chapter_base = measured_base(imgs, [pg for pg in doc["pages"]
                                            if pg["index"] in imgs], mask_fn)
        src = "측정"
        if not chapter_base:
            chapter_base = base_glyph_px(doc["pages"], "chapter")
            src = "면적 역산(측정 실패)"
        if chapter_base:
            print(f"챕터 기준 글자 크기 {chapter_base * args.size_scale:.0f}px [{src}]")

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
        page_base = base_glyph_px([pg], "page") if args.size_scope == "page" else None
        placed = widened = overflow = 0
        # 자리를 먼저 전부 정하고 겹침을 푼다. 하나씩 그리면서 정하면 뒤에 오는
        # 대사가 앞 대사 위에 겹쳐 그려진다.
        placements = []
        for t in targets:
            if args.no_bubble:
                rect, used = tuple(int(round(v)) for v in t["bbox"]), False
            else:
                rect, used = typeset_rect_multi(img, t["bbox"])
            widened += bool(used)
            placements.append((rect, t.get("id", 0), len((t.get("target") or "").strip())))
        placements = split_shared_rects(placements)

        for t, (rect, _id, _n) in zip(targets, placements):
            x1, y1, x2, y2 = rect
            bw, bh = x2 - x1, y2 - y1
            mx = int(bw * args.margin)
            my = int(bh * args.margin)
            iw, ih = max(8, bw - 2 * mx), max(8, bh - 2 * my)

            base = chapter_base if args.size_scope == "chapter" else page_base
            if base:
                font, lines, lh, of = fit_at_base(
                    draw, t["target"].strip(), iw, ih, args.font,
                    base * args.size_scale, args.size_fill, args.line_gap,
                    args.max_font, args.min_font)
                overflow += of
            else:
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
        print(f"  p{pno:02d} 번역 {placed}개 (말풍선 {widened}, 넘침 {overflow}) | "
              f"마스크 {100.0*(mask>0).sum()/(H*W):.2f}% → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
