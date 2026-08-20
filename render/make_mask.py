#!/usr/bin/env python3
"""지우기용 마스크를 만든다 — 박스가 아니라 **글자 획**만 고른다.

왜 세그멘테이션 모델이 필요 없나:
  Magi 는 박스만 주고 마스크를 주지 않는다. 그래서 처음엔 세그멘테이션 모델을
  하나 더 붙여야 한다고 봤는데, comic-translate 를 보니 고전 CV 로 풀고 있었다 —
  박스 안에서 연결요소를 찾아 글자 획만 남기고, 안티에일리어싱을 잡을 만큼만
  팽창시킨다. 입력이 박스 좌표라 우리 Magi 출력과 그대로 맞는다.

박스를 통째로 칠하면 안 되는 이유:
  말풍선은 대개 흰 바탕이라 박스를 칠해도 티가 덜 나지만, 그림 위에 얹힌 대사나
  효과음은 박스를 칠하는 순간 그림이 사라진다. 획만 지우고 인페인팅에 넘겨야
  복원이 자연스럽다.

양극성을 모두 보는 이유:
  만화는 흰 바탕에 검은 글자도, 검은 바탕에 흰 글자도 흔하다. Otsu 문턱 하나로
  한쪽만 취하면 절반을 놓친다.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def _components(binary, min_area, max_area_ratio, box_rect):
    """연결요소 중 글자 획으로 볼 만한 것만 남긴다.

    거르는 것:
      - 너무 작은 것 → 스크린톤 점·노이즈
      - 크롭 면적의 상당 부분을 차지하는 것 → 배경이나 말풍선 안쪽 면
      - 크롭 가장자리를 가로지르는 가늘고 긴 것 → 말풍선 윤곽선
      - 원래 박스(패딩 제외) 밖에만 있는 것 → 옆 말풍선의 글자
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros(binary.shape, np.uint8)
    h, w = binary.shape
    crop_area = h * w
    bx1, by1, bx2, by2 = box_rect
    keep = np.zeros(binary.shape, np.uint8)

    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        if area > crop_area * max_area_ratio:
            continue
        # 크롭의 거의 전폭/전고를 차지하면서 얇으면 윤곽선이다.
        if (cw > w * 0.92 or ch > h * 0.92) and area < 0.25 * cw * ch:
            continue
        # 원래 박스와 겹치지 않으면 남의 글자다.
        ix1, iy1 = max(x, bx1), max(y, by1)
        ix2, iy2 = min(x + cw, bx2), min(y + ch, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        keep[labels == i] = 255
    return keep


def mask_for_box(image, bbox, pad_ratio, min_area, max_area_ratio,
                 close_px, dilate_px):
    """박스 하나의 마스크와 그 크롭 위치를 돌려준다."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    pad = int(pad_ratio * max(2, min(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return None, None

    crop = image[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Otsu 문턱을 구하고 양쪽 극성을 모두 후보로 삼는다.
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray < thr).astype(np.uint8) * 255
    light = (gray > thr).astype(np.uint8) * 255

    box_rect = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    m = cv2.bitwise_or(
        _components(dark, min_area, max_area_ratio, box_rect),
        _components(light, min_area, max_area_ratio, box_rect))
    if not m.any():
        return None, None

    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        m = cv2.dilate(m, k, iterations=1)
    return m, (cx1, cy1, cx2, cy2)


def main():
    p = argparse.ArgumentParser(description="글자 획 마스크 생성")
    p.add_argument("--read-json", required=True, help="박스가 들어 있는 파이프라인 JSON")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pages", type=int, nargs="+", help="이 페이지(1-base)만")
    p.add_argument("--pad-ratio", type=float, default=0.12,
                   help="박스 짧은 변 대비 크롭 여백. 획이 박스를 살짝 넘는 경우를 잡는다")
    p.add_argument("--min-area", type=int, default=12, help="이보다 작은 연결요소는 노이즈")
    p.add_argument("--max-area-ratio", type=float, default=0.35,
                   help="크롭 면적의 이 비율을 넘으면 배경으로 본다")
    p.add_argument("--close", type=int, default=3, help="닫힘 커널 픽셀")
    p.add_argument("--dilate", type=int, default=5,
                   help="팽창 커널 픽셀. 안티에일리어싱 테두리를 덮을 만큼만")
    p.add_argument("--overlay", action="store_true",
                   help="원본 위에 마스크를 얹은 확인용 이미지도 저장")
    args = p.parse_args()

    doc = json.load(open(args.read_json, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)

    total_boxes = covered = 0
    for pg in doc["pages"]:
        pno = pg["index"] + 1
        if args.pages and pno not in args.pages:
            continue
        img = cv2.imread(pg["file"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"  p{pno}: 이미지를 못 읽었습니다: {pg['file']}", file=sys.stderr)
            continue
        H, W = img.shape[:2]
        mask = np.zeros((H, W), np.uint8)

        for t in pg["texts"]:
            total_boxes += 1
            m, box = mask_for_box(img, t["bbox"], args.pad_ratio, args.min_area,
                                  args.max_area_ratio, args.close, args.dilate)
            if m is None:
                continue
            covered += 1
            cx1, cy1, cx2, cy2 = box
            roi = mask[cy1:cy2, cx1:cx2]
            mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(roi, m)

        stem = os.path.splitext(os.path.basename(pg["file"]))[0]
        dest = os.path.join(args.out_dir, f"{stem}_mask.png")
        cv2.imwrite(dest, mask)
        ratio = 100.0 * (mask > 0).sum() / (H * W)
        line = f"  p{pno:02d} 박스 {len(pg['texts']):2d}개 | 마스크 {ratio:5.2f}% → {dest}"

        if args.overlay:
            ov = img.copy()
            ov[mask > 0] = (0, 0, 255)
            ov = cv2.addWeighted(img, 0.45, ov, 0.55, 0)
            odest = os.path.join(args.out_dir, f"{stem}_overlay.jpg")
            cv2.imwrite(odest, ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
            line += f" | {odest}"
        print(line, flush=True)

    print(f"\n박스 {total_boxes}개 중 획을 찾은 것 {covered}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
