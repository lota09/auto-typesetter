#!/usr/bin/env python3
"""말풍선 영역을 찾아 조판할 사각형을 준다 — 모델 없이 flood fill 로.

왜 필요한가:
  Magi 는 **텍스트 박스**만 준다. 원문이 세로쓰기면 그 박스도 세로로 길쭉하다.
  그런데 한국어는 가로쓰기라 그 모양에 조판하면 한두 글자씩 끊겨 세로로 쌓인다.
  실제로 첫 결과물에서 그렇게 나왔다.

  지울 때는 텍스트 박스로 충분했다. 글자 획만 찾으면 되니까. 하지만 **얹을 때는
  말풍선 모양**이 필요하다. 같은 좌표를 쓰는데 요구가 다르다.

왜 검출 모델을 안 붙이나:
  말풍선은 대개 균일한 밝은 면이다. 글자 사이의 배경에서 시작해 색이 비슷한
  곳으로 번져 나가면 경계까지 저절로 도달한다. 모델을 하나 더 얹기 전에 이쪽을
  먼저 시도하는 것이 맞다.

실패하면 조용히 원래 박스로 돌아간다. 말풍선이 없는 대사(그림 위 글자)나 열린
말풍선에서는 번짐이 페이지 전체로 새는데, 그 경우를 넓이로 판정해 버린다.
"""

import cv2
import numpy as np


def bubble_mask(img, bbox, search_scale=3.0, tol=18, max_area_ratio=0.5):
    """텍스트 박스 주변에서 말풍선 안쪽 면을 찾는다.

    반환: (마스크, 탐색창 좌표) 또는 (None, None)

    tol 은 flood fill 의 색 허용치다. 크게 잡으면 말풍선 밖으로 새고, 작게 잡으면
    스크린톤이나 그라데이션에서 멈춘다.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    # 탐색창을 박스보다 넉넉히 잡되 페이지를 벗어나지 않게 한다.
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = int(bw * search_scale / 2) + 20
    half_h = int(bh * search_scale / 2) + 20
    sx1, sy1 = max(0, cx - half_w), max(0, cy - half_h)
    sx2, sy2 = min(W, cx + half_w), min(H, cy + half_h)
    win = img[sy1:sy2, sx1:sx2]
    if win.size == 0:
        return None, None

    wh, ww = win.shape[:2]
    gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)

    # 씨앗: 텍스트 박스 안쪽에서 **글자가 아닌** 밝은 화소를 고른다.
    # 글자 획에서 시작하면 획만 번지고 끝난다.
    tb = (x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1)
    inner = gray[max(0, tb[1]):min(wh, tb[3]), max(0, tb[0]):min(ww, tb[2])]
    if inner.size == 0:
        return None, None
    # 박스 안 화소의 상위 밝기 쪽이 배경(말풍선 면)일 가능성이 높다.
    bg_level = int(np.percentile(inner, 80))
    ys, xs = np.where(inner >= bg_level)
    if len(xs) == 0:
        return None, None

    flood = np.zeros((wh + 2, ww + 2), np.uint8)
    filled = np.zeros((wh, ww), np.uint8)
    # 씨앗을 몇 개 흩어 놓는다. 한 점이 우연히 글자에 걸려도 나머지가 건진다.
    for i in np.linspace(0, len(xs) - 1, num=min(8, len(xs))).astype(int):
        sx = int(xs[i]) + max(0, tb[0])
        sy = int(ys[i]) + max(0, tb[1])
        if filled[sy, sx]:
            continue
        m = np.zeros((wh + 2, ww + 2), np.uint8)
        cv2.floodFill(win.copy(), m, (sx, sy), 255,
                      (tol,) * 3, (tol,) * 3,
                      cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
        filled |= m[1:-1, 1:-1]

    if not filled.any():
        return None, None
    # 페이지로 새어 나간 경우를 버린다. 말풍선이 탐색창을 다 채울 수는 없다.
    if filled.mean() > max_area_ratio * 255:
        return None, None

    # **구멍을 메운다.** 번진 것은 배경뿐이라 글자 획이 구멍으로 남는다. 그대로
    # 두면 내접 사각형이 글자 사이를 비집고 들어가 원래 박스보다도 작아진다
    # (실측: 0.04~0.37배). 바깥 윤곽만 취해 속을 채우면 말풍선 면 전체가 된다.
    cnts, _ = cv2.findContours((filled > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cxr, cyr = (x1 + x2) // 2 - sx1, (y1 + y2) // 2 - sy1
    # 원래 글자 자리를 품는 윤곽을 고른다. 없으면 가장 큰 것.
    inside = [c for c in cnts
              if cv2.pointPolygonTest(c, (float(cxr), float(cyr)), False) >= 0]
    pick = max(inside or cnts, key=cv2.contourArea)
    solid = np.zeros(filled.shape, np.uint8)
    cv2.drawContours(solid, [pick], -1, 1, thickness=-1)
    # **가둠 검사.** 채운 영역이 탐색창 테두리에 닿았으면 말풍선 안에서 멈춘 게
    # 아니라 페이지로 샌 것이다. 말풍선 없는 대사(열린 배경 위 글자)에서 이런 일이
    # 난다 — 조판 자리로는 그럭저럭 쓸 수 있지만 **마스크로 쓰면 그림을 지운다**
    # (실측 ep11cn 3쪽: 마스크 1.6% → 19.9%, 패널 두 개의 그림이 뭉개졌다).
    enclosed = not (solid[0, :].any() or solid[-1, :].any()
                    or solid[:, 0].any() or solid[:, -1].any())
    return solid, (sx1, sy1, sx2, sy2, enclosed)


def largest_inscribed_rect(mask, downscale=4):
    """마스크 안에 들어가는 가장 큰 축 정렬 사각형.

    히스토그램 기반 최대 직사각형 알고리즘을 축소본에 적용한다. 원본 해상도로
    돌리면 느리고, 조판 자리를 정하는 데 그 정도 정밀도는 필요 없다.
    """
    small = mask[::downscale, ::downscale]
    h, w = small.shape
    if h == 0 or w == 0:
        return None
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)          # area, x1, y1, x2, y2  (축소 좌표)

    for r in range(h):
        heights = np.where(small[r] > 0, heights + 1, 0)
        stack = []
        for c in range(w + 1):
            cur = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][1] >= cur:
                sc, sh = stack.pop()
                area = sh * (c - sc)
                if area > best[0]:
                    best = (area, sc, r - sh + 1, c, r + 1)
                start = sc
            stack.append((start, cur))

    if best[0] == 0:
        return None
    _, x1, y1, x2, y2 = best
    return (x1 * downscale, y1 * downscale, x2 * downscale, y2 * downscale)


TOL_LADDER = (12, 18, 26, 36)


# 탐색창 배수 사다리. **텍스트 박스가 작으면 창도 작아져 말풍선을 통째로 놓친다** —
# 그게 "글자가 불합리하게 작다" 의 원인이었다. 실측(maid2 12쪽):
#   `早安` 74x128 → 배수 3·5 실패, 배수 8 에서 11.27 배 사각형을 찾았다
#   `我為您送上餐點` 95x223 → 배수 3 실패, 배수 8 에서 9.50 배
# 원문이 두 글자여도 말풍선은 크다. 한국어는 길어지므로 그 자리를 써야 한다.
SCALE_LADDER = (3.0, 5.0, 8.0)


def typeset_rect_multi(img, bbox, min_gain=1.15, tols=TOL_LADDER,
                       scales=SCALE_LADDER, require_enclosed=False, **kw):
    """허용치와 탐색창 배수를 훑어 가장 넓은 유효 사각형을 고른다.

    실패 원인이 반대 방향으로 갈리기 때문이다 (실측 92 개 중):
      대비 부족 9 개  — 허용치가 작아 말풍선 안에서 멈춘다 → 키워야 한다
      번짐 유출 7 개  — 허용치가 커서 페이지로 샌다 → 줄여야 한다
    고정값 하나로 둘 다 맞출 수 없다. 사다리로 훑고 **안전 검사를 통과한 것 중
    가장 넓은 것**을 쓴다. 유출은 기존 면적 검사(max_area_ratio)와 중심 포함
    검사가 걸러내므로 큰 값을 시도해도 위험이 늘지 않는다.
    """
    best, best_area, used = None, 0, False
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    kw.pop("search_scale", None)
    for scale in scales:
        for tol in tols:
            rect, ok = typeset_rect(img, bbox, min_gain=min_gain, tol=tol,
                                    search_scale=scale,
                                    require_enclosed=require_enclosed, **kw)
            if not ok:
                continue
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            if area > best_area:
                best, best_area, used = rect, area, True
    return (best, True) if used else ((x1, y1, x2, y2), False)


def typeset_rect(img, bbox, min_gain=1.15, require_enclosed=False, **kw):
    """조판할 사각형을 돌려준다. 말풍선을 못 찾으면 원래 박스를 그대로.

    min_gain: 말풍선에서 얻은 사각형이 원래 박스보다 이 배수 이상 넓어야 채택한다.
    비슷하면 굳이 바꿀 이유가 없고, 잘못 번진 결과를 받을 위험만 는다.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    base = (x1, y1, x2, y2)
    base_area = max(1, (x2 - x1) * (y2 - y1))

    m, win = bubble_mask(img, bbox, **kw)
    if m is None:
        return base, False
    rect = largest_inscribed_rect(m)
    if rect is None:
        return base, False

    sx1, sy1 = win[0], win[1]
    enclosed = win[4] if len(win) > 4 else True
    rx1, ry1, rx2, ry2 = rect[0] + sx1, rect[1] + sy1, rect[2] + sx1, rect[3] + sy1
    area = max(1, (rx2 - rx1) * (ry2 - ry1))
    if area < base_area * min_gain:
        return base, False
    # 원래 글자 자리를 포함하지 않으면 엉뚱한 곳을 잡은 것이다.
    if not (rx1 <= (x1 + x2) // 2 <= rx2 and ry1 <= (y1 + y2) // 2 <= ry2):
        return base, False
    return (rx1, ry1, rx2, ry2), (True if not require_enclosed else enclosed)
