#!/usr/bin/env python3
"""원문 글자 크기를 **측정**한다 — 추정이 아니라.

왜 따로 만드나:
  그동안 글자 크기를 `√(박스 넓이 ÷ 글자 수)` 로 역산했다. 이건 측정이 아니라
  추정이고, 두 가지에 기댄다 — 박스가 글자에 딱 맞는다는 가정, 그리고 OCR 이 센
  글자 수가 맞다는 가정. 둘 다 자주 틀린다. 게다가 말풍선 크기에서 상한을 낼 때도
  같은 공식을 써서, 말풍선 크기와 글자 크기가 서로 얽혀 있었다.

  획 마스크가 있으면 글자 크기를 직접 잴 수 있다. 말풍선 크기와 완전히 독립이다.

방법 — 투영 프로파일:
  세로쓰기는 글자가 열을 이루므로, 마스크를 세로로 합치면 열마다 봉우리가 서고
  열 사이는 골이 된다. 봉우리 수를 세면 열 수이고, 박스 폭 ÷ 열 수 가 글자 크기다.
  가로쓰기는 같은 것을 90 도 돌려서 한다.

  글자 수를 세지 않으므로 OCR 오류에 영향받지 않는다.
"""

import cv2
import numpy as np


def _runs(profile, thresh):
    """프로파일에서 문턱을 넘는 구간(봉우리)의 폭 목록."""
    on = profile > thresh
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append(i - start)
            start = None
    if start is not None:
        runs.append(len(on) - start)
    return runs


def measure_from_mask(stroke_mask, min_run=3):
    """획 마스크에서 (글자 크기, 방향, 근거) 를 잰다. 못 재면 (None, None, 사유).

    두 방향 모두 프로파일을 내서, **봉우리가 더 규칙적인 쪽**을 글줄 방향으로 본다.
    세로쓰기면 열 폭이 고르고, 가로쓰기면 행 높이가 고르다.
    """
    if stroke_mask is None or stroke_mask.size == 0 or not stroke_mask.any():
        return None, None, "마스크 없음"

    m = (stroke_mask > 0).astype(np.uint8)
    h, w = m.shape

    col_prof = m.sum(axis=0).astype(float)   # 세로로 합침 → 열 구조
    row_prof = m.sum(axis=1).astype(float)   # 가로로 합침 → 행 구조

    out = {}
    for name, prof, span in (("vertical", col_prof, w), ("horizontal", row_prof, h)):
        if prof.max() <= 0:
            continue
        runs = [r for r in _runs(prof, prof.max() * 0.12) if r >= min_run]
        if len(runs) < 1:
            continue
        # 봉우리 폭의 중앙값이 글자 크기다. 변동계수가 작을수록 그 방향이 글줄 방향.
        med = float(np.median(runs))
        cv_ = float(np.std(runs) / med) if med > 0 and len(runs) > 1 else 1.0
        out[name] = (med, cv_, len(runs))

    if not out:
        return None, None, "봉우리 없음"
    # 봉우리가 여러 개이고 고른 쪽을 고른다. 하나뿐이면 그 방향은 글줄이 아니다.
    best = min(out.items(),
               key=lambda kv: (kv[1][2] < 2, kv[1][1]))
    name, (med, cv_, n) = best
    return med, name, f"{n}개 봉우리, 변동계수 {cv_:.2f}"


def measure_box(img, bbox, mask_fn):
    """페이지와 박스에서 원문 글자 크기를 잰다. mask_fn 은 make_mask.mask_for_box."""
    m, box = mask_fn(img, bbox)
    if m is None:
        return None, None, "획 검출 실패"
    return measure_from_mask(m)
