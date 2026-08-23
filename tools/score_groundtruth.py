FIELD = 'ocr'

#!/usr/bin/env python3
"""Pepper&Carrot 정답과 우리 출력을 대조해 채점한다.

왜 필요한가:
  이전까지 모델 평가에 "크롭 패스 전사"를 기준선으로 썼는데 그 기준선도 모델이
  만든 것이었다. 같은 모델의 두 실행은 오류 습관을 공유하므로 **틀려도 일치**한다.
  지표가 "정확함"과 "그 모델과 비슷함"을 섞어버렸다.

  여기서는 SVG 소스에서 뽑은 원문·공식 번역을 정답으로 쓴다. 모델이 개입하지
  않았으므로 어느 모델에도 유리하지 않다.

정렬:
  정답은 말풍선 단위, 우리 것은 Magi 박스 단위다. 좌표계가 같으므로(SVG width/
  height 가 페이지 이미지와 1:1) IoU 로 맞춘다. Magi 가 한 말풍선을 여러 박스로
  쪼갤 수 있어서 **여러 박스 → 한 말풍선** 을 허용하고, 그 경우 읽는 순서대로
  이어 붙여 비교한다.

채점:
  transcription  우리가 읽은 원문 vs 정답 원문 → 완전일치율, 문자오류율(CER)
  translation    우리 한국어 vs 공식 한국어 → **말투 일치율**이 핵심이다.
                 번역은 정답이 하나가 아니라 문자 일치로 재면 안 된다. 하지만
                 존댓말/반말은 객관적으로 비교할 수 있고, 이 프로젝트가 지키려는
                 것이 바로 그것이다.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vlm"))
from validate_translation import classify_register, HONORIFIC  # noqa: E402

WS = re.compile(r"\s+")
PUNCT = re.compile(r"[。、．，！？!?…‥「」『』（）()\"'’”\[\]~〜ー・．,.\-—]+")


def norm(s, drop_punct=True):
    s = WS.sub("", s or "")
    return PUNCT.sub("", s) if drop_punct else s


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_ratio(box, bubble):
    """박스가 말풍선 안에 얼마나 들어가는가. 쪼개진 박스를 모으는 데 쓴다."""
    bx1, by1, bx2, by2 = box
    ix1, iy1 = max(bx1, bubble[0]), max(by1, bubble[1])
    ix2, iy2 = min(bx2, bubble[2]), min(by2, bubble[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (bx2 - bx1) * (by2 - by1)
    return ((ix2 - ix1) * (iy2 - iy1)) / area if area > 0 else 0.0


def similarity(a, b):
    """0~1. 편집거리를 긴 쪽 길이로 정규화한 유사도."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def align_by_text(page_boxes, bubbles, min_sim=0.25):
    """**글자 유사도**로 짝짓는다. 좌표를 쓰지 않는다.

    좌표 정렬을 먼저 시도했다가 버린 이유:
      정답의 <flowRegion> rect 는 글자가 그려지는 자리가 아니라 넉넉하게 잡은
      흐름 영역이다. 한 페이지에서 rect 두 개가 서로 크게 겹치는 경우가 있었고
      (P04: [874-1371] 과 [990-1488]), 렌더 결과와 위아래가 뒤바뀐 경우도 있었다
      (P02). 그 상태로 짝지으면 전사가 맞았는데도 짝이 어긋나 오답으로 집계된다.

      그리고 우리가 채점하려는 것은 **위치가 아니라 글자**다. 좌표를 거칠 이유가
      없다. 위치 정확도를 재고 싶으면 그건 별도 지표로 다뤄야 한다.

    Magi 가 한 말풍선을 여러 박스로 쪼개는 경우가 있으므로, 먼저 이어 붙인 조합도
    후보에 넣는다. 탐욕적으로 가장 닮은 짝부터 확정한다.
    """
    cands = []
    n = len(page_boxes)
    for i in range(n):
        # 연속한 박스 1~3개를 이어 붙인 것까지 후보로 둔다.
        for k in range(1, min(3, n - i) + 1):
            group = page_boxes[i:i + k]
            cands.append((group, norm("".join((t.get(FIELD) or "") for t in group))))

    pairs, used_box, used_bub = [], set(), set()
    scored = []
    for gi, (group, gtext) in enumerate(cands):
        for bi, b in enumerate(bubbles):
            scored.append((similarity(gtext, norm(b["text"])), gi, bi))
    scored.sort(reverse=True)

    # 유사도가 바닥인 짝은 받지 않는다. 받으면 박스 3개를 이어붙인 긴 문자열이
    # 짧은 말풍선에 붙어 편집거리가 폭발한다 (CER 780% 가 그렇게 나왔다).
    for sim, gi, bi in scored:
        if sim < min_sim or bi in used_bub:
            continue
        group = cands[gi][0]
        ids = {id(t) for t in group}
        if ids & used_box:
            continue
        used_box |= ids
        used_bub.add(bi)
        pairs.append((bi, group, sim))

    leftover = [t for t in page_boxes if id(t) not in used_box]
    missing = [i for i in range(len(bubbles)) if i not in used_bub]
    return pairs, leftover, missing


def page_number(pg):
    """정답의 page 번호에 맞춘다.

    예전에는 `index + 1` 이었다. 표지(00.jpg)를 뺀 7장 글롭으로만 돌리던 시절의
    가정인데, 표지를 포함해 8장을 돌리면 전 페이지가 한 칸씩 밀려 정답과
    한 개도 짝이 지어지지 않는다 (실제로 매칭 3% 가 나왔다). 파일명이 페이지
    번호인 소재(Pepper&Carrot 의 00.jpg…07.jpg)에서는 그 번호를 쓴다.
    숫자가 아니면 예전 규칙으로 돌아간다.
    """
    stem = os.path.splitext(os.path.basename(pg.get("file", "")))[0]
    return int(stem) if stem.isdigit() else pg["index"] + 1


def main():
    p = argparse.ArgumentParser(description="정답 대조 채점")
    p.add_argument("--ours", required=True, help="우리 파이프라인 출력 JSON")
    p.add_argument("--gt", required=True, help="pc_groundtruth.py 출력 (대상 언어)")
    p.add_argument("--pivot-gt",
                   help="정렬용 원문 정답. 번역 채점은 이걸 써야 한다 — 우리 한국어와 "
                        "공식 한국어를 글자로 맞추면 좋은 번역이라도 표현이 다르면 "
                        "짝이 안 지어져 표본이 편향된다. 원문끼리 맞추고 그 인덱스로 "
                        "대상 언어를 가져오면 표현 차이와 무관하게 정렬된다")
    p.add_argument("--mode", choices=["transcription", "translation"], required=True)
    p.add_argument("--field", help="비교할 우리 쪽 필드 (기본: ocr/target)")
    p.add_argument("--min-sim", type=float, default=0.25,
                   help="이 유사도 미만이면 짝으로 인정하지 않는다")
    p.add_argument("--show", type=int, default=8, help="어긋난 사례를 몇 개 보일지")
    args = p.parse_args()

    global FIELD
    ours = json.load(open(args.ours, encoding="utf-8"))
    gt = json.load(open(args.gt, encoding="utf-8"))
    pivot = json.load(open(args.pivot_gt, encoding="utf-8")) if args.pivot_gt else None
    pivot_pages = {g["page"]: g["bubbles"] for g in pivot["pages"]} if pivot else None
    field = args.field or ("ocr" if args.mode == "transcription" else "target")
    FIELD = field
    gt_pages = {g["page"]: g["bubbles"] for g in gt["pages"]}

    tot = matched = exact = 0
    ed_sum = ch_sum = 0
    reg_same = reg_cmp = 0
    misses, samples = [], []
    box_over = box_under = 0

    for pg in ours["pages"]:
        pno = page_number(pg)
        bubbles = gt_pages.get(pno)
        if bubbles is None:
            continue
        # 정렬은 원문(pivot)으로, 채점은 대상 언어(bubbles)로 한다.
        align_against = pivot_pages.get(pno, bubbles) if pivot_pages else bubbles
        if pivot_pages:
            globals()["FIELD"] = "ocr"          # 정렬은 우리 원문 판독으로
            pairs, leftover, missing = align_by_text(pg["texts"], align_against, args.min_sim)
            globals()["FIELD"] = field
        else:
            pairs, leftover, missing = align_by_text(pg["texts"], bubbles, args.min_sim)
        box_over += len(leftover)
        box_under += len(missing)
        tot += len(bubbles)
        for i in missing:
            misses.append((pno, bubbles[i]["text"][:36]))

        for bi, boxes, _sim in pairs:
            if bi >= len(bubbles):
                continue
            b = bubbles[bi]
            matched += 1
            got = "".join((t.get(field) or "") for t in boxes)
            ref = b["text"]
            g, r = norm(got), norm(ref)

            if args.mode == "transcription":
                if g == r:
                    exact += 1
                else:
                    samples.append((pno, r, g))
                ed_sum += levenshtein(g, r)
                ch_sum += max(len(r), 1)
            else:
                cg, cr = classify_register(got), classify_register(ref)
                if cg and cr:
                    reg_cmp += 1
                    if HONORIFIC[cg] == HONORIFIC[cr]:
                        reg_same += 1
                    else:
                        samples.append((pno, f"[{cr}] {ref[:30]}", f"[{cg}] {got[:30]}"))

    print(f"정답 말풍선 {tot}개 | 우리 박스가 붙은 것 {matched}개 "
          f"({100*matched/max(tot,1):.0f}%)")
    print(f"검출 누락 {box_under}개 | 말풍선 밖 박스 {box_over}개")

    if args.mode == "transcription":
        cer = ed_sum / max(ch_sum, 1)
        print(f"\n완전일치 {exact}/{matched} ({100*exact/max(matched,1):.0f}%)")
        print(f"문자오류율(CER) {100*cer:.1f}%  → 문자 정확도 {100*(1-cer):.1f}%")
    else:
        print(f"\n말투 비교 가능 {reg_cmp}개 | 일치 {reg_same}개 "
              f"({100*reg_same/max(reg_cmp,1):.0f}%)")

    if samples and args.show:
        print(f"\n어긋난 사례 (앞 {min(args.show, len(samples))}개):")
        for pno, ref, got in samples[:args.show]:
            print(f"  P{pno:02d}\n    정답: {ref[:60]}\n    우리: {got[:60]}")
    if misses and args.show:
        print(f"\n검출 누락 (앞 {min(args.show, len(misses))}개):")
        for pno, ref in misses[:args.show]:
            print(f"  P{pno:02d} {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
