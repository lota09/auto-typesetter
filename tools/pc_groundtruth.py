#!/usr/bin/env python3
"""Pepper&Carrot 번역 소스(SVG)에서 정답 데이터를 뽑는다.

왜 필요한가:
  지금까지 모델 평가에 "크롭 패스 전사"를 기준선으로 썼는데, 그 기준선을 만든
  것도 모델이었다. `unknown1` 의 경우 Qwen3.6 이 만들었고, 그래서 Qwen3.6 은
  자기 전사와 비교되고 Gemma 는 남의 전사와 비교됐다. 같은 모델의 두 실행은
  오류 습관을 공유하므로 **틀려도 일치**한다 — 지표가 "정확함"과 "Qwen3.6 과
  비슷함"을 섞어버린다.

  Pepper&Carrot 은 번역을 SVG 소스로 관리한다. 말풍선 텍스트가 그 안에 문자로
  들어 있어서, 이미지를 읽지 않고 원문을 그대로 얻을 수 있다. 모델이 개입하지
  않은 진짜 정답이다. ja/cn/kr/en 이 모두 있으므로 **전사 정확도와 번역 품질을
  동시에** 채점할 수 있다.

구조:
  Inkscape flowed text 라 <text> 가 아니라 <flowRoot> 다.
    <flowRoot transform="matrix(...)">
      <flowRegion><rect x y width height/></flowRegion>
      <flowPara>줄</flowPara> ...
  SVG 의 width/height 가 페이지 이미지 해상도와 1:1 이라 좌표를 그대로 픽셀로
  쓸 수 있다. 다만 transform 을 rect 에 적용해야 실제 위치가 나온다.

출처: Pepper&Carrot — David Revoy, CC-BY 4.0 (https://www.peppercarrot.com)
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.peppercarrot.com/0_sources"
UA = "auto-typesetter/0.1 (local research; CC-BY material)"

EPISODES = {
    1: "ep01_Potion-of-Flight",
    11: "ep11_The-Witches-of-Chaosah",
    13: "ep13_The-Pyjama-Party",
    20: "ep20_The-Picnic",
    26: "ep26_Books-Are-Great",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_matrix(transform):
    """transform 속성에서 아핀 행렬 (a,b,c,d,e,f) 을 얻는다.

    matrix() 외에 translate()/scale() 도 쓰이므로 셋 다 처리한다. 없으면 항등.
    """
    if not transform:
        return (1, 0, 0, 1, 0, 0)
    m = re.search(r"matrix\(([^)]*)\)", transform)
    if m:
        v = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
        if len(v) == 6:
            return tuple(v)
    a, d, e, f = 1.0, 1.0, 0.0, 0.0
    m = re.search(r"translate\(([^)]*)\)", transform)
    if m:
        v = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
        e = v[0]
        f = v[1] if len(v) > 1 else 0.0
    m = re.search(r"scale\(([^)]*)\)", transform)
    if m:
        v = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
        a = v[0]
        d = v[1] if len(v) > 1 else v[0]
    return (a, 0.0, 0.0, d, e, f)


def apply_matrix(mat, x, y):
    a, b, c, d, e, f = mat
    return (a * x + c * y + e, b * x + d * y + f)


def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def extract(svg):
    """flowRoot 단위로 (bbox, 줄 목록) 을 뽑는다."""
    out = []
    for fr in re.findall(r"<flowRoot\b.*?</flowRoot>", svg, re.S):
        tm = re.search(r'transform="([^"]*)"', fr)
        mat = parse_matrix(tm.group(1) if tm else None)

        rect = re.search(r"<rect\b[^>]*>", fr)
        bbox = None
        if rect:
            g = lambda k: float(re.search(k + r'="([-\d.eE]+)"', rect.group(0)).group(1)) \
                if re.search(k + r'="([-\d.eE]+)"', rect.group(0)) else 0.0
            x, y, w, h = g("x"), g("y"), g("width"), g("height")
            # 변환이 회전·기울임을 포함할 수 있으므로 네 꼭짓점을 모두 옮긴 뒤
            # 축 정렬 경계상자를 다시 만든다.
            pts = [apply_matrix(mat, px, py)
                   for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        lines = [strip_tags(p) for p in
                 re.findall(r"<flowPara\b[^>]*>(.*?)</flowPara>", fr, re.S)]
        lines = [l for l in lines if l]
        if not lines:
            continue
        out.append({"bbox": bbox, "lines": lines, "text": " ".join(lines)})
    # 읽는 순서를 흉내 낸다 (위→아래, 왼→오른). 서양식 레이아웃 기준이다.
    out.sort(key=lambda b: (round(b["bbox"][1] / 100) if b["bbox"] else 0,
                            b["bbox"][0] if b["bbox"] else 0))
    return out


def main():
    p = argparse.ArgumentParser(description="Pepper&Carrot 정답 데이터 추출")
    p.add_argument("--episodes", type=int, nargs="+", default=[11])
    p.add_argument("--langs", nargs="+", default=["ja", "kr", "cn", "en"])
    p.add_argument("--out", default="assets/groundtruth/peppercarrot")
    p.add_argument("--max-pages", type=int, default=40)
    p.add_argument("--delay", type=float, default=0.3)
    args = p.parse_args()

    total_b = 0
    for ep in args.episodes:
        ep_dir = EPISODES.get(ep)
        if not ep_dir:
            print(f"ep{ep:02d}: EPISODES 에 없습니다", file=sys.stderr)
            continue
        for lang in args.langs:
            pages = []
            for page in range(0, args.max_pages):
                url = f"{BASE}/{ep_dir}/lang/{lang}/E{ep:02d}P{page:02d}.svg"
                svg = fetch(url)
                if svg is None:
                    if page == 0:
                        continue
                    break
                bubbles = extract(svg)
                pages.append({"page": page, "bubbles": bubbles})
                total_b += len(bubbles)
                time.sleep(args.delay)
            if not pages:
                print(f"  ep{ep:02d} [{lang}] 없음")
                continue
            os.makedirs(args.out, exist_ok=True)
            dest = os.path.join(args.out, f"ep{ep:02d}_{lang}.json")
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump({"episode": ep, "lang": lang, "source": ep_dir,
                           "license": "CC-BY 4.0, David Revoy, peppercarrot.com",
                           "pages": pages}, fh, ensure_ascii=False, indent=1)
            nb = sum(len(x["bubbles"]) for x in pages)
            print(f"  ep{ep:02d} [{lang}] {len(pages)}장, 말풍선 {nb}개 → {dest}")
    print(f"\n합계 말풍선 {total_b}개")
    print("출처: Pepper&Carrot — David Revoy, CC-BY 4.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
