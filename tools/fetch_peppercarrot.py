#!/usr/bin/env python3
"""Pepper&Carrot 에피소드를 언어별로 받는다.

왜 이 작품인가:
  우리 소재(assets/examples)는 전부 발췌라 "작품 전체 스토리"를 뽑을 수 없고,
  번역 결과를 대조할 정답도 없다. Pepper&Carrot 은 그 두 가지를 동시에 준다.
    - 39편 완결. 각 편이 자기완결 스토리다
    - **같은 페이지를 ja / cn / kr / en 으로 제공**한다. 일본어를 원문으로 번역한
      결과를 공식 한국어 렌더링과 대조해 점수를 낼 수 있다
    - CC-BY 4.0 이라 법적으로 깨끗하다 (표기만 하면 상용도 허용)

한계도 분명하다. 서양식 레이아웃(가로쓰기, 좌→우)이라 세로쓰기 열 순서나 만화
우→좌 읽기 순서는 전혀 시험하지 못한다. 기존 소재를 대체하지 않고 보완한다.

언어 코드 주의: 한국어는 `ko` 가 아니라 **`kr`**, 중국어는 `zh` 가 아니라
**`cn`**(간체)이다. 번체는 없다. 웹사이트 UI 언어 경로(`/ko/`)와 만화 번역
코드가 다르므로 UI 쪽으로 판단하면 틀린다.

URL 패턴:
  https://www.peppercarrot.com/0_sources/{ep}/{res}/{lang}_Pepper-and-Carrot_by-David-Revoy_E{NN}P{PP}.jpg
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.peppercarrot.com/0_sources"
STEM = "Pepper-and-Carrot_by-David-Revoy"
UA = "auto-typesetter/0.1 (local research; CC-BY material)"

# 사이트의 실제 디렉터리 이름. 번호만으로는 만들 수 없어서 필요한 것만 적어둔다.
# 전체 목록은 /en/webcomics/peppercarrot.html 의 링크에서 얻을 수 있다.
EPISODES = {
    1: "ep01_Potion-of-Flight",
    11: "ep11_The-Witches-of-Chaosah",
    13: "ep13_The-Pyjama-Party",
    20: "ep20_The-Picnic",
    26: "ep26_Books-Are-Great",
}


def fetch(url, dest, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(dest, "wb") as fh:
                fh.write(data)
            return len(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # 페이지 끝. 오류가 아니다
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(1.5 * attempt)
    return None


def fetch_episode(ep_no, lang, res, out_root, max_pages=40, delay=0.4):
    ep_dir = EPISODES.get(ep_no)
    if not ep_dir:
        print(f"  ep{ep_no:02d}: 디렉터리 이름을 모릅니다. EPISODES 에 추가하세요",
              file=sys.stderr)
        return 0
    dest_dir = os.path.join(out_root, f"ep{ep_no:02d}_{lang}")
    os.makedirs(dest_dir, exist_ok=True)

    got = 0
    # P00 은 표지다. 없는 편도 있으므로 404 를 정상 종료로 취급한다.
    for page in range(0, max_pages):
        fname = f"{lang}_{STEM}_E{ep_no:02d}P{page:02d}.jpg"
        url = f"{BASE}/{ep_dir}/{res}/{fname}"
        dest = os.path.join(dest_dir, f"{page:02d}.jpg")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            got += 1
            continue
        size = fetch(url, dest)
        if size is None:
            # 첫 장(P00)이 없는 것은 표지 없음이라 계속 시도한다.
            if page == 0:
                continue
            if os.path.exists(dest):
                os.remove(dest)
            break
        got += 1
        print(f"    P{page:02d} {size/1024:.0f}KB", flush=True)
        time.sleep(delay)
    print(f"  ep{ep_no:02d} [{lang}] {got}장 → {dest_dir}")
    return got


def main():
    p = argparse.ArgumentParser(description="Pepper&Carrot 에피소드 다운로드")
    p.add_argument("--episodes", type=int, nargs="+", default=[11, 13],
                   help="에피소드 번호 (EPISODES 에 등록된 것만)")
    p.add_argument("--langs", nargs="+", default=["ja", "cn", "kr"],
                   help="언어 코드. 한국어는 kr, 중국어(간체)는 cn 이다")
    p.add_argument("--res", choices=["hi-res", "low-res"], default="hi-res",
                   help="hi-res 는 2481x3503 급이다. 판독 시험에는 이쪽이 맞다")
    p.add_argument("--out", default="assets/examples/peppercarrot",
                   help="저장 위치")
    p.add_argument("--delay", type=float, default=0.4,
                   help="요청 간 간격(초). 공개 서버에 부담을 주지 않는다")
    args = p.parse_args()

    total = 0
    for ep in args.episodes:
        for lang in args.langs:
            total += fetch_episode(ep, lang, args.res, args.out, delay=args.delay)
    print(f"\n합계 {total}장")
    print("출처: Pepper&Carrot — David Revoy, CC-BY 4.0 (https://www.peppercarrot.com)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
