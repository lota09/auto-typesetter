#!/usr/bin/env python3
"""크롭 패스와 페이지 패스를 합친다.

두 패스를 다 돌리는 이유는 실측으로 우열이 갈렸기 때문이다 (66개 중 23개 불일치):

  중국어 세로쓰기  크롭이 우세. 밀집 다열은 확대해서 봐야 읽힌다.
                   `雖然我討厭服務主人` (크롭) vs `雖然討厭服務主人…` (페이지, 我 누락)
  일본어 짧은 발화  페이지가 우세. 앞뒤 맥락이 글자를 확정해 준다.
                   `よっしゃ〜` (페이지) vs `はちわら` (크롭, 파손)

화자·수신자·종류는 항상 페이지 패스를 쓴다. 크롭만 봐서는 알 수 없는 정보다.

그리고 불일치 자체가 쓸모 있다. 정답 데이터가 없어도 **두 독립 판독이 어긋난
지점**은 오류일 확률이 높으므로, disagree 플래그를 남겨 뒤 단계가 쓰게 한다.
"""

import argparse
import json
import os
import re
import sys

HAN = re.compile(r"[㐀-䶿一-鿿]")
KANA = re.compile(r"[぀-ヿ]")


def pick(crop_text, page_text, lang, prefer):
    """어느 판독을 본문으로 쓸지 정한다. 돌려주는 값은 (본문, 출처)."""
    if not crop_text and not page_text:
        return None, None
    if not crop_text:
        return page_text, "page"
    if not page_text:
        return crop_text, "crop"
    if prefer == "crop":
        return crop_text, "crop"
    if prefer == "page":
        return page_text, "page"
    # auto: 중국어면 크롭, 그 외에는 페이지. 위 docstring 의 실측 근거를 따른다.
    if lang == "zh":
        return crop_text, "crop"
    return page_text, "page"


def main():
    p = argparse.ArgumentParser(description="크롭 판독과 페이지 판독 병합")
    p.add_argument("--crop-json", required=True, help="read_texts.py 출력")
    p.add_argument("--page-json", required=True, help="read_page.py 출력")
    p.add_argument("--out", required=True)
    p.add_argument("--prefer", choices=["auto", "crop", "page"], default="auto",
                   help="본문으로 쓸 판독 (기본 auto: 중국어는 크롭, 그 외 페이지)")
    args = p.parse_args()

    crop = json.load(open(args.crop_json, encoding="utf-8"))
    doc = json.load(open(args.page_json, encoding="utf-8"))

    crop_ocr = {(pg["index"], t["id"]): (t.get("ocr") or "").strip()
                for pg in crop["pages"] for t in pg["texts"]}
    # 열 정보도 함께 옮긴다. 크롭 패스가 세로쓰기 순서를 지키려고 나눠둔 것인데,
    # 뒤 단계(글자 크기 추정)가 열 수를 쓸 수 있다. 안 넘겨서 열 수가 늘 1 로
    # 잡히는 버그가 있었다.
    crop_cols = {(pg["index"], t["id"]): (t.get("ocr_columns") or [])
                 for pg in crop["pages"] for t in pg["texts"]}

    n = disagree = filled = 0
    for pg in doc["pages"]:
        for t in pg["texts"]:
            n += 1
            c = crop_ocr.get((pg["index"], t["id"]), "")
            g = (t.get("ocr") or "").strip()
            lang = t.get("lang") or pg.get("source_lang")

            t["ocr_crop"], t["ocr_page"] = c or None, g or None
            cols = crop_cols.get((pg["index"], t["id"]))
            if cols:
                t["ocr_columns"] = cols
            text, src = pick(c, g, lang, args.prefer)
            t["ocr"], t["ocr_source"] = text, src
            if text:
                filled += 1
            # 공백만 다른 경우는 불일치로 세지 않는다. 세로쓰기 열 구분에서
            # 생기는 차이라 판독이 어긋난 것이 아니다.
            if c and g and re.sub(r"\s+", "", c) != re.sub(r"\s+", "", g):
                t["disagree"] = True
                disagree += 1
            else:
                t.pop("disagree", None)

    doc["merge"] = {"prefer": args.prefer,
                    "crop_json": os.path.abspath(args.crop_json),
                    "page_json": os.path.abspath(args.page_json)}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    by_src = {}
    for pg in doc["pages"]:
        for t in pg["texts"]:
            by_src[t.get("ocr_source")] = by_src.get(t.get("ocr_source"), 0) + 1
    print(f"박스 {n}개 | 본문 확보 {filled}개 | 불일치 {disagree}개")
    print(f"출처: {', '.join(f'{k}={v}' for k, v in sorted(by_src.items(), key=lambda x: str(x[0])))}")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
