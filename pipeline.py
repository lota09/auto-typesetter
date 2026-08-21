#!/usr/bin/env python3
"""파이프라인 드라이버 — 단계를 순서대로 돌린다.

지금까지 단계를 손으로 이어 붙였다. 그러다 보니 중간 산출물 이름이 실행마다
달라지고, 어느 단계가 어떤 파일을 먹는지 기억에 의존하게 됐다. 여기에 모은다.

--fast 가 있는 이유:
  일본어 크롭 전사는 manga-ocr 이 VLM 보다 **25 배 빠르다** (5 초 vs 123 초).
  정확도는 본문 기준 94.4% 대 98.0% 로 VLM 이 앞서지만, 3.6%p 차이로 25 배를
  사는 것이 늘 옳지는 않다. 초안을 빨리 보거나 분량이 많을 때는 fast 가 낫다.

  manga-ocr 은 transformers >= 4.45 를 요구하고 Magiv2 는 4.44.1 에서만 뜬다.
  그래서 ocr/.venv 로 격리했고, 여기서 다른 인터프리터로 호출한다.

  일본어 전용이라는 점에 주의. 중국어·영어 소재에는 쓰지 않는다.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MAGI_PY = os.path.join(ROOT, "magi", ".venv", "bin", "python")
OCR_PY = os.path.join(ROOT, "ocr", ".venv", "bin", "python")

STAGES = ["magi", "read", "merge", "cast", "translate", "validate", "render"]


def run(cmd, label):
    print(f"\n── {label}", flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"   실패 (exit {r.returncode})", file=sys.stderr)
        return False
    print(f"   {time.time()-t:.1f}초", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description="만화 번역 파이프라인")
    p.add_argument("--pages", required=True, help="페이지 glob (읽는 순서대로 정렬됨)")
    p.add_argument("--work", required=True, help="중간 산출물을 둘 디렉터리")
    p.add_argument("--out", help="완성 페이지 디렉터리 (기본: <work>/out)")
    p.add_argument("--styleguide", help="작품 단위 시트. 있으면 물려 쓰고 없으면 만든다")
    p.add_argument("--fast", action="store_true",
                   help="크롭 전사를 manga-ocr 로. 25배 빠르고 정확도는 3.6%%p 낮다. "
                        "일본어 전용")
    p.add_argument("--skip-pages", type=int, nargs="*", default=[],
                   help="처리에서 뺄 페이지(1-base). 후원자 명단·판권지 등")
    p.add_argument("--from-stage", choices=STAGES, default="magi")
    p.add_argument("--only", choices=STAGES, help="이 단계만 돌린다")
    args = p.parse_args()

    w = args.work
    os.makedirs(w, exist_ok=True)
    out_dir = args.out or os.path.join(w, "out")
    sg = args.styleguide or os.path.join(w, "styleguide.json")
    f = lambda n: os.path.join(w, n)

    start = 0 if args.only else STAGES.index(args.from_stage)
    todo = [args.only] if args.only else STAGES[start:]

    if "magi" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "magi", "magi_worker.py"),
                    "--pages", args.pages, "--out", f("magi.json"),
                    "--batch-size", "10", "--no-ocr"], "① Magi 기하학"):
            return 1

    if "read" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_page.py"),
                    "--magi-json", f("magi.json"), "--out", f("page.json"),
                    "--no-cast-memory"], "② 페이지 판독 (화자·언어)"):
            return 1
        if args.fast:
            if not run([OCR_PY, os.path.join(ROOT, "ocr", "manga_ocr_pass.py"),
                        "--magi-json", f("magi.json"), "--out", f("crop.json")],
                       "② 크롭 전사 [fast: manga-ocr]"):
                return 1
        else:
            if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_texts.py"),
                        "--magi-json", f("magi.json"), "--out", f("crop.json")],
                       "② 크롭 전사 [VLM]"):
                return 1

    if "merge" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "merge_reads.py"),
                    "--crop-json", f("crop.json"), "--page-json", f("page.json"),
                    "--out", f("merged.json"), "--prefer", "crop"], "③ 병합"):
            return 1
        if args.skip_pages:
            import json
            d = json.load(open(f("merged.json"), encoding="utf-8"))
            before = len(d["pages"])
            d["pages"] = [pg for pg in d["pages"]
                          if (pg["index"] + 1) not in args.skip_pages]
            json.dump(d, open(f("merged.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"   페이지 {before} → {len(d['pages'])} (제외 {args.skip_pages})")

    if "cast" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "build_cast.py"),
                    "--read-json", f("merged.json"), "--out", f("cast.json")],
                   "④ 인물·스토리"):
            return 1

    if "translate" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "translate_chapter.py"),
                    "--page-json", f("cast.json"), "--out", f("translated.json"),
                    "--styleguide", sg], "⑤ 시트 + 번역"):
            return 1

    if "validate" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "validate_translation.py"),
                    "--translated-json", f("translated.json"), "--styleguide", sg,
                    "--out", f("final.json"), "--repair"], "⑥ 검사·자동수정"):
            return 1

    if "render" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "render", "compose.py"),
                    "--translated-json", f("final.json"), "--out-dir", out_dir],
                   "⑦⑧⑨ 마스크·인페인팅·조판"):
            return 1
        print(f"\n완성 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
