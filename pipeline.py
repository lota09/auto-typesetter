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

ROUTER_LOG = os.path.join(ROOT, "logs", "llama-router.log")
LLAMA_DIR = "/home/myubuntu/llamacpp/llama_server_rocm"
# pkill 패턴을 통짜 문자열로 두면, 이 파일을 인자로 실행하는 셸의 명령줄에도
# 그 문자열이 있어서 pkill 이 자기 자신을 죽인다. 실제로 세 번 당했다.
ROUTER_PAT = "llama_server" + "_rocm/llama-" + "server"


def router_stop():
    """라우터를 내려 VRAM 을 비운다.

    Magi 는 PyTorch 로 GPU 를 직접 쓰는데 llama-server 가 28~30GB 를 쥐고 있으면
    2GB 도 못 얻어 OOM 으로 죽는다. 실제로 그렇게 실패했다.
    /models/unload 엔드포인트와 자식 프로세스만 죽이는 방법을 둘 다 시도했지만
    VRAM 이 반환되지 않았다. 프로세스를 내리는 것이 확실하다.
    """
    subprocess.run(["pkill", "-f", ROUTER_PAT], stderr=subprocess.DEVNULL)
    time.sleep(5)


def router_start(timeout=180):
    """라우터를 띄우고 응답할 때까지 기다린다."""
    import urllib.request
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LLAMA_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    os.makedirs(os.path.dirname(ROUTER_LOG), exist_ok=True)
    with open(ROUTER_LOG, "w") as log:
        subprocess.Popen(
            [os.path.join(LLAMA_DIR, "llama-server"),
             "--models-preset", os.path.join(ROOT, "config", "models.ini"),
             "--models-max", "1", "--models-autoload",
             "--host", "127.0.0.1", "--port", "8081"],
            stdout=log, stderr=log, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:8081/v1/models", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def is_fresh(out_path, *inputs):
    """산출물이 이미 있고 입력보다 새로우면 그 단계를 건너뛴다.

    77 페이지를 1441 초 걸려 판독했는데 한 장이 깨져 전체가 실패로 끝난 적이 있다.
    다시 돌리면 처음부터 하게 되므로, 끝난 단계는 넘어가야 한다.
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    t = os.path.getmtime(out_path)
    return all(not os.path.exists(i) or os.path.getmtime(i) <= t for i in inputs)


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
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="이미 끝난 단계도 다시 돌린다")
    p.add_argument("--no-manage-router", dest="manage_router", action="store_false",
                   help="라우터를 직접 띄우고 내리지 않는다 (직접 관리할 때)")
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

    if "magi" in todo and not (args.resume and is_fresh(f("magi.json"))):
        if args.manage_router:
            print("── 라우터 정지 (Magi 가 VRAM 을 써야 한다)")
            router_stop()
        if not run([MAGI_PY, os.path.join(ROOT, "magi", "magi_worker.py"),
                    "--pages", args.pages, "--out", f("magi.json"),
                    "--batch-size", "10", "--no-ocr"], "① Magi 기하학"):
            return 1

    if args.skip_pages and os.path.exists(f("magi.json")):
        import json as _json
        d = _json.load(open(f("magi.json"), encoding="utf-8"))
        before = len(d["pages"])
        d["pages"] = [pg for pg in d["pages"] if (pg["index"] + 1) not in args.skip_pages]
        if len(d["pages"]) != before:
            _json.dump(d, open(f("magi.json"), "w", encoding="utf-8"),
                       ensure_ascii=False, indent=1)
            print(f"   제외 적용: 페이지 {before} → {len(d['pages'])} {args.skip_pages}")

    if args.manage_router and set(todo) & {"read", "cast", "translate", "validate"}:
        print("── 라우터 기동")
        if not router_start():
            print("   라우터가 응답하지 않습니다", file=sys.stderr)
            return 1

    if "read" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_page.py"),
                    "--magi-json", f("magi.json"), "--out", f("page.json"),
                    "--no-cast-memory"] + ([] if args.resume else ["--no-resume"]),
                   "② 페이지 판독 (화자·언어)"):
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

    if "merge" in todo and not (args.resume and is_fresh(f("merged.json"), f("crop.json"), f("page.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "merge_reads.py"),
                    "--crop-json", f("crop.json"), "--page-json", f("page.json"),
                    "--out", f("merged.json"), "--prefer", "crop"], "③ 병합"):
            return 1

    if "cast" in todo and not (args.resume and is_fresh(f("cast.json"), f("merged.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "build_cast.py"),
                    "--read-json", f("merged.json"), "--out", f("cast.json")],
                   "④ 인물·스토리"):
            return 1

    if "translate" in todo and not (args.resume and is_fresh(f("translated.json"), f("cast.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "translate_chapter.py"),
                    "--page-json", f("cast.json"), "--out", f("translated.json"),
                    "--styleguide", sg], "⑤ 시트 + 번역"):
            return 1

    if "validate" in todo and not (args.resume and is_fresh(f("final.json"), f("translated.json"))):
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
