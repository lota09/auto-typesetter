#!/usr/bin/env python3
"""파이프라인 드라이버 — 단계를 순서대로 돌린다.

지금까지 단계를 손으로 이어 붙였다. 그러다 보니 중간 산출물 이름이 실행마다
달라지고, 어느 단계가 어떤 파일을 먹는지 기억에 의존하게 됐다. 여기에 모은다.

기본값 두 개는 "빠른 쪽"이다. 느린 쪽이 필요하면 명시적으로 켠다.

--transcribe (기본 ocr):
  일본어 크롭 전사는 manga-ocr 이 VLM 보다 **12 배 빠르다** (4.6 초 vs 55.9 초).
  정확도는 본문 기준 95.2% 대 98.0% 로 VLM 이 앞서지만, 2.8%p 로 12 배를 사는
  것이 늘 옳지는 않다.

  단 **일본어 전용**이고, 다른 언어에서는 빈 출력이 아니라 **그럴듯하게 틀린
  글**을 낸다(번체 중국어에 돌리니 `眞的狼危険` 처럼 한자를 일본 자형으로 바꿔
  내놓았다). 뒤에서 검사로 걸러지지 않으므로 시작 전에 막아야 한다 —
  판독 패스가 본 원문 언어가 일본어가 아니면 자동으로 vlm 으로 물러난다.
  `--transcribe ocr` 을 명시하면 그 판단을 접고 강행한다.

  manga-ocr 은 transformers >= 4.45 를 요구하고 Magiv2 는 4.44.1 에서만 뜬다.
  그래서 ocr/.venv 로 격리했고, 여기서 다른 인터프리터로 호출한다.
  (옛 이름은 --fast 였다. 무엇이 빨라지는지가 이름에 없어서 바꿨다.)

--thinking (기본 off, 값 없이 주면 on):
  추출 단계에서 추론을 켜면 토큰 예산을 사고에 다 쓰고 content 를 빈 채
  finish_reason=length 로 돌아온다. 그건 처음부터 알고 있었다.

  새로 잰 것은 **인물·시트 단계도 추론이 값을 못 한다**는 것이다 — ep11 에서
  텍스트 단계가 767 초에서 43 초로 줄었는데(14 배) 말투 오류 개수는 양쪽 다
  정확히 3 개였다. 켜서 얻은 것은 말투를 판정할 수 있는 문장이 17 → 19 개로
  는 것뿐이고, 그 값이 555 초와 토큰 15.5 배다. 자세한 것은 docs/MODELS.md.

  auto 는 옛 기본값이다(추출은 끄고 인물·시트는 켠다). 비교용으로 남겨 둔다.
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
LLAMA_DIR = os.environ.get(
    "LLAMA_SERVER_DIR", "/home/lota/Developments/llamacpp/llama_server_cuda")
# pkill 패턴을 통짜 문자열로 두면, 이 파일을 인자로 실행하는 셸의 명령줄에도
# 그 문자열이 있어서 pkill 이 자기 자신을 죽인다. 실제로 세 번 당했다.
# 쪼갠 형태를 유지할 것.
ROUTER_PAT = os.path.basename(LLAMA_DIR) + "/llama-" + "server"


def router_stop():
    """라우터를 내려 VRAM 을 비운다.

    32GB MI50 시절에는 Magi 앞에서 이걸 반드시 해야 했다. Magi 는 PyTorch 로
    GPU 를 직접 쓰는데 llama-server 가 28~30GB 를 쥐고 있으면 2GB 도 못 얻어
    OOM 으로 죽었다. /models/unload 엔드포인트도, 자식 프로세스만 죽이는 것도
    VRAM 을 반환하지 않아서 프로세스를 내리는 방식으로 갔다.

    64GB 로 바뀐 뒤로는 기본값이 아니다 (--stop-router-for-magi 참조).
    """
    subprocess.run(["pkill", "-f", ROUTER_PAT], stderr=subprocess.DEVNULL)
    time.sleep(5)


def router_alive(timeout=3):
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8081/v1/models", timeout=timeout)
        return True
    except Exception:
        return False


def router_start(timeout=180):
    """라우터를 띄우고 응답할 때까지 기다린다.

    이미 떠 있으면 그대로 쓴다. 확인 없이 또 띄우면 8081 바인드에 실패한
    좀비가 하나 더 생기는데, 아래 대기 루프는 **먼저 떠 있던** 쪽의 응답을
    보고 성공으로 판정해 버려서 실패가 조용히 묻힌다.
    """
    if router_alive():
        print("   라우터가 이미 떠 있습니다 — 그대로 사용")
        return True
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


def detected_langs(page_json):
    """판독 패스가 본 원문 언어의 페이지 분포.

    manga-ocr 은 일본어 전용인데 다른 언어에서 **빈 출력이 아니라 그럴듯하게 틀린
    글**을 낸다(번체 중국어에 돌리니 `眞的狼危険` 처럼 한자를 일본 자형으로 바꿔
    내놓았다). 그래서 뒤에서 검사로 걸러지지 않는다 — 전사를 시작하기 전에
    막는 수밖에 없다.

    최빈값이 아니라 **분포**를 돌려주는 이유: 한 챕터가 섞여 있을 수 있다.
    실측 maid2 는 일본어 49페이지 / 중국어 24페이지였고, 다수결로 고르면
    중국어 24페이지가 조용히 망가진다. 한 페이지라도 일본어가 아니면 안 쓴다.
    """
    import json as _json
    from collections import Counter
    try:
        d = _json.load(open(page_json, encoding="utf-8"))
    except Exception:
        return Counter()
    return Counter(pg.get("source_lang") for pg in d.get("pages", [])
                   if pg.get("source_lang"))


def run(cmd, label):
    print(f"\n── {label}", flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"   실패 (exit {r.returncode})", file=sys.stderr)
        return False
    print(f"   {time.time()-t:.1f}초", flush=True)
    return True


# 단계 스크립트마다 추론 플래그의 이름과 기본값이 다르다. read_* 와 번역은
# `--thinking` 으로 켜고(기본 꺼짐), cast 와 시트는 `--no-*thinking` 으로 끈다
# (기본 켜짐). 그 차이는 의도적이다 — 추출 단계에서 추론을 켜면 토큰 예산을
# 사고에 다 쓰고 본문을 비운 채 돌아오고, 인물·시트는 관계를 저울질하는
# 추론 과제다. 부르는 쪽에서 그 이름을 외우게 하지 않으려고 여기서 옮긴다.
#
#   (단계) → (켜는 플래그, 끄는 플래그, auto 일 때 켜지는가)
THINKING = {
    "read_page":  (["--thinking"], [],                        False),
    "read_texts": (["--thinking"], [],                        False),
    "cast":       ([],             ["--no-thinking"],         True),
    "styleguide": ([],             ["--no-sheet-thinking"],   True),
    "translate":  (["--thinking"], [],                        False),
    "repair":     (["--thinking"], [],                        False),
}


def thinking_flags(stage, mode):
    on_flags, off_flags, default_on = THINKING[stage]
    want = default_on if mode == "auto" else (mode == "on")
    return on_flags if want else off_flags


def main():
    p = argparse.ArgumentParser(description="만화 번역 파이프라인")
    p.add_argument("--pages", required=True, help="페이지 glob (읽는 순서대로 정렬됨)")
    p.add_argument("--work", required=True, help="중간 산출물을 둘 디렉터리")
    p.add_argument("--out", help="완성 페이지 디렉터리 (기본: <work>/out)")
    p.add_argument("--styleguide", help="작품 단위 시트. 있으면 물려 쓰고 없으면 만든다")
    p.add_argument("--transcribe", choices=["ocr", "vlm"], default=None,
                   help="크롭 전사를 무엇으로 할지. **기본 ocr** — manga-ocr 이 12배 빠르다"
                        "(4.6초 vs 55.9초). 다만 **일본어 전용**이라, 판독이 일본어가 아닌 "
                        "것을 보면 자동으로 vlm 으로 물러난다. 여기에 ocr 을 명시하면 그 "
                        "판단을 접고 강행한다(중/영 소재에서는 그럴듯하게 틀린 글이 나온다). "
                        "vlm=판독 모델, 전 언어에서 되고 본문 정확도가 2.8%%p 높다")
    p.add_argument("--fast", action="store_true",
                   help="--transcribe ocr 의 옛 이름. 남겨 두었지만 새로 쓰지 말 것")
    p.add_argument("--thinking", nargs="?", choices=["auto", "on", "off"],
                   const="on", default="off",
                   help="모델 추론(thinking)을 단계별로 켜고 끈다. **기본 off** — 실측에서 "
                        "텍스트 단계가 14배 빨라지면서 말투 오류 개수는 같았다(docs/MODELS.md). "
                        "값 없이 --thinking 만 주면 on. "
                        "auto=단계마다 옛 기본값(추출은 끄고 인물·시트는 켠다)")
    p.add_argument("--workers", type=int, default=4,
                   help="판독 두 단계에서 동시에 던질 요청 수. 페이지·박스가 서로 "
                        "독립이라 겹쳐 던질 수 있다. 이득은 GPU 배칭이 아니라 "
                        "크롭·인코딩·HTTP 를 GPU 작업과 겹치는 데서 온다 — "
                        "서버(models.ini)의 parallel 은 올릴 필요가 없다. "
                        "자세한 것은 docs/PARALLELISM.md")
    p.add_argument("--model-read", help="판독 두 단계(read_page/read_texts)의 모델을 강제한다")
    p.add_argument("--model-text",
                   help="텍스트 단계(cast/시트/번역/수정)의 모델을 강제한다")
    p.add_argument("--skip-pages", type=int, nargs="*", default=[],
                   help="처리에서 뺄 페이지(1-base). 후원자 명단·판권지 등")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="이미 끝난 단계도 다시 돌린다")
    p.add_argument("--no-manage-router", dest="manage_router", action="store_false",
                   help="라우터를 직접 띄우고 내리지 않는다 (직접 관리할 때)")
    p.add_argument("--stop-router-for-magi", action="store_true",
                   help="Magi 전에 라우터를 내려 VRAM 을 비운다. 32GB 시절에는 "
                        "필수였다. 64GB(CMP 170HX)에서는 판독 모델 43.5GB 를 "
                        "얹은 채로 Magi 가 정상 동작하는 것을 확인했으므로 기본 꺼짐")
    p.add_argument("--from-stage", choices=STAGES, default="magi")
    p.add_argument("--only", choices=STAGES, help="이 단계만 돌린다")
    args = p.parse_args()

    transcribe_explicit = args.transcribe is not None
    if args.transcribe is None:
        args.transcribe = "ocr"
    if args.fast:
        print("경고: --fast 는 --transcribe ocr 의 옛 이름입니다", file=sys.stderr)
        args.transcribe, transcribe_explicit = "ocr", True

    w = args.work
    os.makedirs(w, exist_ok=True)
    out_dir = args.out or os.path.join(w, "out")
    sg = args.styleguide or os.path.join(w, "styleguide.json")
    f = lambda n: os.path.join(w, n)
    think = lambda stage: thinking_flags(stage, args.thinking)
    model_read = ["--model", args.model_read] if args.model_read else []
    workers = ["--workers", str(args.workers)]
    model_text = ["--model", args.model_text] if args.model_text else []

    start = 0 if args.only else STAGES.index(args.from_stage)
    todo = [args.only] if args.only else STAGES[start:]

    if "magi" in todo and not (args.resume and is_fresh(f("magi.json"))):
        if args.manage_router and args.stop_router_for_magi:
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
                    "--no-cast-memory"] + model_read + workers + think("read_page")
                   + ([] if args.resume else ["--no-resume"]),
                   "② 페이지 판독 (화자·언어)"):
            return 1
        if args.transcribe == "ocr":
            langs = detected_langs(f("page.json"))
            other = {k: v for k, v in langs.items() if k != "ja"}
            if other:
                mix = ", ".join(f"{k} {v}장" for k, v in sorted(langs.items()))
                msg = (f"   판독이 일본어가 아닌 페이지를 봤습니다 ({mix}). "
                       "manga-ocr 은 일본어 전용이고 다른 언어에서는 빈 출력이 아니라 "
                       "**그럴듯하게 틀린 글**을 내놓아 검사로 걸러지지 않습니다")
                if transcribe_explicit:
                    print("경고: " + msg + " — ocr 을 명시했으므로 그대로 진행합니다",
                          file=sys.stderr)
                else:
                    print(msg + "\n   → 크롭 전사를 vlm 으로 돌립니다 "
                          "(강행하려면 --transcribe ocr)")
                    args.transcribe = "vlm"

        if args.transcribe == "ocr":
            # manga_ocr_pass 에는 --fill-from 이 없다. 부분 재개가 불가능하므로
            # 이미 신선한 crop.json 이 있으면 아예 부르지 않는다. 그냥 부르면
            # 멀쩡한 전사를 통째로 덮어쓴다 — VLM 으로 읽어 둔 것이었다면 손해다.
            if args.resume and is_fresh(f("crop.json"), f("magi.json")):
                print("── ② 크롭 전사 [manga-ocr] — 이미 있는 crop.json 을 씁니다")
            elif not run([OCR_PY, os.path.join(ROOT, "ocr", "manga_ocr_pass.py"),
                          "--magi-json", f("magi.json"), "--out", f("crop.json")],
                         "② 크롭 전사 [manga-ocr]"):
                return 1
        else:
            # 크롭 전사는 이 파이프라인에서 가장 비싼 축이다 (박스당 2.2초 ×
            # 692박스 ≈ 25분). read_page 는 스스로 재개하는데 여기는 그게 없어서,
            # 뒷단계가 죽어 다시 돌리면 이미 읽은 박스를 통째로 다시 읽었다.
            #
            # read_texts 에 이미 --fill-from 이 있다 — 주어진 JSON 에서 채워진
            # 박스는 옮겨 오고 **빈 것만** 모델에 묻는다. 그게 곧 재개다.
            # magi.json 보다 새로울 때만 쓴다. 박스가 다시 잡혔는데 옛 전사를
            # 끌어오면 좌표와 글이 어긋난다.
            fill = (["--fill-from", f("crop.json")]
                    if args.resume and is_fresh(f("crop.json"), f("magi.json"))
                    else [])
            if fill:
                print("   이미 전사된 crop.json 이 있습니다 — 빈 박스만 읽습니다")
            if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_texts.py"),
                        "--magi-json", f("magi.json"), "--out", f("crop.json")]
                       + fill + model_read + workers + think("read_texts"),
                       "② 크롭 전사 [VLM]"):
                return 1

    if "merge" in todo and not (args.resume and is_fresh(f("merged.json"), f("crop.json"), f("page.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "merge_reads.py"),
                    "--crop-json", f("crop.json"), "--page-json", f("page.json"),
                    "--out", f("merged.json"), "--prefer", "crop"], "③ 병합"):
            return 1

    if "cast" in todo and not (args.resume and is_fresh(f("cast.json"), f("merged.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "build_cast.py"),
                    "--read-json", f("merged.json"), "--out", f("cast.json")]
                   + model_text + think("cast"),
                   "④ 인물·스토리"):
            return 1

    if "translate" in todo and not (args.resume and is_fresh(f("translated.json"), f("cast.json"))):
        # 시트와 번역은 모델을 따로 지정할 수 있다. 파이프라인에서는 한 값으로 묶는다.
        tr_model = (["--sheet-model", args.model_text, "--translate-model", args.model_text]
                    if args.model_text else [])
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "translate_chapter.py"),
                    "--page-json", f("cast.json"), "--out", f("translated.json"),
                    "--styleguide", sg] + tr_model
                   + think("styleguide") + think("translate"), "⑤ 시트 + 번역"):
            return 1

    if "validate" in todo and not (args.resume and is_fresh(f("final.json"), f("translated.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "validate_translation.py"),
                    "--translated-json", f("translated.json"), "--styleguide", sg,
                    "--out", f("final.json"), "--repair"] + model_text + think("repair"),
                   "⑥ 검사·자동수정"):
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
