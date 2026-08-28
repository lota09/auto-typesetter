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
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "vlm"))
import progress as PROG  # noqa: E402
MAGI_PY = os.path.join(ROOT, "magi", ".venv", "bin", "python")
OCR_PY = os.path.join(ROOT, "ocr", ".venv", "bin", "python")

STAGES = ["magi", "read", "merge", "cast", "translate", "validate", "render"]

CONFIG_DIR = os.path.join(ROOT, "config")
LOG_DIR = os.path.join(ROOT, "logs")

# build_cast.py 는 자체 단계 키가 없고 styleguide 를 빌려 쓴다.
STAGE_KEY = {"cast": "styleguide"}


def load_cfg():
    import json as _json
    return _json.load(open(os.path.join(CONFIG_DIR, "config.json"), encoding="utf-8"))


# ── 백엔드 ───────────────────────────────────────────────────────────────
#
# 파이프라인은 **아무 서버도 띄우지 않는다.** 로컬 vLLM 이든 llama.cpp 든 원격
# API 든 전부 같은 것으로 본다 — OpenAI 호환 주소 하나. 기동은 사용자 책임이다.
#
# 시작할 때 한 번만 확인하고, 안 되더라도 **경고만 하고 진행한다.** ① Magi 는
# LLM 이 없어도 돌고, 그동안 사용자가 서버를 띄울 수 있다. 정말 필요한 단계에서
# 못 쓰면 그때 그 단계가 실패한다 — 그게 정확한 시점이다.

ROLE_OF_STAGE = {"read_page": "read", "read_texts": "read",
                 "styleguide": "translate", "translate": "translate",
                 "repair": "translate", "judge": "translate"}


def role_spec(cfg, role, model_read=None, model_text=None):
    """역할 설정 + CLI 덮어쓰기."""
    spec = dict(cfg.get(role) or {})
    over = model_read if role == "read" else model_text
    if over:
        spec["model"] = over
    return spec


def probe_role(cfg, role, spec, timeout=5):
    """이 역할의 주소가 그 모델을 내놓는가. (상태문구, 정상여부)."""
    import json as _json
    import urllib.request
    url = (spec.get("base_url") or "").rstrip("/")
    if not url:
        return f"{role}: base_url 이 없습니다", False
    want = spec.get("model")
    if not want:
        return f"{role}: model 이 없습니다", False
    env = spec.get("api_key_env")
    if env and not os.environ.get(env):
        return f"{role}: 환경변수 {env} 가 비어 있습니다", False
    req = urllib.request.Request(url + "/models")
    if env:
        req.add_header("Authorization", f"Bearer {os.environ[env]}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cards = _json.loads(r.read()).get("data") or []
    except Exception as e:
        return f"{role}: {url} 에 닿지 못했습니다 ({type(e).__name__})", False
    card = next((c for c in cards if c.get("id") == want), None)
    if card is None:
        have = ", ".join(sorted(c.get("id", "?") for c in cards)[:3]) or "(없음)"
        return f"{role}: '{want}' 을 내놓지 않습니다 — 지금 있는 것: {have}", False
    ctx = card.get("max_model_len")
    return f"{role}: {want}{f' · ctx {ctx}' if ctx else ''}", True


def health_banner(cfg, model_read=None, model_text=None):
    """시작 배너. 문제가 있어도 **막지 않는다** — 경고만 남기고 진행한다."""
    ok_all = True
    for role in ("read", "translate"):
        line, ok = probe_role(cfg, role, role_spec(cfg, role, model_read, model_text))
        print(f"── {line}" if ok else f"── 경고 {line}", file=sys.stderr if not ok else sys.stdout)
        ok_all &= ok
    if not ok_all:
        print("   LLM 이 필요한 단계에서 실패합니다. 그 전에 서버를 띄우면 이어집니다.",
              file=sys.stderr)
    return ok_all


def is_fresh(out_path, *inputs):
    """산출물이 이미 있고 입력보다 새로우면 그 단계를 건너뛴다.

    77 페이지를 1441 초 걸려 판독했는데 한 장이 깨져 전체가 실패로 끝난 적이 있다.
    다시 돌리면 처음부터 하게 되므로, 끝난 단계는 넘어가야 한다.
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    t = os.path.getmtime(out_path)
    return all(not os.path.exists(i) or os.path.getmtime(i) <= t for i in inputs)


def page_langs(page_json):
    """쪽별 원어. read_page 가 채운 전사를 코드가 세서 정한다.

    모델에게 `lang` 을 묻던 것을 버린 이유: 같은 페이지 안에서 자기와 불일치했다.
    한자만 든 박스는 글자만으로 일본어·중국어가 갈리지 않기 때문이다. 쪽 단위는
    근거가 쌓여 안정적이다 — 일본어 한 쪽에는 거의 반드시 가나가 나온다.
    """
    import json as _json
    sys.path.insert(0, os.path.join(ROOT, "vlm"))
    import language as LANG
    try:
        d = _json.load(open(page_json, encoding="utf-8"))
    except Exception:
        return {}, {}
    chapter, _ = LANG.chapter_language(d)
    return LANG.page_languages(d, chapter)


def run(cmd, label):
    PROG.log(f"\n── {label}")
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        PROG.log(f"   실패 (exit {r.returncode})")
        return False
    PROG.log(f"   {time.time()-t:.1f}초")
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
    p.add_argument("--model", help="판독·번역 양쪽을 이 모델 id 로")
    p.add_argument("--model-read", help="판독 두 단계(read_page/read_texts)의 모델을 강제한다")
    p.add_argument("--model-text",
                   help="텍스트 단계(cast/시트/번역/수정)의 모델을 강제한다")
    p.add_argument("--skip-pages", type=int, nargs="*", default=[],
                   help="처리에서 뺄 페이지(1-base). 후원자 명단·판권지 등")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="이미 끝난 단계도 다시 돌린다")
    p.add_argument("--from-stage", choices=STAGES, default="magi")
    p.add_argument("--only", choices=STAGES, help="이 단계만 돌린다")
    args = p.parse_args()

    cfg = load_cfg()

    # --model 은 판독·텍스트 양쪽을 한 번에 정한다. vLLM 은 프로세스당 모델이
    # 하나라 이게 정상 사용법이고, 나누고 싶으면 --model-read/--model-text 가 이긴다.
    args.model_read = args.model_read or args.model
    args.model_text = args.model_text or args.model

    def alias_for(stage):
        role = ROLE_OF_STAGE.get(STAGE_KEY.get(stage, stage), "translate")
        return role_spec(cfg, role, args.model_read, args.model_text).get("model")

    def need(stage):
        """예전에는 여기서 서버를 맞췄다. 지금은 아무것도 하지 않는다 —
        확인은 시작할 때 한 번 했고, 못 쓰면 그 단계가 스스로 실패한다."""
        return True

    transcribe_explicit = args.transcribe is not None
    if args.transcribe is None:
        args.transcribe = "ocr"
    if args.fast:
        print("경고: --fast 는 --transcribe ocr 의 옛 이름입니다", file=sys.stderr)
        args.transcribe, transcribe_explicit = "ocr", True

    w = args.work
    os.makedirs(w, exist_ok=True)
    # 중간 산출물 옆에 전체 기록을 남긴다. 터미널은 짧게 유지하고(반복되는 진행
    # 줄은 한 자리에서 갱신), 파일에는 한 줄도 빠뜨리지 않는다.
    log_path = os.path.join(w, "run.log")
    PROG.open_log(log_path)
    PROG.log(f"\n{'='*66}\n{time.strftime('%Y-%m-%d %H:%M:%S')}  "
             f"{' '.join(sys.argv)}\n{'='*66}")
    health_banner(cfg, args.model_read, args.model_text)
    out_dir = args.out or os.path.join(w, "out")
    sg = args.styleguide or os.path.join(w, "styleguide.json")
    f = lambda n: os.path.join(w, n)
    think = lambda stage: thinking_flags(stage, args.thinking)
    workers = ["--workers", str(args.workers)]


    start = 0 if args.only else STAGES.index(args.from_stage)
    todo = [args.only] if args.only else STAGES[start:]

    if "magi" in todo and not (args.resume and is_fresh(f("magi.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "magi", "magi_worker.py"), "--log", log_path,
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


    if "read" in todo:
        if not need("read_page"):
            return 1
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_page.py"), "--log", log_path,
                    "--magi-json", f("magi.json"), "--out", f("page.json"),
                    "--no-cast-memory", "--model", alias_for("read_page")]
                   + workers + think("read_page")
                   + ([] if args.resume else ["--no-resume"]),
                   "② 페이지 판독 (화자·언어)"):
            return 1
        # ── 전사 라우팅 ────────────────────────────────────────────────
        #
        # 쪽마다 맞는 모델로 보낸다. 규칙으로 결과를 고치려 하지 않는다.
        #
        # 근거(실측, maid2 9쪽 일본어): 판독 VLM 이 손글씨를 숫자로 읽었다 —
        #   できたよ、       → 1124.146
        #   ありがとうございます → 16=4×2人いわつ16か
        # 같은 크롭을 manga-ocr 에 주면 **둘 다 정확히** 읽는다. 전사는 모델의
        # 능력 문제이고, 파이프라인이 할 수 있는 일은 **맞는 모델로 보내는 것**이다.
        #
        # manga-ocr 은 일본어 전용이라 다른 언어에서는 빈 출력이 아니라 그럴듯하게
        # 틀린 글을 낸다. 그래서 **일본어로 판정된 쪽에만** 쓴다.
        per, dist = page_langs(f("page.json"))
        ja_pages = sorted(i + 1 for i, l in per.items() if l == "ja")
        if dist:
            PROG.log(f"   쪽 원어 {dist}")

        use_ocr = args.transcribe == "ocr" and ja_pages
        if args.transcribe == "ocr" and not ja_pages:
            PROG.log("   일본어로 판정된 쪽이 없습니다 — 전사를 vlm 으로 돌립니다")
            args.transcribe = "vlm"

        if use_ocr:
            # 일본어 쪽만 manga-ocr 로 채운다. 나머지는 비워 두고 VLM 이 맡는다.
            other = [i + 1 for i, l in per.items() if l != "ja"]
            if other:
                PROG.log(f"   manga-ocr 로 일본어 {len(ja_pages)}쪽, "
                         f"VLM 으로 나머지 {len(other)}쪽")
            if args.resume and is_fresh(f("crop.json"), f("page.json")):
                PROG.log("── ② 크롭 전사 [manga-ocr] — 이미 있는 crop.json 을 씁니다")
            elif not run([OCR_PY, os.path.join(ROOT, "ocr", "manga_ocr_pass.py"),
                          "--magi-json", f("page.json"), "--out", f("crop.json"),
                          "--pages"] + [str(n) for n in ja_pages],
                         "② 크롭 전사 [manga-ocr · 일본어 쪽]"):
                return 1

        # manga-ocr 이 못 맡은 쪽(일본어가 아닌 쪽)과, ocr 을 아예 안 쓴 경우를
        # VLM 이 맡는다. read_texts 의 --fill-from 이 **이미 채워진 박스는 옮기고
        # 빈 것만** 모델에 묻는다 — 그게 곧 두 모델을 한 파일에 합치는 방법이다.
        vlm_needed = (args.transcribe == "vlm"
                      or (use_ocr and any(l != "ja" for l in per.values())))
        if vlm_needed:
            fill = (["--fill-from", f("crop.json")]
                    if use_ocr or (args.resume
                                   and is_fresh(f("crop.json"), f("page.json")))
                    else [])
            if fill and not use_ocr:
                PROG.log("   이미 전사된 crop.json 이 있습니다 — 빈 박스만 읽습니다")
            if not need("read_texts"):
                return 1
            if not run([MAGI_PY, os.path.join(ROOT, "vlm", "read_texts.py"), "--log", log_path,
                        "--magi-json", f("page.json"), "--out", f("crop.json"),
                        "--model", alias_for("read_texts")]
                       + fill + workers + think("read_texts"),
                       "② 크롭 전사 [VLM]"):
                return 1

    if "merge" in todo and not (args.resume and is_fresh(f("merged.json"), f("crop.json"), f("page.json"))):
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "merge_reads.py"), "--log", log_path,
                    "--crop-json", f("crop.json"), "--page-json", f("page.json"),
                    "--out", f("merged.json"), "--prefer", "crop"], "③ 병합"):
            return 1

    if "cast" in todo and not (args.resume and is_fresh(f("cast.json"), f("merged.json"))):
        if not need("cast"):
            return 1
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "build_cast.py"), "--log", log_path,
                    "--read-json", f("merged.json"), "--out", f("cast.json"),
                    "--model", alias_for("cast")] + think("cast"),
                   "④ 인물·스토리"):
            return 1

    if "translate" in todo and not (args.resume and is_fresh(f("translated.json"), f("cast.json"))):
        # 시트와 번역은 모델을 따로 지정할 수 있다. 파이프라인에서는 한 값으로 묶는다.
        if not need("styleguide"):
            return 1
        tr_model = ["--sheet-model", alias_for("styleguide"),
                    "--translate-model", alias_for("translate")]
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "translate_chapter.py"), "--log", log_path,
                    "--page-json", f("cast.json"), "--out", f("translated.json"),
                    "--styleguide", sg] + tr_model
                   + think("styleguide") + think("translate"), "⑤ 시트 + 번역"):
            return 1

    if "validate" in todo and not (args.resume and is_fresh(f("final.json"), f("translated.json"))):
        if not need("repair"):
            return 1
        if not run([MAGI_PY, os.path.join(ROOT, "vlm", "validate_translation.py"), "--log", log_path,
                    "--translated-json", f("translated.json"), "--styleguide", sg,
                    "--out", f("final.json"), "--repair",
                    "--model", alias_for("repair")] + think("repair"),
                   "⑥ 검사·자동수정"):
            return 1

    if "render" in todo:
        if not run([MAGI_PY, os.path.join(ROOT, "render", "compose.py"), "--log", log_path,
                    "--translated-json", f("final.json"), "--out-dir", out_dir],
                   "⑦⑧⑨ 마스크·인페인팅·조판"):
            return 1
        print(f"\n완성 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
