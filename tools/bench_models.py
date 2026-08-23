#!/usr/bin/env python3
"""모델 라인업 비교 — 같은 입력·같은 프롬프트로 정확도와 속도를 나란히 잰다.

왜 스크립트로 만드나:
  하드웨어가 바뀌면 이전 세션의 모델 비교표는 전부 무효가 된다. 실제로 그랬다.
  손으로 돌리면 매번 조건이 조금씩 달라지고(모델 적재 시간이 섞인다든지),
  그러면 표가 "어느 모델이 나은가" 대신 "언제 쟀는가"를 재게 된다.

두 가지를 통제한다:

  **적재 시간을 뺀다.** 라우터는 요청이 오면 모델을 갈아 끼운다. 43.6GB 짜리를
  콜드로 올리면 170초가 걸리고, 그게 첫 호출에 통째로 붙는다. 그래서 재기 전에
  1 토큰짜리 요청으로 예열한 뒤 시작한다.

  **토큰으로도 잰다.** "추론을 켜면 느리다"를 초로만 재면 적재·교체·디스크
  캐시에 오염된다. 생성 토큰 수는 그 모델이 실제로 얼마나 생각했는지만 말한다.
  Qwen 계열의 과사고를 확인하려면 이쪽이 맞다.

두 과제:

  transcribe  크롭 전사(②). 정답은 Pepper&Carrot 공식 원문. 비전 모델만 해당.
              Gemma 처럼 비전 상한이 있는 모델이 크롭에서도 막히는지 여기서 드러난다.

  text        인물·스토리(④) + 시트·번역(⑤). 정답은 공식 한국어판이고, 재는 것은
              **말투 일치율**이다. 번역은 정답이 하나가 아니라 글자로 재면 안 되지만
              존댓말/반말은 객관적으로 비교된다 — 이 프로젝트가 지키려는 것이 그것이다.

예:
  magi/.venv/bin/python tools/bench_models.py --task transcribe \\
    --models qwen-vl q38-uncens gemma \\
    --thinking off on \\
    --magi-json work/ep11cn/magi.json \\
    --gt assets/groundtruth/peppercarrot/ep11_cn.json \\
    --bench-dir work/bench
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "magi", ".venv", "bin", "python")
ROUTER = "http://127.0.0.1:8081/v1"

USAGE_RE = re.compile(r"\[usage\] calls=(\d+) prompt=(\d+) completion=(\d+)")


def warm(model, timeout=900):
    """모델을 미리 올려 둔다. 적재 시간이 측정에 섞이지 않게.

    **이것만으로는 부족하다.** 텍스트 1토큰 요청은 비전 경로(mmproj)를 건드리지
    않아서, 모델 적재 후 **첫 이미지 요청**이 별도의 워밍업 비용을 문다.
    실측 ep11_cn 34박스: 워밍업을 안 한 첫 실행 55초, 그다음 24초 —
    **1회성 30초**가 첫 측정에 통째로 붙는다.
    그래서 아래 warm_task() 로 과제 자체를 한 번 버리고 시작한다.
    이걸 몰랐을 때 동시성 이득을 2.5배로 잘못 읽었다 (실제 1.17배).
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 1, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(ROUTER + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except (urllib.error.URLError, TimeoutError) as e:
        return None, f"예열 실패: {e}"
    return time.time() - t, None


def warm_task(args, model, wdir):
    """측정할 과제를 그대로 한 번 돌리고 버린다.

    비전 경로 워밍업은 해상도·경로마다 다를 수 있어, 흉내내는 것보다 **같은 일을
    한 번 하는 것**이 확실하다. 결과는 쓰지 않는다.
    """
    if args.task != "transcribe":
        return
    out = os.path.join(wdir, "warm.json")
    subprocess.run([PY, os.path.join(ROOT, "vlm", "read_texts.py"),
                    "--magi-json", args.magi_json, "--out", out, "--model", model],
                   capture_output=True, text=True)


def run(cmd, log_path):
    """돌리고 (성공, 소요초, stdout) 을 준다. 로그는 파일로도 남긴다."""
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    out = r.stdout + "\n" + r.stderr
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n" + out)
    return r.returncode == 0, dt, out


def usage_of(out):
    m = None
    for m in USAGE_RE.finditer(out):
        pass
    if not m:
        return {}
    return {"calls": int(m.group(1)), "prompt": int(m.group(2)),
            "completion": int(m.group(3))}


def score(ours, gt, mode, pivot=None, field=None):
    cmd = [PY, os.path.join(ROOT, "tools", "score_groundtruth.py"),
           "--ours", ours, "--gt", gt, "--mode", mode]
    if pivot:
        cmd += ["--pivot-gt", pivot]
    if field:
        cmd += ["--field", field]
    r = subprocess.run(cmd, capture_output=True, text=True)
    o = r.stdout
    g = lambda pat: (re.search(pat, o) or [None, None])[1]
    return {
        "matched_pct": g(r"우리 박스가 붙은 것 \d+개 \((\d+)%\)"),
        "exact_pct": g(r"완전일치 \d+/\d+ \((\d+)%\)"),
        "char_acc": g(r"문자 정확도 ([\d.]+)%"),
        "register_pct": g(r"일치 \d+개 \((\d+)%\)"),
        "raw": o.strip().splitlines()[:6],
    }


def bench_transcribe(args, model, thinking, wdir):
    out = os.path.join(wdir, "crop.json")
    cmd = [PY, os.path.join(ROOT, "vlm", "read_texts.py"),
           "--magi-json", args.magi_json, "--out", out, "--model", model]
    if thinking == "on":
        cmd.append("--thinking")
    ok, dt, log = run(cmd, os.path.join(wdir, "run.log"))
    row = {"seconds": round(dt, 1), "usage": usage_of(log), "ok": ok}
    if ok:
        row.update(score(out, args.gt, "transcription"))
    else:
        row["error"] = log.strip().splitlines()[-1] if log.strip() else "실패"
    return row


def bench_text(args, model, thinking, wdir):
    cast = os.path.join(wdir, "cast.json")
    sg = os.path.join(wdir, "styleguide.json")
    tr = os.path.join(wdir, "translated.json")

    cmd = [PY, os.path.join(ROOT, "vlm", "build_cast.py"),
           "--read-json", args.read_json, "--out", cast, "--model", model]
    if thinking == "off":
        cmd.append("--no-thinking")
    ok, dt_cast, log_cast = run(cmd, os.path.join(wdir, "cast.log"))
    if not ok:
        return {"ok": False, "seconds": round(dt_cast, 1),
                "error": (log_cast.strip().splitlines() or ["실패"])[-1]}

    cmd = [PY, os.path.join(ROOT, "vlm", "translate_chapter.py"),
           "--page-json", cast, "--out", tr, "--styleguide", sg,
           "--sheet-model", model, "--translate-model", model]
    if thinking == "off":
        cmd.append("--no-sheet-thinking")
    elif thinking == "on":
        cmd.append("--thinking")
    ok, dt_tr, log_tr = run(cmd, os.path.join(wdir, "translate.log"))

    u1, u2 = usage_of(log_cast), usage_of(log_tr)
    row = {
        "ok": ok,
        "seconds": round(dt_cast + dt_tr, 1),
        "cast_seconds": round(dt_cast, 1),
        "translate_seconds": round(dt_tr, 1),
        "usage": {k: (u1.get(k, 0) + u2.get(k, 0)) for k in ("calls", "prompt", "completion")},
    }
    if ok:
        row.update(score(tr, args.gt, "translation", args.pivot_gt, "target"))
    else:
        row["error"] = (log_tr.strip().splitlines() or ["실패"])[-1]
    return row


def table(task, rows):
    if task == "transcribe":
        head = ("| 모델 | thinking | 매칭 | 완전일치 | 문자 정확도 | 소요 | 생성 토큰 |\n"
                "|---|---|---|---|---|---|---|")
        line = lambda k, r: (
            f"| {k[0]} | {k[1]} | {r.get('matched_pct','-')}% | {r.get('exact_pct','-')}% | "
            f"{r.get('char_acc','-')}% | {r['seconds']}s | "
            f"{r.get('usage',{}).get('completion','-')} |")
    else:
        head = ("| 모델 | thinking | 말투 일치 | ④ 인물 | ⑤ 시트+번역 | 합계 | 생성 토큰 |\n"
                "|---|---|---|---|---|---|---|")
        line = lambda k, r: (
            f"| {k[0]} | {k[1]} | {r.get('register_pct','-')}% | "
            f"{r.get('cast_seconds','-')}s | {r.get('translate_seconds','-')}s | "
            f"{r['seconds']}s | {r.get('usage',{}).get('completion','-')} |")
    out = [head]
    for k, r in rows:
        out.append(line(k, r) if r.get("ok") else
                   f"| {k[0]} | {k[1]} | **실패** | | | {r.get('seconds','-')}s | |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="모델 정확도·속도 비교")
    p.add_argument("--task", choices=["transcribe", "text"], required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--thinking", nargs="+", default=["off"],
                   choices=["off", "on", "auto"])
    p.add_argument("--magi-json", help="transcribe 용 입력")
    p.add_argument("--read-json", help="text 용 입력 (merged.json)")
    p.add_argument("--gt", required=True)
    p.add_argument("--pivot-gt", help="text 채점의 정렬용 원문 정답")
    p.add_argument("--bench-dir", default="work/bench")
    p.add_argument("--no-warm-task", action="store_true",
                   help="과제 예열을 건너뛴다. 첫 측정이 비전 워밍업 비용을 문다")
    p.add_argument("--out", help="결과 JSON 경로 (기본: <bench-dir>/<task>.json)")
    args = p.parse_args()

    if args.task == "transcribe" and not args.magi_json:
        p.error("--task transcribe 에는 --magi-json 이 필요합니다")
    if args.task == "text" and not args.read_json:
        p.error("--task text 에는 --read-json 이 필요합니다")

    os.makedirs(args.bench_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.bench_dir, f"{args.task}.json")

    rows = []
    # 모델을 바깥 고리에 두어 교체 횟수를 최소로 한다. 43.6GB 를 왕복시키면
    # 벤치마크 자체가 몇 시간 늘어난다.
    for model in args.models:
        wt, err = warm(model)
        print(f"\n══ {model} — 예열 {('%.0f초' % wt) if wt else err}", flush=True)
        if err:
            for th in args.thinking:
                rows.append(((model, th), {"ok": False, "seconds": 0, "error": err}))
            continue
        if not args.no_warm_task:
            wdir0 = os.path.join(args.bench_dir, f"{args.task}_{model}_warm")
            os.makedirs(wdir0, exist_ok=True)
            wt = time.time()
            warm_task(args, model, wdir0)
            print(f"   과제 예열 {time.time()-wt:.0f}초 (버림)", flush=True)
        for th in args.thinking:
            wdir = os.path.join(args.bench_dir, f"{args.task}_{model}_{th}")
            os.makedirs(wdir, exist_ok=True)
            print(f"── {model} / thinking={th}", flush=True)
            fn = bench_transcribe if args.task == "transcribe" else bench_text
            r = fn(args, model, th, wdir)
            rows.append(((model, th), r))
            print("   " + json.dumps({k: v for k, v in r.items() if k != "raw"},
                                     ensure_ascii=False), flush=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump([{"model": k[0], "thinking": k[1], **v} for k, v in rows],
                          fh, ensure_ascii=False, indent=1)

    print("\n" + table(args.task, rows))
    print(f"\n→ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
