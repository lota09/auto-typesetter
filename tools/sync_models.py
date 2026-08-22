#!/usr/bin/env python3
"""config/models.ini 를 읽어 config/config.json 의 models 를 맞춘다.

왜 필요한가:
  모델 설정이 두 곳에 나뉘어 있다.
    config/models.ini   경로와 실행 플래그  → llama-server 라우터가 읽는다
    config/config.json  단계별 모델 배치    → 우리 파이프라인이 읽는다
  새 모델을 넣으려면 양쪽을 다 손봐야 하는데, 한쪽만 고쳐도 **조용히 어긋난다**.
  실제로 겪었다 — .ini 에만 추가했더니 라우터는 별칭을 인식했는데 파이프라인이
  "models 에 'q38-uncens' 이 없습니다" 로 거부했고, 판독 세 건이 통째로 날아갔다.

  .ini 를 단일 출처로 삼고 config.json 을 여기서 생성한다. stages 와 기존
  모델의 note·max_image_pixels 같은 수기 설정은 보존한다.
"""

import argparse
import configparser
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="models.ini → config.json 동기화")
    p.add_argument("--ini", default=os.path.join(ROOT, "config", "models.ini"))
    p.add_argument("--config", default=os.path.join(ROOT, "config", "config.json"))
    p.add_argument("--backend", default="local", help="새 항목에 붙일 백엔드 이름")
    p.add_argument("--check", action="store_true",
                   help="쓰지 않고 어긋난 항목만 보고한다 (CI·사전 점검용)")
    args = p.parse_args()

    ini = configparser.ConfigParser()
    ini.read(args.ini, encoding="utf-8")
    # 실행 플래그만 있고 model 키가 없는 섹션은 모델이 아니다 ([common] 같은 것).
    aliases = [s for s in ini.sections() if ini.has_option(s, "model")]

    cfg = json.load(open(args.config, encoding="utf-8"))
    models = cfg.setdefault("models", {})

    missing = [a for a in aliases if a not in models]
    stale = [a for a in models if a not in aliases
             and models[a].get("backend") == args.backend]

    for a in missing:
        models[a] = {"backend": args.backend, "model": a, "vision": True,
                     "max_image_pixels": None,
                     "note": f"models.ini 의 [{a}] 에서 자동 등록"}

    if args.check:
        if missing or stale:
            print(f"어긋남 — config.json 에 없음: {missing} | ini 에 없음: {stale}")
            return 1
        print(f"동기화 상태 정상 (모델 {len(aliases)}개)")
        return 0

    json.dump(cfg, open(args.config, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"모델 {len(aliases)}개 | 추가 {len(missing)}개 {missing}")
    if stale:
        print(f"⚠ config.json 에만 있는 항목 (직접 확인 필요): {stale}")
    print("stages 는 건드리지 않았다. 새 모델을 쓰려면 stages 를 손으로 바꾼다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
