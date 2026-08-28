#!/usr/bin/env python3
"""Magiv2 챕터 분석 워커 — 페이지 묶음에서 "누가 무슨 말을 하는가"를 뽑는다.

Magi 는 이미지를 만들지 않는다. 검출·OCR·화자 귀속만 하고 JSON 을 내놓는다.
번역과 조판은 이 JSON 을 받아 뒷단이 한다. 그래서 이 워커의 산출물은 그림이
아니라 **스키마**이고, 뒷단(Carrot 러너든 다른 무엇이든)이 거기에만 의존하게
만드는 것이 요점이다.

챕터 단위인 이유:
  Magiv2 는 페이지를 하나씩 보지 않고 챕터를 통째로 받아야 인물 동일성을
  유지한다. 페이지마다 프로세스를 띄우면 그 능력이 사라지고 2GB 가중치를
  매번 다시 읽는다.

배치 크기에 관한 함정 (--batch-size):
  do_chapter_wide_prediction 은 페이지를 batch_size 씩 잘라 클러스터링한다.
  즉 **인물 클러스터 ID 는 배치 안에서만 일관된다.** 챕터 전체를 관통하는
  동일성은 오직 캐릭터 뱅크가 만든다. 뱅크 없이 돌리면 이름은 전부 "Other"
  가 되고 cluster ID 도 배치 경계에서 끊긴다 — 뱅크를 만들기 위한 1차 정찰
  로는 쓸 수 있지만, 그 출력을 번역에 바로 먹이면 안 된다.

입력: --pages 로 준 페이지 경로들 (읽는 순서 그대로. 정렬은 호출자 책임)
출력: --out JSON 하나. --transcript 로 사람이 읽는 형식도 같이 낼 수 있다.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

MODEL_ID = "ragavsachdeva/magiv2"


def read_image(path):
    # 공식 사용법이 흑백을 거쳐 RGB 로 되돌린다. 학습 분포가 그러하니 맞춘다.
    with open(path, "rb") as fh:
        return np.array(Image.open(fh).convert("L").convert("RGB"))


def load_character_bank(bank_dir):
    """<이름>.png 로 캐릭터 뱅크를 만든다.

    한 인물에 여러 크롭을 주려면 `이름__2.png` 처럼 `__` 뒤에 아무거나 붙인다.
    뱅크가 없으면 빈 뱅크를 돌려준다 (Magiv2 가 이 경우를 처리해 전부 "Other").
    """
    bank = {"images": [], "names": []}
    if not bank_dir or not os.path.isdir(bank_dir):
        return bank
    for path in sorted(glob.glob(os.path.join(bank_dir, "*"))):
        if not path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        name = os.path.splitext(os.path.basename(path))[0].split("__")[0]
        bank["images"].append(read_image(path))
        bank["names"].append(name)
    return bank


def build_page_record(index, path, size, result):
    """Magiv2 의 원시 출력을 뒷단이 쓸 형태로 옮긴다.

    원시 출력은 인덱스 쌍(text_idx, char_idx)으로 관계를 표현한다. 뒷단에서
    매번 그 조인을 다시 하게 만들 이유가 없으니 여기서 한 번만 풀어둔다.
    """
    names = result.get("character_names") or []
    clusters = result.get("character_cluster_labels") or []
    ocr = result.get("ocr") or []

    speaker_of_text = {t: c for t, c in result["text_character_associations"]}
    tail_of_text = {t: l for t, l in result["text_tail_associations"]}

    texts = []
    for i, bbox in enumerate(result["texts"]):
        char_idx = speaker_of_text.get(i)
        name = names[char_idx] if char_idx is not None and char_idx < len(names) else None
        texts.append({
            "id": i,                                    # 읽는 순서로 정렬돼 있다
            "bbox": bbox,
            "ocr": ocr[i] if i < len(ocr) else None,
            "essential": bool(result["is_essential_text"][i]),
            # 뱅크가 없으면 이름이 전부 "Other" 다. 그건 이름이 아니니 지운다.
            "speaker": name if name and name != "Other" else None,
            "speaker_cluster": (clusters[char_idx] if char_idx is not None
                                and char_idx < len(clusters) else None),
            "has_tail": i in tail_of_text,              # 꼬리가 잡힌 대사가 귀속이 더 믿을 만하다
        })

    characters = [
        {"bbox": bbox,
         "cluster": clusters[i] if i < len(clusters) else None,
         "name": names[i] if i < len(names) else None}
        for i, bbox in enumerate(result["characters"])
    ]

    return {
        "index": index,
        "file": path,
        "size": list(size),
        "panels": result["panels"],
        "characters": characters,
        "texts": texts,
    }


def main():
    p = argparse.ArgumentParser(description="Magiv2 챕터 분석")
    p.add_argument("--log", help="이 경로에 전체 로그를 덧붙인다")
    p.add_argument("--pages", nargs="+", required=True,
                   help="페이지 경로 또는 glob. 읽는 순서대로 준다")
    p.add_argument("--out", required=True, help="결과 JSON 경로")
    p.add_argument("--bank-dir", help="캐릭터 뱅크 디렉터리 (<이름>.png). 없으면 이름 없이 진행")
    p.add_argument("--transcript", help="사람이 읽는 대사록 경로 (선택)")
    p.add_argument("--visualise-dir", help="검출 시각화 PNG 를 떨어뜨릴 디렉터리 (선택)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="클러스터링 단위. 크게 잡을수록 인물 동일성이 좋아지고 VRAM 을 먹는다")
    p.add_argument("--eta", type=float, default=0.75,
                   help="뱅크에 없는 인물로 판정하는 임계값. 낮추면 이름을 더 적게 붙인다")
    p.add_argument("--no-ocr", action="store_true", help="검출·귀속만 하고 OCR 은 건너뛴다")
    p.add_argument("--text-threshold", type=float,
                   help="텍스트 검출 임계값 (기본 0.3). 낮추면 놓친 글자를 더 잡지만 "
                        "오검출이 는다. do_chapter_wide_prediction 이 이 값을 인자로 "
                        "받지 않아서 내부 메서드를 감싸 주입한다")
    args = p.parse_args()

    import sys as _s, os as _o
    _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), 'vlm'))
    import progress as _P; _P.open_log(getattr(args, 'log', None))
    pages = []
    for spec in args.pages:
        hits = sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
        if not hits:
            print(f"경고: 매칭되는 파일이 없습니다: {spec}", file=sys.stderr)
        pages.extend(hits)
    if not pages:
        print("페이지가 없습니다", file=sys.stderr)
        return 2

    bank = load_character_bank(args.bank_dir)
    if not bank["images"]:
        print("캐릭터 뱅크 없음 → 이름 없이 진행합니다. "
              f"인물 클러스터는 {args.batch_size}장 배치 안에서만 일관됩니다.", file=sys.stderr)
    else:
        print(f"캐릭터 뱅크 {len(bank['images'])}장: {sorted(set(bank['names']))}")

    if not torch.cuda.is_available():
        print("경고: GPU 를 못 찾았습니다. CPU 로는 실용적이지 않습니다.", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

    if args.text_threshold is not None:
        # 상류 API 가 임계값을 위로 노출하지 않는다. 기본값만 바꿔 감싼다.
        _orig = model.predict_detections_and_associations

        def _patched(images, *a, **kw):
            kw.setdefault("text_detection_threshold", args.text_threshold)
            return _orig(images, *a, **kw)

        model.predict_detections_and_associations = _patched
        print(f"텍스트 검출 임계값 {args.text_threshold} (기본 0.3)")
    print(f"모델 로드 {time.time()-t0:.1f}초 ({len(pages)}장에 1회)", flush=True)

    sizes = [Image.open(p).size for p in pages]
    images = [read_image(p) for p in pages]

    t = time.time()
    with torch.no_grad():
        results = model.do_chapter_wide_prediction(
            images, bank, eta=args.eta,
            batch_size=args.batch_size, use_tqdm=True, do_ocr=not args.no_ocr,
        )
    print(f"추론 {time.time()-t:.1f}초 ({(time.time()-t)/len(pages):.1f}초/장)", flush=True)

    records = [build_page_record(i, path, size, r)
               for i, (path, size, r) in enumerate(zip(pages, sizes, results))]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "model": MODEL_ID,
            "batch_size": args.batch_size,
            "eta": args.eta,
            "has_character_bank": bool(bank["images"]),
            "pages": records,
        }, fh, ensure_ascii=False, indent=1)

    n_text = sum(len(r["texts"]) for r in records)
    n_ess = sum(1 for r in records for t in r["texts"] if t["essential"])
    n_spk = sum(1 for r in records for t in r["texts"] if t["essential"] and t["speaker"])
    n_clu = sum(1 for r in records for t in r["texts"]
                if t["essential"] and t["speaker_cluster"] is not None)
    print(f"텍스트 {n_text}개 (대사 {n_ess}개) | 인물 귀속 {n_clu}개 "
          f"({100*n_clu/max(n_ess,1):.0f}%) | 이름까지 붙은 것 {n_spk}개")

    if args.transcript:
        with open(args.transcript, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(f"--- page {r['index']+1} ({os.path.basename(r['file'])})\n")
                for t in r["texts"]:
                    if not t["essential"]:
                        continue
                    who = t["speaker"] or (f"#{t['speaker_cluster']}"
                                           if t["speaker_cluster"] is not None else "unsure")
                    fh.write(f"<{who}>: {t['ocr']}\n")
        print(f"대사록 → {args.transcript}")

    if args.visualise_dir:
        os.makedirs(args.visualise_dir, exist_ok=True)
        for img, path, r in zip(images, pages, results):
            stem = os.path.splitext(os.path.basename(path))[0]
            model.visualise_single_image_prediction(
                img, r, os.path.join(args.visualise_dir, f"{stem}.png"))
        print(f"시각화 → {args.visualise_dir}")

    print(f"완료  총 {time.time()-t0:.1f}초 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
