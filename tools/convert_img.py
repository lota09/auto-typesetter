#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_images.py - 폴더 안의 모든 이미지를 원하는 포맷으로 '병렬' 일괄 변환합니다.

이 스크립트는 직접 디코더를 구현하지 않고, 검증된 오픈소스 프로젝트 위에서 동작합니다.
  - Pillow (PIL Fork)      : 사실상 파이썬 이미지 처리의 표준. JPEG/PNG/WEBP/GIF/TIFF/BMP/ICO ...
      https://github.com/python-pillow/Pillow
  - pillow-heif            : libheif 바인딩. HEIC/HEIF(아이폰 사진) 읽기/쓰기 + AVIF
      https://github.com/bigcat88/pillow_heif
  - pillow-avif-plugin     : libavif 기반 AVIF 플러그인 (선택 사항, 없어도 동작)
      https://github.com/fdintino/pillow-avif-plugin

병렬 처리는 표준 라이브러리 concurrent.futures 의 ProcessPoolExecutor 를 사용합니다.
이미지 인코딩/디코딩은 CPU 바운드 작업이라 스레드보다 프로세스가 확실하게 빠릅니다.
(GIL 영향을 받지 않음. 기본 워커 수 = CPU 코어 수)

결과물은 원본을 건드리지 않고 항상 현재 디렉터리 아래
"<대상폴더이름>_converted" 폴더에 저장됩니다. (-r 사용 시 하위 폴더 구조도 그대로 유지)

설치:
    pip install pillow pillow-heif pillow-avif-plugin

사용 예:
    python convert_images.py                    # 현재 폴더, 포맷은 실행 중 선택
    python convert_images.py --to webp          # 현재 폴더 -> ./<폴더이름>_converted
    python convert_images.py --path ~/사진 -r   # 하위 폴더까지(구조 유지)
    python convert_images.py --to jpg -j 8      # 워커 8개로 병렬 변환
    python convert_images.py --to png -j 1      # 순차 처리(디버깅용)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. 의존성 로드 (검증된 외부 프로젝트 우선)
#    * 이 블록은 모듈 최상단에 있어야 spawn 방식(macOS/Windows)의 워커 프로세스에서도
#      플러그인이 다시 등록됩니다.
# --------------------------------------------------------------------------- #
try:
    from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError
except ImportError:  # pragma: no cover
    sys.exit(
        "[에러] Pillow 가 설치되어 있지 않습니다.\n"
        "       pip install pillow pillow-heif pillow-avif-plugin"
    )

HEIF_OK = False
try:  # HEIC / HEIF (아이폰 기본 포맷) 지원
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
    try:  # pillow-heif 가 AVIF 도 처리 가능한 버전이면 같이 등록
        pillow_heif.register_avif_opener()
    except Exception:
        pass
except ImportError:
    pass

AVIF_OK = False
try:  # AVIF 전용 플러그인 (있으면 더 안정적)
    import pillow_avif  # noqa: F401

    AVIF_OK = True
except ImportError:
    AVIF_OK = "AVIF" in Image.registered_extensions().values()

# 큰 사진에서 나는 경고성 예외 방지 (디컴프레션 폭탄 한도 상향)
Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# 1. 포맷 테이블
# --------------------------------------------------------------------------- #
INPUT_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png", ".apng",
    ".webp",
    ".gif",
    ".bmp", ".dib",
    ".tif", ".tiff",
    ".ico",
    ".tga", ".ppm", ".pgm", ".pbm", ".pnm",
    ".jp2", ".j2k", ".jpx", ".jpf",
    ".heic", ".heif", ".hif",
    ".avif",
}

# 저장 가능한 대상 포맷:  키(사용자 입력) -> (Pillow 포맷명, 실제 확장자)
OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    "jpg":  ("JPEG", ".jpg"),
    "jpeg": ("JPEG", ".jpg"),
    "png":  ("PNG",  ".png"),
    "webp": ("WEBP", ".webp"),
    "avif": ("AVIF", ".avif"),
    "heic": ("HEIF", ".heic"),
    "heif": ("HEIF", ".heif"),
    "tiff": ("TIFF", ".tiff"),
    "tif":  ("TIFF", ".tif"),
    "bmp":  ("BMP",  ".bmp"),
    "gif":  ("GIF",  ".gif"),
    "ico":  ("ICO",  ".ico"),
    "pdf":  ("PDF",  ".pdf"),
}

ANIMATED_TARGETS = {"GIF", "WEBP", "PNG", "TIFF"}      # 다중 프레임 저장 가능
NO_ALPHA_TARGETS = {"JPEG", "PDF", "BMP"}              # 투명도 미지원 -> 배경 합성 필요

MENU_ORDER = ["webp", "jpg", "png", "avif", "tiff", "bmp", "gif", "heic", "ico", "pdf"]


# --------------------------------------------------------------------------- #
# 2. 워커에 넘길 설정 (프로세스 간 전달되므로 순수 값만 담습니다)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cfg:
    fmt: str
    quality: int = 90
    lossless: bool = False
    background: str = "#FFFFFF"
    keep_exif: bool = True
    no_animation: bool = False


@dataclass(frozen=True)
class Job:
    src: Path
    dst: Path
    copy_only: bool = False  # 이미 대상 포맷이면 재인코딩 없이 그대로 복사


@dataclass
class Result:
    src: Path
    dst: Path
    ok: bool
    msg: str = ""
    in_size: int = 0
    out_size: int = 0
    copied: bool = False


# --------------------------------------------------------------------------- #
# 3. 유틸
# --------------------------------------------------------------------------- #
def human(size: float) -> str:
    """바이트를 사람이 읽기 좋은 문자열로."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if size < step:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= step
    return f"{size:.1f}TB"


def collect_images(root: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    """대상 폴더에서 변환할 이미지 목록을 수집."""
    it = root.rglob("*") if recursive else root.glob("*")
    files = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in INPUT_EXTS:
            continue
        if p.name.startswith("."):  # 숨김/시스템 파일 제외
            continue
        # 출력 폴더가 대상 폴더 안에 있을 때 결과물을 다시 읽지 않도록 제외
        if exclude is not None and exclude in p.resolve().parents:
            continue
        files.append(p)
    return sorted(files)


def ask_format() -> str:
    """대상 포맷을 대화식으로 선택."""
    print("\n어떤 포맷으로 변환할까요?")
    for i, key in enumerate(MENU_ORDER, 1):
        note = ""
        if key == "avif" and not (AVIF_OK or HEIF_OK):
            note = "  (플러그인 없음 - 실패할 수 있음)"
        if key == "heic" and not HEIF_OK:
            note = "  (pillow-heif 없음 - 실패할 수 있음)"
        print(f"  {i:>2}. {key}{note}")
    while True:
        raw = input("번호 또는 포맷 이름 입력 > ").strip().lower()
        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(MENU_ORDER):
            return MENU_ORDER[int(raw) - 1]
        raw = raw.lstrip(".")
        if raw in OUTPUT_FORMATS:
            return raw
        print(f"  '{raw}' 는 지원하지 않는 포맷입니다. 다시 입력해 주세요.")


def confirm(question: str, default_yes: bool = False) -> bool:
    """y/N 확인 프롬프트. 비대화형 환경이면 기본값 반환."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    if not sys.stdin or not sys.stdin.isatty():
        return default_yes
    while True:
        try:
            raw = input(f"{question} {suffix} ").strip().lower()
        except EOFError:
            return default_yes
        if raw == "":
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def unique_path(path: Path, reserved: set[Path]) -> Path:
    """
    같은 이름이 있으면 name (1).ext, name (2).ext ... 로 회피.
    reserved 는 '이번 실행에서 이미 배정된 경로' 집합 - 병렬 실행 시
    두 워커가 같은 파일명을 잡는 경쟁 상태를 막기 위해 메인에서 미리 배정합니다.
    """
    def taken(p: Path) -> bool:
        return p.exists() or p in reserved

    if not taken(path):
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 1
    while True:
        cand = parent / f"{stem} ({n}){suffix}"
        if not taken(cand):
            return cand
        n += 1


# --------------------------------------------------------------------------- #
# 4. 변환 본체 (워커 프로세스에서 실행됨 - 모듈 최상위 함수라 pickle 가능)
# --------------------------------------------------------------------------- #
def build_save_kwargs(cfg: Cfg) -> dict:
    """포맷별 저장 옵션."""
    fmt, kw = cfg.fmt, {}
    if fmt == "JPEG":
        kw.update(quality=cfg.quality, optimize=True, progressive=True, subsampling="4:2:0")
    elif fmt == "WEBP":
        kw.update(quality=cfg.quality, method=6)
        if cfg.lossless:
            kw.update(lossless=True)
    elif fmt in ("AVIF", "HEIF"):
        kw.update(quality=cfg.quality)
    elif fmt == "PNG":
        kw.update(optimize=True, compress_level=9)
    elif fmt == "TIFF":
        kw.update(compression="tiff_lzw")
    return kw


def prepare_frame(img: "Image.Image", cfg: Cfg) -> "Image.Image":
    """대상 포맷에 맞게 색상 모드를 정리."""
    fmt = cfg.fmt
    if fmt in NO_ALPHA_TARGETS:
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            base = Image.new("RGB", img.size, cfg.background)
            rgba = img.convert("RGBA")
            base.paste(rgba, mask=rgba.split()[-1])
            return base
        return img if img.mode == "RGB" else img.convert("RGB")

    if fmt == "GIF":
        return img if img.mode in ("P", "L") else img.convert("RGB")

    if fmt in ("WEBP", "AVIF", "HEIF"):
        if img.mode in ("RGB", "RGBA"):
            return img
        has_alpha = "A" in img.mode or "transparency" in img.info
        return img.convert("RGBA" if has_alpha else "RGB")

    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    return img


def convert_one(job: Job, cfg: Cfg) -> Result:
    """이미지 한 장 변환. 워커 프로세스에서 호출됩니다."""
    src, dst = job.src, job.dst
    try:
        in_size = src.stat().st_size

        # 이미 대상 포맷이면 재인코딩(화질 손실) 없이 그대로 복사
        if job.copy_only:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return Result(src, dst, True, "", in_size, dst.stat().st_size, copied=True)

        with Image.open(src) as im:
            im.load()

            n_frames = getattr(im, "n_frames", 1)
            animated = n_frames > 1 and not cfg.no_animation

            save_kw = build_save_kwargs(cfg)
            if im.info.get("icc_profile"):
                save_kw["icc_profile"] = im.info["icc_profile"]
            if cfg.keep_exif and im.info.get("exif"):
                save_kw["exif"] = im.info["exif"]

            dst.parent.mkdir(parents=True, exist_ok=True)

            if animated and cfg.fmt in ANIMATED_TARGETS:
                frames = [prepare_frame(f.copy(), cfg) for f in ImageSequence.Iterator(im)]
                save_kw.update(
                    save_all=True,
                    append_images=frames[1:],
                    loop=im.info.get("loop", 0),
                    duration=im.info.get("duration", 100),
                    disposal=2,
                )
                frames[0].save(dst, format=cfg.fmt, **save_kw)
            else:
                base = im
                if animated:  # 대상이 애니메이션 미지원 -> 첫 프레임만
                    base = ImageSequence.Iterator(im)[0].copy()
                if cfg.keep_exif:
                    # EXIF Orientation 을 실제 픽셀에 반영 (세로 사진 눕는 문제 방지)
                    base = ImageOps.exif_transpose(base) or base
                out = prepare_frame(base, cfg)
                if cfg.fmt == "ICO":
                    save_kw["sizes"] = [(256, 256)]
                out.save(dst, format=cfg.fmt, **save_kw)

        return Result(src, dst, True, "", in_size, dst.stat().st_size)
    except UnidentifiedImageError:
        return Result(src, dst, False, "이미지로 인식할 수 없음")
    except Exception as e:  # noqa: BLE001
        return Result(src, dst, False, f"{type(e).__name__}: {e}")


def _worker(payload: tuple[Job, Cfg]) -> Result:
    """ProcessPoolExecutor 진입점."""
    job, cfg = payload
    return convert_one(job, cfg)


# --------------------------------------------------------------------------- #
# 5. 실행 (순차 / 병렬)
# --------------------------------------------------------------------------- #
def run_serial(jobs: list[Job], cfg: Cfg, on_done) -> list[Result]:
    results = []
    for job in jobs:
        res = convert_one(job, cfg)
        results.append(res)
        on_done(res)
    return results


def run_parallel(jobs: list[Job], cfg: Cfg, workers: int, on_done) -> list[Result]:
    results: list[Result] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (job, cfg)): job for job in jobs}
        try:
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # 워커가 죽은 경우까지 방어
                    res = Result(job.src, job.dst, False, f"{type(e).__name__}: {e}")
                results.append(res)
                on_done(res)
        except KeyboardInterrupt:
            for fut in futures:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return results


# --------------------------------------------------------------------------- #
# 6. CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="convert_images.py",
        description="폴더 안의 모든 이미지(heic, webp, jpg ...)를 원하는 포맷으로 병렬 일괄 변환합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""결과물은 항상 현재 디렉터리 아래 '<대상폴더이름>_converted' 폴더에 저장됩니다.

예시:
  python convert_images.py                       현재 폴더, 포맷은 실행 중 선택
  python convert_images.py --to webp             현재 폴더 -> ./<폴더이름>_converted
  python convert_images.py --path ~/Pictures -r  하위 폴더까지(구조 유지)
  python convert_images.py --to jpg -q 90 -j 8   워커 8개로 병렬 변환
  python convert_images.py --to png -j 1         순차 처리(디버깅용)
  python convert_images.py --to png --dry-run    변환 없이 목록만 확인
""",
    )
    p.add_argument("--path", "-p", default=".",
                   help="대상 폴더 (기본: 현재 디렉터리)")
    p.add_argument("--to", "-t", default=None,
                   help=f"대상 포맷 ({', '.join(MENU_ORDER)}). 생략하면 실행 중에 물어봅니다.")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="하위 폴더까지 처리 (기본: 현재 폴더만). 출력 시 폴더 구조를 그대로 유지합니다.")
    p.add_argument("--jobs", "-j", type=int, default=0,
                   help="동시에 돌릴 워커 프로세스 수 (기본: CPU 코어 수, 1이면 순차 처리)")
    p.add_argument("--out", "-o", default=None,
                   help="결과 저장 폴더를 직접 지정 "
                        "(기본: 현재 디렉터리 아래 '<대상폴더이름>_converted')")
    p.add_argument("--quality", "-q", type=int, default=90,
                   help="손실 압축 품질 1-100 (기본: 90)")
    p.add_argument("--lossless", action="store_true",
                   help="WEBP 등에서 무손실 저장")
    p.add_argument("--background", default="#FFFFFF",
                   help="투명 -> 불투명(JPEG 등) 변환 시 배경색 (기본: 흰색)")
    p.add_argument("--overwrite", action="store_true",
                   help="같은 이름의 파일이 있으면 덮어쓰기 (기본: 이름 뒤에 (1) 붙여 저장)")
    p.add_argument("--no-exif", dest="keep_exif", action="store_false",
                   help="EXIF/회전 정보를 유지하지 않음 (기본: 유지)")
    p.add_argument("--no-animation", action="store_true",
                   help="애니메이션 GIF/WEBP 를 첫 프레임만 변환")
    p.add_argument("--delete-original", action="store_true",
                   help="변환 성공한 원본을 삭제 (마지막 확인 프롬프트를 건너뜁니다)")
    p.add_argument("--keep-original", action="store_true",
                   help="원본을 무조건 유지 (마지막 확인 프롬프트를 건너뜁니다)")
    p.add_argument("--dry-run", action="store_true",
                   help="실제로 변환하지 않고 대상 목록만 출력")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"[에러] 폴더를 찾을 수 없습니다: {root}", file=sys.stderr)
        return 2

    # 결과물은 항상 별도 폴더에: 현재 디렉터리 아래 "<대상폴더이름>_converted"
    # (--out 으로 다른 위치를 직접 지정할 수 있습니다)
    if args.out:
        out_root = Path(args.out).expanduser().resolve()
    else:
        out_root = (Path.cwd() / f"{root.name or 'images'}_converted").resolve()

    # 1) 대상 파일 수집 -- 없으면 실패로 종료
    files = collect_images(root, args.recursive, exclude=out_root)
    if not files:
        scope = "하위 폴더 포함" if args.recursive else "현재 폴더"
        print(f"[에러] 변환할 이미지가 없습니다 ({scope}): {root}", file=sys.stderr)
        print("       다른 폴더라면 --path 로 지정하세요. 예) "
              "python convert_images.py --path ~/Pictures", file=sys.stderr)
        return 1

    # 1-1) 플러그인 누락 경고
    if not HEIF_OK and any(f.suffix.lower() in {".heic", ".heif", ".hif"} for f in files):
        print("[경고] HEIC/HEIF 파일이 있지만 pillow-heif 가 없어 읽을 수 없습니다.\n"
              "       pip install pillow-heif", file=sys.stderr)
    if not (AVIF_OK or HEIF_OK) and any(f.suffix.lower() == ".avif" for f in files):
        print("[경고] AVIF 파일이 있지만 플러그인이 없어 읽지 못할 수 있습니다.\n"
              "       pip install pillow-avif-plugin", file=sys.stderr)

    # 2) 대상 포맷 결정 (--to 없으면 항상 물어봄)
    key = (args.to or "").strip().lower().lstrip(".")
    if key and key not in OUTPUT_FORMATS:
        print(f"[에러] 지원하지 않는 대상 포맷: {args.to}", file=sys.stderr)
        print(f"       가능한 값: {', '.join(MENU_ORDER)}", file=sys.stderr)
        return 2
    if not key:
        if not sys.stdin or not sys.stdin.isatty():
            print("[에러] 비대화형 실행에서는 --to 로 대상 포맷을 지정해야 합니다.", file=sys.stderr)
            return 2
        key = ask_format()

    fmt, out_ext = OUTPUT_FORMATS[key]
    if fmt == "HEIF" and not HEIF_OK:
        print("[에러] HEIC/HEIF 저장에는 pillow-heif 가 필요합니다: pip install pillow-heif",
              file=sys.stderr)
        return 3

    cfg = Cfg(
        fmt=fmt,
        quality=max(1, min(100, args.quality)),
        lossless=args.lossless,
        background=args.background,
        keep_exif=args.keep_exif,
        no_animation=args.no_animation,
    )

    # 3) 작업 목록 만들기 (출력 경로 배정은 메인에서 -- 병렬 경쟁 방지)
    jobs: list[Job] = []
    reserved: set[Path] = set()
    for src in files:
        rel = src.relative_to(root)
        dst_dir = out_root / rel.parent          # -r 일 때 폴더 구조 그대로 유지
        dst = dst_dir / (src.stem + out_ext)

        # 이미 대상 포맷이면 재인코딩하지 않고 복사만 (화질 손실 방지)
        copy_only = src.suffix.lower() == out_ext.lower()

        if not args.overwrite:
            dst = unique_path(dst, reserved)
        reserved.add(dst)
        jobs.append(Job(src, dst, copy_only))

    # 4) 워커 수 결정
    cpu = os.cpu_count() or 1
    workers = args.jobs if args.jobs > 0 else cpu
    workers = max(1, min(workers, len(jobs)))
    mode = "순차" if workers == 1 else f"병렬 워커 {workers}개 (CPU {cpu}코어)"

    print(f"\n대상 폴더 : {root}{'  (하위 폴더 포함)' if args.recursive else ''}")
    print(f"대상 포맷 : {key}  (품질 {cfg.quality}{', 무손실' if cfg.lossless else ''})")
    print(f"저장 위치 : {out_root}")
    print(f"처리 방식 : {mode}")
    print(f"찾은 이미지: {len(files)}개\n")

    if args.dry_run:
        for job in jobs:
            tag = "  (복사)" if job.copy_only else ""
            print(f"  {job.src.relative_to(root)}  ->  "
                  f"{job.dst.relative_to(out_root)}{tag}")
        print("\n" + "-" * 60)
        print(f"dry-run: {len(jobs)}개가 변환 대상입니다. (실제 변환은 하지 않았습니다)")
        return 0

    # 5) 실행 -- 완료되는 순서대로 진행 상황 출력
    total = len(jobs)
    width = len(str(total))
    done = 0

    def on_done(res: Result) -> None:
        nonlocal done
        done += 1
        rel = res.src.relative_to(root)
        if res.ok:
            if res.copied:
                print(f"[{done:>{width}}/{total}] {rel}  ->  {res.dst.name}  "
                      f"(이미 {key} - 원본 그대로 복사)")
                return
            diff = (res.out_size - res.in_size) / res.in_size * 100 if res.in_size else 0
            print(f"[{done:>{width}}/{total}] {rel}  ->  {res.dst.name}  "
                  f"({human(res.in_size)} → {human(res.out_size)}, {diff:+.0f}%)")
        else:
            print(f"[{done:>{width}}/{total}] {rel}  ->  실패: {res.msg}")

    started = time.perf_counter()
    if workers == 1:
        results = run_serial(jobs, cfg, on_done)
    else:
        results = run_parallel(jobs, cfg, workers, on_done)
    elapsed = time.perf_counter() - started

    # 6) 요약
    converted = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_in = sum(r.in_size for r in converted)
    total_out = sum(r.out_size for r in converted)

    copied = sum(1 for r in converted if r.copied)
    print("\n" + "-" * 60)
    print(f"성공 {len(converted)}개"
          f"{f' (그중 복사 {copied}개)' if copied else ''} / 실패 {len(failed)}개")
    print(f"결과 폴더: {out_root}")
    if converted:
        diff = (total_out - total_in) / total_in * 100 if total_in else 0
        print(f"용량 {human(total_in)} → {human(total_out)} ({diff:+.1f}%)")
    rate = len(results) / elapsed if elapsed > 0 else 0
    print(f"소요 시간 {elapsed:.2f}초  ({rate:.1f}장/초, {mode})")
    for r in failed:
        print(f"  실패: {r.src.relative_to(root)} - {r.msg}")

    # 7) 마지막에 원본 삭제 여부 확인 (기본값 n = 유지)
    if converted and not args.keep_original:
        do_delete = args.delete_original or confirm(
            f"\n변환에 성공한 원본 {len(converted)}개를 삭제할까요? (기본: 유지)",
            default_yes=False,
        )
        if do_delete:
            removed = 0
            for r in converted:
                try:
                    if (r.dst.exists() and r.dst.stat().st_size > 0
                            and r.dst.resolve() != r.src.resolve()):
                        r.src.unlink()
                        removed += 1
                except OSError as e:
                    print(f"  삭제 실패: {r.src.name} - {e}")
            print(f"원본 {removed}개를 삭제했습니다.")
        else:
            print("원본을 그대로 두었습니다.")

    return 0 if not failed else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        sys.exit(130)