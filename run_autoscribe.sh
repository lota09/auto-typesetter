#!/usr/bin/env bash
#
# 에피소드 하나를 역식한다. 물어보는 것은 **입력 디렉터리 하나**뿐이고,
# 나머지 인자는 pipeline.py 에 그대로 넘긴다.
#
#   ./run_autoscribe.sh                          # 대화형
#   ./run_autoscribe.sh assets/examples/ep11_cn  # 디렉터리 지정
#   ./run_autoscribe.sh ep11_cn --model gpt-5-mini --workers 8
#   ./run_autoscribe.sh --help                   # pipeline.py 의 전체 옵션
#
# 어느 모델을 쓸지는 config/config.json 의 stages 가 정한다. 한 번만 바꾸려면
# --model 을 넘긴다. **로컬 백엔드(vLLM·llama.cpp)는 미리 띄워 두어야 한다** —
# 이 파이프라인은 서버를 기동하지 않는다. `pipeline.py --list-models` 로 확인.
#
# 작업 디렉터리는 입력 이름에서 만든다 (work/<이름>). 중간에 끊겨도 같은
# 명령을 다시 주면 끝난 단계는 건너뛴다 — 그게 재개의 전부다.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BOLD=$'\e[1m'; DIM=$'\e[2m'; ERR=$'\e[31m'; OFF=$'\e[0m'
die() { printf '%s✗%s %s\n' "$ERR" "$OFF" "$*" >&2; exit 1; }

PY="$ROOT/magi/.venv/bin/python"
[[ -x "$PY" ]] || die "magi/.venv 가 없습니다. 먼저 ./setup_autoscribe.sh 를 돌리세요"

# --help 는 pipeline.py 것을 그대로 보여준다. 옵션 표를 두 곳에 두지 않는다.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s사용:%s ./run_autoscribe.sh [에피소드 디렉터리] [pipeline.py 옵션...]\n\n' "$BOLD" "$OFF"
    exec "$PY" pipeline.py --help
fi

# ── 입력 디렉터리 ──────────────────────────────────────────────────────────
EPISODE="${1:-}"
[[ -n "$EPISODE" && "$EPISODE" != -* ]] && shift || EPISODE=""

if [[ -z "$EPISODE" ]]; then
    printf '%s어느 에피소드를 역식할까요?%s\n' "$BOLD" "$OFF"
    # assets/examples 아래를 후보로 보여준다. 이미지가 든 디렉터리만.
    mapfile -t CANDS < <(
        find assets/examples -mindepth 1 -maxdepth 2 -type d 2>/dev/null |
        while read -r d; do
            n=$(find "$d" -maxdepth 1 -type f \
                \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)
            (( n > 0 )) && printf '%s\t%s\n' "$d" "$n"
        done | sort)
    if (( ${#CANDS[@]} )); then
        i=1
        for row in "${CANDS[@]}"; do
            printf '  %2d) %-48s %s%s장%s\n' "$i" "${row%%$'\t'*}" "$DIM" "${row##*$'\t'}" "$OFF"
            ((i++))
        done
        printf '   %s번호를 고르거나 경로를 직접 입력하세요%s\n' "$DIM" "$OFF"
    fi
    read -r -p "> " EPISODE </dev/tty || true
    # 번호로 골랐으면 경로로 바꾼다.
    if [[ "$EPISODE" =~ ^[0-9]+$ ]] && (( EPISODE >= 1 && EPISODE <= ${#CANDS[@]} )); then
        EPISODE="${CANDS[$((EPISODE-1))]%%$'\t'*}"
    fi
fi

[[ -n "$EPISODE" ]] || die "에피소드 디렉터리가 필요합니다"

# 이름만 준 경우 assets/examples 아래에서 찾아 준다.
if [[ ! -d "$EPISODE" && -d "assets/examples/$EPISODE" ]]; then
    EPISODE="assets/examples/$EPISODE"
fi
[[ -d "$EPISODE" ]] || die "디렉터리가 없습니다: $EPISODE"

EPISODE="${EPISODE%/}"
shopt -s nullglob nocaseglob
PAGES=("$EPISODE"/*.jpg "$EPISODE"/*.jpeg "$EPISODE"/*.png "$EPISODE"/*.webp)
shopt -u nullglob nocaseglob
(( ${#PAGES[@]} )) || die "$EPISODE 에 이미지가 없습니다"

WORK="work/$(basename "$EPISODE")"

# ── 확인 ───────────────────────────────────────────────────────────────────
printf '\n  %-10s %s (%d장)\n' "입력" "$EPISODE" "${#PAGES[@]}"
printf '  %-10s %s%s\n' "작업" "$WORK" \
    "$([[ -d "$WORK" ]] && printf ' %s(이어서 — 끝난 단계는 건너뜁니다)%s' "$DIM" "$OFF")"
printf '  %-10s %s\n' "완성" "$WORK/out"
(( $# )) && printf '  %-10s %s\n' "추가 옵션" "$*"
printf '\n'

# glob 은 셸이 아니라 파이썬이 확장하게 둔다 — **정렬 순서가 곧 읽는 순서**이고,
# 그 정렬은 magi_worker 가 한다. 다만 `dir/*` 로 넘기면 JSON·txt 까지 잡히므로
# 이 디렉터리에서 가장 많은 확장자로 좁힌다.
EXT="$(printf '%s\n' "${PAGES[@]}" | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')"
exec "$PY" pipeline.py --pages "$EPISODE/*.$EXT" --work "$WORK" "$@"
