#!/usr/bin/env bash
#
# git clone 직후 상태에서 돌릴 수 있는 상태까지 만든다.
#   ① 가상환경 둘 (magi / ocr)
#   ② config/config.json 에 붙을 LLM 백엔드를 적는다
#
# **LLM 서버는 이 스크립트도, 파이프라인도 띄우지 않는다.** 로컬 vLLM·llama.cpp 를
# 쓸 생각이면 직접 기동해 두고, 여기서는 그 주소만 적는다. 원격 API 를 쓰면
# 띄울 것 자체가 없다.
#
# 여러 번 돌려도 안전하다. 이미 있는 것은 묻고 건너뛴다.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BOLD=$'\e[1m'; DIM=$'\e[2m'; OK=$'\e[32m'; WARN=$'\e[33m'; ERR=$'\e[31m'; OFF=$'\e[0m'
say()  { printf '%s\n' "$*"; }
head2(){ printf '\n%s%s%s\n' "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$OK" "$OFF" "$*"; }
warn() { printf '  %s!%s %s\n' "$WARN" "$OFF" "$*"; }
die()  { printf '  %s✗%s %s\n' "$ERR" "$OFF" "$*" >&2; exit 1; }

# 값이 있으면 그것을, 없으면 기본값을. -y 면 묻지 않는다.
ASSUME_YES=0
[[ "${1:-}" == "-y" || "${1:-}" == "--yes" ]] && ASSUME_YES=1

ask() {  # ask <프롬프트> <기본값>
    local prompt="$1" default="${2:-}" reply
    if (( ASSUME_YES )); then printf '%s' "$default"; return; fi
    read -r -p "$prompt${default:+ [$default]}: " reply </dev/tty || true
    printf '%s' "${reply:-$default}"
}
confirm() {  # confirm <프롬프트>
    (( ASSUME_YES )) && return 0
    local reply; read -r -p "$1 [y/N] " reply </dev/tty || true
    [[ "$reply" == [yY]* ]]
}

# ── 0. 전제 확인 ────────────────────────────────────────────────────────────
head2 "0. 전제 확인"

[[ "$(uname -s)" == "Linux" ]] || warn "리눅스가 아닙니다 ($(uname -s)). pipeline.py 가 돌지 않습니다"
command -v nvidia-smi >/dev/null || warn "nvidia-smi 가 없습니다. GPU 없이는 ① Magi 단계가 실용적이지 않습니다"
command -v nvidia-smi >/dev/null && ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# 파이썬 3.12 를 어디서 얻을지. 시스템 파이썬이 3.13+ 이면 옛 핀이 안 깔린다 —
# Magiv2 는 transformers 4.44.1 에서만 뜨고 그건 3.12 까지다.
PY_MAKER=""
if command -v conda >/dev/null; then PY_MAKER="conda"
elif command -v uv >/dev/null; then PY_MAKER="uv"
elif command -v python3.12 >/dev/null; then PY_MAKER="venv"
else
    die "python 3.12 을 만들 수단이 없습니다. conda / uv / python3.12 중 하나가 필요합니다"
fi
ok "python 3.12 공급: $PY_MAKER"

# ── 1. 가상환경 ────────────────────────────────────────────────────────────
#
# 둘로 나누는 이유: manga-ocr 은 transformers>=4.45 를 요구하고 Magiv2 는
# 4.44.1 에서만 뜬다. 같은 환경에 살 수 없다.
head2 "1. 가상환경"

make_env() {  # make_env <경로>
    local path="$1"
    case "$PY_MAKER" in
        conda) conda create -p "$path" python=3.12 -y >/dev/null ;;
        uv)    uv venv --python 3.12 --seed "$path" >/dev/null ;;
        venv)  python3.12 -m venv "$path" >/dev/null ;;
    esac
}

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

setup_env() {  # setup_env <경로> <설명> <pip 인자들...>
    local path="$1" label="$2"; shift 2
    if [[ -x "$path/bin/python" ]]; then
        ok "$label 이미 있음 ($("$path/bin/python" -V 2>&1))"
        confirm "  다시 만들까요? (기존 삭제)" || return 0
        rm -rf "$path"
    fi
    say "  $label 만드는 중… (torch 내려받기가 오래 걸립니다)"
    make_env "$path"
    "$path/bin/pip" install -q --no-input torch $EXTRA_TORCH --index-url "$TORCH_INDEX"
    "$path/bin/pip" install -q --no-input "$@"
    ok "$label 완료"
}

EXTRA_TORCH="torchvision"
# matplotlib 은 magiv2 의 remote code 가 최상단에서 import 한다. 없으면 모델
# 로드가 ModuleNotFoundError 로 죽는다 — 어느 문서에도 안 적혀 있던 항목이다.
setup_env "$ROOT/magi/.venv" "magi/.venv (Magi + 파이프라인 본체)" \
    "transformers==4.44.1" timm scipy shapely pulp opencv-python einops matplotlib requests

EXTRA_TORCH=""
setup_env "$ROOT/ocr/.venv" "ocr/.venv (manga-ocr, --transcribe ocr 용)" \
    "transformers==4.46.3" manga-ocr

# ── 2. 폰트 ────────────────────────────────────────────────────────────────
head2 "2. 조판 폰트"
FONT_DEFAULT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if [[ ! -f "$FONT_DEFAULT" ]]; then
    FOUND="$(fc-list 2>/dev/null | grep -iE 'NotoSansCJK|NotoSerifCJK' | head -1 | cut -d: -f1 || true)"
    [[ -n "$FOUND" ]] && FONT_DEFAULT="$FOUND"
fi
if [[ -f "$FONT_DEFAULT" ]]; then ok "폰트: $FONT_DEFAULT"
else warn "CJK 폰트를 못 찾았습니다. 조판 단계에서 --font 로 지정해야 합니다"; fi

# ── 3. 백엔드 ──────────────────────────────────────────────────────────────
#
# 이 파이프라인은 **어떤 LLM 서버도 띄우지 않는다.** 로컬이든 원격이든 전부
# OpenAI 호환 주소 하나로 본다. 여기서 정하는 것은 "어디에 붙을지"뿐이다.
head2 "3. LLM 백엔드"
say "  ${DIM}로컬 서버(vLLM·llama.cpp)를 쓰려면 **직접 띄워 두어야 합니다.**${OFF}"
say "  ${DIM}원격 API 를 쓰려면 주소와 키 환경변수만 있으면 됩니다.${OFF}"

BK="$(ask '  기본 백엔드 (vllm/llamacpp/openai/openrouter)' 'vllm')"
BASE_URL="$(ask "  $BK 의 base_url" "$(
  case "$BK" in
    vllm)       echo 'http://127.0.0.1:8000/v1' ;;
    llamacpp)   echo 'http://127.0.0.1:8081/v1' ;;
    openai)     echo 'https://api.openai.com/v1' ;;
    openrouter) echo 'https://openrouter.ai/api/v1' ;;
  esac)")"
KEY_ENV=""
case "$BK" in openai) KEY_ENV=OPENAI_API_KEY ;; openrouter) KEY_ENV=OPENROUTER_API_KEY ;; esac
if [[ -n "$KEY_ENV" ]]; then
    KEY_ENV="$(ask '  API 키를 담은 환경변수 이름' "$KEY_ENV")"
    [[ -n "${!KEY_ENV:-}" ]] && ok "$KEY_ENV 설정됨" \
        || warn "$KEY_ENV 가 비어 있습니다. 실행 전에 export 하세요"
fi
MODEL="$(ask '  기본 모델 (비우면 config 의 stages 를 그대로 둡니다)' '')"

# ── 4. config.json 반영 ────────────────────────────────────────────────────
head2 "4. config/config.json 반영"
python3 - "$BK" "$BASE_URL" "$KEY_ENV" "$MODEL" "$FONT_DEFAULT" <<'PY'
import json, sys
bk, base_url, key_env, model, font = sys.argv[1:6]
p = "config/config.json"
c = json.load(open(p, encoding="utf-8"))
b = c.setdefault("backends", {}).setdefault(bk, {})
b["base_url"] = base_url
b["api_key_env"] = key_env or None
b.setdefault("supports_json_schema", True)
b.setdefault("thinking_style", "reasoning_effort" if bk == "openai" else
             "none" if bk == "openrouter" else "llama_cpp")
b.setdefault("timeout", 1800 if base_url.startswith("http://") else 600)
if model:
    # 별칭이 없으면 만들어 준다 — 모델 id 자체를 별칭으로 쓴다.
    alias = next((a for a, m in c.get("models", {}).items()
                  if m.get("backend") == bk and m.get("model") == model), model)
    c.setdefault("models", {}).setdefault(
        alias, {"backend": bk, "model": model, "vision": True,
                "max_image_pixels": None})
    c["stages"] = {k: alias for k in c.get("stages", {})} or {
        k: alias for k in ["read_page", "read_texts", "styleguide",
                           "translate", "repair", "judge"]}
    print(f"  전 단계 모델 = {alias} ({bk})")
if font:
    c["font"] = font
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"  백엔드 {bk} = {base_url}")
PY
ok "config 갱신"

# ── 5. 점검 ────────────────────────────────────────────────────────────────
head2 "5. 점검"
"$ROOT/magi/.venv/bin/python" -c "
import torch, transformers, cv2
print(f'  torch {torch.__version__} | CUDA {torch.cuda.is_available()} | transformers {transformers.__version__}')" || die "magi/.venv 가 정상이 아닙니다"
"$ROOT/ocr/.venv/bin/python" -c "
import transformers, manga_ocr
print(f'  ocr/.venv: transformers {transformers.__version__} | manga-ocr OK')" || warn "ocr/.venv 가 정상이 아닙니다 (--transcribe ocr 만 못 씁니다)"
# 백엔드가 실제로 그 모델을 내놓는지 본다. 로컬 서버가 안 떠 있으면 여기서
# "닿지 않음" 으로 나오는데, 그건 지금 고칠 일이 아니라 실행 전에 띄우면 될 일이다.
"$ROOT/magi/.venv/bin/python" pipeline.py --list-models || \
    warn "모델 목록을 못 읽었습니다. config/config.json 을 확인하세요"

head2 "끝났습니다"
say "  ./run_autoscribe.sh   로 시작하세요"
say "  ${DIM}Magi 가중치(2GB)는 첫 실행 때 자동으로 내려받습니다${OFF}"
say "  ${DIM}로컬 백엔드를 쓴다면 먼저 서버를 띄우세요 — 이 도구는 띄우지 않습니다${OFF}"
