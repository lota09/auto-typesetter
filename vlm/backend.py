"""단계별 모델 호출을 한 곳으로 모은다.

지금까지 스크립트 네 개가 각자 requests.post 를 하고 있었고, 그 안에
llama.cpp 전용 규약(`chat_template_kwargs.enable_thinking`)이 박혀 있었다.
다른 백엔드로 갈 길을 열어두려면 그 규약을 한 곳에 가둬야 한다.

제공자마다 다른 것은 사실상 셋뿐이다:
  1. base_url 과 인증
  2. **추론을 끄는 방법** — 이게 진짜 문제다. llama.cpp 는
     chat_template_kwargs.enable_thinking, OpenAI 계열은 reasoning_effort 로
     제어한다. 그리고 이 파이프라인에서 추출 단계의 추론 끄기는 취향이 아니라
     필수다: 추론형 모델은 토큰 예산을 사고에 다 쓰고 content 를 빈 채
     finish_reason=length 로 돌려준다.
  3. 구조화 출력 지원 여부 — 스키마 강제가 없으면 형식이 깨진다 (프롬프트로
     부탁했을 때 8페이지 중 5페이지가 파싱 실패했다).

모델 선택은 요청의 model 필드로 한다. 로컬 llama-server 를 라우터 모드로 띄우면
그것만으로 온디맨드 교체가 되므로 (config/models.ini) 파이프라인이 서버를
관리하지 않는다.
"""

import json
import os
import re

import requests

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config", "config.json")


class ConfigError(RuntimeError):
    pass


class EmptyResponse(RuntimeError):
    """본문 없이 돌아왔다. 대개 추론에 예산을 다 쓴 경우다."""


def load_config(path=None):
    path = path or os.environ.get("AUTOTYPESET_CONFIG") or DEFAULT_CONFIG
    if not os.path.exists(path):
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Client:
    """한 단계가 쓰는 모델 하나에 대한 호출기."""

    def __init__(self, name, model_cfg, backend_cfg):
        self.name = name
        self.model = model_cfg.get("model", name)
        self.vision = bool(model_cfg.get("vision"))
        self.max_image_pixels = model_cfg.get("max_image_pixels")
        self.base_url = backend_cfg["base_url"].rstrip("/")
        self.thinking_style = backend_cfg.get("thinking_style", "none")
        self.supports_json_schema = backend_cfg.get("supports_json_schema", True)
        self.timeout = backend_cfg.get("timeout", 1800)
        self.session = requests.Session()

        key_env = backend_cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env) if key_env else None
        if key_env and not self.api_key:
            raise ConfigError(f"백엔드가 {key_env} 환경변수를 요구합니다 (모델 {name})")

    # ── 상태 확인 ────────────────────────────────────────────────────────
    def health(self):
        """로컬 서버만 /health 를 준다. 원격은 확인을 생략한다."""
        if not self.base_url.startswith("http://127.0.0.1") and \
           not self.base_url.startswith("http://localhost"):
            return True
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            r = self.session.get(root + "/health", timeout=10)
            if r.status_code == 200:
                return True
            # 라우터 모드는 모델이 적재되기 전에는 /health 가 200 이 아닐 수 있다.
            return self.session.get(self.base_url + "/models", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    # ── 요청 조립 ────────────────────────────────────────────────────────
    def _apply_thinking(self, payload, thinking):
        if self.thinking_style == "llama_cpp":
            payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        elif self.thinking_style == "reasoning_effort":
            payload["reasoning_effort"] = "medium" if thinking else "none"
        # "none": 제어 수단이 없다. 끌 수 없으므로 max_tokens 를 넉넉히 두는 것으로
        # 대응한다 — 호출하는 쪽 책임이다.

    def chat(self, content, schema=None, schema_name="result", thinking=False,
             max_tokens=2048, temperature=0.0, timeout=None,
             retry_without_thinking=True):
        """content 는 문자열이거나 OpenAI 형식의 content 파트 리스트다.

        schema 를 주면 구조화 출력을 강제하고 파싱해서 돌려준다. 문법 제약이
        없으면 모델이 형식을 깬다 — 정규식으로 뒤쫓지 않고 여기서 막는다.

        추론을 켠 호출이 예산을 다 쓰고 본문을 못 내면(finish_reason=length,
        content='') 추론을 끄고 한 번 더 시도한다. 이 실패는 스크립트마다 따로
        겪었으므로 백엔드에서 처리한다 — 결과 없이 죽는 것보다 낫다.
        """
        try:
            return self._chat_once(content, schema, schema_name, thinking,
                                   max_tokens, temperature, timeout)
        except EmptyResponse:
            if not (thinking and retry_without_thinking):
                raise
            return self._chat_once(content, schema, schema_name, False,
                                   max_tokens, temperature, timeout)

    def _chat_once(self, content, schema, schema_name, thinking,
                   max_tokens, temperature, timeout):
        messages = [{"role": "user",
                     "content": content if isinstance(content, list) else str(content)}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        self._apply_thinking(payload, thinking)
        if schema is not None and self.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            }

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self.session.post(self.base_url + "/chat/completions", json=payload,
                              headers=headers, timeout=timeout or self.timeout)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text:
            raise EmptyResponse(
                f"[{self.name}] 본문이 비었습니다 "
                f"(finish_reason={choice.get('finish_reason')}). "
                "추론을 끄거나 max_tokens 를 올리세요")
        if schema is None:
            return text
        # 스키마를 강제해도 코드펜스를 붙여 오는 경우가 있어 방어적으로 벗긴다.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        return json.loads(re.sub(r"\s*```$", "", text))


def client_for(stage, config=None, override=None):
    """단계 이름으로 호출기를 만든다. override 로 모델을 강제할 수 있다."""
    cfg = config if isinstance(config, dict) else load_config(config)
    name = override or (cfg.get("stages") or {}).get(stage)
    if not name:
        raise ConfigError(f"stages 에 '{stage}' 항목이 없습니다")
    models = cfg.get("models") or {}
    if name not in models:
        raise ConfigError(f"models 에 '{name}' 이 없습니다")
    mc = models[name]
    backends = cfg.get("backends") or {}
    bname = mc.get("backend")
    if bname not in backends:
        raise ConfigError(f"backends 에 '{bname}' 이 없습니다 (모델 {name})")
    return Client(name, mc, backends[bname])
