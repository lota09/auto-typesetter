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
import sys
import threading

import requests

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config", "config.json")


class ConfigError(RuntimeError):
    pass


class EmptyResponse(RuntimeError):
    """본문 없이 돌아왔다. 대개 추론에 예산을 다 쓴 경우다."""


class TruncatedResponse(RuntimeError):
    """본문은 왔는데 max_tokens 에서 잘렸다.

    빈 응답과 갈라 두는 이유: 증상도 대처도 다르다. 빈 응답은 추론이 예산을
    먹은 것이라 추론을 끄면 풀린다. 잘린 응답은 **출력 자체가 예산보다 큰**
    것이라 추론을 꺼도 그대로다 — 예산을 키우거나 출력을 줄여야 한다.

    이걸 구분하지 않으면 잘린 JSON 이 그대로 json.loads 로 들어가
    `Unterminated string starting at line 2334` 같은, 원인을 전혀 알려주지 않는
    예외가 난다. 실제로 77페이지 챕터에서 그렇게 죽었다.
    """


def load_config(path=None):
    path = path or os.environ.get("AUTOTYPESET_CONFIG") or DEFAULT_CONFIG
    if not os.path.exists(path):
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Client:
    # 잘림 재시도가 올라갈 수 있는 천장. 출력만으로 컨텍스트를 다 먹으면
    # 입력이 잘려 나가 결과가 더 나빠진다 (지금 ctx-size 는 65536).
    MAX_RETRY_TOKENS = 32768

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
        # 세션은 **스레드마다** 따로 둔다. requests.Session 은 스레드 안전이
        # 보장되지 않는다 (커넥션 풀과 쿠키 저장소를 공유한다). 크롭 전사를
        # 동시에 던지기 시작하면 그게 곧 간헐적이고 재현 안 되는 실패가 된다.
        self._local = threading.local()
        self._usage_lock = threading.Lock()
        # 호출 사용량을 쌓아 둔다. "추론을 켜면 느리다"를 시간이 아니라 토큰으로
        # 재기 위한 것이다 — 시간은 모델 적재·교체에 오염되지만 생성 토큰 수는
        # 그 모델이 실제로 얼마나 생각했는지만 말한다.
        self.usage = {"calls": 0, "prompt": 0, "completion": 0}

        key_env = backend_cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env) if key_env else None
        if key_env and not self.api_key:
            raise ConfigError(f"백엔드가 {key_env} 환경변수를 요구합니다 (모델 {name})")

    @property
    def session(self):
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = self._local.session = requests.Session()
        return sess

    # ── 상태 확인 ────────────────────────────────────────────────────────
    def served_models(self, timeout=10):
        """서버가 지금 내놓는 모델 이름 목록. 실패하면 None."""
        try:
            r = self.session.get(self.base_url + "/models", timeout=timeout)
            if r.status_code != 200:
                return None
            return [m.get("id") for m in (r.json().get("data") or [])]
        except (requests.RequestException, ValueError):
            return None

    def health(self):
        """살아 있고 **우리가 원하는 모델을 내놓는지**까지 본다.

        200 만 보면 안 된다. 새 서버를 기다리다 **옛 서버**의 200 을 보고 통과하거나,
        기동에 실패했는데 이전 서버가 살아남아 그걸로 측정한 사고가 실제로 있었다
        (vLLM 학습 문서의 측정 함정 목록). 엔진을 갈아 끼우는 지금은 더 위험하다 —
        포트는 같은데 모델이 다를 수 있다.
        """
        names = self.served_models()
        if names is None:
            return False
        # 라우터(llama.cpp)는 적재 전에도 프리셋 전체를 나열한다. 이름만 맞으면 된다.
        # 원격 API 도 같은 방식으로 본다 — 로컬이라고 다르게 취급하지 않는다.
        return self.model in names

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
             retry_without_thinking=True, retry_on_truncation=True):
        """content 는 문자열이거나 OpenAI 형식의 content 파트 리스트다.

        schema 를 주면 구조화 출력을 강제하고 파싱해서 돌려준다. 문법 제약이
        없으면 모델이 형식을 깬다 — 정규식으로 뒤쫓지 않고 여기서 막는다.

        예산이 모자란 실패는 두 가지이고 대처가 다르다. 둘 다 스크립트마다 따로
        겪었으므로 여기에 모은다 — 결과 없이 죽는 것보다 낫다.

          본문이 빔 (추론이 예산을 다 먹음) → 추론을 끄고 한 번 더
          본문이 잘림 (출력이 예산보다 큼)  → 예산을 2배로 하고 한 번 더
        """
        try:
            return self._chat_once(content, schema, schema_name, thinking,
                                   max_tokens, temperature, timeout)
        except EmptyResponse:
            if not (thinking and retry_without_thinking):
                raise
            return self._chat_once(content, schema, schema_name, False,
                                   max_tokens, temperature, timeout)
        except TruncatedResponse:
            # 예산을 키우는 것은 **출력이 예산보다 조금 큰** 경우에만 답이다.
            # 모델이 배열을 끝없이 뱉는 폭주라면 두 배를 줘도 두 배로 탈 뿐이다.
            # 실제로 24576 → 49152 를 태우고 똑같이 잘린 적이 있다(13분 낭비).
            # 그래서 한 번만, 그리고 천장 아래에서만 다시 시도한다. 폭주는
            # 여기서 못 고친다 — 스키마에 maxItems 를 주는 것이 답이다.
            if not retry_on_truncation or max_tokens >= self.MAX_RETRY_TOKENS:
                raise
            bigger = min(max_tokens * 2, self.MAX_RETRY_TOKENS)
            print(f"[{self.name}] 출력이 잘렸습니다 — 예산 {max_tokens} → {bigger} "
                  "로 1회 재시도", file=sys.stderr, flush=True)
            return self._chat_once(content, schema, schema_name, thinking,
                                   bigger, temperature, timeout)

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
        body = r.json()
        u = body.get("usage") or {}
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["prompt"] += u.get("prompt_tokens") or 0
            self.usage["completion"] += u.get("completion_tokens") or 0
        choice = body["choices"][0]
        finish = choice.get("finish_reason")
        text = (choice["message"].get("content") or "").strip()
        if not text:
            raise EmptyResponse(
                f"[{self.name}] 본문이 비었습니다 (finish_reason={finish}). "
                "추론을 끄거나 max_tokens 를 올리세요")
        if schema is None:
            return text
        # 스키마를 강제해도 코드펜스를 붙여 오는 경우가 있어 방어적으로 벗긴다.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if finish == "length":
                raise TruncatedResponse(
                    f"[{self.name}] 출력이 max_tokens={max_tokens} 에서 잘렸습니다 "
                    f"(생성 {u.get('completion_tokens')} 토큰, {len(text)}자). "
                    "예산을 키우거나 출력 크기를 줄이세요")
            raise


def bound_schema(node, max_str=80, max_items=60):
    """스키마의 모든 문자열·배열에 상한을 채워 넣는다. 이미 있으면 건드리지 않는다.

    상한이 없으면 문법이 끝을 강제하지 않는다. 그래서 모델이 배열을 계속 뱉거나
    문자열을 계속 이어도 아무것도 막지 못한다. 실측으로 두 번 당했다 —
    인물 6명짜리 시트가 188,360자로, 인물 목록 하나가 19,118자로 폭주했다.

    llama.cpp 는 json_schema 를 GBNF 로 바꿔 **디코딩 중에** 강제하므로,
    상한은 프롬프트의 부탁이 아니라 물리적 제약이 된다. maxLength 10 을 주고
    긴 글을 요청하면 정확히 10자에서 끊긴다 (실측).

    필드를 하나씩 손으로 막는 방식은 빠뜨린 하나가 곧 같은 사고다. 실제로
    evidence 와 story 만 막았다가 id·aliases·notPeople 이 뚫렸다. 그래서 훑는다.
    유별나게 길어야 하는 필드는 **스키마에 직접** 값을 적어 두면 보존된다.
    """
    if isinstance(node, dict):
        t = node.get("type")
        if t == "string" and "maxLength" not in node and "enum" not in node:
            node["maxLength"] = max_str
        elif t == "array" and "maxItems" not in node:
            node["maxItems"] = max_items
        for v in node.values():
            bound_schema(v, max_str, max_items)
    elif isinstance(node, list):
        for v in node:
            bound_schema(v, max_str, max_items)
    return node


def usage_line(*clients):
    """벤치마크가 파싱할 한 줄. 스크립트마다 같은 형식으로 낸다."""
    c = sum(x.usage["calls"] for x in clients)
    p = sum(x.usage["prompt"] for x in clients)
    g = sum(x.usage["completion"] for x in clients)
    return f"[usage] calls={c} prompt={p} completion={g}"


# 단계 → 역할. read 는 이미지를 보고, translate 는 글만 다룬다.
ROLE_OF_STAGE = {
    "read_page": "read", "read_texts": "read",
    "styleguide": "translate", "translate": "translate",
    "repair": "translate", "judge": "translate",
}

# 역할 설정에서 생략할 수 있는 것들의 기본값. config 에는 base_url·api_key_env·
# model 만 있으면 된다.
ROLE_DEFAULTS = {
    "thinking_style": "llama_cpp",
    "supports_json_schema": True,
    "timeout": 1800,
    "vision": True,
    "max_image_pixels": None,
}


def client_for(stage, config=None, override=None):
    """단계 이름으로 호출기를 만든다. override 로 모델 id 를 강제할 수 있다."""
    cfg = config if isinstance(config, dict) else load_config(config)
    role = ROLE_OF_STAGE.get(stage)
    if role is None:
        raise ConfigError(f"'{stage}' 는 read/translate 어느 역할에도 속하지 않습니다")
    spec = cfg.get(role)
    if not isinstance(spec, dict):
        raise ConfigError(f"config 에 '{role}' 역할이 없습니다 "
                          f"(base_url · api_key_env · model 이 필요합니다)")
    merged = dict(ROLE_DEFAULTS)
    merged.update(spec)
    if not merged.get("base_url"):
        raise ConfigError(f"'{role}' 역할에 base_url 이 없습니다")
    model = override or merged.get("model")
    if not model:
        raise ConfigError(f"'{role}' 역할에 model 이 없습니다")
    # Client 는 모델 설정과 백엔드 설정을 나눠 받는다. 지금은 한 덩어리라 둘로 준다.
    return Client(model, {"model": model, "vision": merged["vision"],
                          "max_image_pixels": merged["max_image_pixels"]}, merged)
