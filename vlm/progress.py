#!/usr/bin/env python3
"""터미널은 짧게, 로그 파일은 전부.

두 가지를 동시에 원한다:
  - 파이프라인이 무엇을 하고 있는지 **한눈에** 보이는 터미널
  - 나중에 뒤져 볼 수 있는 **빠짐없는** 기록

그래서 반복되는 진행 줄(페이지마다·박스마다 한 줄)은 터미널에서는 **같은 자리를
덮어쓰고**, 파일에는 한 줄씩 쌓는다. 진행바는 쓰지 않는다 — 파일에 남는 기록이
제어문자로 지저분해지고, [44/66] 이 그 자체로 이미 진행도다.

터미널이 아니면(리다이렉트·CI) 덮어쓰기를 끄고 전부 그냥 찍는다.
"""

import os
import sys

_fh = None
_live = False          # 지금 덮어쓰기 줄이 떠 있는가
_width = 0


class _Tee:
    """터미널로 나가는 것을 파일에도 흘린다.

    단계 스크립트마다 print 를 수백 개 고치는 대신 stdout 을 한 번 감싼다.
    step() 이 쓰는 덮어쓰기 줄만 예외다 — 그건 파일에 따로 온전한 줄로 남기고
    터미널에는 제어문자로 나가므로, 여기서는 걸러야 한다.
    """

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, data):
        self._s.write(data)
        # 캐리지리턴이 든 것은 step() 의 덮어쓰기다. 파일에는 이미 들어갔다.
        if self._f and "\r" not in data:
            self._f.write(data)

    def flush(self):
        self._s.flush()
        if self._f:
            self._f.flush()

    def isatty(self):
        return self._s.isatty()

    def __getattr__(self, name):
        return getattr(self._s, name)


def open_log(path):
    """work 디렉터리의 로그 파일을 연다. 이어붙인다 — 재개해도 이력이 남는다."""
    global _fh
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _fh = open(path, "a", encoding="utf-8", buffering=1)   # 줄 단위 flush
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, _fh)
    return _fh


def _raw():
    """tee 를 거치지 않는 원래 스트림. 터미널에만 보낼 때 쓴다."""
    return getattr(sys.stdout, "_s", sys.stdout)


def _tty():
    return sys.stdout.isatty()


def _term_cols():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 200


def _clear():
    """떠 있는 덮어쓰기 줄을 지운다."""
    global _live, _width
    if _live:
        sys.stdout.write("\r" + " " * _width + "\r")
        sys.stdout.flush()
        _live = False
        _width = 0


def log(msg=""):
    """보통 줄. stdout 이 tee 되어 있으므로 print 하면 파일에도 남는다."""
    _clear()
    print(msg, flush=True)


def step(msg):
    """반복되는 진행 줄. 터미널에서는 한 줄을 덮어쓰고, 파일에는 쌓인다."""
    global _live, _width
    if not _tty():
        # 리다이렉트·CI: 덮어쓸 터미널이 없다. 그냥 찍는다 — stdout 이 tee 되어
        # 있으므로 파일에는 print 한 번으로 들어간다. 여기서 _fh 에 직접 쓰면
        # 같은 줄이 두 번 남는다.
        print(msg, flush=True)
        return
    if _fh:
        _fh.write(msg + "\n")
    line = msg.replace("\n", " ")
    cols = _term_cols() - 1
    if len(line) > cols:
        line = line[:cols - 1] + "…"
    sys.stdout.write("\r" + " " * _width + "\r" + line)
    sys.stdout.flush()
    _live, _width = True, len(line)


def done():
    """덮어쓰기 구간을 끝낸다. 마지막 줄을 지우고 커서를 내린다."""
    _clear()


def prompt_block(stage, text, extras=None):
    """이 단계가 **모든 요청에 공통으로** 넣는 프롬프트를 남긴다.

    왜 찍는가: 실제로 모델에게 무엇이 갔는지 보지 않으면, 결과가 이상할 때
    프롬프트 탓인지 모델 탓인지 가를 수 없다. 인물 명부·용어집·줄거리처럼
    단계마다 따로 찍히던 것들도 여기로 모은다 — 그것들도 공통 프롬프트의 일부다.
    """
    bar = "─" * 66
    parts = [f"{bar}\n[공통 프롬프트] {stage}\n{bar}", (text or "").rstrip()]
    for name, body in (extras or {}).items():
        body = (body or "").rstrip()
        if body:
            parts.append(f"── {name} ──\n{body}")
    parts.append(bar)
    block = "\n".join(parts)
    _clear()
    if _fh:
        _fh.write(block + "\n")
    # 터미널에는 파일에 남겼다는 사실만 알린다. 프롬프트가 수십 줄이라
    # 터미널을 덮으면 진행 상황이 안 보인다.
    n = block.count("\n") + 1
    if _fh:
        # 요약 한 줄은 **터미널에만**. 파일에는 이미 전문이 들어갔다.
        _raw().write(f"[공통 프롬프트] {stage} — {n}줄, 로그 파일에 기록\n")
        _raw().flush()
    else:
        print(block, flush=True)
