#!/usr/bin/env python3
"""원어 판정과 **전사 변환** 검출.

왜 규칙 기반인가:
  모델에게 박스마다 `lang` 을 물었더니 같은 페이지 안에서 자기와 불일치했다.
  한자만 든 일본어 박스(「大丈夫」)는 글자만으로는 중국어와 구별할 수 없어서
  동전 던지기가 된다. 그런데 **챕터 하나에는 반드시 가나가 나온다.** 판정 단위를
  박스에서 챕터로 올리면 결정론적으로 풀린다.

⚠ 폐기된 가설 — "모델이 일본어를 중국어로 옮겨 적는다":
  가나 0 · 한자만 있는 박스가 많아 **전사 변환**이라고 판단했는데, 틀렸다.
  `assets/examples/maid2` 는 **중국어 번역본(스캔레이션)** 이고 그 중국어가
  실제로 인쇄된 글자다. 근거: SFX 305박스 중 73%에만 가나가 있고(원문 일본어
  의성어를 안 지운 것), 나레이션 8박스는 가나 0%다. 대사가 중국어인 것이 정상이다.
  게다가 이 디렉터리에는 **두 작품이 섞여** 있다 — 일본어 61쪽 · 중국어 12쪽.

  그래서 변환 검출은 **기본으로 끈다**(`scan(..., conversion=True)` 로만 켠다).
  켜 두면 멀쩡한 중국어 132박스를 다시 부르게 된다.

살아남은 것 — **쓰레기 전사 검출**:
  모델이 손글씨를 숫자로 읽는 실패는 실재한다. `できたよ。`→`11241146`,
  `ありがとうございます`→`15=4×2+1いわつ116号`. 이 규칙은 llama.cpp 회차 692박스에서
  오탐 0건이었다.
"""

import unicodedata

# 일본어에는 존재할 수 없는 글자. 간체·번체를 섞어 둔 이유는 모델이 어느 쪽으로
# 바꿀지 모르기 때문이다. 일본 신자체와 겹치는 글자(国·学·会·来)는 **넣지 않는다** —
# 오탐이 나면 멀쩡한 전사를 다시 부르게 된다.
CHINESE_ONLY = set(
    "你您妳咱吗嗎呢嘛啦咧唷囉啰们們这這么麼儿兒沒别別讓让给給从從"
    "个個几幾么乜咋啥甭俩倆仨"
)


def _block(ch):
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
        return "hangul"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "han"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return None


def counts(text):
    """문자종별 개수."""
    out = {}
    for ch in text or "":
        b = _block(ch)
        if b:
            out[b] = out.get(b, 0) + 1
    return out


def kana(text):
    c = counts(text)
    return c.get("hiragana", 0) + c.get("katakana", 0)


def han(text):
    return counts(text).get("han", 0)


def iter_texts(doc):
    for pg in doc.get("pages", []):
        for t in pg.get("texts", []):
            s = (t.get("ocr") or "").strip()
            if s:
                yield pg, t, s


def chapter_language(doc):
    """챕터 전체의 원어. (언어, 근거) 를 돌려준다.

    **박스나 페이지가 아니라 챕터 단위로 한 번만 정한다.** 사용자 규칙대로 한
    작품 안에서 언어는 섞이지 않는다(영어는 어디서나 나올 수 있으므로 판정에서
    제외한다).
    """
    agg = {}
    for _, _, s in iter_texts(doc):
        for k, v in counts(s).items():
            agg[k] = agg.get(k, 0) + v
    kn = agg.get("hiragana", 0) + agg.get("katakana", 0)
    hg = agg.get("hangul", 0)
    hn = agg.get("han", 0)
    cn = sum(1 for _, _, s in iter_texts(doc) for ch in s if ch in CHINESE_ONLY)

    # **비율로 정한다.** "있으면 이긴다" 로 했더니 중국어 챕터의 한글 6자가
    # 한자 449자를 이겼다 — 전사 잡음 몇 글자가 챕터 언어를 뒤집으면 안 된다.
    total = kn + hg + hn
    if total == 0:
        lang = "en" if agg.get("latin", 0) else None
    elif hg / total >= 0.25:
        # 한국어 본문은 한글이 대부분이다. 한자 몇 자가 섞여도 비율이 유지된다.
        lang = "ko"
    elif kn and kn / total >= 0.05:
        # 중국어에는 가나가 없다. 잡음 한두 자를 배제할 만큼만 문턱을 둔다.
        lang = "ja"
    elif hn:
        lang = "zh"
    elif agg.get("latin", 0):
        lang = "en"
    else:
        lang = None
    share = {"ja": kn / total if total else 0, "ko": hg / total if total else 0,
             "zh": hn / total if total else 0}
    return lang, {"kana": kn, "hangul": hg, "han": hn, "latin": agg.get("latin", 0),
                  "chinese_only": cn, "share": round(share.get(lang or "zh", 0), 3)}


def page_language(pg, fallback=None):
    """**쪽 하나**의 원어. 근거가 없으면 fallback.

    챕터 단위로만 정하면 안 되는 경우가 실재한다 — `assets/examples/maid2` 는
    한 디렉터리에 두 작품이 섞여 있어서 일본어 61쪽 · 중국어 12쪽이었다.
    챕터를 ja 로 못박으면 그 12쪽이 전부 "일본어가 중국어로 변환됨" 으로 오인돼
    멀쩡한 전사를 다시 부르게 된다(실측 168박스).

    쪽 단위는 박스 단위와 달리 안정적이다. 박스 하나는 한자만 있을 수 있어도,
    **일본어 한 쪽에는 거의 반드시 가나가 나온다.**
    """
    txt = "".join((t.get("ocr") or "") for t in pg.get("texts", []))
    c = counts(txt)
    kn = c.get("hiragana", 0) + c.get("katakana", 0)
    hg, hn = c.get("hangul", 0), c.get("han", 0)
    total = kn + hg + hn
    if total == 0:
        return fallback
    if hg / total >= 0.25:
        return "ko"
    if kn:
        return "ja"
    if hn:
        return "zh"
    return fallback


def conversion_reason(text, lang, min_han=3):
    """이 전사가 원어에서 **다른 문자체계로 바뀌었는지**. 아니면 None.

    두 가지 증거를 쓴다:
      확정 — 그 언어에 존재할 수 없는 글자가 있다
      의심 — 일본어인데 가나가 하나도 없고 한자가 min_han 개 이상이다
             (「了解」같은 짧은 한자어는 정상이므로 길이 문턱을 둔다)
    """
    s = (text or "").strip()
    if not s or lang != "ja":
        return None
    bad = sorted({ch for ch in s if ch in CHINESE_ONLY})
    if bad:
        return f"일본어에 없는 글자 {''.join(bad)}"
    if kana(s) == 0 and han(s) >= min_han:
        return f"가나 0 · 한자 {han(s)}"
    return None


def garbage_reason(text, kind=None):
    """전사가 **글이 아니라 기호 덩어리**로 나왔는지.

    손글씨나 장식체를 모델이 숫자로 읽어 버리는 일이 있다. 실측(maid2 9쪽):
      `できたよ。`      → `11241146`
      `ありがとうございます` → `15=4×2+1いわつ116号`
    번역기는 그걸 그대로 통과시키므로 결과 페이지에 숫자가 박힌다. 대사인데
    숫자가 글자보다 많으면 정상적인 전사가 아니다 — llama.cpp 회차 692박스에서
    이 규칙의 오탐은 0건이었다.
    """
    s = (text or "").strip()
    if not s or kind not in (None, "dialogue", "thought", "narration"):
        return None
    letters = sum(1 for c in s
                  if _block(c) in ("hiragana", "katakana", "han", "hangul"))
    digits = sum(1 for c in s if c.isdigit())
    if digits >= 3 and digits > letters:
        return f"숫자 {digits} > 글자 {letters}"
    return None


def scan(doc, lang, min_han=3, conversion=False):
    """재전사가 필요한 박스. [(page_index, text_id, 이유)]

    conversion 은 기본으로 끈다 — 문서 맨 위의 폐기된 가설 참조.
    """
    out = []
    for pg, t, s in iter_texts(doc):
        # **쪽 언어로 본다.** 챕터 언어로 보면 섞인 자료에서 멀쩡한 쪽이
        # 통째로 변환 의심이 된다.
        pl = page_language(pg, lang)
        r = (conversion_reason(s, pl, min_han) if conversion else None) \
            or garbage_reason(s, t.get("kind"))
        if r:
            out.append((pg["index"], t["id"], r))
    return out


def page_languages(doc, chapter=None):
    """쪽별 언어와 분포. 섞여 있으면 부르는 쪽이 알아야 한다."""
    per = {pg["index"]: page_language(pg, chapter) for pg in doc.get("pages", [])}
    dist = {}
    for v in per.values():
        if v:
            dist[v] = dist.get(v, 0) + 1
    return per, dist
