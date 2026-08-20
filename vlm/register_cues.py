"""원문에서 화자별 경어 사용 증거를 계산한다.

왜 모델에게 맡기지 않는가:
  Gemma 가 클로에를 반말 화자로 판정한 적이 있다. 근거 문구가 "동료 메이드이고
  나이·지위가 비슷하다" 였다 — 즉 **사회적 추론을 하고 원문 증거를 무시했다**.
  정작 원문에는 `参りました`, `ございます`, `お導きいただけます` 가 있었다.
  일본어 경어 표지는 유한하게 열거되므로, 추론에 맡기지 않고 세어서 주면 된다.

두 곳에서 쓴다:
  1) 시트를 만들 때 프롬프트에 증거로 주입해 오판 여지를 없앤다
  2) 완성된 시트가 이 증거와 모순되는지 검사한다 (시트 자체의 오류는 출력
     일관성 검사로는 절대 잡히지 않는다 — 일관되게 틀리기 때문이다)

한 줄 단위로 판정한다. `最初は誰だって慣れないものですから` 처럼 반말 표지
(`だって`)와 경어 표지(`です`)가 한 줄에 같이 오는 경우가 있고, 이때는 경어가
이긴다 — 문말이 말투를 정하기 때문이다.
"""

import re

# 일본어 표지는 **문말에서만** 본다. 아무 위치나 찾으면 오탐이 난다 —
# `愚かな` 가 반말 표지 `かな` 로 잡혀 판정이 뒤집힌 적이 있다.
#
# 앵커를 고치면서 목록도 채웠다. 처음 목록은 현대 표준 반말/정중체만 담아서,
# 노인·권위체가 많은 챕터에서 표지를 하나도 못 찾고 "증거 없음"을 냈다. 정작
# 그 챕터는 `じゃ`·`ぞ`·`わ`·`の` 가 가득했다.
JA_POLITE = [
    "です", "ます", "ました", "ません", "ませんか", "でした", "でしょう", "ましょう",
    "ございます", "ございません", "いらっしゃる", "いらっしゃいます", "おります",
    "いただきます", "いただけます", "ください", "くださいませ",
    "申します", "存じます", "参りました", "であります",
]
# 상대를 높이는 호칭은 문말이 아니어도 화자의 경의를 뜻한다. 다만 `様` 는 넣지
# 않는다 — `王様`(임금)처럼 평범한 명사에도 붙어서, 남을 화제로 삼은 반말 문장이
# 정중체로 잡힌다. 실제로 그 오탐이 났다.
JA_POLITE_ANYWHERE = ["貴方", "どうぞ", "恐れ入り"]
JA_PLAIN = [
    "だ", "だな", "だぞ", "だよ", "だぜ", "だろ", "だろう", "だっけ", "なんだ",
    # じゃ 계열은 だ 의 고풍·방언형이다. 노인·권위체에서 흔하다.
    "じゃ", "じゃな", "のじゃ", "じゃぞ", "じゃろ",
    "ぞ", "ぜ", "もん", "もの",
    # 여성체·설명체
    "わ", "のよ", "のね", "かしら", "の",
    "じゃん", "っしょ", "ってさ", "ってね", "てよ", "だって", "かな", "かい",
]
# `なさい` 는 형태는 정중하지만 윗사람이 아랫사람에게 쓰는 명령이라 경의의 증거가
# 아니다. 정중 쪽에 넣으면 권위체 화자가 정중체로 잡힌다. 어느 쪽에도 넣지 않는다.

# 중국어: 경어 표지가 훨씬 적다. 2인칭 존칭과 청유가 사실상 전부다.
ZH_POLITE = ["您", "請", "请", "敬", "恭", "拜託", "麻煩您"]
ZH_PLAIN = ["你", "咱", "啦", "唄", "吧"]

# 영어: 호칭 정도만 신호가 된다.
EN_POLITE = ["sir", "madam", "ma'am", "my lord", "my lady", "please", "would you"]
EN_PLAIN = ["gonna", "wanna", "yeah", "hey", "dude", "ain't"]

TABLE = {"ja": (JA_POLITE, JA_PLAIN), "zh": (ZH_POLITE, ZH_PLAIN),
         "en": (EN_POLITE, EN_PLAIN)}

# 한쪽이 이 개수 이상이고 상대의 RATIO 배 이상일 때만 단정한다. 근거가 빈약할 때
# 단정하면 시트를 잘못된 방향으로 못박게 된다.
MIN_HITS = 2
RATIO = 3


TRAIL_JA = re.compile(r"[\s。、．，！？!?…‥「」『』（）()~〜ー・\u3000]+$")


def _ends_with(body, markers):
    """문말 표지 매칭. 가장 긴 표지를 우선해 `だ` 가 `だろう` 를 가리지 않게 한다."""
    hits = [m for m in markers if body.endswith(m)]
    return sorted(hits, key=len, reverse=True)[:1]


def line_verdict(text, lang):
    """한 줄이 경어인지 반말인지. 표지가 없으면 None.

    문말을 본다. 한 줄에 정중·반말 표지가 같이 오면 정중이 이긴다 — 말투를
    정하는 것은 문말이고, 반말 표지는 인용이나 관용구일 수 있다.
    """
    lang = lang or "ja"
    polite_markers, plain_markers = TABLE.get(lang, (JA_POLITE, JA_PLAIN))
    if lang == "en":
        low = text.lower()
        hit = [m for m in polite_markers if m in low]
        if hit:
            return "polite", hit
        hit = [m for m in plain_markers if m in low]
        return ("plain", hit) if hit else (None, [])

    if lang == "zh":
        # 중국어는 문말 어미로 경어를 표시하지 않는다. 존칭·청유가 문중에 온다.
        hit = [m for m in polite_markers if m in text]
        if hit:
            return "polite", hit
        hit = [m for m in plain_markers if m in text]
        return ("plain", hit) if hit else (None, [])

    # 일본어: 줄이 여러 문장으로 쪼개져 있을 수 있어 마지막 조각의 문말을 본다.
    parts = [p for p in re.split(r"[\n。！？!?]+", text) if p.strip()]
    body = TRAIL_JA.sub("", (parts[-1] if parts else text).strip())

    hit = _ends_with(body, polite_markers)
    if hit:
        return "polite", hit
    anywhere = [m for m in JA_POLITE_ANYWHERE if m in text]
    if anywhere:
        return "polite", anywhere
    hit = _ends_with(body, plain_markers)
    if hit:
        return "plain", hit
    return None, []


def evidence_by_speaker(doc):
    """화자별 증거를 모은다. {화자: {verdict, polite, plain, markers}}"""
    acc = {}
    for pg in doc["pages"]:
        for t in pg["texts"]:
            src = (t.get("ocr") or "").strip()
            spk = t.get("speaker_name")
            # 나레이션과 효과음은 화자 말투의 증거가 아니다.
            if not src or not spk or t.get("kind") not in ("dialogue", "thought"):
                continue
            lang = t.get("lang") or pg.get("source_lang")
            verdict, markers = line_verdict(src, lang)
            if not verdict:
                continue
            e = acc.setdefault(spk, {"polite": 0, "plain": 0, "markers": set()})
            e[verdict] += 1
            e["markers"].update(markers)

    for spk, e in acc.items():
        p, q = e["polite"], e["plain"]
        if p >= MIN_HITS and p >= RATIO * max(q, 1) * (1 if q else 1) and p > q:
            e["verdict"] = "honorific"
        elif q >= MIN_HITS and q >= RATIO * max(p, 1) * (1 if p else 1) and q > p:
            e["verdict"] = "plain"
        else:
            e["verdict"] = None
        e["markers"] = sorted(e["markers"])
    return acc


def render_evidence(acc):
    """시트 생성 프롬프트에 넣을 문장. 단정할 수 있는 화자만 싣는다."""
    lines = []
    for spk, e in sorted(acc.items()):
        if not e["verdict"]:
            continue
        what = ("uses HONORIFIC / polite speech" if e["verdict"] == "honorific"
                else "uses PLAIN / casual speech")
        lines.append(f"- {spk} {what} in the source "
                     f"({e['polite']} polite vs {e['plain']} plain lines; "
                     f"markers: {', '.join(e['markers'][:8])})")
    if not lines:
        return "(no decisive register markers found in the source)"
    return "\n".join(lines)


# 시트의 speechStyle / koreanEnding 을 경어-반말 두 층위로 접는다. 합니다체와
# 해요체의 차이는 여기서 따지지 않는다 — 둘 다 경어이고, 그 안의 변주는 정상이다.
HONORIFIC_STYLES = {"formal", "polite", "elderly"}
PLAIN_STYLES = {"casual", "rough", "childish"}


def contradictions(sg, acc):
    """시트가 원문 증거와 어긋나는 지점을 찾는다."""
    out = []
    for c in sg.get("characters", []):
        keys = [c.get("displayName"), c.get("targetName"),
                *(c.get("sourceNames") or []), *(c.get("visualDescriptors") or [])]
        ev = None
        for k in keys:
            if k and k in acc and acc[k].get("verdict"):
                ev = acc[k]
                break
        if not ev:
            continue
        style = c.get("speechStyle")
        declared = ("honorific" if style in HONORIFIC_STYLES
                    else "plain" if style in PLAIN_STYLES else None)
        if declared and declared != ev["verdict"]:
            out.append((c["displayName"], f"기본 말투 {style}({declared})인데 "
                                          f"원문 증거는 {ev['verdict']} "
                                          f"({ev['polite']}:{ev['plain']}, "
                                          f"{', '.join(ev['markers'][:5])})"))
        for r in c.get("registerTo") or []:
            end = r.get("koreanEnding") or ""
            # 해라체는 문어적 권위체지만 경어가 아니다 — 반말 층위로 접는다.
            # 이 목록이 좁아서 해라체가 적힌 시트는 검사에서 통째로 건너뛰어졌다.
            r_declared = ("plain" if any(k in end for k in
                                         ("반말", "해체", "해라", "한다체"))
                          else "honorific" if any(k in end for k in
                                                  ("합니다", "해요", "하십시오", "습니다"))
                          else None)
            if r_declared and r_declared != ev["verdict"]:
                out.append((c["displayName"],
                            f"{r.get('toward')} 에게 {end}({r_declared}) 인데 "
                            f"원문 증거는 {ev['verdict']}"))
    return out
