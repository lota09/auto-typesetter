# auto-typesetter

만화를 **한국어로 자동 역식**한다 — 대사를 읽고, 누가 말했는지 정하고, 번역하고,
원문을 지운 자리에 다시 얹는다. 사람이 승인하는 단계는 없다.

**LLM 백엔드는 갈아 끼울 수 있다.** 로컬 GPU 한 장(vLLM·llama.cpp)으로도, 원격
API(OpenAI·OpenRouter)로도 돈다 — 코드에 로컬 분기가 없고 전부 OpenAI 호환 주소
하나로 본다. 다만 **LLM 서버는 이 도구가 띄우지 않는다**(로컬이라면 직접 기동).
말풍선 기하(Magi)와 선택적 일본어 OCR 은 언제나 로컬 GPU 에서 돈다.

---

## 무엇을 우선하는가

순서가 곧 설계다.

1. **맥락 유지** — 챕터 전체를 관통하는 이해
2. **화자 구분**
3. **말투(존댓말/반말) 개연성** — "그 인물이 할 법한 말투로"
4. **일관성** — 반말과 존댓말을 오가지 않는 것

**타이포그래피와 SFX 레터링은 우선순위가 낮다.** 대사가 아닌 것의 처리는 중요하지 않다.

사람이 승인하는 단계가 없으므로, 파이프라인이 오탐하면 그대로 나간다. 대신 명백한
오류는 **결정론적 규칙**으로 걸러낸다 (⑥ 검사 단계).

---

## 핵심 설계 두 가지

### Magi 는 기하학만, VLM 이 의미 전부

Magi(magiv2)가 VLM 으로 대체 불가능한 것은 **픽셀 정확한 좌표** 하나뿐이다. 지우고
다시 얹을 영역이라 몇 px 도 틀리면 안 된다. 의미(전사·화자·수신자·대사/나레이션
구분·언어)는 전부 VLM 이 맡는다. 실측 근거:

- Magi 의 OCR 은 **영어 전용**이다. 디코더 vocab 이 RoBERTa BPE 50265 이고 영어판
  만화로 학습돼, CJK 페이지에서 유창한 영어를 환각한다. 가중치 로딩 실패가 아니다
- Magi 는 입력을 **흑백으로 변환**한다 → 머리색 같은 인물 식별 단서가 사라진다
- 인물 클러스터 ID 가 배치 안에서만 일관된다 → 챕터를 관통하는 동일성은 만들지 못한다

둘을 잇는 방법: **박스 번호를 페이지에 그려 넣고**(박스 바깥에 — 안에 그리면 전사가
망가진다) VLM 이 번호로 영역을 지칭하게 한다.

### 2단 구조 — 말투는 페이지 단위로 지킬 수 없다

번역을 시작하기 **전에** 인물별 말투를 한 번 확정하고(스타일 시트), 그 시트를 전
페이지에 똑같이 적용한다. 컨텍스트는 이미지가 아니라 **텍스트로** 채운다 — 페이지
이미지는 장당 1~2K 토큰인데, 화자 라벨이 붙은 챕터 전사는 692줄이 16K 토큰이다.

`vlm/register_cues.py` 가 **원문에서 경어 표지를 세어** 시트 프롬프트에 주입한다.
모델의 사회적 추론보다 원문 증거가 이긴다 — 메이드에게 평어를 배정하던 것이,
원문의 `参りました`·`ございます` 개수를 넣자 한 번에 고쳐졌다.

---

## 필요한 것

| | |
|---|---|
| GPU | ① Magi 기하 단계용. 없으면 실용적이지 않다. 개발·측정은 NVIDIA CMP 170HX 64GB |
| LLM 백엔드 | **OpenAI 호환 `/v1` 주소 하나.** 로컬(vLLM·llama.cpp)이든 원격(OpenAI·OpenRouter)이든 상관없다 |
| 모델 | 비전이 되는 모델 1개면 전 단계를 감당한다. 나누고 싶으면 판독용·텍스트용 2개 |
| VRAM | **로컬 백엔드를 쓸 때만.** 27B 모델 기준 18~30GB + Magi 몫 |
| 폰트 | `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` (`--font` 으로 교체 가능) |

### 가상환경 두 개

`manga-ocr` 은 `transformers >= 4.45` 를 요구하고 **Magiv2 는 4.44.1 에서만 뜬다.**
같은 환경에 살 수 없어 분리한다. 시스템 파이썬이 3.13 이상이면 옛 핀이 안 깔리므로
**3.12** 를 따로 만든다.

```bash
# ① Magi + 파이프라인 본체
conda create -p magi/.venv python=3.12 -y
magi/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
magi/.venv/bin/pip install "transformers==4.44.1" timm scipy shapely pulp \
                           opencv-python einops matplotlib

# ② manga-ocr (--transcribe ocr 용, 일본어 전용)
conda create -p ocr/.venv python=3.12 -y
ocr/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
ocr/.venv/bin/pip install "transformers==4.46.3" manga-ocr
```

> `matplotlib` 은 magiv2 의 remote code 가 최상단에서 import 한다. 없으면 모델 로드가
> `ModuleNotFoundError` 로 죽는다. 문서마다 빠져 있는 항목이니 주의.

### 모델 설정

### ★ 이 파이프라인은 LLM 서버를 띄우지 않는다

로컬이든 원격이든 **OpenAI 호환 주소 하나로만** 본다. 코드에 로컬 분기가 없고,
로컬 백엔드의 기동·종료는 사용자 몫이다.

`config/config.json` 은 **역할 둘**이 전부다:

```json
{
  "read":      { "base_url": "http://127.0.0.1:8000/v1", "api_key_env": null, "model": "..." },
  "translate": { "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "model": "..." }
}
```

`read` 는 ② 페이지 판독·크롭 전사(이미지를 본다), `translate` 는 ④⑤⑥ 인물·시트·
번역·검사다. 같은 값을 두 번 적으면 한 모델이 전 단계를 맡고, 다르게 적으면 갈린다 —
서버가 달라도, 한쪽이 원격 API 여도 상관없다.

나머지는 전부 선택이고 기본값이 있다: `thinking_style`(기본 `llama_cpp`) ·
`supports_json_schema`(true) · `timeout`(1800) · `vision`(true) · `max_image_pixels`(없음).

**시작할 때 한 번 확인하지만 막지는 않는다.** 서버가 없으면 경고만 내고 ① Magi 를
돌린다 — 그동안 서버를 띄우면 이어진다. 정말 필요한 단계에서 못 쓰면 그때 실패한다.

```
── read: orcarouter_Qwen3.8-27B_AL_NVFP4-FP8 · ctx 32768
── 경고 translate: http://127.0.0.1:8000/v1 에 닿지 못했습니다 (URLError)
   LLM 이 필요한 단계에서 실패합니다. 그 전에 서버를 띄우면 이어집니다.
```

`config/models.ini` 는 llama-server 라우터를 **직접 띄울 때 쓰는** 프리셋이다.
파이프라인은 이 파일을 읽지 않는다.

```bash
# .ini 를 단일 출처로 config.json 을 생성한다
python3 tools/sync_models.py

# 어긋남과 가중치 실존을 확인만 한다 (CI·사전 점검용)
python3 tools/sync_models.py --check
```

`.ini` 에서 주의할 것: **섹션 상속이 없다.** `[common]` 을 만들어도 별개 모델로
등록될 뿐 다른 섹션에 반영되지 않는다. 섹션마다 전부 명시할 것.

---

## 실행

```bash
magi/.venv/bin/python pipeline.py \
  --pages 'assets/examples/peppercarrot/ep11_cn/*.jpg' \
  --work work/ep11cn
```

라우터는 파이프라인이 알아서 띄운다(이미 떠 있으면 그대로 쓴다). 완성 페이지는
`<work>/out/` 에 나온다. 중간에 끊겨도 다시 같은 명령을 주면 **끝난 단계는 건너뛴다.**

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--pages GLOB` | *필수* | 페이지 glob. 정렬 순서가 곧 읽는 순서 |
| `--work DIR` | *필수* | 중간 산출물 디렉터리 |
| `--out DIR` | `<work>/out` | 완성 페이지 |
| `--styleguide PATH` | `<work>/styleguide.json` | 작품 단위 시트. 있으면 물려 쓰고 없으면 만든다 |
| `--transcribe {ocr,vlm}` | `ocr` | 크롭 전사 방식. **일본어가 아닌 페이지가 하나라도 보이면 자동으로 `vlm` 으로 물러난다** |
| `--thinking [auto\|on\|off]` | `off` | 모델 추론. 값 없이 주면 `on`. `auto` 는 단계별 옛 기본값 |
| `--workers N` | `4` | 판독 두 단계에서 동시에 던질 요청 수 |
| `--model ID` | config | 판독·번역 양쪽을 이 모델 id 로 |
| `--model-read ID` | config `read` | ② 판독 두 단계 (`--model` 보다 우선) |
| `--model-text ID` | config `translate` | ④⑤⑥ 텍스트 단계 (`--model` 보다 우선) |
| `--skip-pages N …` | 없음 | 뺄 페이지(1-base). **판독 전에** 적용된다 |
| `--from-stage` / `--only` | `magi` | 단계 지정 (`magi read merge cast translate validate render`) |
| `--no-resume` | 재개함 | 끝난 단계도 다시 |
| `--fast` | 꺼짐 | `--transcribe ocr` 의 옛 이름. 경고를 내며 동작한다. 새로 쓰지 말 것 |

**`--transcribe ocr` 의 안전망:** manga-ocr 은 일본어 전용인데, 다른 언어에서는 빈
출력이 아니라 **그럴듯하게 틀린 글**을 낸다(번체 중국어에서 `眞的狼危険` 처럼 한자를
일본 자형으로 바꿔 놓는다). 뒤에서 검사로 걸러지지 않으므로 시작 전에 막는다.
판독 패스가 본 언어가 전부 일본어일 때만 쓰고, `ocr` 을 명시하면 경고 후 강행한다.

---

## 파이프라인

```
① Magi 기하학          magi/magi_worker.py        → magi.json
   박스·인물·패널·꼬리 좌표 (검출 임계값 0.3)
        │
   ┌────┴──────────────────────┬────────────────────┐
   ▼                           ▼                    ▼
② 판독 2패스                ⑦ 마스크             ⑨ 조판
  vlm/read_page.py            render/make_mask.py    render/bubble.py
   → page.json (화자·언어)     획 단위(연결요소)      말풍선 flood fill
  vlm/read_texts.py                                  render/glyph_size.py
   → crop.json (전사)                                투영 프로파일로 글자 크기 실측
  (--transcribe ocr: ocr/manga_ocr_pass.py)
   │                           │
   ▼                           ▼
③ 병합  vlm/merge_reads.py  ⑧ 인페인팅
   → merged.json               render/compose.py (고전 CV)
   ▼
④ 인물·스토리  vlm/build_cast.py  → cast.json
   흩어진 화자 라벨을 정규화 + 줄거리. 텍스트로만 한다
   ▼
⑤ 시트+번역  vlm/translate_chapter.py  → styleguide.json, translated.json
   register_cues.py 가 원문 경어 증거를 세어 프롬프트에 주입
   ▼
⑥ 검사·수정  vlm/validate_translation.py  → final.json
   결정론적 8종 위반 검출 → 걸린 박스만 재요청
   ▼
⑨ 완성 페이지 → <work>/out/
```

**병렬화 가능한 곳과 아닌 곳이 갈린다.** ②는 페이지·박스가 서로 독립이라 겹쳐 던진다
(`--workers`). ④→⑤시트→⑤번역은 **챕터 전체를 봐야 하므로 직렬**이고, 그건 우회 대상이
아니라 1순위(맥락 유지)를 지키는 방법 그 자체다.

---

## 실측

Pepper&Carrot 공식 다국어판을 정답으로 쓴다. SVG 소스에서 뽑은 것이라 모델이 개입하지
않았고 어느 모델에도 유리하지 않다.

| | |
|---|---|
| 크롭 전사 문자 정확도 | **98.2%** (중국어), **98.0%** (일본어) — 본문 기준 |
| 말투 일치율 | 82~84% (공식 한국어판 대비) |
| 조판 넘침 | 0 (maid2 77페이지 70장) |
| 챕터 소요 | ep11_cn 8페이지 약 345초 |

자세한 것은 **[docs/MODELS.md](docs/MODELS.md)** (모델 비교·과사고 측정),
**[docs/PARALLELISM.md](docs/PARALLELISM.md)** (병렬화 실측),
**[docs/PROGRESS.md](docs/PROGRESS.md)** (전체 진행 기록·미해결 목록).

```bash
# 모델 비교를 직접 재려면
magi/.venv/bin/python tools/bench_models.py --task transcribe \
  --models qwen-vl gemma --thinking off on \
  --magi-json work/ep11cn/magi.json \
  --gt assets/groundtruth/peppercarrot/ep11_cn.json --bench-dir work/bench

# 정답 대조 채점
magi/.venv/bin/python tools/score_groundtruth.py \
  --ours work/ep11cn/merged.json \
  --gt assets/groundtruth/peppercarrot/ep11_cn.json --mode transcription
```

---

## 알려진 한계

사람 눈으로 본 평가(maid2 77페이지):

| | 성격 |
|---|---|
| 일부 대사의 **인물 매칭**이 틀린다 | 의미 판단. 완벽할 수 없음을 감수 |
| 일부 대사가 **역식되지 않고 남는다** | 의미 판단. 일부는 기계적 원인 |
| **원문이 완전히 지워지지 않은** 곳이 있다 | 기하·영상처리. 고칠 수 있다 |
| 일부 **글자 크기가 불합리하게 작다** | 기하·조판. 고칠 수 있다 |

뒤 둘은 **말풍선 검출 실패 23%** 로 수렴한다 — 검출이 실패하면 확장이 안 돼 자리가
좁아지고(→작은 글자), 같은 대비 문제가 마스크에서도 나타난다(→지워지지 않은 원문).

그밖에: 작품 단위 시트 누적이 아직 검증되지 않았고, 세로쓰기 소재의 정답 데이터가
없다. 전체 목록은 [docs/PROGRESS.md](docs/PROGRESS.md) §7.

---

## 저장소에 없는 것

`.gitignore` 가 막는다.

- **`assets/`** — 제3자 저작물인 만화 원고. `assets/secrets/` 는 열람 금지
- **`work/`, `logs/`** — 실행하면 다시 만들어진다
- **`*.venv/`** — 기계 종속. 재구축 방법은 위에 있다
- **모델 가중치** (`*.gguf`, `*.safetensors` …)
- **`refs/`** — 참고용 클론. CarrotMangaTranslator 는 GPL-3.0 이라 포함시키면
  라이선스가 얽힌다. 시트 스키마만 차용했고 파이프라인은 자체 구현이다

**Magi 라이선스:** HF 모델 카드가 "personal, research, non-commercial" 사용을
허용한다(개인 사용 명시). GitHub README 의 "academic research only" 와 충돌하지만,
가중치와 함께 배포되는 것은 모델 카드다.
