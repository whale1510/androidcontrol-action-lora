# AndroidControl 행동 LoRA

`Qwen3-VL-4B-Instruct`에 LoRA를 붙여 AndroidControl로 학습하는 코드다. 모델은 목표·지난 스텝 기록·현재 스크린샷을 받아 "생각 한 줄 + 행동 JSON 하나"를 낸다. 최종 평가는 AndroidWorld에서 할 예정이고, 이 저장소는 데이터 준비·학습·AndroidControl 오프라인 채점까지 다룬다.

**상태 (2026-11-18)**: 데이터 생성, 학습, 예측, 채점 코드가 있고 CPU에서 작은 입력으로 끝까지 도는 것을 확인했다. **GPU로는 아직 한 번도 돌리지 않았다.** 메모리 사용량과 학습 시간은 추정이다.

처음부터 따라 하는 순서는 [GUIDE.md](GUIDE.md)에 있다.

## 빠른 시작

```bash
git clone <이 저장소> && cd <폴더>
python -m venv .venv && source .venv/bin/activate          # Python 3.11
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

bash scripts/get_androidcontrol.sh       # 다운로드 + MD5 확인 + 추출 (약 100GB)
python build_sft.py                      # 학습 데이터 만들기 (수 초 ~ 수 분)
python check_data.py                     # 원본과 대조, "failures: none"이어야 한다
sha256sum -c reference/sha256.txt        # 이 저장소를 만든 서버의 결과와 같은지

python train.py --config configs/lora_4b.yaml --set max_steps=30 output_dir=runs/try   # 짧은 시험
python train.py --config configs/lora_4b.yaml                                           # 본 학습 1에폭
python predict.py --data data/val.jsonl --out runs/lora_4b/pred_val_high.jsonl --adapter runs/lora_4b/final
python score.py   --data data/val.jsonl --pred runs/lora_4b/pred_val_high.jsonl
```

## 입력과 출력

```
[system]  역할 + 행동 9종의 JSON 형식 + 좌표 설명 + 기록 형식 + 출력 형식   (전문: prompts.py)

[user]
Goal: In the HealthifyMe app, view your today activity
History:
1. Open the HealthifyMe app | {"action_type":"open_app","app_name":"HealthifyMe"}
2. Open the HealthifyMe app | {"action_type":"wait"}
3. Click on the I DON'T WANT TO AVAIL THIS OFFER at the bottom of the screen | {"action_type":"click","x":499,"y":944}
4. Scroll down to view more options | {"action_type":"scroll","direction":"down"}
<image>

[assistant]
Thought: Click on the running icon in the Other Trackers section
Action: {"action_type":"click","x":258,"y":541}
```

- **행동 공간**: AndroidControl의 8개(`click`, `long_press`, `scroll`, `input_text`, `open_app`, `navigate_back`, `navigate_home`, `wait`)를 키 이름까지 그대로 쓰고, 끝 행동 `{"action_type":"status","goal_status":"successful"}`을 에피소드마다 하나 붙였다.
- **좌표**: `click`과 `long_press`만 좌표가 있다. 원본의 픽셀 좌표를 0~999 상대값으로 바꿔 쓴다. 그림을 어떤 크기로 줄여도 좌표는 그대로다.
- **Thought**: AndroidControl의 스텝 지시문(사람이 쓴 한 문장)이다. 추론할 때는 모델이 쓴 생각이 다음 스텝의 `History`로 들어간다.
- **입력**: 글과 현재 스크린샷 한 장. 접근성 트리는 넣지 않는다 (데이터를 만들고 채점할 때만 쓴다).

## 파일

| 파일 | 하는 일 |
|---|---|
| `scripts/get_androidcontrol.sh` | AndroidControl 다운로드, MD5 확인, 추출 |
| `scripts/extract_androidcontrol.py` | TFRecord 샤드 하나를 PNG·접근성 트리·jsonl로 푼다 (표준 라이브러리만) |
| `build_sft.py` | 추출한 데이터 → `data/{train,val,test}.jsonl` |
| `check_data.py` | 만든 데이터를 원본과 전수 대조 |
| `prompts.py` | 시스템 프롬프트, 입력 만들기, 출력 읽기, 좌표 변환, 그림 크기 (학습·추론이 같이 씀) |
| `modeling.py` | 프로세서, chat template 적용, 그림 읽기 |
| `train.py` | LoRA SFT. LLaMA-Factory의 Qwen3-VL SFT 경로를 따라 직접 짰다 |
| `predict.py` | 기본 모델 또는 기본 모델 + 어댑터로 답 생성 (high / low 모드) |
| `score.py` | AndroidControl 논문 방식 채점 |
| `configs/lora_4b.yaml` | 학습 설정 |
| `reference/` | 이 저장소를 만든 서버에서 나온 결과: 체크섬, 집계, scroll 수정 목록 |
| `third_party/android_env/` | 접근성 트리 proto 정의 (android_env 1.2.3, Apache-2.0) |

## 학습 데이터

| 파일 | 예제 | 내용 |
|---|---|---|
| `data/train.jsonl` | 88,062 | 공식 train 13,603 에피소드: 스텝 74,459 + 끝 예제 13,603 |
| `data/val.jsonl` | 827 | 공식 validation 137 에피소드 (전부) |
| `data/test.jsonl` | 9,987 | 공식 test 1,543 에피소드 (전부, 하위 분할 표시) |

형식은 LLaMA-Factory의 sharegpt(`messages` + `images`)이고, `data/dataset_info.json`이 같이 만들어져 LLaMA-Factory에서도 그대로 읽힌다. 예제마다 `meta`에 원본 행동, 화면 크기, 정답 요소 영역이 들어 있다.

train에서 정답으로 쓰지 않은 스텝은 255개다 (뒤 스텝의 `History`에는 남는다).

| 이유 | 수 |
|---|---|
| 스텝 지시문이 비어 있음 | 136 |
| scroll 방향을 정할 수 없음 | 74 |
| click이 어떤 요소에도 닿지 않음 | 34 |
| click 좌표가 화면 밖 | 11 |

### scroll 방향 라벨 수정

AndroidControl의 scroll 방향 라벨 일부가 틀려 있어서 고쳤다. scroll마다 기록된 라벨, 앞뒤 접근성 트리에서 잰 실제 움직임, 스텝 지시문의 문구("swipe up" 등)를 비교한다. 움직임과 지시문이 함께 라벨을 반박할 때만 고치고, 지시문이 라벨을 확인해 주면 그대로 두고, 정할 수 없으면 정답에서 뺀다. 규칙 전문은 `build_sft.py` 맨 위에 있다.

- train에서 59개를 고쳤다: `left`→`down` 39, `right`→`up` 19, `right`→`down` 1. val 2개, test 28개.
- 건별 근거는 `data/scroll_fixes.jsonl`에 남는다 (`reference/scroll_fixes.jsonl`과 같아야 한다).
- 채점은 고친 라벨 기준이다. 원래 라벨은 `meta.source_action`에 있다.

## 학습 설정

| 항목 | 값 |
|---|---|
| 백본 | `Qwen/Qwen3-VL-4B-Instruct` |
| LoRA | r=64, alpha=128, dropout 0.05, 언어 모델의 선형층 전부 (252개 층). 이미지 인코더와 연결층은 얼림 |
| 배치 | 8 × 누적 2 = 업데이트당 16건 |
| 학습률 | 1e-4, 워밍업 100스텝 후 코사인 감소 |
| 에폭 | 1 (5,504 업데이트) |
| 그림 크기 | `image_max_pixels: 1302528` → 1080×2400 화면이 736×1696 (그림 토큰 1,219개) |
| 저장 | 10%마다 체크포인트 + val 손실 |

손실은 정답(생각 + 행동)과 턴 끝 토큰에만 건다.

## 채점

`score.py`는 행동만 채점한다 (생각은 채점하지 않는다).

| 행동 | 정답 조건 |
|---|---|
| click, long_press | 예측 좌표가 정답 요소 영역 안 (정답 점을 포함하는 가장 작은 접근성 요소) |
| scroll | 방향이 같음 |
| input_text, open_app | 글자가 같음 (대소문자·공백 무시) |
| status | `goal_status`가 같음 |
| navigate_back, navigate_home, wait | 종류가 같음 |

`strict`는 위 조건 그대로이고, `relaxed`는 AndroidControl 논문이 인정한 두 가지를 더 받는다: `open_app` ↔ 그 앱 이름이 붙은 요소 누르기, `navigate_back` ↔ 화면의 뒤로 버튼 누르기. 결과는 AndroidControl의 8개 행동만 센 값과 `status`까지 넣은 값을 따로 낸다.

## 알아 둘 것

- GPU에서 돌린 적이 없다. 본 학습 전에 30스텝 시험으로 메모리와 속도를 먼저 잰다 ([GUIDE.md](GUIDE.md) 5절).
- 학습하지 않은 기본 모델은 이 형식에서 JSON이 깨진 답을 내기도 한다 (작은 그림 시험에서 관찰). 학습 전 기준선은 그 때문에 낮게 나올 수 있다.
- AndroidControl에는 답을 말하는 행동, Enter 키, 두 번 누르기가 없다. 이 행동 공간으로는 그런 AndroidWorld 과제를 풀 수 없다.

## 출처

- AndroidControl: Li et al., "On the Effects of Data Scale on UI Control Agents" (https://arxiv.org/abs/2406.03679), 데이터 https://github.com/google-research/google-research/tree/master/android_control
- 학습 코드의 방식: LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory)
- 좌표와 출력 형식: Qwen3-VL mobile agent 쿡북 (https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/mobile_agent.ipynb)
- 접근성 트리 proto: android_env (https://github.com/google-deepmind/android_env), Apache-2.0. `third_party/android_env/`에 정의와 라이선스가 있다.
