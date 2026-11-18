# 가이드: 새 서버에서 처음부터 돌리기

저장소를 받은 뒤 환경 설치 → AndroidControl 다운로드·추출 → 학습 데이터 만들기 → 학습 → 평가 순서다. 명령은 모두 저장소 폴더에서 실행한다.

각 단계에 적힌 "확인된 것"은 이 저장소를 만든 서버에서 실제로 돌려 본 결과다. GPU가 필요한 5~7절은 **돌려 본 적이 없고**, 시간과 메모리는 추정이다.

## 0. 필요한 것

| 항목 | 필요량 |
|---|---|
| OS | Linux, `wget`과 `md5sum` |
| Python | 3.11 |
| 디스크 | 데이터 약 100GB (원본 47 + 스크린샷 48 + 접근성 트리 4.9), 모델 약 9GB, 체크포인트 10~20GB |
| GPU | NVIDIA 1장. 필요한 메모리는 재지 않았다 (4절 끝 참고) |
| 인터넷 | 데이터(Google Cloud Storage)와 모델(Hugging Face) 다운로드 |

## 1. 환경 설치

```bash
git clone <이 저장소> action_lora && cd action_lora
python3.11 -m venv .venv && source .venv/bin/activate

# torch는 서버의 CUDA에 맞는 빌드를 먼저 깐다. 아래는 CUDA 13.0 빌드의 예.
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

python -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
```

- conda를 쓴다면 `conda create -n action_lora python=3.11` 뒤에 같은 `pip install` 두 줄을 실행한다.
- 데이터를 만드는 단계(2~3절)는 `protobuf`만 있으면 된다. GPU 서버가 아닌 곳에서 데이터만 만들 수도 있다.
- 다른 버전을 쓸 때 주의: `transformers`의 Qwen3-VL 지원과 chat template이 필요하다. 버전을 바꿨다면 4절의 점검을 꼭 돌린다.

## 2. AndroidControl 받기

```bash
bash scripts/get_androidcontrol.sh
```

하는 일: 공개 버킷(`gs://gresearch/android_control/`)에서 22개 파일을 받고, MD5를 확인하고, 20개 샤드를 PNG·접근성 트리·jsonl로 푼다.

끝나면 이렇게 나와야 한다.

```
[..] done: 15283 episodes, 99131 screenshots, 99131 trees
expected for the full dataset: 15283 episodes, 99131 screenshots, 99131 trees
```

결과 폴더 (`data/androidcontrol/`):

```
raw/              TFRecord 샤드 20개 + splits.json + test_subsplits.json   47GB
images/           episode_<id>_step_<i>.png (원본 해상도)                   48GB, 99,131개
a11y/             episode_<id>_step_<i>.pb (접근성 트리, 원본 proto 그대로)  4.9GB, 99,131개
episodes.jsonl    에피소드마다 한 줄: 목표, 스텝 지시문, 행동, 파일 경로
meta/, logs/      샤드별 중간 파일과 로그
```

옵션 (환경 변수):

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `ANDROIDCONTROL_DIR` | 데이터를 둘 폴더. 다른 디스크에 두려면 이것을 정한다 | `data/androidcontrol` |
| `SHARDS` | 받을 샤드. `SHARDS="00000"`이면 하나만 받아 빨리 시험해 볼 수 있다 (2.5GB) | 20개 전부 |
| `DOWNLOADS` | 동시에 받는 파일 수 | 4 |
| `WORKERS` | 동시에 푸는 프로세스 수 | 8 |
| `PYTHON` | 추출에 쓸 파이썬 | `python3` |

- `ANDROIDCONTROL_DIR`을 정했다면 **이후 모든 명령에서도 같은 값으로 정해 둔다** (`export ANDROIDCONTROL_DIR=...`).
- 중간에 끊기면 같은 명령을 다시 실행한다. 다 받은 파일은 건너뛰고, 받다 만 파일은 이어받는다.
- `raw/`는 추출이 끝나면 학습에 쓰이지 않는다. 디스크가 모자라면 지워도 되지만, 다시 풀려면 다시 받아야 한다.

확인된 것: 받는 데 약 10분(초당 약 80MB), 푸는 데 약 30초 (64코어, 8~10개 동시). 샤드 하나로 스크립트를 다시 돌려 기존 추출 결과와 파일이 바이트 단위로 같음을 확인했다. MD5가 다르면 멈추는 것도 확인했다.

## 3. 학습 데이터 만들기

```bash
python build_sft.py
python check_data.py
sha256sum -c reference/sha256.txt
```

`build_sft.py`가 끝나면 분할마다 한 줄씩 나온다. 전체 데이터라면 숫자가 아래와 같아야 한다.

```
train  written 88062   excluded_as_target {no_instruction 136, scroll_unclear 74, no_element 34, oob 11}   scroll_fixed {left->down 39, right->up 19, right->down 1}
val    written 827     excluded_as_target {scroll_unclear 1}                                              scroll_fixed {left->down 2}
test   written 9987    excluded_as_target {no_element 6, no_instruction 16, scroll_unclear 12, oob 1}     scroll_fixed {left->down 27, right->up 1}
```

`check_data.py`는 만든 예제를 원본과 전수 대조한다. 마지막 줄이 `failures: none`이어야 한다.

```
train: rows 88062, scroll fixes applied 59, click/long_press 46517, max pixel error after 0-999 round trip 2, clicks leaving their element after rounding 60
val: rows 827, ...
test: rows 9987, ...
failures: none
```

`sha256sum -c reference/sha256.txt`는 만든 파일이 이 저장소를 만든 서버의 것과 글자 하나까지 같은지 본다. 다섯 줄 모두 `OK`여야 한다. `OK`가 아니면 2절의 episodes 수와 위 숫자부터 비교한다.

확인된 것: `protobuf`만 설치한 새 가상환경(Python 3.11.18, protobuf 7.36.2)에서 다시 만들어 체크섬이 모두 같았다. 만드는 데 4초 걸렸다 (24프로세스, `--workers`로 조절).

만들어지는 파일 (`data/`):

| 파일 | 내용 |
|---|---|
| `train.jsonl`, `val.jsonl`, `test.jsonl` | 학습·평가 예제 |
| `scroll_fixes.jsonl` | 방향을 고치거나 뺀 scroll과 근거 |
| `report.json` | 분할별·행동별·규칙별 수 |
| `dataset_info.json` | LLaMA-Factory용 등록 정보 |

## 4. 코드 점검 (GPU 쓰기 전에)

모델을 처음 쓰면 Hugging Face에서 `Qwen/Qwen3-VL-4B-Instruct`(약 9GB)를 받는다. 이미 받아 둔 폴더가 있으면 `configs/lora_4b.yaml`의 `model:`을 그 경로로 바꾼다.

채점기 점검 (정답을 예측으로 넣기, 몇 초):

```bash
python score.py --data data/val.jsonl --pred gold
# androidcontrol steps   n=  689  parsed 1.000  type 1.000  strict 1.000 ...
```

GPU에서 학습과 예측이 이어지는지 2~3분 안에 보는 점검:

```bash
python train.py --config configs/lora_4b.yaml --set max_steps=4 max_train_samples=64 max_eval_samples=16 output_dir=runs/smoke
python predict.py --data data/val.jsonl --out runs/smoke/pred.jsonl --adapter runs/smoke/final --limit 16
python score.py --data data/val.jsonl --pred runs/smoke/pred.jsonl
```

- `train.py`의 첫 줄들에 `LoRA layers: 252`와 `trainable params: 132,120,576`이 나와야 한다.
- 4스텝만 학습한 것이라 점수는 의미가 없다. 오류 없이 끝나는지만 본다.
- GPU가 없는 곳에서 코드만 보려면 `--set device=cpu image_max_pixels=200000 per_device_train_batch_size=1 dataloader_num_workers=0 max_steps=2 max_train_samples=4 max_eval_samples=2`를 붙이고, `predict.py`에는 `--device cpu --image_max_pixels 200000 --limit 4`를 붙인다.

확인된 것: 위 CPU 방식으로 학습 2스텝, 예측(두 모드), 채점이 끝까지 돌았다. **GPU에서는 돌려 보지 않았다.**

## 5. 학습 전 기준선과 30스텝 시험

기준선 (학습하지 않은 기본 모델, val 827건):

```bash
python predict.py --data data/val.jsonl --out runs/base/pred_val_high.jsonl
python score.py   --data data/val.jsonl --pred runs/base/pred_val_high.jsonl --out runs/base/score_val_high.json
```

그림 크기를 정하려면 같은 명령에 `--image_max_pixels`와 `--out`만 바꿔 세 번 돌려 비교한다.

| 이름 | `image_max_pixels` | 1080×2400 화면 | 그림 토큰 |
|---|---|---|---|
| S | 700000 | 544×1216 | 646 |
| M (설정 파일의 값) | 1302528 | 736×1696 | 1,219 |
| L | 2611200 | 1088×2400 (원본) | 2,550 |

정확도가 구별되지 않으면 더 작은 쪽을 쓴다. 정한 값은 `configs/lora_4b.yaml`의 `image_max_pixels`에 적는다. val은 690스텝뿐이라 95% 구간이 약 ±3.5%p다.

30스텝 시험:

```bash
python train.py --config configs/lora_4b.yaml --set max_steps=30 output_dir=runs/try
```

볼 것:

- 손실이 내려가는가, `nan`이 없는가.
- GPU 메모리 최대값 (`nvidia-smi`).
- 스텝당 시간. 진행 표시줄의 `s/it` × 5,504가 본 학습 시간이다.

## 6. 본 학습

```bash
tmux new -s action_lora
python train.py --config configs/lora_4b.yaml 2>&1 | tee runs/lora_4b.log
```

- 결과는 `runs/lora_4b/`에 쌓인다: `checkpoint-<스텝>/` 10개(10%마다), `final/`(마지막 어댑터), `train_config.json`, `train_log.json`.
- 체크포인트마다 val 손실(`eval_loss`)이 찍힌다.
- 중단됐으면 마지막 체크포인트에서 잇는다: `python train.py --config configs/lora_4b.yaml --set resume=runs/lora_4b/checkpoint-<스텝>`
- 설정은 `--set 키=값`으로 바꾼다. 쓸 수 있는 키는 `train.py`의 `DEFAULTS`에 있다.

더 큰 GPU에서 빨리 돌리려면:

- 메모리가 남으면 배치를 키우고 누적을 줄인다. 업데이트당 16건을 유지하면 다른 설정은 그대로 둘 수 있다: `--set per_device_train_batch_size=16 gradient_accumulation_steps=1`
- `dataloader_num_workers`(기본 8)를 CPU 코어에 맞춰 올린다. 그림 읽기와 줄이기가 여기서 돈다.
- 업데이트당 건수를 16보다 키우면 업데이트 수가 줄어든다. 그 조합은 시험해 보지 않았다.
- GPU 여러 장으로 돌리는 것은 시험해 보지 않았다.

시간 추정 (RTX 5090 한 장에서 같은 백본을 학습한 다른 작업의 속도, 초당 약 1,800토큰으로 계산):

| 그림 크기 | 예제당 토큰 | 1에폭 |
|---|---|---|
| S | 약 1,140 | 약 16시간 |
| M | 약 1,710 | 약 23시간 |
| L | 약 3,040 | 약 41시간 |

## 7. 평가

```bash
A=runs/lora_4b/final      # 또는 runs/lora_4b/checkpoint-<스텝>

# high: 목표만 주고 생각과 행동을 받는다 (주 지표)
python predict.py --data data/val.jsonl --out runs/lora_4b/pred_val_high.jsonl --adapter $A
python score.py   --data data/val.jsonl --pred runs/lora_4b/pred_val_high.jsonl --out runs/lora_4b/score_val_high.json

# low: 정답 생각을 미리 써 주고 행동만 받는다
python predict.py --data data/val.jsonl --out runs/lora_4b/pred_val_low.jsonl --adapter $A --mode low
python score.py   --data data/val.jsonl --pred runs/lora_4b/pred_val_low.jsonl --out runs/lora_4b/score_val_low.json
```

- `--adapter`를 주면 모델과 그림 크기를 그 학습의 `train_config.json`에서 읽는다. 학습 때와 다른 크기를 넣지 않도록 `--image_max_pixels`는 따로 주지 않는다.
- 같은 `--out`으로 다시 실행하면 끝난 데서 이어 간다. 처음부터 다시 하려면 파일을 지운다.
- 체크포인트를 고를 때는 val만 쓴다. test(`data/test.jsonl`, 9,987건)는 설정을 다 정한 뒤 한 번만 돌린다. test 결과에는 하위 분할 4개(IDD, app_unseen, task_unseen, category_unseen)가 따로 나온다.

출력 읽는 법:

```
rows 827 | scored 826 | missing prediction 0 | unscorable {'scroll_unclear': 1}
androidcontrol steps   n=  689  parsed ...  type ...  strict ...  95% [.., ..]  relaxed ...
all steps (+status)    n=  826  ...
  click                n=  420  ...
```

- `androidcontrol steps`: AndroidControl의 8개 행동만 센 값. 다른 연구와 견줄 때 쓴다.
- `all steps (+status)`: 끝 행동까지 넣은 값.
- `parsed`: 출력이 규칙에 맞는 행동으로 읽힌 비율. `type`: 행동 종류가 맞은 비율. `strict` / `relaxed`: README의 채점 절 참고.

## 8. 문제가 생기면

| 증상 | 할 일 |
|---|---|
| 다운로드가 끊김, `missing or incomplete` | `get_androidcontrol.sh`를 다시 실행한다 |
| `md5 mismatch` | 그 파일을 `raw/`에서 지우고 다시 실행한다 |
| `build_sft.py`가 `episodes.jsonl`을 못 찾음 | `ANDROIDCONTROL_DIR`이 2절과 같은 값인지 본다 |
| 체크섬이 `OK`가 아님 | 샤드를 일부만 받았는지(`SHARDS`), `build_sft.py`의 숫자가 3절과 같은지 본다 |
| GPU 메모리 부족 | `per_device_train_batch_size`를 줄이고 `gradient_accumulation_steps`를 그만큼 늘린다 (4 × 4). 그래도 안 되면 그림 크기를 줄인다 |
| `example of N tokens exceeds cutoff_len` | 나오면 안 된다 (가장 긴 예제가 M 크기에서 약 2,450토큰). 그림 크기를 크게 바꿨다면 `cutoff_len`을 올린다 |
| 모델을 못 받음 | 미리 받아 둔 폴더 경로를 `model:`에 적는다. `predict.py`는 `--model` |

## 9. 무엇이 확인됐고 무엇이 안 됐나

| 항목 | 상태 |
|---|---|
| 다운로드·MD5·추출 스크립트 | 전체 데이터로 한 번, 샤드 하나로 다시 한 번 돌렸다 |
| 데이터 생성과 대조 | 새 가상환경에서 다시 만들어 체크섬이 같았다. 대조 실패 0건 |
| 채점기 | 정답을 넣으면 val 100%, test 99.9% (0~999 반올림으로 얇은 요소를 벗어나는 click 6개). 틀린 예측 9종이 의도대로 채점됐다 |
| 학습·예측 코드 | CPU에서 작은 그림으로 2스텝 / 4건이 끝까지 돌았다 |
| GPU 학습 | **안 돌려 봤다.** 메모리, 속도, 손실 곡선 모두 모른다 |
| 학습된 모델의 성능 | **모른다** |
| vLLM에 어댑터 올리기 | 안 해 봤다. 어댑터에는 LoRA 행렬만 들어 있다 (r=64이므로 `--max-lora-rank 64` 필요) |
