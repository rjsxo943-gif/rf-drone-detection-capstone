# 2026-05-26 RF 드론 탐지 실시간 파이프라인 개선 및 Live Gain15 재학습 보고서

## 1. 작업 개요

본 작업은 RF 기반 드론 탐지 프로젝트에서 실시간 스캔 및 AoA 방향 추정 파이프라인을 실제 드론 신호 환경에 맞게 안정화하고, 기존 RF4 CNN 모델이 live 드론 신호를 WiFi로 오분류하던 문제를 개선하기 위해 수행하였다.

기존 모델은 오프라인 학습 데이터에서는 정상적으로 동작했으나, 실제 드론을 정면에 배치한 live 환경에서는 `cnn=WiFi(0.999...)`로 강하게 오분류되는 문제가 있었다. 이에 따라 실시간 파이프라인의 입력 전처리, 스캔 게이트, 정밀 분석 구조, 데이터 수집 방식, 재학습 데이터셋 구성을 전반적으로 점검하고 수정하였다.

---

## 2. 초기 문제 상황

### 2.1 RF4 모델 로딩 및 inference 경로 문제

초기에는 `configs/ml.yaml`에서 최근 학습된 RF4 모델을 지정했음에도 불구하고, 기존 범용 Torch CNN inference 경로를 타면서 모델 구조가 맞지 않는 문제가 발생하였다.

대표 에러는 다음과 같았다.

```text
Missing key(s) in state_dict: ... features.*.block.*
Unexpected key(s) in state_dict: ... features.*.net.*
```

이는 현재 사용 중인 RF4 모델 체크포인트가 `RF3SmallCNN` 기반의 `features.*.net.*` 구조인데, 기존 inference 코드가 다른 CNN 구조를 기대했기 때문에 발생한 문제였다.

### 2.2 Spectrogram shape 불일치

RF4 live 모델은 `128 x 509` spectrogram을 입력으로 학습되었지만, AoA sector scan의 `PrecisionAnalyzer`에서는 기존 기본값인 `512 x 125` spectrogram을 생성하고 있었다.

대표 에러는 다음과 같았다.

```text
Unexpected spectrogram shape: (512, 125), expected: (128, 509)
```

### 2.3 무신호 상태 false positive 문제

초기 scan threshold가 낮아 all-off 상태에서도 모든 주파수 후보가 trigger되었고, 이 잡음성 spectrogram이 CNN으로 들어가 `WiFi(0.999...)`로 출력되었다.

즉, 당시 구조는 다음과 같았다.

```text
무신호/잡음 block → trigger 통과 → CNN 입력 → WiFi false positive
```

따라서 CNN 결과를 해석하기 전에 scan gate를 먼저 정상화해야 했다.

### 2.4 실제 드론 live 신호의 WiFi 오분류

게이트와 shape 문제가 해결된 뒤에도 실제 드론을 정면에 배치했을 때 기존 모델은 드론 신호를 대부분 WiFi로 분류하였다.

저장된 live spectrogram의 통계는 기존 학습 데이터와 분포가 크게 달랐다.

```text
기존 학습 Drone-like 평균: 약 0.27
live 드론 spectrogram 평균: 약 0.67~0.68
```

따라서 단순 코드 문제가 아니라 실제 live 환경 데이터 분포가 기존 학습 데이터와 달라 모델이 일반화하지 못하는 문제로 판단하였다.

---

## 3. 주요 코드 수정 내용

### 3.1 RF4 inference backend 연결

`configs/ml.yaml`에서 inference backend를 RF4 전용 경로로 설정하였다.

```yaml
inference:
  backend: rf4
  model_path: outputs/ml/rf4_cnn_live_gain15_v1/best_model.pt
  device: cpu
  general_threshold: 0.50
  drone_threshold: 0.70
```

이를 통해 `build_cnn_classifier()`가 기존 Torch CNN 경로가 아니라 `RF4Classifier` 기반 adapter를 사용하도록 정리하였다.

### 3.2 `scan_loop.py`에서 STFT 설정 전달

`PrecisionAnalyzer`가 `configs/ml.yaml`의 STFT 설정을 사용하도록 수정하였다.

적용된 주요 설정은 다음과 같다.

```yaml
nperseg: 128
noverlap: 96
nfft: 128
expected_freq_bins: 128
expected_time_frames: 509
```

그 결과 실시간 scan에서도 RF4 모델 입력과 동일한 `128 x 509` spectrogram을 생성할 수 있게 되었다.

### 3.3 CNN Dataset Capture와 동일한 spectrogram 생성 경로 적용

기존 `PrecisionAnalyzer`는 `features/spectrogram.py`의 branch spectrogram을 CNN 입력으로 사용하고 있었다. 그러나 학습 데이터는 `cnn_capture_actions.py`의 `_compute_cnn_spectrogram_numpy()` 함수로 생성되었기 때문에 입력 분포 차이가 발생할 수 있었다.

이를 해결하기 위해 AoA용 complex STFT와 CNN용 spectrogram을 분리하였다.

```text
AoA branch:
  remove_dc_offset → normalize_iq → compute_dual_channel_stft_branch

CNN branch:
  remove_dc_offset → get_cnn_input_iq → peak normalize → _compute_cnn_spectrogram_numpy
```

이로써 실시간 CNN 입력 생성 방식이 학습 데이터 생성 방식과 맞춰졌다.

### 3.4 PrecisionAnalyzer multi-block selection 적용

기존 구조는 trigger 후 precision 단계에서 1 block만 읽어 CNN/AoA를 수행하였다. 드론 신호는 burst/hopping 특성이 있으므로 1 block만 보면 드론 burst 타이밍을 놓칠 수 있었다.

이를 개선하기 위해 다음 구조를 적용하였다.

```text
trigger 발생
→ 해당 center frequency로 retune
→ precision block 10개 수집
→ 각 block의 Drone-like 확률 계산
→ Drone-like 확률이 가장 높은 block 선택
→ 선택된 block으로 CNN/AoA/sector 결과 출력
```

출력에는 다음 디버그 항목이 추가되었다.

```text
sel_block=<selected>/<total>
sel_score=<Drone-like probability>
```

### 3.5 Center frequency retune 및 settle 적용

`PrecisionAnalyzer.analyze(center_freq)` 시작 시 해당 center frequency로 receiver를 다시 tuning하도록 수정하였다. 또한 LO 변경 직후 안정화를 위해 `settle_sec`를 적용하였다.

현재 설정은 다음과 같다.

```yaml
settle_sec: 0.02
```

### 3.6 Background 수집용 gate bypass 추가

`[s]` CNN Dataset Capture 모드는 기본적으로 scan candidate를 통과한 신호만 저장하는 구조였기 때문에 all-off Background 데이터가 저장되지 않았다.

이를 해결하기 위해 label에 `background`가 포함되면 candidate gate와 precision 저장 gate를 우회하도록 수정하였다.

```python
is_background_capture = "background" in capture_cfg.label.lower()
```

적용 후 `background_live_gain15_alloff` 라벨에서는 trigger가 없어도 background spectrogram을 저장할 수 있게 되었다.

### 3.7 저장 파일명 충돌 방지

기존 저장 파일명은 세션 내부에서 `sample_000001.npz` 형태였기 때문에, 파일을 한 폴더에 모을 경우 이름이 충돌할 수 있었다.

이를 방지하기 위해 파일명에 다음 정보를 포함하도록 수정하였다.

```text
<index>__<session_id>__<label>__cf<center_freq_mhz>__rx<rx_index>.npz
```

예시:

```text
0001__20260526_202815__background_live_gain15_alloff__cf2450__rx0.npz
```

또한 동일 파일명이 이미 존재하는 경우 `__dupXX` suffix를 붙이도록 처리하였다.

---

## 4. 최종 수집 조건

Live gain15 데이터셋은 다음 조건을 기준으로 수집하였다.

```yaml
receiver gain: 15
scan range: 2435 MHz ~ 2465 MHz
step: 5 MHz
threshold: 4000000.0
scan_blocks: 5
min_pass_blocks: 2
precision_blocks_per_candidate: 10
settle_sec: 0.02
STFT: 128 / 96 / 128
rx_index: 0
```

Bluetooth 수집의 경우 신호가 너무 드물게 잡혀 `min_pass_blocks: 1` 조건으로 별도 수집하였고, 라벨명에 `minpass1`을 명시하여 구분하였다.

---

## 5. Live gain15 데이터 수집 결과

최종 수집된 live 데이터셋 구성은 다음과 같다.

| Class | Source label | Count |
|---|---:|---:|
| Background | `background_live_gain15_alloff` | 300 |
| WiFi | `wifi_live_gain15_ch6_range2425_2450` | 300 |
| Bluetooth | `bluetooth_live_gain15_airpods_music_minpass1` | 125 |
| Bluetooth | `bluetooth_live_gain15_airpods_call_minpass1` | 125 |
| Bluetooth | `bluetooth_live_gain15_airpods_pairing_event_minpass1` | 50 |
| Drone-like | `drone_like_live_gain15_front` | 137 |
| Drone-like | `drone_like_live_gain15_front_center` | 150 |
| Drone-like | `drone_like_live_gain15_front_left30` | 75 |
| Drone-like | `drone_like_live_gain15_front_right30` | 75 |

클래스별 총합은 다음과 같다.

```text
Background : 300
WiFi       : 300
Bluetooth  : 300
Drone-like : 437
Total      : 1337
```

---

## 6. Live gain15 RF4 모델 재학습

새로 수집한 live gain15 데이터셋만으로 RF4 CNN 모델을 재학습하였다.

학습 스크립트:

```text
scripts/ml/train_rf4_live_gain15.py
```

출력 모델 경로:

```text
outputs/ml/rf4_cnn_live_gain15_v1/best_model.pt
```

데이터 split은 다음과 같이 생성되었다.

```text
Train: 935 samples
Val  : 200 samples
Test : 202 samples
```

클래스별 split은 다음과 같다.

```text
Train:
  Background  210
  WiFi        210
  Bluetooth   210
  Drone-like  305

Val:
  Background   45
  WiFi         45
  Bluetooth    45
  Drone-like   65

Test:
  Background   45
  WiFi         45
  Bluetooth    45
  Drone-like   67
```

학습 데이터 기준 mean/std는 다음과 같다.

```text
mean = 0.61798335
std  = 0.14522279
```

---

## 7. 재학습 결과

최고 validation accuracy는 다음과 같다.

```text
best_val_acc = 0.935
```

`best_model.pt` 기준 test accuracy는 다음과 같다.

```text
test_acc = 0.8960
```

Best model 기준 confusion matrix는 다음과 같다.

| True \ Pred | Background | WiFi | Bluetooth | Drone-like |
|---|---:|---:|---:|---:|
| Background | 44 | 0 | 1 | 0 |
| WiFi | 0 | 33 | 11 | 1 |
| Bluetooth | 1 | 1 | 43 | 0 |
| Drone-like | 0 | 0 | 6 | 61 |

클래스별 성능 해석은 다음과 같다.

```text
Background : 44/45 = 97.8%
WiFi       : 33/45 = 73.3%
Bluetooth  : 43/45 = 95.6%
Drone-like : 61/67 = 91.0%
```

드론 탐지 관점에서 중요한 결과는 다음과 같다.

```text
Drone-like recall = 약 91.0%
Background → Drone-like false positive = 0
Bluetooth  → Drone-like false positive = 0
WiFi       → Drone-like false positive = 1
```

기존 모델에서는 실제 live 드론 신호를 대부분 WiFi로 오분류했으나, live gain15 데이터 기반 재학습 후 Drone-like recall이 약 91%까지 개선되었다.

---

## 8. 현재 남은 문제

### 8.1 WiFi와 Bluetooth 혼동

현재 모델은 WiFi와 Bluetooth 사이에서 일부 혼동이 존재한다.

대표 오분류:

```text
WiFi → Bluetooth: 11개
Drone-like → Bluetooth: 6개
```

다만 프로젝트 목표가 드론 탐지인 점을 고려하면, Drone-like false positive가 낮고 Drone-like recall이 높은 것은 긍정적이다.

### 8.2 같은 세션 기반 split의 한계

현재 train/val/test split은 파일 단위 random split이다. 같은 수집 세션의 데이터가 train과 test에 섞일 수 있으므로, 최종 보고서용 성능은 다소 과대평가될 가능성이 있다.

향후에는 세션 단위 split 또는 별도 날짜/별도 환경의 holdout test set을 구성하는 것이 바람직하다.

### 8.3 실시간 검증 필요

재학습 모델은 오프라인 test set에서 개선된 성능을 보였으나, 실제 `[a] AoA sector scan` 실시간 출력에서 `cnn=Drone-like(...)`가 안정적으로 나오는지는 아직 확인하지 않았다.

---

## 9. 내일 수행할 작업

### 9.1 설정 확인

내일 실험 전 다음 명령어로 설정을 확인한다.

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

grep -n "backend\|model_path\|general_threshold\|drone_threshold" configs/ml.yaml
grep -n "threshold\|scan_blocks\|min_pass_blocks\|precision_blocks_per_candidate\|settle_sec" configs/scan.yaml
```

기대 설정:

```text
model_path: outputs/ml/rf4_cnn_live_gain15_v1/best_model.pt
threshold: 4000000.0
scan_blocks: 5
min_pass_blocks: 2
precision_blocks_per_candidate: 10
settle_sec: 0.02
```

### 9.2 실시간 AoA sector scan 테스트

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

CLI에서 다음을 선택한다.

```text
select> a
```

드론 정면 배치 시 기대 출력:

```text
cnn=Drone-like(...)
sel_score=0.7 근처 또는 이상
angle=-15~15 deg 근처
sector=Center Left 또는 Center Right 근처
```

### 9.3 threshold 조정 여부 판단

초기 테스트는 다음 threshold를 유지한다.

```text
general_threshold = 0.50
drone_threshold   = 0.70
```

만약 raw CNN이 `Drone-like`로 나오지만 confidence가 0.5~0.7 사이에 머문다면, `drone_threshold`를 0.60으로 낮추는 방안을 검토한다.

---

## 10. Git 변경 사항

이번 작업에서 커밋 대상으로 정리한 파일은 다음과 같다.

```text
configs/ml.yaml
configs/receiver.yaml
configs/scan.yaml
src/ml/inference.py
src/runtime/cnn_capture_actions.py
src/runtime/scan_loop.py
src/scan/precision_analyzer.py
scripts/ml/train_rf4_live_gain15.py
```

커밋 메시지 예시는 다음과 같다.

```bash
git commit -m "feat: integrate live gain15 RF4 runtime pipeline"
```

푸시 대상 브랜치는 다음과 같다.

```text
experiment/runtime-integrated-pipeline
```

푸시 명령어:

```bash
git push -u origin experiment/runtime-integrated-pipeline
```

---

## 11. 결론

오늘 작업을 통해 실시간 RF 드론 탐지 파이프라인은 다음 단계까지 개선되었다.

```text
1. RF4 모델 inference 경로 정상화
2. 실시간 spectrogram shape 및 전처리 경로 정합성 확보
3. scan gate 정상화로 all-off false positive 억제
4. PrecisionAnalyzer multi-block selection 적용
5. Background 수집 gate bypass 구현
6. 파일명 충돌 방지 저장 구조 개선
7. live gain15 데이터셋 1337장 구축
8. live gain15 RF4 모델 재학습 완료
9. Drone-like recall 약 91% 확보
```

따라서 기존의 `실제 드론 → WiFi 오분류` 문제는 데이터 기반으로 상당 부분 개선되었으며, 다음 단계는 새 모델을 적용한 실시간 드론 탐지 및 AoA sector 출력 검증이다.
