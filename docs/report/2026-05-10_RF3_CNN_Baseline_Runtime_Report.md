# 2026-05-10 RF 3분류 CNN Baseline 구축 및 Runtime 연동 준비 보고서

## 1. 작업 개요

본 작업에서는 2.4GHz 대역 RF spectrogram 데이터를 이용하여 `Background`, `Bluetooth`, `WiFi` 세 클래스를 분류하는 RF 3분류 CNN baseline 모델을 구축하였다. 현재 프로젝트에는 아직 실제 드론 RF 데이터가 확보되지 않았기 때문에, 본 모델은 드론 탐지 모델이 아니라 **2.4GHz 대역에서 비드론 RF 신호를 분류하기 위한 baseline CNN 모델**로 정의하였다.

이번 작업의 주요 목적은 다음과 같다.

```text
1. 수집된 RF spectrogram 데이터 정리
2. manifest 기반 데이터 관리 체계 구축
3. train / validation / test split 생성
4. RF 3분류 CNN 학습
5. learning rate 비교 실험
6. 오분류 샘플 분석
7. ambiguous sample 후보 관리
8. 단일 파일 예측 스크립트 작성
9. runtime 연동을 위한 RF3 inference 구조 설계
10. pipeline spectrogram shape를 RF3 입력 기준인 128 x 509로 정렬
```

---

## 2. 데이터셋 구성

현재 사용한 데이터는 모두 `.npy` 형식의 RF spectrogram이며, 전체 데이터 shape는 다음과 같이 확인되었다.

```text
shape: (128, 509)
dtype: float32
```

이는 IQ 원본 데이터가 아니라, 이미 STFT 또는 유사한 방식으로 변환된 spectrogram 데이터이다. 따라서 별도의 IQ → STFT 변환 과정을 학습 파이프라인에 추가하지 않고, `.npy` spectrogram을 CNN 입력으로 바로 사용하였다.

데이터셋의 전체 selected sample 수는 다음과 같다.

| Class | Sample 수 |
|---|---:|
| Background | 1500 |
| Bluetooth | 554 |
| WiFi | 518 |
| Total | 2572 |

초기 baseline 학습에서는 클래스 불균형을 줄이기 위해 각 클래스당 500개씩 샘플링하여 balanced dataset을 구성하였다.

| Class | Balanced sample 수 |
|---|---:|
| Background | 500 |
| Bluetooth | 500 |
| WiFi | 500 |
| Total | 1500 |

---

## 3. Manifest 기반 데이터 관리

데이터 파일을 직접 train, validation, test 폴더로 이동하지 않고, `manifest.csv` 기반으로 관리하였다. 이를 통해 원본 데이터 구조를 유지하면서도 학습용 데이터 목록을 재현 가능하게 관리할 수 있도록 하였다.

생성한 주요 manifest 파일은 다음과 같다.

```text
data/processed/cnn_capture/manifests/manifest_all.csv
data/processed/cnn_capture/manifests/manifest_rf3_balanced_v1.csv
```

`manifest_all.csv`에는 selected 폴더 내부의 전체 `.npy` 파일이 포함되며, `manifest_rf3_balanced_v1.csv`에는 각 클래스 500개씩 총 1500개의 balanced sample이 포함된다.

이후 balanced manifest를 기준으로 random split을 적용하였다.

| Split | Sample 수 |
|---|---:|
| Train | 1050 |
| Validation | 225 |
| Test | 225 |

각 split은 클래스별로 균등하게 분배되었다.

```text
Train: Background 350 / Bluetooth 350 / WiFi 350
Validation: Background 75 / Bluetooth 75 / WiFi 75
Test: Background 75 / Bluetooth 75 / WiFi 75
```

단, 이번 split은 random split 방식이므로 동일 수집 세션에서 나온 유사한 sample이 train과 test에 함께 포함될 수 있다. 따라서 본 실험 결과는 현재 수집 데이터셋 내부에서의 baseline 성능으로 해석해야 하며, 새로운 장소, 거리, 주파수, 장비 조건에 대한 일반화 성능은 추가 검증이 필요하다.

---

## 4. RF3 CNN 모델 구성

RF 3분류 모델은 PyTorch 기반의 작은 CNN 구조로 구성하였다. 입력은 `(128, 509)` 크기의 spectrogram이며, PyTorch Conv2D 입력을 위해 `(1, 128, 509)` 형태로 변환하여 사용하였다.

모델 구조는 다음과 같다.

```text
Input: 1 x 128 x 509

ConvBlock 1: Conv2D → BatchNorm → ReLU → MaxPool
ConvBlock 2: Conv2D → BatchNorm → ReLU → MaxPool
ConvBlock 3: Conv2D → BatchNorm → ReLU → MaxPool
ConvBlock 4: Conv2D → BatchNorm → ReLU → MaxPool

AdaptiveAvgPool2D
Dropout
Linear
Output: 3 classes
```

출력 클래스는 다음과 같다.

```text
0: Background
1: Bluetooth
2: WiFi
```

학습 시에는 train set 기준으로 mean/std를 계산한 뒤, 모든 spectrogram에 동일한 정규화를 적용하였다.

---

## 5. 대표 모델 실험: rf3_cnn_baseline_v1

대표 모델은 learning rate `0.001` 조건에서 학습하였다.

| 항목 | 설정 |
|---|---|
| 모델명 | rf3_cnn_baseline_v1 |
| 입력 shape | `(128, 509)` |
| 클래스 | Background / Bluetooth / WiFi |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epoch | 20 |
| Learning rate | 0.001 |
| Model selection | validation accuracy 최고 epoch의 checkpoint 저장 |

Test 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| Test accuracy | 98.67% |
| Test loss | 0.0880 |
| Test sample 수 | 225 |
| 정분류 수 | 222 |
| 오분류 수 | 3 |

클래스별 성능은 다음과 같다.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Background | 1.0000 | 0.9867 | 0.9933 | 75 |
| Bluetooth | 0.9615 | 1.0000 | 0.9804 | 75 |
| WiFi | 1.0000 | 0.9733 | 0.9865 | 75 |

Confusion matrix는 다음과 같다.

| True \\ Pred | Background | Bluetooth | WiFi |
|---|---:|---:|---:|
| Background | 74 | 1 | 0 |
| Bluetooth | 0 | 75 | 0 |
| WiFi | 0 | 2 | 73 |

오분류 유형은 다음과 같다.

```text
Background → Bluetooth: 1개
WiFi → Bluetooth: 2개
```

대표 모델은 test set 225개 중 222개를 정확히 분류하여 98.67%의 정확도를 기록하였다. 특히 Bluetooth 클래스는 recall 1.0000을 기록하여 test set 내 Bluetooth sample을 모두 올바르게 탐지하였다.

---

## 6. Learning Rate 비교 실험

학습률 변화에 따른 성능을 비교하기 위해 learning rate `0.0003` 조건의 보조 실험을 수행하였다.

| 항목 | lr=0.001 | lr=0.0003 |
|---|---:|---:|
| Test accuracy | 98.67% | 97.33% |
| Test loss | 0.0880 | 0.2213 |
| 오분류 수 | 3 | 6 |

`lr=0.0003` 실험의 confusion matrix는 다음과 같다.

| True \\ Pred | Background | Bluetooth | WiFi |
|---|---:|---:|---:|
| Background | 74 | 1 | 0 |
| Bluetooth | 1 | 74 | 0 |
| WiFi | 1 | 3 | 71 |

`lr=0.0003` 모델은 97.33%의 높은 정확도를 기록했지만, 대표 모델인 `lr=0.001` 조건보다 오분류 수가 증가하였다. 따라서 현재 RF3 baseline 모델로는 `lr=0.001` 조건의 `rf3_cnn_baseline_v1`을 대표 모델로 선정하였다.

---

## 7. 오분류 분석

대표 모델의 오분류 sample 3개를 분석하였다.

| True label | Predicted label | Confidence | File |
|---|---|---:|---|
| Background | Bluetooth | 0.9464 | `home_bg_2450_gain10_block_0049.npy` |
| WiFi | Bluetooth | 0.9706 | `0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |
| WiFi | Bluetooth | 0.8717 | `0949__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |

오분류 sample을 PNG로 변환하여 시각적으로 확인한 결과, 해당 sample들은 사람의 눈으로 보아도 Bluetooth burst와 유사한 시간-주파수 패턴을 포함하고 있었다. 따라서 본 오분류는 단순한 모델 오류라기보다는, WiFi 또는 Background sample 내에 Bluetooth와 유사한 burst 성분이 포함된 **ambiguous sample**로 해석하였다.

이를 위해 원본 데이터는 이동하거나 삭제하지 않고, review 폴더에 ambiguous 후보 목록을 별도로 보관하였다.

```text
data/processed/cnn_capture/review/ambiguous_candidates_rf3_v1.csv
data/processed/cnn_capture/review/ambiguous_png_rf3_v1/
```

이 방식은 원본 데이터의 재현성을 유지하면서도, 경계 sample을 추후 분석할 수 있도록 하기 위한 관리 방식이다.

---

## 8. 단일 파일 예측 스크립트 작성

학습된 `best_model.pt`를 이용해 spectrogram `.npy` 파일 하나를 바로 예측할 수 있는 단일 파일 예측 스크립트를 작성하였다.

스크립트 위치는 다음과 같다.

```text
scripts/ml/predict_rf3_file.py
```

이 스크립트는 다음 기능을 수행한다.

```text
1. best_model.pt 로드
2. 입력 spectrogram .npy 로드
3. train mean/std 기준 정규화
4. RF3 CNN 추론
5. 예측 클래스 출력
6. confidence 및 class별 probability 출력
7. 선택적으로 JSON / PNG 저장
8. confidence threshold 기반 Unknown 처리 지원
```

이를 통해 새로 수집한 spectrogram 파일이나 오분류 sample을 개별적으로 확인할 수 있게 되었다.

---

## 9. RF3 Inference 모듈 설계

단일 파일 예측뿐 아니라 runtime에서도 동일한 RF3 모델을 재사용할 수 있도록, `src/ml` 내부에 RF3 inference 구조를 설계하였다.

추가한 핵심 구조는 다음과 같다.

```text
src/ml/rf3_inference.py
```

주요 클래스는 다음과 같다.

```text
RF3Classifier
RF3Result
```

`RF3Classifier`는 다음 역할을 수행한다.

```text
1. best_model.pt 로드
2. checkpoint 내부의 mean/std 로드
3. expected input shape 확인
4. spectrogram ndarray 입력 처리
5. CNN 추론
6. class_name, confidence, probabilities 반환
7. threshold 이하 confidence에 대해 Unknown 처리
```

이 구조를 통해 향후 runtime에서는 다음과 같이 RF3 모델을 호출할 수 있다.

```python
from src.ml.rf3_inference import RF3Classifier

classifier = RF3Classifier("outputs/ml/rf3_cnn_baseline_v1/best_model.pt")
result = classifier.predict_array(spectrogram)

print(result.class_name, result.confidence)
```

즉, runtime에서는 모델 로드나 정규화 로직을 다시 작성할 필요 없이 `RF3Classifier`만 호출하면 된다.

---

## 10. Pipeline Spectrogram Shape 정렬

기존 `scripts/run_pipeline.py`에는 STFT 파라미터가 `nperseg=512`, `noverlap=384`, `nfft=512`로 설정되어 있어, 생성되는 CNN spectrogram shape가 `[512, 125]`였다.

하지만 현재 RF3 모델은 `(128, 509)` 입력을 기준으로 학습되었으므로, runtime pipeline에서도 동일한 shape를 생성해야 한다. 이에 따라 `run_pipeline.py`의 STFT 파라미터를 다음과 같이 수정하였다.

```text
nperseg = 128
noverlap = 96
nfft = 128
```

해당 설정에서 hop size는 다음과 같다.

```text
hop = nperseg - noverlap = 128 - 96 = 32
```

입력 block size가 16384 samples일 때 time frame 수는 다음과 같다.

```text
(16384 - 128) / 32 + 1 = 509
```

따라서 최종 spectrogram shape는 다음과 같이 된다.

```text
freq bins = 128
time frames = 509
spectrogram shape = (128, 509)
```

이는 RF3 CNN 학습 입력과 동일한 shape이므로, 향후 runtime inference와 연결하기 위한 기본 조건을 만족한다.

---

## 11. Runtime 연동 방향

현재 RF3 CNN은 드론 탐지 모델이 아니라, 2.4GHz 비드론 RF 신호 분류기이다. 따라서 runtime에서 RF3 모델은 다음과 같은 역할로 사용하는 것이 적절하다.

```text
Background → 신호 없음 또는 잡음
Bluetooth → 비드론 간섭 신호
WiFi → 비드론 간섭 신호
Unknown → confidence가 낮은 미확정 신호
```

중요한 점은 RF3 모델에 Drone-like 클래스가 없기 때문에, Unknown 또는 비분류 신호를 바로 드론으로 판단해서는 안 된다는 것이다.

향후 runtime 흐름은 다음과 같이 설계할 수 있다.

```text
PlutoReceiver
↓
IQ block 수신
↓
DC offset 제거 및 정규화
↓
STFT spectrogram 생성
shape = (128, 509)
↓
RF3Classifier.predict_array()
↓
Background / Bluetooth / WiFi / Unknown
↓
event log 또는 dashboard에 결과 표시
```

초기 runtime 연결은 모든 block에 대해 CNN을 호출하기보다는, scan 또는 precision analysis 단계에서 후보 신호가 확인된 뒤 RF3 CNN을 호출하는 방식이 더 적절하다. 이렇게 하면 CPU 부담을 줄이고, 의미 있는 신호에 대해서만 분류를 수행할 수 있다.

---

## 12. 현재까지의 산출물

오늘 작업을 통해 생성 또는 정리된 주요 산출물은 다음과 같다.

### 데이터 관리

```text
data/processed/cnn_capture/manifests/manifest_all.csv
data/processed/cnn_capture/manifests/manifest_rf3_balanced_v1.csv
data/processed/cnn_capture/splits/rf3_random_v1/train.csv
data/processed/cnn_capture/splits/rf3_random_v1/val.csv
data/processed/cnn_capture/splits/rf3_random_v1/test.csv
```

### 학습 결과

```text
outputs/ml/rf3_cnn_baseline_v1/best_model.pt
outputs/ml/rf3_cnn_baseline_v1/summary.json
outputs/ml/rf3_cnn_baseline_v1/history.json
outputs/ml/rf3_cnn_baseline_v1/classification_report.txt
outputs/ml/rf3_cnn_baseline_v1/confusion_matrix.csv
outputs/ml/rf3_cnn_baseline_v1/confusion_matrix.png
```

### 오분류 분석

```text
outputs/ml/rf3_cnn_baseline_v1/error_analysis/predictions.csv
outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified.csv
outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified_png/
```

### 문서

```text
docs/experiments/rf3_cnn_baseline_v1.md
docs/experiments/rf3_cnn_baseline_lr3e4.md
docs/planning/RF3_CNN_COMMANDS.md
```

### 코드

```text
src/ml/rf3_dataset.py
src/ml/rf3_labels.py
src/ml/rf3_metrics.py
src/ml/rf3_model.py
src/ml/rf3_inference.py

scripts/ml/build_rf3_manifest.py
scripts/ml/split_rf3_manifest.py
scripts/ml/train_rf3_cnn.py
scripts/ml/find_rf3_misclassified.py
scripts/ml/plot_rf3_history.py
scripts/ml/predict_rf3_file.py
```

---

## 13. 한계점

본 실험에는 다음과 같은 한계가 있다.

첫째, 현재 실험은 random split 기반이다. 따라서 동일한 수집 조건에서 얻어진 유사 sample이 train과 test에 함께 포함되었을 가능성이 있다. 이로 인해 test accuracy가 실제 새로운 환경에서의 일반화 성능보다 높게 나타날 수 있다.

둘째, WiFi 데이터는 현재 단일 세션에 가깝다. 즉, WiFi에 대해서는 다양한 채널, 거리, gain, 장소 조건을 반영한 세션 분리 평가가 아직 어렵다.

셋째, 현재 모델은 Drone-like 클래스를 포함하지 않는다. 따라서 RF3 모델은 드론 탐지 모델이 아니라 Background, Bluetooth, WiFi를 구분하는 비드론 RF 분류 baseline 모델이다.

넷째, Unknown 처리는 confidence threshold 기반으로만 가능하다. 실제 드론 또는 기타 신호가 입력될 경우, 모델은 세 클래스 중 하나로 강제 분류할 수 있으므로 추후 Drone-like 데이터 확보와 4분류 모델 확장이 필요하다.

---

## 14. 향후 작업

향후 작업은 다음과 같다.

```text
1. RF3 inference 모듈을 runtime 또는 scan precision 단계에 연결
2. pipeline에서 생성되는 spectrogram shape가 항상 (128, 509)인지 검증
3. dashboard 또는 event log에 RF3 class, confidence, probabilities 추가
4. WiFi 데이터를 다른 거리, 채널, gain, 장소 조건에서 추가 수집
5. Bluetooth / Background에 대해서도 세션 분리 평가 수행
6. Drone-like 데이터 확보 후 4분류 모델로 확장
7. RF3 모델을 비드론 간섭 신호 필터로 활용하는 시스템 구조 정리
```

---

## 15. 결론

오늘 작업을 통해 2.4GHz RF spectrogram 기반 `Background / Bluetooth / WiFi` 3분류 CNN baseline을 구축하였다. Balanced dataset 1500개를 이용해 학습한 대표 모델 `rf3_cnn_baseline_v1`은 test set 225개에서 98.67%의 정확도를 기록하였다. 오분류는 총 3개였으며, 모두 Bluetooth 방향으로 발생하였다. 해당 오분류 sample은 사람이 보아도 Bluetooth burst와 유사한 ambiguous sample로 판단되어 별도 review 목록으로 관리하였다.

또한 단일 파일 예측 스크립트와 RF3 inference 모듈을 설계하여, 향후 runtime pipeline에서 RF3 모델을 재사용할 수 있는 기반을 마련하였다. 마지막으로 기존 pipeline의 spectrogram shape를 RF3 입력 기준인 `(128, 509)`로 정렬하여, 학습 모델과 runtime 입력 간의 shape mismatch 문제를 해결하였다.

따라서 현재 단계에서는 RF3 CNN이 비드론 RF 신호 분류 baseline으로 정상 동작함을 확인했으며, 다음 단계는 이를 실제 Pluto 기반 수신 pipeline에 연결하고, 추후 드론 데이터 확보 후 4분류 모델로 확장하는 것이다.
