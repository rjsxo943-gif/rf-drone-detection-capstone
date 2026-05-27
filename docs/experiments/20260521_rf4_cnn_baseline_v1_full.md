
# 2026-05-21 RF 4분류 CNN Baseline 구축 및 실제 드론 RF 데이터 적용 보고서

## 1. 작업 개요

본 작업에서는 기존 Background, WiFi, Bluetooth 3분류 RF CNN baseline에 실제 드론-조종기 RF 데이터를 추가하여 Drone-like 클래스를 포함하는 RF 4분류 CNN baseline 모델을 구축하였다.

기존 RF3 모델은 2.4GHz 대역에서 비드론 RF 신호를 분류하기 위한 baseline 모델이었다. 이번 작업에서는 실제 소형 2.4GHz 드론을 이용해 드론 RF 데이터를 직접 수집하고, 이를 CNN 학습 데이터셋에 통합하여 Background, WiFi, Bluetooth, Drone-like 네 클래스를 분류하는 RF4 모델로 확장하였다.

이번 작업의 주요 목적은 다음과 같다.

- 실제 드론-조종기 RF 데이터 수집
- Drone-like 클래스 데이터셋 구성
- 기존 RF3 데이터셋을 RF4 데이터셋으로 확장
- manifest 기반 RF4 데이터 관리 체계 구축
- train / validation / test split 생성
- RF 4분류 CNN baseline 학습
- Drone-like 클래스 분리 가능성 확인
- WiFi / Bluetooth / Drone-like 간 confusion 분석
- RF4 학습 결과 문서화
- 향후 runtime 및 일반화 실험 방향 정리

---

## 2. 데이터셋 구성

현재 사용한 데이터는 모두 .npy 형식의 RF spectrogram이다. 저장된 데이터는 raw IQ가 아니라, 이미 STFT 기반으로 변환된 CNN 입력용 spectrogram이다.

데이터 shape는 다음 기준을 따른다.

- shape: 128 x 509
- dtype: float32

따라서 학습 파이프라인에서는 IQ 원본을 다시 STFT로 변환하지 않고, .npy spectrogram을 바로 CNN 입력으로 사용하였다.

RF4 데이터셋은 다음 네 개의 학습용 통합 폴더로 구성하였다.

- data/processed/cnn_capture/Background
- data/processed/cnn_capture/Wifi
- data/processed/cnn_capture/Bluetooth
- data/processed/cnn_capture/Drone-like

주의할 점은 실제 폴더명은 Wifi이지만, manifest 생성 과정에서 label은 WiFi로 정규화하였다.

---

## 3. 실제 학습에 사용된 데이터 수

이번 RF4 baseline v1 학습에 사용된 manifest 기준 sample 수는 다음과 같다.

| Class | Sample 수 |
|---|---:|
| Background | 150 |
| WiFi | 298 |
| Bluetooth | 298 |
| Drone-like | 298 |
| Total | 1044 |

초기 의도는 각 클래스 약 298개 수준의 balanced dataset을 구성하는 것이었으나, 현재 실험에서는 Background 통합 폴더에 150개만 반영된 상태로 split이 생성되었다. 따라서 이번 RF4 baseline v1은 완전한 balanced dataset이 아니라, Background 클래스가 상대적으로 적은 조건의 4분류 baseline 실험으로 해석해야 한다.

향후에는 Background 원본 세션 2개가 서로 덮어쓰기 되지 않도록 파일명을 세션명과 함께 재구성한 뒤, Background / WiFi / Bluetooth / Drone-like가 각각 약 298개 이상 반영되도록 다시 학습하는 것이 바람직하다.

---

## 4. 드론 데이터 수집 조건

Drone-like 데이터는 실제 소형 2.4GHz 드론과 조종기를 이용하여 수집하였다. 드론 비행은 안정적으로 유지하기 어려웠기 때문에, 안전성과 재현성을 위해 대부분 바닥 기반 조건에서 데이터를 수집하였다.

### SDR 수신 조건

| 항목 | 값 |
|---|---:|
| SDR | PlutoSDR / Pluto+ |
| Center frequency | 2437 MHz |
| Sample rate | 5 MSPS |
| RF bandwidth | 5 MHz |
| Gain | 20 dB |
| Distance | 50 cm |
| Block size | 16384 samples |
| STFT nperseg | 128 |
| STFT noverlap | 96 |
| STFT nfft | 128 |
| Window | Hann |

STFT 설정에서 hop size는 32이다. 입력 block size가 16384 samples일 때 time frame 수는 509이며, 최종 spectrogram shape는 128 x 509이다.

---

## 5. Drone-like 데이터 수집 상태

드론 데이터는 다음 세 가지 상태로 나누어 수집하였다.

| State | 설명 |
|---|---|
| connected_idle | 드론과 조종기를 모두 켠 상태, 스틱 조작 없음 |
| motor_ground | 드론을 바닥에 둔 상태에서 프로펠러만 회전 |
| control_move_ground | 드론을 바닥에 둔 상태에서 오른쪽 스틱을 조작 |

수집된 원본 드론 세션 폴더는 다음과 같다.

- data/processed/cnn_capture/20260521_drone_cf2437_g20p0_d50/connected_idle
- data/processed/cnn_capture/20260521_drone_cf2437_g20p0_d50/motor_ground
- data/processed/cnn_capture/20260521_drone_cf2437_g20p0_d50/control_move_ground

총 수집된 Drone-like sample 수는 300개이며, 이 중 학습 manifest에는 Drone-like 클래스 기준 298개가 사용되었다.

본 실험에서는 안정적인 hover를 필수 조건으로 사용하지 않았다. 소형 입문용 드론이 매우 민감하게 반응하여 제자리 유지가 어렵고, 실내 환경에서 안전 문제가 발생할 수 있기 때문이다. 대신 조종기-드론 링크 유지, 모터 회전, 조종 입력에 따른 RF burst를 확보하는 것을 목표로 하였다.

---

## 6. Manifest 기반 RF4 데이터 관리

데이터 파일을 직접 train, validation, test 폴더로 이동하지 않고, manifest.csv 기반으로 관리하였다.

생성한 주요 manifest 파일은 다음과 같다.

- data/processed/cnn_capture/manifests/manifest_all.csv
- data/processed/cnn_capture/manifests/manifest_rf4_balanced_v1.csv

단, 파일명은 balanced_v1이지만, 실제 이번 학습에서는 Background 클래스가 150개만 포함되어 완전한 클래스 균형은 아니다.

split 결과는 다음과 같다.

| Split | Total | Background | WiFi | Bluetooth | Drone-like |
|---|---:|---:|---:|---:|---:|
| Train | 729 | 105 | 208 | 208 | 208 |
| Validation | 154 | 22 | 44 | 44 | 44 |
| Test | 161 | 23 | 46 | 46 | 46 |

이번 split은 random split 방식이므로 동일 수집 조건에서 얻어진 유사 sample이 train과 test에 함께 포함되었을 가능성이 있다.

---

## 7. RF4 CNN 모델 구성

RF4 모델은 기존 RF3 baseline에서 사용하던 작은 PyTorch CNN 구조를 그대로 사용하되, 출력 class 수를 4개로 확장하였다.

입력은 128 x 509 크기의 spectrogram이며, PyTorch Conv2D 입력을 위해 1 x 128 x 509 형태로 변환하여 사용하였다.

모델 구조는 다음과 같다.

- Input: 1 x 128 x 509
- ConvBlock 1: Conv2D, BatchNorm, ReLU, MaxPool
- ConvBlock 2: Conv2D, BatchNorm, ReLU, MaxPool
- ConvBlock 3: Conv2D, BatchNorm, ReLU, MaxPool
- ConvBlock 4: Conv2D, BatchNorm, ReLU, MaxPool
- AdaptiveAvgPool2D
- Dropout
- Linear
- Output: 4 classes

출력 클래스 순서는 다음과 같다.

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | WiFi |
| 2 | Bluetooth |
| 3 | Drone-like |

이번 학습에서 계산된 train set statistics는 다음과 같다.

| 항목 | 값 |
|---|---:|
| mean | -12.066824 |
| std | 12.501995 |

---

## 8. 대표 모델 실험: rf4_cnn_baseline_v1

대표 모델은 learning rate 0.001 조건에서 학습하였다.

| 항목 | 설정 |
|---|---|
| 모델명 | rf4_cnn_baseline_v1 |
| 입력 shape | 128 x 509 |
| 클래스 | Background / WiFi / Bluetooth / Drone-like |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epoch | 20 |
| Learning rate | 0.001 |
| Device | CPU |
| Model selection | validation accuracy 최고 epoch의 checkpoint 저장 |

---

## 9. 학습 과정 요약

학습 정확도는 epoch이 증가함에 따라 전반적으로 상승하였고, 학습 손실은 감소하였다.

| 항목 | 시작 | 마지막 |
|---|---:|---:|
| Train loss | 0.6405 | 0.0747 |
| Train accuracy | 72.29% | 98.90% |

Validation accuracy는 일부 epoch에서 큰 변동을 보였다.

| Epoch | Validation accuracy |
|---:|---:|
| 1 | 37.66% |
| 8 | 100.00% |
| 15 | 80.52% |
| 18 | 100.00% |
| 20 | 86.36% |

이는 검증 데이터 수가 제한적이고, RF 데이터가 수집 세션 및 환경 조건에 따라 편차를 가지기 때문으로 판단된다.

---

## 10. Test 결과

최종 test 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| Test accuracy | 96.27% |
| Test loss | 0.2509 |
| Test sample 수 | 161 |

클래스별 성능은 다음과 같다.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Background | 0.9583 | 1.0000 | 0.9787 | 23 |
| WiFi | 0.9000 | 0.9783 | 0.9375 | 46 |
| Bluetooth | 1.0000 | 0.8913 | 0.9425 | 46 |
| Drone-like | 1.0000 | 1.0000 | 1.0000 | 46 |

전체 정확도는 96.27%를 기록하였다. 특히 Drone-like 클래스는 현재 test split에서 precision, recall, F1-score 모두 1.0000을 기록하였다.

---

## 11. Confusion Matrix 분석

Confusion matrix는 다음과 같다. 행은 실제 label, 열은 예측 label을 의미한다.

| True / Pred | Background | WiFi | Bluetooth | Drone-like |
|---|---:|---:|---:|---:|
| Background | 23 | 0 | 0 | 0 |
| WiFi | 1 | 45 | 0 | 0 |
| Bluetooth | 0 | 5 | 41 | 0 |
| Drone-like | 0 | 0 | 0 | 46 |

오분류 유형은 다음과 같다.

- WiFi -> Background: 1개
- Bluetooth -> WiFi: 5개
- Drone-like -> 다른 클래스: 0개

가장 많은 오분류는 Bluetooth -> WiFi 방향으로 발생하였다. 반면 Drone-like는 현재 test set에서 다른 클래스로 오분류되지 않았다. 이는 이번 실험의 가장 중요한 결과이다.

---

## 12. 산출물

### 데이터 관리

- data/processed/cnn_capture/manifests/manifest_all.csv
- data/processed/cnn_capture/manifests/manifest_rf4_balanced_v1.csv
- data/processed/cnn_capture/splits/rf4_random_v1/train.csv
- data/processed/cnn_capture/splits/rf4_random_v1/val.csv
- data/processed/cnn_capture/splits/rf4_random_v1/test.csv
- data/processed/cnn_capture/splits/rf4_random_v1/split_manifest.csv

### 학습 결과

- outputs/ml/rf4_cnn_baseline_v1/best_model.pt
- outputs/ml/rf4_cnn_baseline_v1/summary.json
- outputs/ml/rf4_cnn_baseline_v1/history.json
- outputs/ml/rf4_cnn_baseline_v1/classification_report.txt
- outputs/ml/rf4_cnn_baseline_v1/confusion_matrix.csv
- outputs/ml/rf4_cnn_baseline_v1/confusion_matrix.png

### 문서

- docs/experiments/20260521_rf4_dataset_summary.md
- docs/experiments/20260521_rf4_cnn_baseline_v1.md
- docs/experiments/20260521_rf4_cnn_baseline_v1_full.md

### 코드

- scripts/capture_drone_dataset.py
- scripts/ml/build_rf4_manifest.py
- scripts/ml/split_rf4_manifest.py
- scripts/ml/train_rf4_cnn_baseline.py
- src/ml/rf3_labels.py

---

## 13. 코드 변경 사항

이번 작업에서 기존 RF3 구조를 RF4 실험에 사용할 수 있도록 일부 코드를 확장하였다.

주요 변경 사항은 다음과 같다.

- Drone-like label 추가
- rf3_labels.py의 label mapping을 4클래스 기준으로 수정
- RF4 manifest 생성 스크립트 추가
- RF4 split 생성 스크립트 추가
- RF4 CNN 학습 스크립트 추가
- 드론 데이터 수집 스크립트 추가
- RF4 데이터셋 및 학습 결과 문서 추가

일부 파일명에는 기존 호환성을 위해 rf3라는 이름이 남아 있다. 예를 들어 src/ml/rf3_labels.py는 파일명은 RF3이지만, 현재 label mapping은 RF4 기준으로 확장되어 있다. 향후 코드 정리 단계에서 rf_labels.py 또는 rf4_labels.py로 분리하는 것이 더 명확하다.

---

## 14. 한계점

본 실험에는 다음과 같은 한계가 있다.

첫째, 이번 실험은 random split 기반이다. 따라서 동일한 수집 조건에서 얻어진 유사 sample이 train과 test에 함께 포함되었을 수 있다. 이로 인해 test accuracy가 실제 새로운 환경에서의 일반화 성능보다 높게 나타날 가능성이 있다.

둘째, Drone-like 데이터는 같은 날, 같은 장소, 같은 거리, 같은 gain, 같은 중심주파수 조건에서 수집되었다. 따라서 모델이 순수한 드론 RF 패턴뿐만 아니라 수집 환경의 특성까지 함께 학습했을 가능성이 있다.

셋째, 사용한 드론은 소형 toy drone 한 종류이다. 따라서 본 결과를 모든 드론 RF 신호에 일반화하기는 어렵다.

넷째, Drone-like 데이터는 안정적인 hover 비행 상태에서 충분히 수집되지 않았다. 안전성과 재현성을 위해 바닥 기반 모터 회전 및 조종 입력 상태를 중심으로 수집하였다.

다섯째, 현재 저장된 학습 데이터는 raw IQ가 아니라 CNN-ready spectrogram이다. 따라서 추후 STFT 파라미터를 변경하거나 다른 feature extraction 방법을 적용하려면 raw IQ를 별도로 저장하는 구조가 필요하다.

여섯째, 이번 RF4 baseline v1은 Background class의 sample 수가 상대적으로 적은 상태에서 학습되었다. 따라서 다음 실험에서는 Background 통합 폴더를 다시 구성하여 클래스 균형을 맞출 필요가 있다.

---

## 15. 향후 작업

향후 작업은 다음과 같다.

1. Background 통합 폴더를 세션명 기반 파일명으로 재구성하여 300장 반영
2. manifest / split을 다시 생성하여 298 samples/class 수준으로 재학습
3. RF 3-Class Classification Report 문구를 RF 4-Class로 수정
4. training history loss / accuracy 그래프 생성
5. confusion_matrix.png를 보고서에 포함
6. RF4 모델 단일 파일 예측 스크립트 작성 또는 RF3 예측 스크립트 확장
7. RF4 inference 모듈 작성
8. runtime pipeline에서 RF4 CNN inference 연결
9. 다른 거리 조건에서 Drone-like 데이터 추가 수집
10. 다른 gain 조건에서 Drone-like 데이터 추가 수집
11. 다른 장소 및 시간 조건에서 Background / WiFi / Bluetooth / Drone-like 재수집
12. session-based split 또는 leave-session-out 평가 수행
13. raw IQ 저장 옵션 추가
14. 최종적으로 Drone-like detection threshold와 Unknown 처리 전략 설계

특히 다음 실험에서는 같은 조건의 random split이 아니라, 수집 세션을 분리한 평가가 필요하다.

---

## 16. 결론

오늘 작업을 통해 기존 RF3 baseline을 Background / WiFi / Bluetooth / Drone-like 4분류 CNN baseline으로 확장하였다. 실제 소형 드론과 조종기를 이용해 Drone-like 데이터를 직접 수집하였고, 이를 기존 WiFi, Bluetooth, Background 데이터와 통합하여 RF4 dataset을 구성하였다.

이번 RF4 baseline v1 학습에는 총 1044개의 sample이 사용되었으며, train 729개, validation 154개, test 161개로 분할하였다. RF4 CNN baseline 모델은 test set에서 96.27%의 정확도를 기록하였다. 특히 Drone-like 클래스는 현재 test split에서 precision, recall, F1-score 모두 1.0000을 기록하여, 현재 조건에서는 드론 RF 데이터가 다른 2.4GHz 신호와 잘 구분됨을 확인하였다.

다만 본 결과는 같은 환경에서 수집된 데이터에 대한 random split 결과이므로, 실제 일반화 성능을 보장하지는 않는다. 또한 이번 실험에서는 Background class의 sample 수가 상대적으로 적어 완전한 class-balanced 조건이 아니었다. 따라서 다음 단계에서는 Background 폴더 재구성, session-based split, 거리 및 gain 조건을 바꾼 추가 수집을 통해 더 엄격한 검증을 수행해야 한다.

결론적으로, 이번 RF4 baseline 실험은 실제 드론 RF 데이터를 프로젝트의 CNN 학습 파이프라인에 성공적으로 통합했다는 점에서 중요한 진전이다. 현재 단계에서는 드론 탐지 가능성을 확인한 baseline 결과로 볼 수 있으며, 향후 runtime 연동 및 일반화 검증을 통해 실제 RF 기반 드론 탐지 시스템으로 확장할 수 있다.


## RF4 baseline v2
RF4 baseline v2는 각 클래스 298개 기준의 balanced dataset으로 학습되었으며, test set 184개에서 99.46%의 정확도를 기록하였다. Drone-like 클래스는 test set 46개 중 46개를 모두 정확히 분류하여 precision, recall, F1-score 모두 1.0000을 기록하였다. 전체 오분류는 Bluetooth sample 1개가 Background로 분류된 경우뿐이었다.