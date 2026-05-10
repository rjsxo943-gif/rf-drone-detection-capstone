## RF 3분류 CNN 대표 모델: rf3_cnn_baseline_v1

### 1. 실험 목적

본 실험은 2.4GHz 대역 RF spectrogram 데이터를 이용하여 `Background`, `Bluetooth`, `WiFi` 세 클래스를 분류하는 baseline CNN 모델을 구축하기 위한 실험이다.  
드론 데이터가 아직 확보되지 않은 상태이므로, 본 모델은 드론 탐지 모델이 아니라 2.4GHz 대역의 비드론 RF 신호를 구분하기 위한 3분류 baseline 모델로 정의하였다.

---

### 2. 실험 설정

| 항목 | 설정 |
|---|---|
| 모델 | RF3 Small CNN |
| 실험명 | `rf3_cnn_baseline_v1` |
| 입력 데이터 | RF spectrogram `.npy` |
| 입력 크기 | `(128, 509)` |
| 데이터 타입 | `float32` |
| 클래스 | Background / Bluetooth / WiFi |
| 데이터 구성 | 각 클래스 500개, 총 1500개 |
| Split 방식 | Random split |
| Train / Val / Test | 1050 / 225 / 225 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epoch | 20 |
| Learning rate | `0.001` |
| 저장 방식 | validation accuracy가 가장 높은 best checkpoint 저장 |

---

### 3. Test 결과

| 항목 | 결과 |
|---|---:|
| Test accuracy | 98.67% |
| Test loss | 0.0880 |
| Test sample 수 | 225개 |
| 정분류 개수 | 222개 |
| 오분류 개수 | 3개 |

---

### 4. 클래스별 성능

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Background | 1.0000 | 0.9867 | 0.9933 | 75 |
| Bluetooth | 0.9615 | 1.0000 | 0.9804 | 75 |
| WiFi | 1.0000 | 0.9733 | 0.9865 | 75 |

---

### 5. Confusion Matrix

| True \ Pred | Background | Bluetooth | WiFi |
|---|---:|---:|---:|
| Background | 74 | 1 | 0 |
| Bluetooth | 0 | 75 | 0 |
| WiFi | 0 | 2 | 73 |

---

### 6. 오분류 분석

총 3개의 오분류가 발생하였다.

| True label | Predicted label | 개수 |
|---|---:|---:|
| Background | Bluetooth | 1 |
| WiFi | Bluetooth | 2 |

세부 오분류 파일은 다음과 같다.

| True | Pred | Confidence | File |
|---|---|---:|---|
| Background | Bluetooth | 0.9464 | `home_bg_2450_gain10_block_0049.npy` |
| WiFi | Bluetooth | 0.9706 | `0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |
| WiFi | Bluetooth | 0.8717 | `0949__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |

---

### 7. 해석

`rf3_cnn_baseline_v1` 모델은 test set 225개 중 222개를 올바르게 분류하여 98.67%의 높은 정확도를 보였다. 특히 Bluetooth 클래스는 75개 test sample을 모두 올바르게 분류하여 recall 1.0000을 기록하였다.

오분류는 총 3건으로, 모두 Bluetooth 방향으로 발생하였다. Background 샘플 1개와 WiFi 샘플 2개가 Bluetooth로 분류되었다. 오분류된 spectrogram을 시각적으로 확인한 결과, 해당 샘플들은 사람이 보아도 Bluetooth burst와 유사한 시간-주파수 패턴을 포함하고 있어 클래스 경계에 위치한 ambiguous sample로 해석할 수 있다.

특히 WiFi 샘플 중 일부는 넓고 지속적인 대역 특성보다 순간적인 burst 성분이 두드러져 Bluetooth 신호와 유사하게 인식된 것으로 판단된다. Background 오분류 샘플 역시 순간적인 잡음 상승 또는 주변 2.4GHz 신호 성분으로 인해 Bluetooth로 오인된 것으로 해석된다.

---

### 8. 보조 실험과 비교

| 항목 | 대표 모델 `lr=0.001` | 보조 실험 `lr=0.0003` |
|---|---:|---:|
| Test accuracy | 98.67% | 97.33% |
| Test loss | 0.0880 | 0.2213 |
| 오분류 개수 | 3개 | 6개 |
| 대표 오분류 경향 | Background/WiFi → Bluetooth | Background/WiFi/Bluetooth 간 일부 혼동 |

두 실험 모두 97% 이상의 높은 test accuracy를 보였지만, `lr=0.001` 모델이 더 높은 정확도와 더 적은 오분류를 기록하였다. 따라서 본 RF 3분류 baseline 실험에서는 `rf3_cnn_baseline_v1`을 대표 모델로 선정하였다.

---

### 9. 한계

본 실험은 동일 수집 데이터셋 내부에서 random split으로 구성한 train/validation/test set을 사용하였다. 따라서 현재 결과는 수집된 데이터셋 내부에서의 분류 성능을 의미하며, 새로운 장소, 거리, 중심주파수, 장비 조건에 대한 일반화 성능을 완전히 보장하지는 않는다.

특히 WiFi 데이터는 현재 `home_wifihot_ch6_on_2437_gain10_1m` 조건의 단일 세션에 기반하고 있으므로, 추후 다양한 WiFi 채널, 거리, gain, 장소 조건에서 추가 데이터를 확보한 뒤 세션 분리 기반 평가가 필요하다.

---

### 10. 결론

`rf3_cnn_baseline_v1`은 Background, Bluetooth, WiFi 세 클래스를 분류하는 2.4GHz RF spectrogram baseline CNN 모델이다. Test set 225개에 대해 98.67%의 정확도를 기록하였으며, 오분류는 3건에 그쳤다. 오분류 샘플은 모두 Bluetooth로 분류되었고, 시각적으로도 Bluetooth burst와 유사한 애매한 패턴을 보였다.

따라서 본 모델은 현재 수집한 2.4GHz RF 데이터셋 내부에서 비드론 RF 신호를 효과적으로 구분할 수 있는 baseline 모델로 판단된다. 향후 드론 RF 데이터가 확보되면 본 구조를 확장하여 `Drone-like` 클래스를 추가한 4분류 모델로 발전시킬 수 있다.