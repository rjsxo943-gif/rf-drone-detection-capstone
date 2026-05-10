## RF 3분류 CNN 보조 실험: Learning Rate 0.0003

### 1. 실험 목적

기존 baseline 모델에서 learning rate를 `0.001`로 설정했을 때 validation accuracy가 epoch마다 다소 크게 변동하는 현상이 나타났다.  
이에 따라 learning rate를 더 낮춘 `0.0003`으로 설정하여 학습 안정성과 분류 성능 변화를 비교하였다.

---

### 2. 실험 설정

| 항목 | 설정 |
|---|---|
| 모델 | RF3 Small CNN |
| 입력 데이터 | RF spectrogram `.npy` |
| 입력 크기 | `(128, 509)` |
| 클래스 | Background / Bluetooth / WiFi |
| 데이터 구성 | 각 클래스 500개, 총 1500개 |
| Split 방식 | Random split |
| Train / Val / Test | 1050 / 225 / 225 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epoch | 20 |
| Learning rate | `0.0003` |
| 저장 모델 | validation accuracy가 가장 높은 best checkpoint |

---

### 3. Test 결과

| 항목 | 결과 |
|---|---|
| Test accuracy | 97.33% |
| Test loss | 0.2213 |
| Test sample 수 | 225개 |
| 정분류 개수 | 219개 |
| 오분류 개수 | 6개 |

---

### 4. 클래스별 성능

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Background | 0.9737 | 0.9867 | 0.9801 | 75 |
| Bluetooth | 0.9487 | 0.9867 | 0.9673 | 75 |
| WiFi | 1.0000 | 0.9467 | 0.9726 | 75 |

---

### 5. Confusion Matrix

| True \ Pred | Background | Bluetooth | WiFi |
|---|---:|---:|---:|
| Background | 74 | 1 | 0 |
| Bluetooth | 1 | 74 | 0 |
| WiFi | 1 | 3 | 71 |

---

### 6. 오분류 분석

총 6개의 오분류가 발생하였다.

| True label | Predicted label | 개수 |
|---|---:|---:|
| Background | Bluetooth | 1 |
| Bluetooth | Background | 1 |
| WiFi | Background | 1 |
| WiFi | Bluetooth | 3 |

세부 오분류 파일은 다음과 같다.

| True | Pred | Confidence | File |
|---|---|---:|---|
| Background | Bluetooth | 0.6254 | `home_bg_2450_gain10_block_0049.npy` |
| Bluetooth | Background | 0.5363 | `home_bt_audio_on_2460_gain10_0.4m_v2_block_7121.npy` |
| WiFi | Background | 0.4287 | `0272__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |
| WiFi | Bluetooth | 0.8375 | `0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |
| WiFi | Bluetooth | 0.5087 | `0442__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |
| WiFi | Bluetooth | 0.5655 | `0949__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy` |

---

### 7. 해석

Learning rate를 `0.0003`으로 낮춘 모델은 test set에서 97.33%의 정확도를 기록하였다. 이는 충분히 높은 성능이지만, 기존 `lr=0.001` baseline 모델의 98.67%보다는 낮은 결과이다.

오분류는 총 6건으로, 특히 WiFi 샘플이 Bluetooth 또는 Background로 분류되는 경우가 나타났다. WiFi 샘플 중 일부는 시간-주파수 영역에서 burst 형태가 두드러져 Bluetooth 신호와 유사하게 보였으며, 이로 인해 Bluetooth로 오분류된 것으로 해석된다.

또한 `lr=0.0003` 모델에서만 발생한 일부 오분류는 confidence가 비교적 낮았다. 예를 들어 WiFi가 Background로 분류된 경우의 confidence는 0.4287, Bluetooth가 Background로 분류된 경우의 confidence는 0.5363으로 나타났다. 이는 모델이 해당 샘플에 대해 확신도가 낮은 상태에서 경계적으로 판단했음을 의미한다.

---

### 8. Baseline 모델과 비교

| 항목 | lr=0.001 | lr=0.0003 |
|---|---:|---:|
| Test accuracy | 98.67% | 97.33% |
| Test loss | 0.0880 | 0.2213 |
| 오분류 개수 | 3개 | 6개 |
| 대표 오분류 경향 | Background/WiFi → Bluetooth | Background/WiFi/Bluetooth 간 일부 혼동 |

두 실험 모두 97% 이상의 높은 test accuracy를 보였으나, `lr=0.001` 모델이 더 높은 정확도와 더 적은 오분류를 보였다. 따라서 현재 RF 3분류 baseline 모델로는 `lr=0.001` 설정을 대표 모델로 선정하는 것이 적절하다.

---

### 9. 결론

Learning rate `0.0003` 실험은 기존 baseline 대비 학습률을 낮췄을 때의 성능 변화를 확인하기 위한 보조 실험이다. 해당 모델은 97.33%의 높은 test accuracy를 기록하였으나, 오분류 수가 기존 baseline보다 증가하였다. 따라서 최종 대표 모델은 `lr=0.001` baseline으로 설정하고, `lr=0.0003` 실험은 learning rate 비교를 위한 보조 실험 결과로 정리한다.