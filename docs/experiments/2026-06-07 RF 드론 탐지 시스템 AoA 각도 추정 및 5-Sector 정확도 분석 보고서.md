# 2026-06-07 RF 드론 탐지 시스템 AoA 각도 추정 및 5-Sector 정확도 분석 보고서

## 1. 실험 목적

본 실험의 목적은 Pluto+ SDR의 RX0/RX1 2채널 IQ 데이터를 이용하여 드론 조종기 또는 드론 관련 RF activity를 탐지한 뒤, AoA 기반 방향 추정 성능을 정량적으로 확인하는 것이다.

기존 시스템은 CNN 기반 Drone / NotDrone 분류 이후 AoA angle을 계산하는 구조였으나, 실시간 시연 및 dashboard 표시를 위해 angle 값을 고정된 sector로 양자화하는 방식이 필요하였다. 이에 따라 본 실험에서는 수집된 sector profile CSV를 기반으로 실제 각도(`true_angle_deg`)와 추정 각도(`angle_median`)를 비교하고, 7-sector 및 5-sector 기준의 방향 추정 정확도를 분석하였다.

본 분석의 핵심 목표는 다음과 같다.

```text
1. AoA 추정각과 실제 각도의 오차 분석
2. 거리별 AoA 추정 안정성 확인
3. 7-sector와 5-sector 방식의 정확도 비교
4. 최종 OpenCV sector dashboard에 적합한 sector 구조 결정
```

---

## 2. 실험 환경

본 실험은 RF 드론 탐지 캡스톤 프로젝트의 실시간 AoA sector profile 저장 기능을 이용하여 수행하였다.

| 항목          | 조건                              |
| ----------- | ------------------------------- |
| SDR 장비      | Pluto+ SDR                      |
| 사용 채널       | RX0 / RX1 2채널                   |
| 중심 주파수      | 2.45 GHz                        |
| Sample Rate | 5 MSPS                          |
| RF Gain     | 35                              |
| 처리 단위       | 1 block = 16,384 samples        |
| CNN 모델      | Drone / NotDrone binary CNN     |
| AoA 방식      | RX0/RX1 phase difference 기반 AoA |
| Sector 방식   | 7-sector 및 5-sector 비교          |
| 저장 조건       | trusted row 자동 저장               |

분석에 사용한 CSV 파일은 다음 4개이다.

```text
20260606_190242_gain35_cf2450000000_sector_profile.csv
20260606_191956_gain35_cf2450000000_sector_profile.csv
20260607_173001_gain35_cf2450000000_sector_profile.csv
20260607_175750_gain35_cf2450000000_sector_profile.csv
```

CSV에는 다음과 같은 주요 필드가 포함되어 있다.

```text
distance_m
true_angle_deg
angle_median
angle_spread
median_coherence
median_raw_p99
locked_sector_name
sector_status
valid_aoa_count
cnn_drone_count
```

분석 시 `distance_m = 0`인 row는 실제 거리 실험 라벨이 아닌 것으로 판단하여 제외하였다. 따라서 최종 분석에 사용한 row 수는 총 2752개이다.

---

## 3. AoA 및 Sector 판단 구조

본 프로젝트의 실시간 처리 흐름은 다음과 같다.

```text
Gain-wise Noise Calibration
→ Gain-wise Phase/Gain Calibration
→ RawNoiseGate
→ Representative / Top-K Block Selection
→ STFT-CNN Drone / NotDrone Classification
→ Temporal Voting
→ Confirmed Drone
→ AoA / Coherence Calculation
→ Fixed-bin Sector Estimation
→ Sector Profile CSV 저장
```

기존에는 단일 block을 바로 CNN 및 AoA 판단에 사용하는 구조였으나, 드론 RF 신호는 burst 특성을 가지므로 background block이 temporal voting history에 섞이는 문제가 있었다. 이를 개선하기 위해 현재 구조에서는 한 번의 update에서 여러 block을 수집한 뒤, raw gate를 통과한 block 중 score가 높은 top-k block을 선택하여 CNN 및 AoA 판단에 사용한다.

현재 `configs/ui.yaml`의 주요 설정은 다음과 같다.

```yaml
live_rf_viewer:
  blocks_per_update: 20
  select_policy: raw_gate_pass_score_max
  cli_log_every_n: 1
  disable_cli_log: false
```

즉, 한 번의 viewer update에서 20개의 block을 읽고, 이 중 대표성이 높은 block을 선택하여 CNN voting 및 AoA 계산에 활용한다.

---

## 4. Sector 정의

### 4.1 기존 7-Sector 구조

기존 AoA sector는 다음 7개 구간으로 정의하였다.

| Sector 이름   | 각도 범위       |
| ----------- | ----------- |
| LEFT_60_45  | -60° ~ -45° |
| LEFT_45_30  | -45° ~ -30° |
| LEFT_30_15  | -30° ~ -15° |
| CENTER      | -15° ~ +15° |
| RIGHT_15_30 | +15° ~ +30° |
| RIGHT_30_45 | +30° ~ +45° |
| RIGHT_45_60 | +45° ~ +60° |

7-sector 방식은 세밀한 방향 표시가 가능하다는 장점이 있으나, sector 경계 부근에서 인접 sector로 흔들리는 문제가 발생할 수 있다.

### 4.2 5-Sector 구조

실시간 dashboard 표시 및 발표용 시각화를 위해 5-sector 구조도 함께 분석하였다.

| 5-Sector 이름 | 각도 범위       | 기존 7-Sector 대응            |
| ----------- | ----------- | ------------------------- |
| LEFT_OUTER  | -60° ~ -30° | LEFT_60_45 + LEFT_45_30   |
| LEFT_INNER  | -30° ~ -15° | LEFT_30_15                |
| CENTER      | -15° ~ +15° | CENTER                    |
| RIGHT_INNER | +15° ~ +30° | RIGHT_15_30               |
| RIGHT_OUTER | +30° ~ +60° | RIGHT_30_45 + RIGHT_45_60 |

5-sector 구조는 outer 방향을 넓게 묶기 때문에, 7-sector보다 경계 흔들림에 강하고 dashboard 시각화에 더 적합하다.

---

## 5. 분석 방법

각 CSV row에 대해 실제 각도와 추정 각도를 비교하였다.

```text
실제 각도: true_angle_deg
추정 각도: angle_median
각도 오차: angle_median - true_angle_deg
절대 오차: |angle_median - true_angle_deg|
```

분석 지표는 다음과 같다.

```text
Mean Error
Median Error
MAE, Mean Absolute Error
Median Absolute Error
RMSE
90% Error
±5° 이내 비율
±10° 이내 비율
±15° 이내 비율
7-sector 일치율
5-sector 일치율
```

Sector 일치율은 실제 각도와 추정 각도를 각각 sector로 양자화한 뒤, 두 sector가 일치하는지를 기준으로 계산하였다.

---

## 6. 전체 AoA 각도 추정 결과

전체 2752개 row에 대한 AoA 각도 추정 결과는 다음과 같다.

| 항목                  |       결과 |
| ------------------- | -------: |
| 사용 row 수            |     2752 |
| 평균 오차               |   +0.84° |
| 중앙 오차               |   +1.05° |
| MAE, 평균 절대 오차       |    3.88° |
| Median AE, 중앙 절대 오차 |    2.89° |
| RMSE                |    5.68° |
| 90% 오차 범위           | 8.02° 이내 |
| ±5° 이내 비율           |   75.51% |
| ±10° 이내 비율          |   94.15% |
| ±15° 이내 비율          |   98.33% |

분석 결과, 전체 데이터의 약 94.15%가 실제 각도 대비 ±10° 이내에 포함되었다. 또한 중앙 절대 오차는 약 2.89°로, trusted 상태에서 AoA 후보들이 비교적 안정적으로 모이는 것을 확인할 수 있었다.

평균 오차는 +0.84°로 나타났으며, 이는 전체적으로 약간 오른쪽 방향으로 bias가 존재함을 의미한다. 그러나 bias의 크기는 1° 미만으로, 5-sector 기반 방향 표시에 큰 영향을 줄 수준은 아니라고 판단된다.

---

## 7. Sector 기준 정확도

실제 각도와 추정 각도를 각각 sector로 양자화하여 sector 일치율을 계산하였다.

| Sector 기준    |    정확도 |
| ------------ | -----: |
| 7-sector 정확도 | 87.14% |
| 5-sector 정확도 | 93.28% |

7-sector 기준 정확도는 87.14%로 나타났으며, 5-sector 기준 정확도는 93.28%로 상승하였다.

이는 7-sector에서는 인접 sector 경계에서 흔들림이 발생하지만, 5-sector에서는 outer sector를 넓게 통합함으로써 이러한 흔들림이 줄어들기 때문으로 해석된다. 따라서 실시간 dashboard 및 발표용 시각화에는 5-sector 방식이 더 적합하다.

---

## 8. 거리별 AoA 추정 결과

거리별 각도 추정 성능은 다음과 같다.

|  거리 | Row 수 |   MAE | ±10° 이내 | 7-sector 정확도 | 5-sector 정확도 |
| --: | ----: | ----: | ------: | -----------: | -----------: |
|  6m |   819 | 3.81° |  96.46% |       86.69% |       94.14% |
|  9m |   586 | 3.47° |  95.22% |       84.81% |       95.90% |
| 12m |   540 | 2.90° |  97.22% |       94.81% |       98.15% |
| 15m |   807 | 4.90° |  88.97% |       84.14% |       87.24% |

거리별 결과를 보면 12m 조건에서 가장 높은 정확도가 나타났다. 12m에서는 MAE가 2.90°로 가장 낮고, ±10° 이내 비율은 97.22%, 5-sector 정확도는 98.15%였다.

6m와 9m 조건에서도 ±10° 이내 비율이 각각 96.46%, 95.22%로 양호하였다. 반면 15m 조건에서는 MAE가 4.90°로 증가하고, ±10° 이내 비율은 88.97%, 5-sector 정확도는 87.24%로 낮아졌다.

이는 거리가 증가할수록 수신 신호 세기가 감소하고, outer angle 및 sector boundary 부근에서 angle compression 또는 sector 흔들림이 증가하기 때문으로 해석된다.

---

## 9. 파일별 분석 결과

CSV 파일별 각도 추정 성능은 다음과 같다.

| 파일              | Row 수 |   MAE | ±10° 이내 | 7-sector 정확도 | 5-sector 정확도 |
| --------------- | ----: | ----: | ------: | -----------: | -----------: |
| 20260606_190242 |   266 | 2.51° | 100.00% |       93.98% |       93.98% |
| 20260606_191956 |   450 | 4.51° |  94.67% |       79.56% |       92.89% |
| 20260607_173001 |   292 | 3.60° |  94.86% |       90.75% |       92.12% |
| 20260607_175750 |  1744 | 3.97° |  93.00% |       87.44% |       93.46% |

`20260606_190242` 파일은 MAE가 2.51°로 가장 낮고, ±10° 이내 비율이 100%로 매우 안정적인 결과를 보였다.

`20260606_191956` 파일은 7-sector 정확도가 79.56%로 낮게 나타났지만, 5-sector 기준으로는 92.89%까지 상승하였다. 이는 완전히 잘못된 방향으로 추정된 것이 아니라, 주로 인접 7-sector 간 경계 흔들림이 발생했음을 의미한다.

따라서 전체적으로 5-sector 구조가 실험 데이터에 대해 더 안정적인 방향 표시 성능을 제공한다고 볼 수 있다.

---

## 10. 5-Sector Dashboard 적용 타당성

본 실험 결과를 바탕으로 실시간 OpenCV dashboard에는 5-sector 구조를 적용하는 것이 적절하다고 판단하였다.

5-sector 구조의 장점은 다음과 같다.

```text
1. 7-sector보다 sector 경계 흔들림에 강함
2. 실시간 시각화에서 방향성이 직관적임
3. outer angle에서 발생하는 angle compression을 완화할 수 있음
4. 5-sector 기준 정확도가 93.28%로 높음
5. 발표 및 시연에서 해석하기 쉬움
```

최종 dashboard 표시 sector는 다음과 같이 정의하는 것이 적절하다.

```text
LEFT_OUTER   : -60° ~ -30°
LEFT_INNER   : -30° ~ -15°
CENTER       : -15° ~ +15°
RIGHT_INNER  : +15° ~ +30°
RIGHT_OUTER  : +30° ~ +60°
```

dashboard에서는 `locked_sector_name`을 내부적으로 5-sector로 변환하여 표시하고, trusted 상태일 때 해당 sector를 점등하는 방식이 적합하다.

---

## 11. 한계 및 주의점

본 실험에는 다음과 같은 한계가 있다.

첫째, 각도 라벨은 실험자가 수동으로 조정하여 저장한 값이므로 일부 row에서는 라벨 입력 실수가 포함될 수 있다. 특히 이전 분석 과정에서 일부 `true_angle_deg = 0` row가 실제로는 다른 각도에서 저장되었을 가능성이 언급되었다. 따라서 최종 보고서에 사용할 때는 의심 구간을 별도로 검토하거나 제외하는 것이 좋다.

둘째, CSV 파일별로 phase live delta 상태가 동일하지 않다. 일부 데이터에는 `phase_offset_live_delta_deg`가 서로 다른 값으로 포함되어 있으므로, 최종 calibration 기준 성능을 평가하려면 phase offset을 고정한 상태에서 다시 검증하는 것이 바람직하다.

셋째, 본 실험은 gain 35, center frequency 2.45GHz 조건에서 수행되었다. 따라서 다른 gain, 다른 주파수, 다른 장소, 다른 안테나 배치에서도 동일한 성능이 유지된다고 단정할 수 없다.

넷째, 15m 조건에서는 상대적으로 오차가 증가하였다. 이는 수신 신호 세기 감소, 멀티패스, 편파 영향, sector boundary 흔들림 등의 영향을 받은 것으로 보인다.

---

## 12. 결론

Gain 35, 2.45GHz 조건에서 수집한 sector profile CSV 4개를 분석한 결과, AoA 추정각 `angle_median`은 실제 각도 `true_angle_deg` 대비 평균 절대 오차 3.88°를 보였다. 전체 row의 94.15%가 ±10° 이내 오차 범위에 포함되었으며, 98.33%는 ±15° 이내에 포함되었다.

Sector 기준으로는 7-sector 일치율이 87.14%, 5-sector 일치율이 93.28%로 나타났다. 따라서 실시간 dashboard 및 발표용 방향 표시는 5-sector 구조가 더 적합하다고 판단된다.

거리별로는 12m 조건에서 가장 우수한 성능을 보였고, 15m 조건에서는 상대적으로 오차가 증가하였다. 이는 원거리 조건에서 수신 세기 저하와 angle/sector 흔들림이 증가하기 때문으로 해석된다.

최종적으로 본 시스템의 AoA 기반 방향 추정은 5-sector 기준에서 실시간 시각화에 충분히 활용 가능한 수준의 성능을 보였으며, 향후 dashboard에는 5-sector sector fan 형태의 방향 표시를 적용하는 것이 적절하다.

---

## 13. 보고서용 요약 문장

본 실험에서는 Pluto+ SDR의 RX0/RX1 2채널 IQ 데이터를 이용하여 드론 RF 신호에 대한 AoA 기반 방향 추정 성능을 분석하였다. Gain 35, 2.45GHz 조건에서 수집한 2752개의 trusted row를 대상으로 실제 각도와 추정 각도를 비교한 결과, 평균 절대 오차는 3.88°였으며 전체 데이터의 94.15%가 ±10° 이내 오차 범위에 포함되었다. 또한 7-sector 기준 정확도는 87.14%, 5-sector 기준 정확도는 93.28%로 나타나, 실시간 dashboard 표시에는 5-sector 구조가 더 안정적인 것으로 판단된다.
