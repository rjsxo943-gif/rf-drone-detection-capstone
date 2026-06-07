# 2026-06-06 AoA Sector Profile CSV 분석 보고서

## 6m 조건에서의 Fixed-bin Sector AoA 및 거리 추정 가능성 검토

## 1. 분석 목적

본 분석의 목적은 2026-06-06 실험에서 수집한 AoA sector profile CSV 데이터를 기반으로 다음 항목을 확인하는 것이다.

첫째, 6m 거리 조건에서 위상차 기반 AoA 결과를 고정 sector로 양자화했을 때 실제 각도 변화에 따라 sector가 안정적으로 변화하는지 확인한다.

둘째, 각 sector별로 raw RF 절대세기 feature가 어느 정도의 분포를 가지는지 확인하고, 이를 향후 거리 구간 추정에 활용할 수 있는지 검토한다.

셋째, 향후 9m, 12m, 15m 데이터를 추가 수집할 때 어떤 방식으로 거리 profile을 구성해야 하는지 방향을 정리한다.

---

## 2. 분석 데이터

분석에 사용한 CSV는 다음 두 개이다.

```text
20260606_190242_gain35_cf2450000000_sector_profile.csv
20260606_191956_gain35_cf2450000000_sector_profile.csv
```

공통 실험 조건은 다음과 같다.

| 항목               |                                                                                                값 |
| ---------------- | -----------------------------------------------------------------------------------------------: |
| Center frequency |                                                                                         2.45 GHz |
| Gain             |                                                                                               35 |
| Distance         |                                                                                              6 m |
| Sample rate      |                                                                                           5 MSPS |
| Block size       |                                                                                   16,384 samples |
| Capture target   |                                                                                  trusted row 30개 |
| Sector 방식        |                                                                                 fixed-bin sector |
| 주요 출력            | angle_median, locked_sector_name, median_raw_p99, raw_abs_p99, frame_power_p99, median_coherence |

전체 데이터 수는 다음과 같다.

| CSV                   | Row 수 | 비고                             |
| --------------------- | ----: | ------------------------------ |
| 20260606_190242...csv |   266 | phase live delta가 일부 적용된 sweep |
| 20260606_191956...csv |   450 | phase live delta 0도 기준 sweep   |
| Total                 |   716 | 6m 조건 sector profile           |

두 번째 CSV의 일부 `true_angle_deg = 0` 데이터는 실험 중 실제 각도 라벨을 변경하지 않은 상태에서 저장된 것으로 확인되었다. 실험자의 설명에 따르면 해당 구간은 실제로 약 -40도 조건이었으므로, 본 분석에서는 해당 구간을 실제 -40도 데이터로 해석하였다.

---

## 3. 분석 방법

각 row는 capture viewer에서 `sector_status == trusted` 조건을 만족한 상태에서 저장된 데이터이다. 따라서 본 분석에서는 각도별로 다음 값을 요약하였다.

```text
1. locked_sector_name
2. angle_median
3. angle_spread
4. median_coherence
5. raw_abs_p99
6. median_raw_p99
7. frame_power_p99
8. raw_rms
```

또한 실제 각도와 sector bin의 관계를 비교하기 위해 다음 기준을 사용하였다.

```text
LEFT_60_45   : -60 ~ -45 deg
LEFT_45_30   : -45 ~ -30 deg
LEFT_30_15   : -30 ~ -15 deg
CENTER       : -15 ~ +15 deg
RIGHT_15_30  : +15 ~ +30 deg
RIGHT_30_45  : +30 ~ +45 deg
RIGHT_45_60  : +45 ~ +60 deg
```

---

## 4. AoA 및 Sector 양자화 결과

### 4.1 전체 경향

6m, gain=35 조건에서 AoA 결과는 전반적으로 실제 각도 변화에 따라 잘 이동하였다.

왼쪽 방향에서는 실제 각도가 커질수록 다음과 같이 sector가 이동하였다.

```text
LEFT_30_15 → LEFT_45_30 → LEFT_60_45
```

오른쪽 방향에서도 다음과 같은 방향 변화가 확인되었다.

```text
CENTER → RIGHT_15_30 → RIGHT_30_45 → RIGHT_45_60
```

즉, 단일 각도값을 그대로 사용하는 방식보다 sector 단위로 양자화했을 때 실제 방향 변화가 비교적 안정적으로 표현되었다.

---

### 4.2 주요 각도별 결과 요약

대표 각도별 결과는 다음과 같다.

| 실제 각도 | 주요 locked sector          | angle_median | median_coherence | 해석             |
| ----: | ------------------------- | -----------: | ---------------: | -------------- |
|  -60° | LEFT_60_45                |      -54.85° |            0.999 | 왼쪽 끝단 방향 검출 가능 |
|  -55° | LEFT_60_45                |      -54.63° |            0.999 | 매우 양호          |
|  -50° | LEFT_60_45                |      -49.19° |            0.999 | 매우 양호          |
|  -40° | LEFT_45_30                |       약 -40° |            0.999 | 매우 양호          |
|  -35° | LEFT_45_30                |      -32.07° |            0.999 | 양호             |
|  -30° | LEFT_30_15                |      -26.31° |            0.999 | 약간 안쪽으로 추정     |
|  -25° | LEFT_30_15                |      -24.66° |            0.999 | 매우 양호          |
|  -20° | LEFT_30_15                |      -19.89° |            0.999 | 매우 양호          |
|    0° | CENTER                    |       -2.28° |            0.999 | 중심 방향 양호       |
|  +15° | RIGHT_15_30               |      +18.16° |            0.999 | 양호             |
|  +20° | RIGHT_15_30               |      +26.79° |            0.999 | 오른쪽 bias 존재    |
|  +25° | RIGHT_15_30 / RIGHT_30_45 |      +29.72° |            0.999 | sector 경계 부근   |
|  +30° | RIGHT_30_45               |      +34.19° |            0.997 | 양호             |
|  +35° | RIGHT_30_45               |      +40.55° |            0.997 | 오른쪽 bias 존재    |
|  +40° | RIGHT_45_60               |      +49.54° |            0.996 | 과대 추정 경향       |

왼쪽 방향은 -20°부터 -55°까지 실제 각도와 angle_median이 비교적 잘 일치하였다. 특히 -25°, -40°, -50°, -55° 조건에서는 sector와 angle_median이 모두 실제 방향과 잘 맞았다.

오른쪽 방향도 각도 증가에 따라 sector가 순차적으로 이동하였으나, +20° 이상에서는 실제 각도보다 angle_median이 더 크게 나타나는 bias가 관찰되었다. 특히 +40° 조건에서는 실제 각도보다 약 9~10° 정도 오른쪽으로 크게 추정되는 경향이 있었다.

---

## 5. Sector 일치율 분석

라벨 보정 후 전체 row 기준으로 실제 각도가 속한 sector와 locked sector가 일치한 비율은 약 84.9%로 나타났다.

다만 이 수치는 sector 경계에 가까운 각도를 포함한 값이다. 예를 들어 -45°, -15°, +25°, +40°와 같이 sector 경계 또는 경계 부근에 위치한 데이터에서는 locked sector가 인접 sector로 넘어가는 경우가 발생하였다.

특히 다음 조건은 경계 또는 bias 영향으로 해석해야 한다.

| 실제 각도 | 관찰 현상                       | 해석                   |
| ----: | --------------------------- | -------------------- |
|  -45° | LEFT_60_45로 주로 판단           | -45°가 sector 경계에 가까움 |
|  -15° | CENTER와 LEFT_30_15 혼재       | CENTER 경계            |
|  +25° | RIGHT_15_30과 RIGHT_30_45 혼재 | 오른쪽 bias 및 경계 접근     |
|  +40° | RIGHT_45_60로 주로 판단          | 오른쪽 고각 과대 추정         |

따라서 향후 성능 평가에서는 sector 경계각보다 sector 중앙 근처 각도를 사용하는 것이 적절하다.

추천 대표 각도는 다음과 같다.

```text
-55°, -40°, -25°, 0°, +25°, +40°, +55°
```

시간이 부족할 경우 최소 대표 각도는 다음과 같다.

```text
-25°, 0°, +25°
```

---

## 6. Coherence 및 Angle Spread 분석

대부분의 trusted row에서 median_coherence는 약 0.997~1.000 수준으로 매우 높게 나타났다. 이는 RX0/RX1 두 채널의 위상 관계가 해당 조건에서 비교적 일관되게 유지되었음을 의미한다.

또한 angle_spread도 대부분 0.2~1.5° 수준으로 작게 나타났다. 이는 top-K 후보 block에서 계산된 AoA들이 서로 크게 흩어지지 않고, trusted sector 내부로 안정적으로 모였다는 의미이다.

따라서 이번 데이터에서 관찰된 각도 오차는 랜덤한 위상 튐보다는 다음 요인에 의한 systematic bias로 해석하는 것이 적절하다.

```text
1. 안테나 배치 비대칭
2. 원형편파 안테나의 위상 중심 차이
3. 드론 조종기 방향 및 손/몸의 영향
4. phase offset 기준 차이
5. 오른쪽 고각에서의 과대 추정 경향
```

---

## 7. 절대세기 Feature 분포 분석

거리 추정에 활용할 수 있는 주요 절대세기 feature는 다음과 같다.

```text
raw_abs_p99
median_raw_p99
frame_power_p99
raw_rms
```

6m 조건에서 sector별 raw strength는 서로 크게 달랐다. phase live delta가 0인 데이터 기준으로 sector별 대표값은 다음과 같이 요약된다.

| Locked sector |   n | raw_abs_p99 median | median_raw_p99 median | frame_power_p99 median | raw_rms median |
| ------------- | --: | -----------------: | --------------------: | ---------------------: | -------------: |
| CENTER        | 138 |            약 200.0 |               약 187.9 |               약 31,639 |         약 76.0 |
| LEFT_45_30    |  60 |            약 190.2 |               약 182.9 |                약 8,107 |         약 74.8 |
| LEFT_60_45    | 119 |            약 168.2 |               약 163.1 |                약 2,204 |         약 63.2 |
| RIGHT_15_30   |  89 |            약 166.2 |               약 159.3 |               약 16,568 |         약 62.2 |
| RIGHT_30_45   |  72 |            약 116.4 |               약 113.0 |                약 4,749 |         약 42.9 |
| RIGHT_45_60   |  31 |            약 124.3 |               약 117.5 |                약 2,296 |         약 44.5 |

분석 결과, 동일한 6m 거리에서도 sector에 따라 절대세기 feature가 크게 달라졌다. 예를 들어 CENTER sector의 raw_abs_p99 median은 약 200 수준이지만, RIGHT_30_45 sector는 약 116 수준으로 나타났다. 이는 같은 거리라도 방향, 안테나 지향성, 편파 조건에 따라 수신 세기가 크게 달라질 수 있음을 의미한다.

따라서 거리 추정을 전체 공통 모델 하나로 수행하는 것은 적절하지 않다. 대신 다음과 같은 sector-specific distance profile이 필요하다.

```text
1. 먼저 AoA 결과를 sector로 양자화
2. locked sector를 기준으로 해당 sector의 거리 profile 선택
3. 해당 sector 내부에서 raw_abs_p99, median_raw_p99, raw_rms, frame_power_p99를 비교
4. 가장 가까운 거리 bin을 추정
```

---

## 8. 거리 구간 추정 가능성 검토

이번 CSV는 모두 6m 조건에서 수집되었기 때문에, 현재 데이터만으로 9m, 12m, 15m 거리 구간을 직접 검증할 수는 없다. 그러나 6m 조건에서 trusted row의 coherence와 angle_spread가 안정적으로 나타났고, sector별 raw strength 분포가 일정한 범위를 형성했기 때문에 거리 bin 추정은 시도할 가치가 있다.

다만 거리 추정은 연속적인 m 단위 추정보다는 거리 구간 classification으로 접근하는 것이 적절하다.

권장 표현은 다음과 같다.

```text
정확한 거리 추정이 아니라,
sector별 raw RF strength profile 기반 거리 구간 추정
```

예상 거리 bin은 다음과 같이 설정할 수 있다.

```text
6m / 9m / 12m / 15m
```

또는 더 안정적인 구간 표현으로는 다음과 같이 설정할 수 있다.

```text
Near / Mid / Far
```

향후 9m, 12m, 15m 데이터를 추가 수집할 경우, 각 sector에서 raw_abs_p99와 median_raw_p99가 거리 증가에 따라 감소하는 경향을 보이는지 확인해야 한다. 만약 같은 sector 내부에서 다음과 같이 일관된 감소가 나타난다면 거리 bin 추정 가능성이 높다.

```text
6m  >  9m  >  12m  >  15m
```

반대로 거리 증가에 따른 세기 감소보다 조종기 방향, 손 위치, 멀티패스, 편파 영향이 더 크게 나타난다면 4단계 거리 분류보다는 Near/Far 정도의 coarse distance bin이 더 적절하다.

---

## 9. 향후 데이터 수집 전략

향후 9m, 12m, 15m 데이터를 수집할 때는 모든 각도를 5도 단위로 촘촘히 수집할 필요는 없다. 이번 6m 데이터 분석 결과, sector 경계각에서는 인접 sector로 넘어가는 현상이 발생하므로 sector 중앙에 가까운 대표각을 수집하는 것이 더 효율적이다.

### 9.1 최소 거리 profile 수집 세트

```text
각도: -25°, 0°, +25°
거리: 6m, 9m, 12m, 15m
각 조건: trusted row 10~30개
```

이 구성은 LEFT_30_15, CENTER, RIGHT_15_30에 대한 거리 profile을 만들 수 있는 최소 세트이다.

### 9.2 확장 거리 profile 수집 세트

```text
각도: -40°, -25°, 0°, +25°, +40°
거리: 6m, 9m, 12m, 15m
각 조건: trusted row 10~30개
```

이 구성은 중간 sector와 바깥쪽 sector까지 포함하므로, 방향에 따른 수신 세기 차이를 더 잘 반영할 수 있다.

### 9.3 수집 시 주의사항

거리 profile 구축을 위해서는 다음 조건을 가능한 한 일정하게 유지해야 한다.

```text
1. phase_offset_live_delta_deg 조건을 동일하게 유지
2. 안테나 위치, 높이, 간격 고정
3. 조종기 방향과 잡는 자세 최대한 일정하게 유지
4. true_angle_deg 라벨 확인 후 capture 시작
5. sector 경계각보다 sector 중앙 근처 각도 사용
6. capture는 trusted row만 저장
```

특히 phase offset live delta가 섞이면 같은 거리에서도 angle_median과 locked sector가 달라질 수 있으므로, 거리 profile 학습용 데이터는 동일 phase offset 조건끼리만 비교해야 한다.

---

## 10. 종합 결론

2026-06-06 6m sector profile CSV 분석 결과, fixed-bin sector 기반 AoA는 실제 방향 변화에 대해 전반적으로 안정적인 결과를 보였다. CENTER와 LEFT 방향에서는 angle_median이 실제 각도와 매우 근접하게 나타났고, RIGHT 방향에서도 각도가 증가함에 따라 sector가 순차적으로 이동하였다.

다만 RIGHT 방향에서는 실제 각도보다 angle_median이 크게 추정되는 bias가 관찰되었다. 특히 +35°와 +40° 조건에서는 오른쪽 고각 방향으로 과대 추정되는 경향이 있었다. 이는 안테나 배치 비대칭, 원형편파 안테나의 위상 중심 차이, 또는 phase offset 기준 차이에 의한 영향으로 판단된다.

절대세기 feature 분석에서는 동일한 6m 거리에서도 sector에 따라 raw_abs_p99, median_raw_p99, frame_power_p99가 크게 달라지는 것을 확인하였다. 따라서 거리 추정은 전체 공통 모델이 아니라, AoA sector로 먼저 방향을 양자화한 뒤 해당 sector 내부에서 거리별 raw strength profile을 비교하는 방식이 적절하다.

결론적으로, 현재 CSV 데이터는 다음 가능성을 보여준다.

```text
1. 6m 조건에서 sector 기반 방향 양자화는 충분히 가능하다.
2. trusted row 기준 coherence와 angle_spread는 안정적이다.
3. 절대세기 feature는 sector별로 의미 있는 분포 차이를 보인다.
4. 향후 9m, 12m, 15m 데이터를 추가하면 sector-specific distance bin 추정을 시도할 수 있다.
5. 정확한 연속 거리 추정보다는 6m / 9m / 12m / 15m 또는 Near / Mid / Far와 같은 거리 구간 추정이 적절하다.
```

따라서 다음 단계에서는 -25°, 0°, +25° 또는 -40°, -25°, 0°, +25°, +40° 대표각에 대해 9m, 12m, 15m 데이터를 추가 수집하고, 각 sector별 raw strength profile이 거리 증가에 따라 얼마나 분리되는지 확인하는 것이 필요하다.
