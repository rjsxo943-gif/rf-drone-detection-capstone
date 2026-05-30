# RF 드론 탐지 실험 보고서: Drone 데이터 보강 및 Gain Control 설계안

- 작성일: 2026-05-29
- 프로젝트: RF 기반 드론 탐지 캡스톤
- 목적: 내일 진행할 드론 데이터 추가 수집 및 gain-distance 보정 알고리즘 설계 방향 정리

---

## 1. 현재 모델 평가 요약

현재 테스트 결과에서 가장 중요한 특징은 **오탐은 낮지만 드론 미검출이 크다**는 점이다.

| 항목 | 값 |
|---|---:|
| Test loss | 0.3306271280557247 |
| 전체 정확도 | 87.45% |
| NotDrone 정확도 | 98.73% |
| Drone 정확도 / Recall | 71.68% |
| False Drone Rate | 1.27% |
| Missed Drone Rate | 28.32% |
| TN | 156 |
| FP | 2 |
| FN | 32 |
| TP | 81 |

해석:

- `False Drone Rate`는 낮으므로 WiFi/Bluetooth/Background를 Drone으로 착각하는 문제는 현재 크지 않다.
- 반대로 `Missed Drone Rate`가 높으므로 실제 드론을 NotDrone으로 놓치는 문제가 더 크다.
- 따라서 다음 실험의 우선순위는 **드론 미검출 감소**이다.

---

## 2. 핵심 결론

이번 실험에서는 WiFi/Bluetooth를 다시 수집하는 것이 우선이 아니다.

현재 목표는 다음과 같다.

> 실제 드론을 켜둔 상태에서 Drone 데이터만 추가 수집하고, CNN 판정 threshold를 기존보다 낮춘 상태에서 약한 드론 신호까지 관찰/저장한다.

단, 데이터 라벨은 CNN 판정 결과로 붙이지 않는다.

| 구분 | 기준 |
|---|---|
| 데이터 라벨 | 실제 실험 상황 |
| Drone 데이터 | 실제 드론 ON 상태에서 캡처한 데이터 |
| CNN threshold 0.4 | 저장/관찰 기준을 완화하기 위한 조건 |
| CNN prediction | 참고용 로그 |

즉, CNN이 `NotDrone`이라고 판단해도 실제 드론이 켜져 있었다면 해당 데이터는 **Drone 데이터**이다.

---

## 3. 이번 실험의 진짜 목적

이번 실험은 단순히 Drone 데이터 개수를 늘리는 작업이 아니다.

목표는 다음 두 가지이다.

1. **CNN이 드론 RF 패턴을 더 잘 학습하도록 Drone 데이터를 보강한다.**
2. **거리 변화에 따라 gain을 어떻게 조절해야 하는지 정규화 전 feature 기반으로 관계식을 만든다.**

핵심 아이디어:

```text
1m 기준 드론 신호를 기준 상태로 잡음
→ 정규화 전 feature들을 target으로 저장
→ 거리가 멀어지면 feature가 약해짐
→ gain을 올려 target feature와 비슷하게 맞춤
→ 정규화된 spectrogram을 CNN에 입력
```

---

## 4. CNN 입력 파이프라인

현재 CNN 입력은 절대 세기 자체가 아니라, 정규화된 spectrogram이다.

```text
IQ 수신
→ DC offset 제거
→ STFT 계산
→ spectrogram magnitude/log magnitude 생성
→ spectrogram 정규화
→ CNN 입력
→ Drone probability 출력
```

따라서 CNN은 주로 다음과 같은 **패턴 정보**를 학습한다.

- 드론 RF burst 모양
- 시간축 반복성
- 주파수축 분포
- 대역폭 형태
- WiFi/Bluetooth/Background와 다른 spectrogram 구조

이 때문에 CNN 학습용 입력은 정규화된 형태를 유지하는 것이 맞다.

---

## 5. Gain Control용 feature는 정규화 전 데이터를 사용

Gain 조절 알고리즘은 정규화 후 spectrogram을 기준으로 만들면 안 된다.

정규화 후에는 절대 세기 차이가 사라지기 때문에, 1m 강한 신호와 5m 약한 신호가 CNN 입력에서는 비슷하게 보일 수 있다.

따라서 gain control은 반드시 **정규화 전 feature**를 기준으로 해야 한다.

추천 feature:

| Feature | 의미 | 사용 목적 |
|---|---|---|
| `median_db` | 전체 배경 레벨 | noise floor 근사 |
| `p95_db` | 상위 에너지 레벨 | 신호 강도 보조 지표 |
| `p99_db` | 강한 burst 성분 | 주 gain 제어 기준 |
| `max_db` | 최대값 | 포화 여부 확인 |
| `burst_score = p99_db - median_db` | 배경 대비 burst 강도 | 드론 신호 대비 확인 |
| `saturation_ratio` | 포화 비율 | gain 과다 상승 방지 |
| `noise_floor` | 배경 잡음 추정 | threshold 계산 기준 |

가장 중요한 feature 후보:

```text
p99_db
burst_score
saturation_ratio
```

---

## 6. 기준 상태 설정

기준 상태는 다음 조건으로 잡는다.

```text
거리: 1m
gain: 25~30
상태: Drone ON
목표: 드론 패턴이 가장 깔끔하게 보이는 정규화 전 feature 저장
```

예시 target feature:

```text
target_p99_db
target_median_db
target_burst_score
target_saturation_ratio
```

이 target은 실험 후 실제 로그를 보고 결정한다.

---

## 7. Drone 데이터 수집 계획

목표 수량:

```text
Drone 추가 수집 목표: 약 1000~1200장
```

권장 분배:

| 조건 | 수량 |
|---|---:|
| Drone / 1m / gain25 | 200장 |
| Drone / 1m / gain30 | 200장 |
| Drone / 2m / gain30~35 | 200장 |
| Drone / 3m / gain35~40 | 200장 |
| Drone / 5m / gain40~45 | 200장 |
| 거리 이동 + gain 조절 실험 | 200장 |

시간이 부족하면 다음처럼 줄여도 된다.

| 조건 | 수량 |
|---|---:|
| Drone / 1m / gain25~30 | 400장 |
| Drone / 3m / gain35~40 | 400장 |
| Drone / 5m / gain40~45 | 400장 |

중요한 점:

- 한 조건에서만 1200장을 몰아서 받지 않는다.
- 거리에 따른 약한 드론 신호를 반드시 포함한다.
- CNN 판정값이 낮게 나오는 실제 Drone 샘플도 버리지 않는다.
- 가능하면 session 단위로 구분해서 저장한다.

---

## 8. Threshold 0.4 사용 목적

이번 실험에서 `Drone probability threshold = 0.4`를 사용하는 이유는 다음과 같다.

```text
기존에는 애매한 드론 신호가 NotDrone으로 빠졌을 가능성이 큼
→ threshold를 0.4로 낮춤
→ 약한 드론 후보를 더 많이 관찰
→ missed drone sample을 분석/보강
```

단, threshold 0.4는 최종 확정값이 아니다.

추후 기존 test/val 데이터에 대해 다음 threshold sweep을 해야 한다.

```text
0.30 / 0.35 / 0.40 / 0.45 / 0.50
```

평가 기준:

| 항목 | 목표 |
|---|---:|
| Drone recall | 85~90% 이상 |
| Missed Drone Rate | 10~15% 이하 |
| False Drone Rate | 가능하면 10% 이하 |

---

## 9. Gain-Distance 관계식 설계 방향

이론적으로 자유공간에서는 거리가 멀어질수록 필요한 보상 gain은 대략 다음과 같다.

```text
gain_compensation_db ≈ 20 log10(distance / 1m)
```

예상값:

| 거리 | 이론상 추가 gain |
|---:|---:|
| 1m | 0 dB |
| 2m | 약 +6 dB |
| 3m | 약 +9.5 dB |
| 5m | 약 +14 dB |

하지만 실제 환경에서는 다음 요소 때문에 이론값과 차이가 난다.

- 실내 반사
- 멀티패스
- 안테나 방향성
- 편파 차이
- 드론 자세
- 주변 WiFi/Bluetooth 간섭
- SDR gain 특성

따라서 최종 알고리즘은 단순 거리 공식보다 **정규화 전 feature 기반 feedback control** 방식이 더 현실적이다.

---

## 10. Gain Control 알고리즘 초안

기준 feature:

```text
target_p99_db
target_burst_score
sat_limit
```

현재 feature:

```text
current_p99_db
current_burst_score
current_saturation_ratio
```

기본 로직:

```python
if current_saturation_ratio > sat_limit:
    gain -= 2

elif current_p99_db < target_p99_db - margin:
    gain += 2

elif current_p99_db > target_p99_db + margin:
    gain -= 1

else:
    gain 유지
```

조금 더 안정적인 버전:

```python
if current_saturation_ratio > sat_limit:
    gain -= 2

elif (
    current_p99_db < target_p99_db - margin
    and current_burst_score < target_burst_score - margin
):
    gain += 2

elif current_p99_db > target_p99_db + margin:
    gain -= 1

else:
    gain 유지
```

주의:

- 한 번에 gain을 크게 바꾸지 않는다.
- 1~2 dB 단위로 천천히 조절한다.
- gain 증가 후에도 SNR이 좋아지는 것은 아니다.
- SDR gain을 올리면 신호뿐 아니라 noise도 같이 커질 수 있다.
- 따라서 `burst_score`와 `saturation_ratio`를 같이 봐야 한다.

---

## 11. 저장해야 할 로그 항목

이번 실험에서는 파일명 또는 별도 CSV 로그에 다음 항목을 남기는 것이 좋다.

```text
timestamp
label
session_id
distance_m
gain
center_freq
sample_rate
rf_bandwidth
block_size
threshold
drone_prob
pred_label
median_db
p95_db
p99_db
max_db
burst_score
saturation_ratio
noise_floor
memo
```

추천 파일명 예시:

```text
0001__Drone__d1m__g25__cf2450__thr04__session01.npy
0002__Drone__d3m__g38__cf2450__thr04__session02.npy
0003__Drone__d5m__g43__cf2450__thr04__session03.npy
```

---

## 12. Train / Validation / Test 분리 주의

연속으로 수집한 같은 session 데이터를 랜덤으로 train/test에 섞으면 성능이 실제보다 좋게 나올 수 있다.

가능하면 session 단위로 나누는 것이 좋다.

예시:

```text
train: 1m, 2m, 3m 일부 session
val: 3m 다른 session
test: 5m 또는 야외 session
```

특히 최종 성능 검증은 다음 조건이 좋다.

```text
학습에 쓰지 않은 거리
학습에 쓰지 않은 session
가능하면 실내와 다른 환경
```

---

## 13. 내일 실험 체크리스트

### 실험 전

- [ ] SDR 연결 확인
- [ ] 안테나 방향 고정
- [ ] center frequency 확인
- [ ] sample rate 확인
- [ ] gain 수동 설정 가능 여부 확인
- [ ] viewer에서 p99/median/burst_score 표시 확인
- [ ] 저장 폴더/session 이름 확인
- [ ] threshold 0.4 설정 확인

### 실험 중

- [ ] Drone ON 상태 확인
- [ ] 1m 기준 feature 저장
- [ ] gain 25, 30 비교
- [ ] 2m/3m/5m 거리별 feature 변화 확인
- [ ] 포화 여부 확인
- [ ] threshold 0.4에서 drone_prob 로그 확인
- [ ] CNN이 낮게 판단한 실제 Drone 샘플도 저장
- [ ] session별 memo 기록

### 실험 후

- [ ] 수집 개수 확인
- [ ] feature 통계 확인
- [ ] 거리별 best gain 정리
- [ ] 기존 NotDrone 데이터로 false drone rate 재평가
- [ ] threshold sweep 수행
- [ ] missed drone sample 분석

---

## 14. 최종 판단 기준

이번 실험 후 성공 기준은 단순히 데이터 1200장을 모으는 것이 아니다.

성공 기준:

```text
1. 1m 기준 target feature를 정의할 수 있어야 함
2. 거리별로 feature가 어떻게 약해지는지 보여야 함
3. gain을 올렸을 때 target feature에 가까워지는 경향이 보여야 함
4. 정규화된 CNN 입력에서 드론 패턴이 유지되어야 함
5. Drone recall이 기존보다 개선되어야 함
6. False Drone Rate가 과도하게 증가하지 않아야 함
```

---

## 15. 요약

이번 실험의 핵심은 다음 한 문장으로 정리할 수 있다.

> CNN은 정규화된 spectrogram에서 드론 패턴을 학습하고, gain control은 정규화 전 절대세기 feature를 1m 기준값에 맞추는 방식으로 설계한다.

따라서 내일 할 일은 다음과 같다.

```text
1. 1m gain25~30에서 기준 드론 feature 확보
2. Drone 데이터 약 1000~1200장 추가 수집
3. 거리별 gain 변화에 따른 정규화 전 feature 기록
4. p99_db, burst_score, saturation_ratio 기반 gain 조절 로직 검증
5. 이후 threshold sweep과 재학습으로 missed drone rate 감소 확인
```
