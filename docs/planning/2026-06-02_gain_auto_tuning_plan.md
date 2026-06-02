# Gain Auto Tuning 및 Raw IQ Safety Calibration 설계 계획서

작성일: 2026-06-02

---

# 1. 배경

현재 RF Drone Detection 프로젝트는 CNN 기반 스펙트로그램 분류를 수행하고 있으며, 실제 운용 환경에서는 드론과 수신기 사이의 거리가 변화한다.

거리가 증가하면 수신 신호 세기가 감소하고, 이를 보상하기 위해 SDR Gain을 증가시켜야 한다.

하지만 Gain을 과도하게 증가시키면 다음과 같은 문제가 발생한다.

* ADC Saturation
* IQ Clipping
* FFT 왜곡
* Spectrogram 왜곡
* CNN 입력 왜곡
* AoA 위상 오차 증가

따라서 거리 변화에 따라 적절한 Gain을 자동으로 선택하는 알고리즘이 필요하다.

---

# 2. 목표

본 설계의 목표는 다음과 같다.

> 포화(Saturation)가 발생하지 않는 범위 내에서 드론 신호가 충분히 관측되도록 적절한 Gain을 자동 결정한다.

즉,

* Gain이 너무 낮아 신호가 보이지 않는 상황 방지
* Gain이 너무 높아 포화되는 상황 방지
* CNN이 학습한 패턴과 유사한 입력 유지

를 동시에 달성하는 것을 목표로 한다.

---

# 3. 전체 구조

```text
Raw IQ Input
      │
      ▼
Raw IQ Safety Check
      │
      ▼
Noise Calibration
      │
      ▼
Signal Feature Analysis
      │
      ▼
Gain Decision Policy
      │
      ▼
Gain Up / Keep / Down
```

---

# 4. Raw IQ Safety Check

## 목적

현재 수신 상태가 정상 범위인지 확인한다.

자동 Gain 조절 알고리즘의 가장 첫 번째 단계이다.

---

## 확인 항목

### 4.1 Maximum Amplitude

```text
max_abs
```

현재 IQ 샘플의 최대 절대값

---

### 4.2 RMS Power

```text
rms
```

전체 신호 세기 측정

---

### 4.3 DC Offset

```text
mean(I)
mean(Q)
```

수신기 이상 여부 확인

---

### 4.4 Saturation Ratio

```text
saturation_ratio
```

포화 샘플 비율

```text
saturation_ratio =
포화 샘플 수 / 전체 샘플 수
```

---

## 상태 분류

### SAFE

```text
saturation_ratio = 0%
```

정상

---

### WARNING

```text
0% < saturation_ratio < 1%
```

주의

---

### CLIPPED

```text
saturation_ratio ≥ 1%
```

Gain 감소 필요

---

# 5. Gain별 Noise Calibration

## 목적

Gain마다 다른 노이즈 레벨을 모델링한다.

Gain 10과 Gain 30은 배경 노이즈 분포가 다르므로 동일 Threshold를 사용할 수 없다.

---

## Gain별 저장 항목

예시

```text
Gain 10
Gain 15
Gain 20
Gain 25
Gain 30
```

각 Gain에 대해

```text
median_noise
p95_noise
p99_noise
max_noise
burst_score_noise
```

를 저장한다.

---

## 결과

실시간 운용 시

```text
현재 Gain = 25
```

이면

```text
Gain25 Noise Profile
```

을 로드하여 Threshold를 결정한다.

---

# 6. Signal Feature Analysis

## 목적

현재 신호가 충분히 관측되는지 평가한다.

---

## 사용 후보 Feature

### Amplitude 계열

```text
median
p95
p99
max
rms
```

---

### Burst 계열

```text
burst_score
```

---

### Spectrogram 계열

```text
spectrogram contrast
spectrogram energy
```

(향후 확장 가능)

---

# 7. Gain Decision Policy

## Gain 증가 조건

```text
SAFE
+
신호 세기 부족
```

↓

```text
Gain + Δ
```

---

## Gain 유지 조건

```text
SAFE
+
목표 범위
```

↓

```text
Keep Gain
```

---

## Gain 감소 조건

```text
WARNING 또는 CLIPPED
```

↓

```text
Gain - Δ
```

---

# 8. 최종 목표

본 알고리즘의 최종 목표는

```text
가장 강한 신호를 받는 것
```

이 아니라

```text
포화되지 않는 범위에서
CNN이 안정적으로 분류 가능한 수준의 신호를 유지하는 것
```

이다.

즉,

* 포화 없음
* 노이즈 대비 충분한 신호 세기
* 학습 데이터와 유사한 Spectrogram 패턴

을 동시에 만족하는 Gain을 자동으로 선택하는 것이 최종 목표이다.

---

# 9. 향후 구현 순서

## Phase 1

Raw IQ Safety Check 구현

```text
max_abs
rms
dc_offset
saturation_ratio
```

---

## Phase 2

Gain별 Noise Calibration 데이터 수집

```text
Gain 10
Gain 15
Gain 20
Gain 25
Gain 30
```

---

## Phase 3

Signal Visibility Score 설계

```text
noise 대비 신호 강도 계산
```

---

## Phase 4

Auto Gain Tuning 구현

```text
Gain Up
Gain Down
Gain Keep
```

---

## Phase 5

실외 거리별 검증

```text
1 m
2 m
3 m
5 m
```

환경에서 적정 Gain 추적 성능 평가
