# 2026-05-31 RF Drone Detection Viewer 및 AoA Phase Calibration 개선 통합 보고서

## 1. 작업 목적

오늘 작업의 목적은 RF 기반 드론 탐지 프로젝트에서 실시간 실험 효율을 높이기 위해 기존 live viewer를 개선하고, 향후 거리별 gain 보정 및 AoA 추정 실험을 수행할 수 있는 보조 도구를 개발하는 것이다.

기존 live viewer는 SDR에서 IQ block을 수신하고, raw feature 및 CNN 입력용 spectrogram을 계산하여 화면과 CSV에 기록하는 구조였다. 그러나 실험 현장에서 gain 조절, 거리 기록, 메모 입력, 일시정지, 기준 feature 비교, AoA 확인 등을 실시간으로 수행하기에는 불편함이 있었다. 따라서 오늘은 실험자가 현장에서 직접 gain과 조건을 조정하면서 RF 신호 상태를 확인할 수 있도록 viewer 기능을 확장하였다.

---

## 2. 기존 문제점

기존 실시간 viewer의 주요 한계는 다음과 같았다.

1. gain 값을 변경하려면 프로그램을 종료하고 명령어를 다시 실행해야 했다.
2. 실험 중 거리와 메모를 즉시 변경하여 기록하기 어려웠다.
3. 신호발생기나 드론 위치를 이동하는 동안에도 수집이 계속되어 불필요한 데이터가 CSV에 저장될 수 있었다.
4. 기준 거리에서의 raw feature를 저장하고, 다른 거리에서 gain을 조절했을 때 기준 feature와 얼마나 비슷한지 판단하는 기능이 필요했다.
5. AoA 실험을 위해 dual RX 채널의 위상차를 이용한 각도 추정 기능이 필요했다.
6. 드론 CNN 판정과 AoA 계산을 강하게 연결하면 신호발생기 기반 AoA 검증에는 부적합하므로, 신호발생기 전용 AoA viewer를 별도로 분리할 필요가 있었다.

---

## 3. 실시간 Viewer 개선 사항

### 3.1 Gain 실시간 조절 기능

viewer 하단에 gain 입력칸과 Apply 버튼을 추가하였다. 사용자는 viewer 실행 중 gain 값을 입력하고 Apply를 누르면 receiver의 gain이 변경되도록 구성하였다.

이를 통해 실험 중 프로그램을 재시작하지 않고도 gain을 조절할 수 있으며, 변경된 gain 값은 현재 화면과 CSV row에 반영된다.

기대 효과는 다음과 같다.

- 거리별 gain 보정 실험 가능
- 실외 실험 중 수신 세기 변화에 빠르게 대응 가능
- gain 25, gain 30 등 조건 비교 시 실험 시간 단축

### 3.2 현재 gain 표시 기능

현재 적용 중인 gain 값을 viewer 화면에 표시하도록 구성하였다. gain을 변경하면 화면의 Current Gain 값도 함께 갱신된다.

### 3.3 Distance / Memo 입력 기능

viewer 하단에 distance와 memo 입력칸을 추가하였다. 실험자는 거리 조건이나 실험 상황을 입력할 수 있으며, 이 값은 CSV 로그에 함께 저장된다.

예시 memo는 다음과 같다.

```text
drone_fixed_gain25_5m
controller_only_drone_off_g25
siggen_aoa_front_0deg_g25
drone_relink_state_transition_5m_gain25
```

### 3.4 Pause / Resume 기능

실험 중 드론 위치 변경, 신호발생기 이동, 안테나 방향 조정 등을 수행할 때 수집을 일시정지할 수 있도록 Pause/Resume 버튼을 추가하였다.

```text
Pause 클릭
→ IQ 수집 중지
→ feature 계산 중지
→ CSV 저장 중지
→ 화면은 마지막 상태 유지

Resume 클릭
→ IQ 수집 재개
→ feature 계산 및 CSV 저장 재개
```

### 3.5 Target feature 저장 및 비교 기능

거리별 gain 보정 실험을 위해 현재 raw feature를 target으로 저장하는 기능을 추가하였다.

기준 거리에서 Use Current 버튼을 누르면 현재 feature가 target 값으로 저장된다. 이후 다른 거리에서 gain을 조절하면서 현재 feature가 target과 얼마나 유사한지 비교할 수 있다.

현재 비교 대상 feature는 다음과 같다.

```text
raw_abs_p99
frame_power_p99
raw_rms
```

판정 상태는 다음과 같이 구성하였다.

```text
NO_TARGET : 기준 feature가 아직 없음
SEARCH    : 현재 feature가 target과 차이가 큼
MATCH     : 현재 feature가 target tolerance 이내
OVERLOAD  : 입력이 과도하여 비교 불가
```

현재는 선형값 기준의 percentage error로 비교하고 있으나, 거리-gain 관계식에는 dB 기반 비교가 더 적합하므로 향후 개선이 필요하다.

---

## 4. Overload 기준 조정

실험 중 기존 overload 기준이 현재 Pluto SDR raw IQ scale에 비해 너무 낮게 설정되어 있음을 확인하였다.

초기 기준은 다음과 같았다.

```text
overload_peak = 0.95
```

하지만 실제 raw_peak 값은 10 이상으로 나오는 경우가 많았기 때문에, 단순히 raw amplitude가 0.95를 넘는다는 이유로 대부분의 block이 OVERLOAD로 판정되는 문제가 있었다.

이에 따라 실험적으로 다음과 같은 기준을 사용하였다.

```text
overload_peak = 20 ~ 30
overload_clip_ratio = 0.001
```

이를 통해 실제 실험 환경의 raw IQ scale에 맞는 overload 판정이 가능해졌다.

다만 이 값은 최종 확정값이 아니며, 향후 background, controller-only, drone-linked, siggen 조건에서 raw_peak와 clip_ratio 분포를 비교하여 다시 조정할 필요가 있다.

---

## 5. 실험 중 발견한 모델 해석 문제

### 5.1 조종기 단독 신호 오탐

실험 중 드론을 끄고 조종기만 켜도 CNN이 Drone으로 판단하는 현상이 확인되었다. 반대로 조종기까지 끄면 NonDrone으로 전환되었다.

현재 관측된 현상은 다음과 같이 정리할 수 있다.

```text
드론 ON + 조종기 ON + 링크 상태
→ Drone 또는 Confirmed Drone

드론 OFF + 조종기 ON
→ Drone으로 탐지됨

드론 OFF + 조종기 OFF
→ NonDrone으로 전환됨
```

이는 현재 이진 분류 모델이 조종기 단독 신호와 드론-linked 신호를 구분하지 못하고 있음을 의미한다.

### 5.2 원인 분석

기존 Drone class 데이터는 대부분 드론과 조종기가 함께 켜져 있고 링크된 상태에서 수집되었다. 따라서 수신된 RF에는 드론 기체에서 발생하는 신호뿐 아니라 조종기 송신 신호도 함께 포함되었을 가능성이 크다.

CNN은 학습 과정에서 가장 강하고 반복적인 패턴을 class feature로 학습하는 경향이 있다. 만약 조종기 신호가 드론 기체 신호보다 훨씬 강하게 수신되었다면, 모델은 조종기 신호를 Drone class의 주요 특징으로 학습했을 수 있다.

특히 패치 안테나를 사용하더라도 측면이나 반사 경로를 통해 조종기 신호가 들어올 수 있으며, 조종기 신호가 드론 자체 신호보다 강한 경우 모델이 조종기 패턴에 더 민감하게 반응하는 것은 자연스러운 결과이다.

### 5.3 향후 데이터셋 개선 방향

사용자의 목표가 조종기 신호 탐지가 아니라 드론 신호 탐지라면, 향후 데이터셋에는 반드시 controller-only 조건을 hard negative로 추가해야 한다.

권장 데이터 구성은 다음과 같다.

```text
Drone class:
- 드론 ON
- 조종기 ON
- 링크 완료
- 드론이 안테나 정면에 위치

NonDrone class:
- 드론 OFF + 조종기 OFF
- 드론 OFF + 조종기 ON
- 조종기 조작 중
- Wi-Fi
- Bluetooth
- Background
```

더 정확한 분리를 위해서는 3-class 구조도 고려할 수 있다.

```text
Class 0: Background / Wi-Fi / Bluetooth
Class 1: Controller only
Class 2: Drone + Controller linked
```

---

## 6. AoA 기능 추가

### 6.1 Dual RX phase difference 기반 AoA

dual RX 채널의 위상차를 이용하여 AoA를 추정하는 기능을 추가하였다.

기본 아이디어는 다음과 같다.

```text
RX0, RX1 복소 IQ 수신
→ cross = mean(RX1 * conj(RX0))
→ phase_diff = angle(cross)
→ sin(theta) = phase_diff * lambda / (2πd)
→ theta = asin(...)
```

여기서 d는 안테나 간격이고, lambda는 중심주파수에 해당하는 파장이다.

단, 현재 AoA는 정밀한 절대각이라기보다는 실험용 상대 방향 추정값에 가깝다. 채널 간 위상 offset, 케이블 길이, 안테나 위치 오차, SDR 내부 경로 차이 등이 보정되지 않으면 정면 0도에서도 AoA가 0도로 나오지 않을 수 있다.

### 6.2 Phase calibration

두 RX 채널은 하드웨어 경로 차이 때문에 정면 0도에서도 위상차가 0이 아닐 수 있다. 이를 보정하기 위해 phase calibration 기능을 추가하였다.

절차는 다음과 같다.

```text
1. 신호원 또는 드론을 안테나 정면 0도에 배치
2. RX1-RX0 평균 위상차 측정
3. 해당 값을 phase_offset_rad로 저장
4. 이후 AoA 계산 시 phase_diff_raw - phase_offset_rad 적용
```

이를 통해 정면 방향의 기준 위상차를 제거하고, 이후 좌우 방향 변화에 따른 상대 AoA를 확인할 수 있다.

### 6.3 AoA smoothing

AoA는 순간 위상차에 민감하기 때문에 단일 update 값이 튈 수 있다. 이를 완화하기 위해 최근 유효 AoA 값들을 이용한 smoothing 기능을 추가하였다.

일반 평균이 아니라 circular mean을 사용하였다. 이유는 각도 값은 -180도와 +180도 근처에서 wrap 문제가 발생하기 때문이다.

현재 구성은 다음과 같다.

```text
aoa_smooth_window = 5
aoa_smooth_min_valid = 3
```

즉 최근 5개 중 유효한 AoA가 3개 이상이면 circular moving average를 출력한다.

### 6.4 AoA gate 문제와 개선

처음에는 AoA가 모든 block에서 계산되어 배경 신호나 NO_SIGNAL 상태에서도 0~4도 수준의 각도가 계속 표시되는 문제가 있었다. 이는 유효한 신호가 없는데도 두 채널 위상차 평균이 계산되기 때문에 발생한다.

따라서 AoA 계산 전 gate가 필요하다는 것을 확인하였다.

드론용 AoA에서는 다음 조건을 사용할 수 있다.

```text
status == VALID_SIGNAL 또는 WEAK_SIGNAL
signal_ratio 기준 이상
CNN raw decision 또는 final decision이 Drone 계열
coherence 기준 이상
OVERLOAD 아님
```

하지만 신호발생기 AoA 실험에서는 CNN 판정이 필요 없으므로, 별도 viewer에서는 CNN gate를 제거해야 한다.

---

## 7. 신호발생기 전용 AoA Viewer 분리

신호발생기로 AoA를 검증할 경우, 드론 CNN 판정에 의존하면 안 된다. 신호발생기는 Drone class가 아니기 때문에 기존 드론 판정 gate를 통과하지 못할 수 있다.

따라서 신호발생기 전용 AoA viewer를 별도 파일로 분리하였다.

```text
scripts/live_siggen_aoa_viewer.py
```

이 viewer의 목적은 다음과 같다.

```text
- CNN drone 판정과 무관하게 AoA 계산
- 신호발생기 CW 또는 고정 신호원 기준 phase calibration 수행
- coherence 기반으로 AoA 신뢰도 판단
- phase offset 및 smoothing 기능 유지
```

이를 통해 드론 신호와 무관하게 AoA 알고리즘 자체를 신호발생기로 먼저 검증할 수 있다.

---

## 8. 경량 스펙트로그램 Viewer 개발

기존 viewer는 CNN, CSV, matplotlib UI, side text, widget 등이 모두 포함되어 있어 무겁게 동작하였다. 특히 실시간으로 드론/조종기/신호발생기 신호를 눈으로 확인하기에는 부담이 있었다.

따라서 spectrogram만 표시하는 경량 viewer를 별도로 만들었다.

```text
scripts/live_spectrogram_only_viewer.py
```

이 viewer의 목적은 다음과 같다.

```text
- CNN inference 없음
- CSV logging 없음
- matplotlib widget 없음
- OpenCV 기반 실시간 spectrogram 표시
- q 종료
- p pause/resume
- [ / ] gain 조절
```

이 도구는 조종기 단독 신호와 드론-linked 상태의 spectrogram 차이를 눈으로 확인하기 위한 보조 도구로 사용할 수 있다.

---

## 9. 현재 한계

현재까지 확인된 한계는 다음과 같다.

1. 기존 CNN 모델은 조종기 단독 신호를 Drone으로 판단한다. 따라서 현재 모델을 그대로 사용하면 드론 기체 신호 탐지라고 주장하기 어렵다.
2. 드론 자체 신호는 조종기 신호보다 약할 가능성이 있다. 조종기를 안테나 수직 방향에 두더라도 조종기 신호가 더 강하게 수신되는 것으로 보인다.
3. AoA 계산은 dual RX coherence와 phase calibration에 크게 의존한다. coherence가 낮은 조건에서는 AoA 값의 신뢰도가 낮다.
4. AoA는 현재 상대 방향 확인용에 가까우며, 정밀한 절대각 추정을 위해서는 0도 기준 calibration, 안테나 간격 정확도, 케이블 위상차 보정, 채널 동기 안정성 검증이 추가로 필요하다.
5. overload 기준은 아직 최종 확정되지 않았다. raw IQ scale에 맞는 overload_peak와 clip_ratio 기준을 추가 실험으로 재설정해야 한다.

---

## 10. 다음 작업 계획

다음 작업은 다음과 같이 정리할 수 있다.

1. Controller-only 데이터를 NonDrone hard negative로 수집한다.
2. 동일한 조종기 위치에서 드론 OFF / 드론 ON linked 조건을 A-B-A 방식으로 수집한다.
3. 조종기 단독 조건과 드론-linked 조건의 spectrogram 차이를 경량 viewer로 직접 확인한다.
4. 차이가 보이면 controller-only를 포함하여 CNN을 재학습한다.
5. 차이가 거의 없으면 현재 하드웨어 조건에서 드론 기체 신호 분리가 어렵다고 판단하고, 안테나 배치 또는 수집 조건을 재설계한다.
6. 신호발생기 전용 AoA viewer로 phase calibration과 각도 추정 안정성을 먼저 검증한다.
7. AoA가 안정적으로 동작하면 드론 또는 조종기 신호에 대해 AoA를 적용한다.
8. 거리별 gain 실험은 dB 기반 feature 비교 방식으로 개선한다.

---

## 11. 중간 정리

오늘 작업을 통해 실시간 viewer의 실험 편의 기능을 크게 확장하였다. Gain 실시간 조절, pause/resume, 거리 및 메모 입력, target feature 비교 기능을 추가하여 거리별 gain 및 수신 세기 실험을 더 효율적으로 수행할 수 있게 되었다.

또한 dual RX 기반 AoA 기능을 추가하면서 phase calibration, coherence 기반 신뢰도 판단, circular smoothing 기능을 도입하였다. 이후 신호발생기 기반 AoA 검증을 위해 CNN 판정 gate를 제거한 별도 viewer를 분리하였다.

실험 과정에서는 기존 CNN 모델이 드론 기체 단독 신호가 아니라 조종기 송신 신호 또는 드론 운용 RF activity에 강하게 반응한다는 중요한 한계를 발견하였다. 이 문제를 해결하기 위해서는 controller-only 데이터를 NonDrone 또는 별도 class로 포함한 재학습이 필요하다.

따라서 오늘의 작업은 단순한 viewer 기능 추가가 아니라, 현재 모델의 한계를 확인하고 향후 데이터셋 재구성 및 AoA 검증 방향을 정리한 중간 단계로 볼 수 있다.

---

## 12. Phase Calibration 실험 중 추가로 확인한 문제

### 12.1 단순 평균 phase calibration의 한계

기존 phase calibration은 일정 개수의 block에서 RX1-RX0 평균 위상차를 구한 뒤 단순 평균에 가까운 방식으로 phase offset을 저장하는 구조였다. 그러나 실제 야외 및 실내 실험에서는 일부 block이 멀티패스, Wi-Fi 간섭, SDR 초기 안정화 문제의 영향을 받아 정상 block과 다른 phase cluster를 형성하는 현상이 관찰되었다.

예를 들어 동일한 gain 조건에서도 다음과 같은 현상이 나타났다.

```text
case 1:
대부분 +84° 근처
일부 +95~105° 근처
→ 단순 평균 시 +90° 근처의 애매한 값 저장 가능

case 2:
초기 run에서는 +102° 근처
반복 run에서는 +82~85° 근처
→ SDR 초기 버퍼 또는 현장 RF 조건에 따라 값 변동

case 3:
신호발생기 2452 MHz, SDR center 2450 MHz, 거리 약 1.5 m 조건
→ +44.2° 근처로 안정
→ coherence median 약 0.81 수준
```

따라서 phase calibration은 단일 block 또는 단순 평균으로 처리하기보다, 다수 block의 phase 분포를 확인하고 dominant cluster를 선택하는 방식이 필요하다고 판단하였다.

### 12.2 2.4 GHz Wi-Fi 및 핫스팟 간섭

실험 중 주변에 2.4 GHz Wi-Fi가 강하게 존재하는 것을 확인하였다. 또한 대화 및 인터넷 연결을 위해 스마트폰 핫스팟을 켜둔 상태였기 때문에, 2.45 GHz 부근의 calibration tone과 주변 Wi-Fi 성분이 함께 수신되었을 가능성이 있다.

이때 phase 분포가 하나의 cluster로 모이지 않고 여러 cluster로 나뉘는 현상이 관찰되었다. 이후 신호발생기 주파수를 2449 MHz에서 2452 MHz로 변경하고, SDR center frequency는 2450 MHz로 유지하여 baseband에서 약 +2 MHz tone으로 수신되도록 조정하였다. 또한 송신원과 RX 간 거리를 약 1.5 m로 줄여 신호발생기 tone이 주변 Wi-Fi보다 우세하게 들어오도록 하였다.

그 결과 coherence median이 약 0.80 이상으로 개선되었고, 반복 측정에서 phase offset이 약 +44.2° 근처로 안정되는 결과가 나타났다.

### 12.3 Gain 변경에 따른 phase offset 변화 가능성

이론적으로 수신 gain은 신호 크기에만 영향을 주고 phase에는 영향을 주지 않아야 한다. 그러나 실제 SDR에서는 gain 값에 따라 내부 RF front-end의 gain stage, VGA 동작점, 채널별 amplitude imbalance 등이 달라질 수 있다.

실험 중 다음과 같은 경향이 확인되었다.

```text
실외:
- 직접파가 상대적으로 우세
- 일정 거리 이내에서는 gain 조절 후에도 phase calibration 값이 비교적 안정

실내:
- 벽, 바닥, 천장, 사람, 장비 등에 의한 반사파가 강함
- gain을 변경하면 직접파/반사파/간섭 성분의 상대 비율이 바뀜
- phase calibration 값이 흔들릴 가능성이 큼
```

따라서 향후 AoA 실험에서는 gain 변경 시 같은 phase offset을 그대로 재사용하지 않고, gain별 phase 변화량을 보정할 수 있는 구조가 필요하다.

---

## 13. Robust Phase Calibration 개선

### 13.1 개선 목표

기존 phase calibration은 멀티패스 또는 간섭 block이 섞여도 단순 평균에 반영되는 문제가 있었다. 이를 개선하기 위해 다음과 같은 robust calibration 방식을 설계하였다.

```text
1. 총 200 block 수집
2. 초기 안정화용 앞 30 block discard
3. 남은 170 block 사용
4. block별 RX1-RX0 phase와 coherence-like 계산
5. coherence 기준 미달 block 제거
6. phase 분포에서 dominant cluster 선택
7. dominant cluster 내부 block만 coherence-weighted circular mean
8. phase_std, coherence_median, cluster_ratio 기반 quality 판정
9. current_phase_offset.json 저장
```

이 방식은 단순히 phase offset 하나를 계산하는 것뿐 아니라, 해당 현장의 RF 환경이 AoA 측정에 적합한지 판단하는 quality check 역할도 수행한다.

### 13.2 Dominant Cluster 방식

phase calibration에서 가장 중요한 개선은 dominant phase cluster 선택이다. 기존에는 coherence threshold를 넘는 모든 block을 평균했기 때문에, 두 개 이상의 phase cluster가 존재하면 중간값으로 왜곡될 수 있었다.

개선된 방식은 각 phase 값을 cluster center 후보로 두고, 일정 window 안에 가장 많은 block이 들어오는 cluster를 선택한다. 같은 개수일 경우 coherence 합이 더 큰 cluster를 선택한다.

```text
예시:
120 blocks → +44° 근처
50 blocks  → +96° 근처

단순 평균:
→ +60~70° 또는 상황에 따라 애매한 값

dominant cluster:
→ +44° cluster 선택
→ +44° 근처를 최종 phase offset으로 저장
```

### 13.3 Circular Weighted Mean

phase는 일반 실수 평균을 사용하면 wrap 문제가 발생할 수 있다. 예를 들어 +179°와 -179°는 실제로 2° 차이지만, 일반 평균에서는 0° 근처로 잘못 해석될 수 있다.

따라서 최종 phase offset은 다음 방식으로 계산한다.

```text
phase_i를 단위원 벡터 exp(j·phase_i)로 변환
coherence_i를 weight로 사용
weighted vector sum 계산
angle(sum)을 최종 phase로 사용
```

이 방식은 phase wrap 문제를 피하면서 coherence가 높은 block이 최종 결과에 더 크게 반영되도록 한다.

### 13.4 Calibration Quality 판정

Robust calibration 결과에는 다음 지표를 저장한다.

```text
phase_offset_deg
phase_offset_rad
phase_std_deg
coherence_mean
coherence_median
coherence_min
coherence_max
valid_blocks
cluster_blocks
cluster_ratio
quality
```

현재 quality 기준은 다음과 같이 설정하였다.

```text
OK:
- phase_std < 3°
- valid_blocks >= 100
- cluster_ratio >= 0.80
- coherence_median >= 0.70

WARNING:
- phase_std < 7°
- valid_blocks >= 50
- cluster_ratio >= 0.60
- coherence_median >= 0.55

FAIL:
- 위 조건 미달
```

quality가 OK이면 해당 phase offset을 현재 실험 세션에 적용할 수 있다. WARNING이면 사용은 가능하지만 재측정을 권장하고, FAIL이면 AoA 결과의 신뢰도가 낮다고 판단한다.

---

## 14. Gain-dependent Phase Offset Table 설계

### 14.1 설계 배경

실험 중 gain 값이 달라질 때 phase calibration 값도 변할 수 있음을 확인하였다. 이는 SDR 내부 gain path 또는 실내 멀티패스 환경의 영향으로 해석할 수 있다.

문제는 실제 현장 또는 드론 실험 중 gain을 바꿀 때마다 신호발생기를 다시 정면 0도에 배치할 수 없다는 점이다. 따라서 사전에 실외에서 gain별 phase 변화량을 측정하고, 현장에서는 기준 gain에서 한 번만 calibration한 뒤 gain별 delta를 적용하는 방식을 설계하였다.

### 14.2 운영 개념

운영 흐름은 다음과 같다.

```text
[실외 사전 작업]
build_gain_phase_table()
→ gain별 200 block 수집
→ 앞 30 block discard
→ dominant_cluster_phase()로 gain별 phase 추출
→ reference gain=30 기준 delta 변환
→ gain_phase_table_2450.json 저장

[현장 시작]
gain=30, 0도에서 robust phase calibration
→ current_ref_phase_offset 저장

[실시간]
gain 변경 시
→ get_phase_offset_to_apply()
→ current_ref_phase_offset + gain_delta_table[current_gain]
→ RX1에 exp(-j·phase_offset_to_apply) 적용
```

### 14.3 절대값 테이블이 아닌 상대 delta 테이블

한 번 측정한 gain별 절대 phase offset을 영구적으로 사용하는 것은 위험하다. SDR 재부팅, LO 재설정, RX 채널 초기화 등에 따라 전체 phase 기준이 달라질 수 있기 때문이다.

따라서 저장해야 하는 핵심 값은 gain별 절대 offset이 아니라 reference gain 대비 상대 변화량이다.

```text
실외 사전 측정:
gain30 = +44.2°
gain25 = +58.0°
gain20 = +70.5°

reference gain = 30
delta30 = 0.0°
delta25 = +13.8°
delta20 = +26.3°
```

현장에서 gain30 기준 calibration 값이 +46.0°로 측정되었다면, gain25 적용값은 다음과 같이 계산한다.

```text
current_offset(gain25)
= current_ref_offset(gain30) + delta25
= +46.0° + +13.8°
= +59.8°
```

이 방식은 부팅 또는 세션마다 바뀔 수 있는 전체 phase 기준은 현장 calibration으로 흡수하고, gain 변경에 따른 상대 phase 변화는 사전 테이블로 보정하는 구조이다.

---

## 15. Runtime Phase Correction 적용

### 15.1 Runtime 적용 목적

phase calibration 결과를 실제 AoA 계산에 반영하기 위해 runtime utility를 추가하였다. 이 utility는 다음을 수행한다.

```text
1. configs/calibration/current_phase_offset.json 로드
2. 필요 시 configs/calibration/gain_phase_table_2450.json 로드
3. 현재 gain에 해당하는 phase offset 계산
4. RX1 채널에 exp(-j·phase_offset)를 곱해 보정
5. AoA 계산부에는 보정된 IQ 전달
```

### 15.2 적용 방식

AoA 계산 전 RX1 채널에 다음 보정을 적용한다.

```python
rx1_corrected = rx1 * np.exp(-1j * phase_offset_to_apply)
```

2채널 IQ block 기준으로는 다음과 같이 적용한다.

```python
iq_corrected = iq.copy()
iq_corrected[1] = iq_corrected[1] * np.exp(-1j * phase_offset_to_apply)
```

이 보정은 AoA phase difference 계산 전에 수행되어야 한다. 보정 후 계산되는 RX1-RX0 phase difference는 하드웨어 phase offset이 제거된 값에 가까워진다.

### 15.3 run_pipeline.py 적용

`scripts/run_pipeline.py`에는 runtime phase calibration을 읽어 `iq_for_aoa`에 적용하는 구조를 추가하였다.

기존에는 처음 한 block으로 phase offset을 추정하고 즉시 적용하는 방식이었으나, 이는 단일 block 기반이라 멀티패스와 잡음에 취약하다. 따라서 robust calibration으로 저장된 `current_phase_offset.json`이 존재하면 해당 값을 우선 적용하도록 구조를 변경하였다.

summary.json에는 다음 항목이 추가되도록 구성하였다.

```text
runtime_phase_calibration_enabled
runtime_phase_calibration_source
runtime_phase_calibration_quality
runtime_phase_offset_to_apply_rad
runtime_phase_offset_to_apply_deg
runtime_phase_uncertainty_deg
```

### 15.4 live_siggen_aoa_viewer.py 적용

`scripts/live_siggen_aoa_viewer.py`에는 기존 수동 `--aoa-phase-offset-rad` 입력 외에도 phase calibration JSON을 자동으로 읽는 옵션을 추가하였다.

추가된 옵션은 다음과 같다.

```text
--aoa-phase-calibration-json
--aoa-gain-phase-table
```

이를 통해 신호발생기 AoA viewer 실행 시 phase offset을 손으로 입력하지 않고, 사전에 저장된 calibration 결과를 자동으로 적용할 수 있다.

---

## 16. 오늘 추가/수정한 파일 브리핑

### 16.1 `src/calibration/gain_phase_table.py`

역할은 gain별 phase calibration table을 생성하고, gain 변경 시 적용할 phase delta를 계산하는 것이다.

주요 기능은 다음과 같다.

```text
wrap_phase_rad()
circular_distance_rad()
circular_weighted_mean_rad()
compute_block_phase_and_coherence()
select_dominant_cluster()
dominant_cluster_phase()
evaluate_table_entry()
build_gain_phase_table()
load_gain_phase_table()
interpolate_phase_delta()
get_phase_offset_to_apply()
```

이 파일은 오늘 추가한 phase calibration 고도화의 핵심 모듈이다.

### 16.2 `src/calibration/__init__.py`

`src.calibration` 패키지에서 `gain_phase_table.py`의 주요 함수들을 import할 수 있도록 export 목록을 추가하였다.

이를 통해 다른 파일에서 다음과 같이 간단히 사용할 수 있다.

```python
from src.calibration import dominant_cluster_phase, build_gain_phase_table
```

### 16.3 `scripts/build_gain_phase_table.py`

실외 사전 작업용 스크립트이다. 여러 gain에 대해 phase offset을 측정하고, reference gain 기준 delta table을 생성한다.

출력 파일 예시는 다음과 같다.

```text
configs/calibration/gain_phase_table_2450.json
```

이 파일은 향후 gain 변경 시 phase offset 보정값을 갱신하는 데 사용된다.

### 16.4 `scripts/calibrate_phase_offset_outdoor.py`

기존 단순 phase calibration 스크립트를 robust 방식으로 개선하였다.

변경된 주요 사항은 다음과 같다.

```text
기존:
- 50 block 또는 지정 block 평균
- coherence 기준 통과 block 전체 평균
- 단순 phase offset 저장

개선:
- 기본 200 block 수집
- 앞 30 block discard
- dominant cluster phase 선택
- circular weighted mean 적용
- phase_std/coherence/cluster_ratio 저장
- quality 판정 저장
```

출력 파일은 다음과 같다.

```text
outputs/calibration/<timestamp>_phase_offset.../phase_offset_calibration.json
configs/calibration/current_phase_offset.json
```

### 16.5 `src/runtime/phase_calibration_runtime.py`

실시간 AoA 계산에서 phase calibration 결과를 적용하기 위한 runtime utility이다.

주요 기능은 다음과 같다.

```text
load_current_phase_offset()
resolve_phase_offset_to_apply()
apply_phase_offset_to_iq()
print_phase_calibration_state()
```

이 파일은 `current_phase_offset.json`과 선택적으로 `gain_phase_table_2450.json`을 읽어 현재 gain에서 적용할 phase offset을 계산한다.

### 16.6 `src/runtime/__init__.py`

runtime phase calibration utility를 다른 모듈에서 쉽게 import할 수 있도록 export를 추가하였다.

```python
from src.runtime import resolve_phase_offset_to_apply, apply_phase_offset_to_iq
```

### 16.7 `scripts/run_pipeline.py`

pipeline 실행 시 phase calibration runtime을 로드하여 AoA 계산 전 `iq_for_aoa`에 phase 보정을 적용하도록 수정하였다.

기존 단일 block 기반 phase offset 적용은 멀티패스에 취약하므로, robust calibration JSON이 존재하는 경우 해당 값을 우선 적용한다.

### 16.8 `scripts/live_siggen_aoa_viewer.py`

신호발생기 기반 AoA viewer에서 phase calibration JSON을 자동 로드할 수 있도록 옵션을 추가하였다.

추가 옵션은 다음과 같다.

```text
--aoa-phase-calibration-json
--aoa-gain-phase-table
```

이를 통해 viewer 실행 시 `--aoa-phase-offset-rad`를 손으로 입력하지 않고 저장된 phase calibration 값을 적용할 수 있다.

---

## 17. 주요 실행 명령어 요약

명령어는 별도 파일인 `2026-05-31_rf_drone_command_cheatsheet.md`에 분리하여 정리하였다. 보고서 본문에는 전체 흐름만 간단히 남긴다.

```text
1. 모듈 문법 검사
2. robust phase calibration 실행
3. current_phase_offset.json 로드 테스트
4. gain phase table 제작
5. run_pipeline.py 적용 확인
6. live_siggen_aoa_viewer.py 실행
```

---

## 18. 현재 한계 및 주의사항

### 18.1 Phase calibration 값의 유효 범위

phase calibration 값은 다음 조건에서만 유효하다.

```text
같은 SDR
같은 RX0/RX1 포트
같은 안테나
같은 케이블
같은 안테나 간격
같은 center frequency
같은 sample rate / RF bandwidth
같은 gain 또는 gain table 보정 적용
같은 실험 세션 또는 동일한 초기화 조건
```

SDR 재부팅, center frequency 변경, gain 변경, 케이블 교체, 안테나 간격 변경이 발생하면 재측정이 필요하다.

### 18.2 Gain table의 한계

gain phase table은 모든 환경에서 절대 보정을 보장하지 않는다. 실외에서 직접파가 우세한 조건에서 측정한 gain별 상대 phase 변화량을 저장한 것이다. 실내처럼 멀티패스가 강한 환경에서는 gain 변경에 따른 phase 변화가 테이블과 다르게 나타날 수 있다.

따라서 gain table은 다음과 같이 사용하는 것이 적절하다.

```text
현장 시작 시 reference gain에서 현재 세션 calibration 수행
→ gain table은 reference gain 대비 delta만 제공
→ 실시간 gain 변경 시 보정값 보조로 사용
```

### 18.3 실내 AoA의 한계

실내에서는 벽, 바닥, 천장, 사람, 장비 등에 의한 반사파가 많기 때문에 AoA 결과가 직접파 기준 각도와 다르게 나올 수 있다. 따라서 실내 AoA는 절대각 추정보다는 알고리즘 디버깅 및 상대 변화 확인용으로 보는 것이 적절하다.

### 18.4 CNN 모델 해석 문제

오늘 확인한 것처럼 기존 CNN 모델은 조종기 단독 신호에도 Drone으로 반응할 수 있다. 따라서 현재 모델 결과를 드론 기체 신호 탐지로 단정하기 어렵다. 향후 controller-only 데이터를 hard negative 또는 별도 class로 추가한 재학습이 필요하다.

---

## 19. 다음 작업 계획

다음 작업은 다음 순서로 진행하는 것이 적절하다.

```text
1. 오늘 수정한 파일 전체 py_compile 검사
2. current_phase_offset.json 로드 테스트
3. robust phase calibration 200 block 방식 재실험
4. run_pipeline.py summary.json 확인
5. live_siggen_aoa_viewer.py phase calibration 자동 로드 확인
6. 실외에서 gain phase table 제작
7. gain 25/30/35에 대한 phase delta table 검증
8. gain 변경 시 AoA 보정값이 정상 갱신되는지 확인
9. 신호발생기 0도/좌/우 각도별 AoA 테스트
10. controller-only hard negative 데이터 수집
11. 드론 ON linked / 드론 OFF controller-only 조건 A-B-A 수집
12. CNN 재학습 또는 3-class 구조 검토
```

---

## 20. 통합 정리

오늘 작업은 단순한 viewer 기능 추가를 넘어, RF 드론 탐지 시스템의 실험 운영 구조를 개선하는 방향으로 진행되었다. 초반에는 gain 실시간 조절, distance/memo 입력, pause/resume, target feature 비교, 경량 spectrogram viewer 등 현장 실험 편의 기능을 강화하였다.

이후 조종기 단독 신호가 CNN에서 Drone으로 탐지되는 문제를 확인하면서, 현재 모델이 드론 기체 신호만을 학습했다고 보기 어렵다는 중요한 한계를 발견하였다. 이 문제는 향후 controller-only 데이터를 hard negative로 추가하거나, Background / Controller-only / Drone-linked의 3-class 구조로 재설계하는 방향으로 개선할 수 있다.

AoA 측면에서는 dual RX phase difference 기반 각도 추정 기능을 추가하였고, 신호발생기 기반 검증을 위해 CNN gate와 분리된 전용 viewer를 구성하였다. 추가 실험 과정에서 phase calibration 값이 단순 평균 방식으로는 멀티패스, Wi-Fi 간섭, gain 변경에 취약하다는 점을 확인하였다.

이에 따라 robust phase calibration 구조를 설계하고 구현하였다. 핵심은 200 block 수집, 앞 30 block discard, coherence filtering, dominant phase cluster 선택, circular weighted mean, quality 판정이다. 또한 gain 변경에 대응하기 위해 실외 사전 gain-dependent phase delta table을 구축하고, 현장 시작 시 reference gain에서 측정한 current phase offset에 gain별 delta를 더해 적용하는 구조를 설계하였다.

최종적으로 `current_phase_offset.json`과 `gain_phase_table_2450.json`을 runtime에서 로드하여 AoA 계산 전 RX1 채널에 phase correction을 적용하는 구조까지 추가하였다. 이로써 향후 실외 AoA 실험에서 phase calibration, gain 변경, quality 평가, runtime 보정 적용을 일관된 흐름으로 수행할 수 있는 기반을 마련하였다.
