# 2026-06-01 RF Drone Detection 다음 실험 계획 보고서

## 1. 실험 목적

본 실험의 목적은 2026-05-31에 개발한 실시간 viewer 개선 기능과 AoA phase calibration 구조를 실제 실외 환경에서 검증하는 것이다. 특히 신호발생기를 이용해 RX0/RX1 채널 간 phase offset calibration이 안정적으로 동작하는지 확인하고, gain 변경에 따른 phase offset 변화량을 gain-dependent phase table로 저장하는 것을 목표로 한다.

또한 기존 CNN 모델이 조종기 단독 신호를 Drone으로 오탐하는 문제를 확인했기 때문에, 조종기 단독 신호와 드론-linked 신호의 spectrogram 차이를 비교하여 향후 데이터셋 재구성 방향을 결정한다.

---

## 2. 실험 핵심 목표

내일 실험의 핵심 목표는 다음 세 가지이다.

```text
1. robust phase calibration이 실외에서 quality OK로 나오는지 확인
2. gain 25 / 30 / 35에 대한 phase delta table 제작
3. 신호발생기 AoA 0도 및 좌우 방향성이 정상적으로 나오는지 검증
```

여유가 있을 경우 다음 항목까지 진행한다.

```text
4. controller-only와 drone-linked spectrogram 비교
5. controller-only hard negative 데이터 수집
```

---

## 3. 실험 준비 조건

### 3.1 장비

```text
- Pluto SDR 또는 현재 사용 중인 dual RX SDR
- RX0/RX1 안테나 2개
- 동일한 안테나 간격 유지용 지그 또는 고정 구조
- 신호발생기
- 노트북
- 드론 및 조종기
- 전원 보조 장치
- 줄자 또는 거리 측정 도구
```

### 3.2 RF 조건

기준 세팅은 다음과 같이 설정한다.

```text
SDR center frequency : 2450 MHz
신호발생기 주파수    : 2452 MHz
sample rate          : 5 MS/s
reference gain       : 30
신호원-RX 거리       : 1.5 ~ 2 m
calibration block    : 총 200 blocks
discard block        : 앞 30 blocks
실제 사용 block      : 170 blocks
```

신호발생기를 SDR center와 완전히 같은 2450 MHz에 두지 않고 2452 MHz로 설정하는 이유는 DC offset 및 LO leakage 근처를 피하고, baseband에서 약 +2 MHz tone으로 수신하기 위함이다.

### 3.3 장소 조건

가능하면 다음 조건을 만족하는 장소에서 수행한다.

```text
- 건물, 차량, 철제 난간 등 큰 반사체가 가까이 없는 개방된 장소
- RX 주변 5~10 m 이내 금속 구조물 최소화
- 핫스팟, 블루투스, 불필요한 2.4 GHz 장치 OFF
- 사람과 스마트폰은 RX 주변에서 멀리 두기
- 신호발생기와 RX 안테나는 비슷한 높이로 배치
```

단, 실제 야외에서는 2.4 GHz Wi-Fi가 완전히 없는 장소를 찾기 어렵기 때문에, 목표는 Wi-Fi가 없는 장소를 찾는 것이 아니라 신호발생기 tone이 주변 RF보다 충분히 우세하게 들어오도록 만드는 것이다.

---

## 4. 실험 전 코드 검증

실험 전에는 먼저 오늘 수정한 파일들이 정상적으로 import 및 compile 되는지 확인한다.

```bash
python -m py_compile src/calibration/gain_phase_table.py
python -m py_compile src/runtime/phase_calibration_runtime.py
python -m py_compile scripts/calibrate_phase_offset_outdoor.py
python -m py_compile scripts/build_gain_phase_table.py
python -m py_compile scripts/run_pipeline.py
python -m py_compile scripts/live_siggen_aoa_viewer.py
```

현재 phase calibration 파일이 정상적으로 로드되는지도 확인한다.

```bash
PYTHONPATH=. python - <<'PY'
from src.runtime import resolve_phase_offset_to_apply, print_phase_calibration_state

state = resolve_phase_offset_to_apply(
    current_phase_path="configs/calibration/current_phase_offset.json",
)
print_phase_calibration_state(state)
PY
```

정상이라면 `apply_offset`, `quality`, `source` 값이 출력되어야 한다.

---

## 5. 1단계: 실외 Robust Phase Calibration

### 5.1 목적

실험 시작 시 reference gain에서 현재 세션의 RX0/RX1 phase offset을 측정한다. 이 값은 해당 세션에서 AoA 계산 전 RX1 채널에 적용할 기준 phase offset이다.

### 5.2 실행 명령어

```bash
PYTHONPATH=. python scripts/calibrate_phase_offset_outdoor.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --num-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --memo "outdoor_0deg_1p5m_gain30_sig2452_robust"
```

### 5.3 성공 기준

성공 기준은 다음과 같다.

```text
quality           : OK
phase_std_deg     : 3도 이하
coherence_median  : 0.70 이상
cluster_ratio     : 0.80 이상
```

오늘 측정에서 안정적으로 나온 값은 약 +44도 근처였지만, 내일 실험에서 반드시 같은 값이 나와야 하는 것은 아니다. SDR 세션, 초기화 상태, 현장 배치에 따라 절대 phase offset은 달라질 수 있다. 중요한 것은 quality가 OK이고 phase 분포가 하나의 dominant cluster로 안정되는 것이다.

---

## 6. 2단계: Gain-dependent Phase Table 제작

### 6.1 목적

gain을 변경할 때마다 신호발생기를 다시 0도에 놓고 calibration할 수 없으므로, 실외에서 gain별 phase offset 변화량을 사전에 측정한다.

이때 저장하는 값은 gain별 절대 phase offset이 아니라 reference gain=30 대비 상대 변화량이다.

### 6.2 1차 gain table 제작

처음에는 gain 25, 30, 35만 측정한다.

```bash
PYTHONPATH=. python scripts/build_gain_phase_table.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 25,30,35 \
  --reference-gain 30 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output configs/calibration/gain_phase_table_2450.json \
  --memo "outdoor_0deg_1p5m_sig2452_gain_phase_table"
```

### 6.3 확인할 값

각 gain에 대해 다음 값을 확인한다.

```text
phase_deg
phase_delta_deg
phase_std_deg
coherence_median
cluster_ratio
quality
```

gain=30의 delta는 reference gain이므로 0도여야 한다.

```text
gain=25 → phase_delta_deg 확인
gain=30 → delta = 0
gain=35 → phase_delta_deg 확인
```

모든 gain이 OK이면 gain table을 실시간 AoA 보정에 사용할 수 있다. 특정 gain이 WARNING 또는 FAIL이면 해당 gain은 AoA 측정용으로 사용하지 않거나 재측정한다.

---

## 7. 3단계: AoA 0도 검증

### 7.1 목적

phase calibration을 적용했을 때 정면 0도에서 AoA가 0도 근처로 나오는지 확인한다.

### 7.2 실행 명령어

```bash
PYTHONPATH=. python scripts/live_siggen_aoa_viewer.py \
  --enable-aoa \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --gain 30
```

### 7.3 성공 기준

```text
정면 0도에서 AoA가 0도 근처
coherence가 안정적으로 높음
AoA smoothing 값이 크게 튀지 않음
angle_valid가 True 또는 사용 가능 상태
```

정면 0도에서도 AoA가 크게 벗어나면 다음 가능성을 점검한다.

```text
- phase offset 적용 부호가 반대일 가능성
- RX0/RX1 채널 순서가 바뀐 가능성
- 안테나 정렬 오차
- 신호원 위치가 RX0/RX1 중앙 정면이 아님
- phase calibration quality가 낮음
```

---

## 8. 4단계: 좌우 각도 검증

### 8.1 목적

AoA 절대 정확도보다 먼저 방향성이 맞는지 검증한다.

### 8.2 측정 각도

다음 순서로 신호발생기를 이동한다.

```text
0도
+10도
-10도
+20도
-20도
```

각 위치에서 다음 값을 기록한다.

```text
실제 배치 각도
AoA 출력 angle_deg
smoothed AoA
coherence
angle_valid
gain
phase_offset_to_apply_deg
memo
```

### 8.3 성공 기준

```text
오른쪽 이동 시 AoA 부호가 한쪽 방향으로 변화
왼쪽 이동 시 반대 방향으로 변화
각도 변화가 부드럽게 따라감
0도 복귀 시 다시 0도 근처로 돌아옴
```

절대 각도 오차가 있더라도 방향성이 일관되면 1차 검증은 성공으로 본다. 각도 크기 오차는 안테나 간격, phase wrapping, 멀티패스, 신호원 위치 오차 등을 통해 추가 보정한다.

---

## 9. 5단계: Gain 변경 시 AoA 유지 검증

### 9.1 목적

gain phase table을 적용했을 때 gain을 변경해도 0도 AoA가 크게 흔들리지 않는지 확인한다.

### 9.2 절차

신호발생기는 정면 0도에 고정한다.

```text
gain 25 측정
gain 30 측정
gain 35 측정
```

각 gain에서 AoA가 0도 근처로 유지되는지 확인한다.

### 9.3 성공 기준

```text
gain 25 → AoA 0도 근처
gain 30 → AoA 0도 근처
gain 35 → AoA 0도 근처
```

gain 변경 시 AoA가 크게 튀면 다음 원인을 의심한다.

```text
- gain table delta가 불안정
- 해당 gain의 phase calibration quality가 낮음
- gain 변경 후 warmup 부족
- 실내/현장 멀티패스 영향
- gain 변경 시 SDR 내부 상태가 크게 바뀜
```

---

## 10. 6단계: Controller-only vs Drone-linked 비교

### 10.1 목적

기존 CNN 모델이 조종기 단독 신호를 Drone으로 오탐하는 문제가 있었기 때문에, 조종기 단독 조건과 드론 linked 조건의 spectrogram 차이를 직접 확인한다.

### 10.2 비교 조건

가능하면 같은 위치와 gain에서 다음 조건을 A-B-C-B-A 방식으로 측정한다.

```text
A: 드론 OFF + 조종기 OFF
B: 드론 OFF + 조종기 ON
C: 드론 ON + 조종기 ON + linked
B: 드론 OFF + 조종기 ON
A: 드론 OFF + 조종기 OFF
```

### 10.3 확인할 내용

```text
controller-only 신호 패턴
drone-linked 신호 패턴
두 조건의 burst 위치/폭/반복성 차이
CNN raw/final decision 변화
signal_ratio 변화
spectrogram 시각적 차이
```

### 10.4 판단 기준

```text
차이가 명확함:
→ controller-only를 hard negative로 추가하여 재학습 가능

차이가 거의 없음:
→ 현재 안테나/거리/하드웨어 조건에서는 드론 기체 신호 분리가 어려울 수 있음
→ 안테나 배치, 조종기 위치, 드론 위치, 주파수 조건 재설계 필요
```

---

## 11. 실험 중 기록해야 할 값

실험 로그 또는 메모에 다음 값을 기록한다.

```text
날짜/장소
환경: 실외/실내, 주변 Wi-Fi 상태
center_freq
signal_freq
sample_rate
gain
거리
신호원 각도
phase_offset_deg
phase_std_deg
coherence_median
cluster_ratio
quality
AoA angle_deg
AoA smoothed angle
angle_valid
CNN raw decision
CNN final decision
signal_ratio
memo
```

---

## 12. 예상 결과 및 판단

### 12.1 성공적인 경우

```text
robust phase calibration quality OK
gain 25/30/35 table 모두 OK
0도 AoA가 0도 근처
좌우 이동 시 AoA 부호와 방향성이 일치
gain 변경 후에도 0도 AoA가 유지
controller-only와 drone-linked spectrogram 차이 일부 확인
```

이 경우 다음 단계는 controller-only hard negative 수집 및 CNN 재학습이다.

### 12.2 부분 성공

```text
phase calibration은 OK
하지만 AoA 각도 크기 오차가 큼
방향성은 맞음
```

이 경우 안테나 간격, phase 부호, phase-to-angle 변환식, smoothing window, coherence gate를 점검한다.

### 12.3 실패

```text
phase calibration quality FAIL
phase cluster가 여러 개로 갈라짐
coherence가 낮음
AoA가 무작위로 튐
```

이 경우 장소, 신호발생기 출력, 거리, Wi-Fi/핫스팟 간섭, 사람/장비 위치를 먼저 개선한다.

---

## 13. 내일 실험의 우선순위

시간이 부족하면 다음 순서만 수행한다.

```text
1순위: robust phase calibration quality OK 확인
2순위: gain 25/30/35 phase table 제작
3순위: 신호발생기 0도 AoA 검증
4순위: 좌우 방향성 검증
5순위: controller-only vs drone-linked 비교
```

내일 실험의 메인 제목은 다음과 같이 정리할 수 있다.

```text
실외 신호발생기 기반 AoA phase calibration 안정성 검증 및 gain-dependent phase table 구축
```

---

## 14. 최종 정리

내일 실험은 AoA 알고리즘의 정확도를 바로 완성하는 실험이 아니라, AoA 실험을 안정적으로 수행하기 위한 phase calibration 체계를 검증하는 단계이다. 특히 robust phase calibration과 gain-dependent phase delta table이 제대로 동작하면, 이후 드론 또는 조종기 신호에 대해 AoA를 적용할 수 있는 기반이 마련된다.

또한 controller-only 조건과 drone-linked 조건을 비교함으로써, 현재 CNN 모델의 오탐 원인이 조종기 신호인지, 드론-linked RF activity인지 구분할 수 있다. 이 결과는 향후 데이터셋 재구성 및 모델 재학습 방향을 결정하는 핵심 근거가 된다.
