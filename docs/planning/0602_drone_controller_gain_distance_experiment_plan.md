# 드론 조종기 기반 거리–Gain 실험 계획서

## 1. 실험 목적

신호발생기가 아니라 **실제 드론 조종기 RF 신호**를 기준으로, 거리와 gain 변화에 따른 수신 상태를 확인한다.

핵심 목표:

```text
2m / gain30 / 드론 조종기 신호를 reference 조건으로 잡고,
gain30 → gain35 → gain40 → gain45 순서로 올리면서
각 gain에서 포화되지 않는 최소 거리와 최대 usable 거리를 찾는다.
```

이 실험은 거리 추정 공식을 만드는 것이 아니라, 실제 runtime에서 사용할 **gain 선택 정책**을 만드는 실험이다.

---

## 2. 기본 판단 기준

각 조건에서 다음 항목을 같이 본다.

```text
1. Raw feature
   - raw_abs_p99
   - raw_rms
   - raw_abs_max
   - frame_power_p99
   - overload flag

2. CNN
   - raw class
   - confidence
   - positive vote
   - confirmed 여부

3. AoA / coherence
   - coherence
   - coherence pass 여부
   - angle 안정성
```

Decision 기준:

| Decision | 의미 | 판단 기준 |
|---|---|---|
| reference | 기준 조건 | 2m / gain30 / drone controller |
| too_strong | 너무 강함 / 포화 위험 | overload=True, raw_abs_max 과도, spectrogram pattern 뭉개짐 |
| good | 가장 적합 | overload=False, CNN confirmed 안정, coherence 높음 |
| usable | 사용 가능 | CNN은 대부분 잡힘, coherence가 중간 이상 |
| weak | 약함 | CNN은 가끔 잡히지만 coherence 낮음 또는 raw feature 낮음 |
| lost | 탐지 한계 | CNN 불안정, coherence 낮음, 패턴 거의 안 보임 |

---

## 3. Reference 조건

가장 먼저 아래 조건을 저장한다.

```text
distance = 2.0 m
gain = 30 dB
source = drone controller
```

이 조건은 CNN 학습 조건에 가장 가까운 기준이다.  
이후 거리/gain 조건의 raw feature와 CNN/coherence 상태를 이 기준과 비교한다.

---

## 4. 전체 실험 절차

### Step 1. 주변 환경 정리

```text
[ ] 핫스팟 OFF
[ ] 블루투스 가능하면 OFF
[ ] 주변 Wi-Fi 간섭 최소화
[ ] 사람/폰/노트북은 RX 안테나 전방 경로에서 멀리
[ ] 드론 조종기 위치와 방향 고정
[ ] RX0/RX1 안테나 위치와 간격 고정
[ ] gain 변경 후 3~5초 안정화 후 기록
```

### Step 2. gain30 reference 측정

```text
2m / gain30
```

에서 full viewer 실행 후 `s` 키를 눌러 profile을 저장한다.

### Step 3. gain30 거리 한계 찾기

```text
gain = 30
distance = 2m → 3m → 4m → 5m ...
```

다음을 찾는다.

```text
- good 유지 거리
- usable 유지 거리
- weak/lost 시작 거리
```

### Step 4. gain35로 올리고 반복

gain30에서 약해진 거리 근처부터 시작한다.

```text
1. gain30에서 마지막 usable이었던 거리 근처에서 시작
2. 너무 강하면 뒤로 이동
3. 포화가 사라지는 최소 거리 기록
4. 멀리 이동하면서 good / usable / weak / lost 지점 기록
```

### Step 5. gain40, gain45 반복

gain45는 가까운 거리에서 포화될 수 있으므로 gain40의 usable 거리 근처 또는 그보다 약간 먼 곳에서 시작한다.

---

## 5. 추천 거리/Gain 조합

| Gain | 추천 시작 거리 | 측정 후보 거리 |
|---:|---:|---|
| 30 | 2m | 2m, 3m, 4m, 5m |
| 35 | 3m | 3m, 4m, 5m, 6m |
| 40 | 4m | 4m, 5m, 6m, 7m |
| 45 | 5m | 5m, 6m, 7m 이상 가능 시 |

운영 방식:

```text
gain30에서 멀리 보냄
→ 약해지는 거리 확인
→ 그 근처에서 gain35 시작
→ 포화되면 뒤로 이동
→ 다시 멀리 보냄
→ gain40, gain45 반복
```

---

## 6. 공통 실행 명령어

아래 명령어에서 바꿀 부분은 두 개다.

```text
--gain <GAIN>
--distance-m <DISTANCE>
--memo "dist<DISTANCE>_gain<GAIN>_drone_controller"
```

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain <GAIN> \
  --distance-m <DISTANCE> \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "dist<DISTANCE>_gain<GAIN>_drone_controller"
```

실행 후:

```text
1. 화면 안정 확인
2. s 키 누르기
3. 50 block profile 저장 대기
4. q 키로 종료
5. 다음 거리/gain 조건 실행
```

---

## 7. 바로 복붙용 명령어 예시

### 2m / gain30 reference

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --distance-m 2.0 \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "ref_dist2p0_gain30_drone_controller"
```

### 3m / gain30

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --distance-m 3.0 \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "dist3p0_gain30_drone_controller"
```

### 4m / gain35

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 35 \
  --distance-m 4.0 \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "dist4p0_gain35_drone_controller"
```

### 5m / gain40

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 40 \
  --distance-m 5.0 \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "dist5p0_gain40_drone_controller"
```

### 6m / gain45

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --cnn-backend torch \
  --model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --cnn-device cpu \
  --class-names "Background,Drone-like" \
  --cnn-positive-class-names "Drone-like,Drone,drone" \
  --cnn-confidence-threshold 0.5 \
  --cnn-smooth-window 5 \
  --cnn-confirm-votes 3 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 45 \
  --distance-m 6.0 \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --aoa-antenna-spacing-m 0.0625 \
  --aoa-coherence-threshold 0.6 \
  --profile-blocks 50 \
  --log-csv outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --log-every-n 1 \
  --display-scale 2 \
  --nperseg 128 \
  --noverlap 96 \
  --nfft 128 \
  --memo "dist6p0_gain45_drone_controller"
```

---

## 8. 현장 기록표 양식


---
PYTHONPATH=. python scripts/summarize_gain_distance_sweep.py \
  --input outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --output-csv outputs/viewer/drone_controller_gain_distance_summary.csv \
  --output-md docs/experiments/drone_controller_gain_distance_summary.md \
  --reference-distance 2.0 \
  --reference-gain 30

outputs/viewer/drone_controller_gain_distance_summary.csv
docs/experiments/drone_controller_gain_distance_summary.md


---

| No. | Distance (m) | Gain (dB) | Source | raw_abs_p99 | raw_rms | raw_abs_max | Overload | CNN raw class | CNN conf | CNN votes | Confirmed | Coherence | AoA angle | Decision | Memo |
|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---|---:|---:|---|---|
| 1 | 2.0 | 30 | Drone controller |  |  |  |  |  |  |  |  |  |  | reference | 2m gain30 기준 |
| 2 | 3.0 | 30 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 | 4.0 | 30 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 | 3.0 | 35 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 | 4.0 | 35 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | 5.0 | 35 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 | 4.0 | 40 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 | 5.0 | 40 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 9 | 6.0 | 40 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | 5.0 | 45 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | 6.0 | 45 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |
| 12 | 7.0 | 45 | Drone controller |  |  |  |  |  |  |  |  |  |  |  |  |

---

## 9. 최종 요약표 양식

| Gain (dB) | Too Strong Range | Good Range | Usable Range | Weak/Lost Range | Runtime Memo |
|---:|---|---|---|---|---|
| 30 |  | 2~3m | 4m | 5m 이상 | reference gain |
| 35 |  | 3~4m | 5m | 6m 이상 | mid gain |
| 40 |  | 4~5m | 6m | 7m 이상 | scan/high gain |
| 45 | 가까운 거리 포화 가능 | 5~6m | 7m |  | far gain 후보 |

최종 runtime 정책 예시:

```text
기본 scan gain = 40 또는 45
후보 신호 발견 후 overload=True면 gain down
raw feature가 2m gain30 reference보다 과도하면 gain down
CNN confirmed + coherence good이면 AoA 수행
CNN confirmed지만 coherence 낮으면 hold
CNN 불안정하면 scan 복귀
```

---

## 10. 내일 실험 체크리스트

```text
[ ] 핫스팟 OFF
[ ] 블루투스 OFF
[ ] 드론 조종기 위치/방향 고정
[ ] RX 안테나 위치 고정
[ ] current_phase_offset 또는 current_session_phase_table 준비
[ ] gain_phase_table_2450.json 준비
[ ] 2m gain30 reference 먼저 저장
[ ] 각 조건 실행 후 s 눌러 profile 저장
[ ] 기록표에 decision 작성
[ ] 포화 시작 거리와 lost 시작 거리 반드시 마크
```
