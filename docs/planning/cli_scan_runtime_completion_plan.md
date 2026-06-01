# CLI Scan Runtime 완성 계획서

## 1. 목적

본 계획서는 RF 드론 탐지 캡스톤 프로젝트에서 남은 핵심 작업을 정리하기 위한 문서이다.

현재 프로젝트의 중심은 live viewer가 아니라 **CLI 기반 scan/runtime pipeline**이다.  
Live viewer는 RF 패턴 확인, gain profile 저장, CNN/AoA 디버깅을 위한 보조 실험 도구로 사용한다.

최종 목표는 다음과 같다.

```text
Noise Calibration
→ Phase Calibration
→ Gain Phase Table 제작
→ Gain Feature Profile 제작
→ CLI Scan Mode
→ 후보 대역 Precision Mode
→ CNN Voting
→ Confirmed Uplink-side RF Activity
→ Coherence 기반 AoA 계산
→ Sector 출력 및 결과 저장
```

---

## 2. 현재 정리된 운영 정책

### 2.1 본 실행 흐름

```text
[Before Runtime]
1. Noise Calibration
2. Robust Phase Calibration
3. Gain-dependent Phase Table 제작
4. Gain Feature Profile 제작

[Runtime]
1. Scan Mode
2. Candidate Band Detection
3. Precision Mode
4. STFT + CNN Voting
5. Confirmed 판단
6. Coherence Check
7. AoA Angle 계산
8. Sector 출력
9. Signal lost 또는 coherence timeout 시 Scan Mode 복귀
```

---

### 2.2 Scan Mode 정책

Scan mode는 2.4GHz 대역을 sweep하면서 후보 신호가 존재하는 center frequency를 찾는 단계이다.

현재 설정 정책은 다음과 같다.

```text
대역 변경
→ 4 block warmup/discard
→ 4 block scan
→ 2 block 이상 threshold 통과 시 candidate band
```

설정 예시는 다음과 같다.

```yaml
# configs/receiver.yaml
warmup_reads: 4
retune_warmup_reads: 4

# configs/scan.yaml
scan_blocks: 4
min_pass_blocks: 2
```

현재 단계에서는 scan 속도보다 retune 직후 안정성을 우선한다.

---

### 2.3 Gain 운용 정책

Scan mode는 낮은 gain에서 시작하지 않는다.  
멀리 있거나 약한 조종기 기원 uplink-side RF activity를 놓치지 않기 위해 **비교적 강한 gain**에서 시작한다.

단, 무조건 최대 gain을 사용하는 것은 아니다. 후보 신호가 들어온 뒤 raw feature를 보고 너무 강하면 gain을 낮춘다.

```text
Scan 시작:
- 낮은 gain X
- 무조건 max gain X
- 실험적으로 안전한 비교적 강한 gain 사용

후보 신호 발견 후:
- raw_abs_max 확인
- raw_abs_p99 확인
- overload flag 확인
- 너무 강하면 gain table step 단위로 gain down
```

Gain은 가능하면 사전에 calibration과 profile이 준비된 table 값만 사용한다.

예시:

```text
사용 gain 후보:
25 dB / 30 dB / 35 dB
```

Gain down은 임의의 1 dB 단위가 아니라, 사전에 준비된 table step 단위로 수행한다.

```text
35 → 30
30 → 25
25 → 20
```

이 정책의 목적은 runtime 신호를 CNN이 학습한 1~2m 수집 조건의 raw feature 분포와 비슷한 영역으로 맞추는 것이다.

---

### 2.4 Raw Feature 계산 위치

각 IQ block은 다음 순서로 처리한다.

```text
RX0/RX1 IQ 수신
→ raw IQ safety check
→ DC Offset Removal
→ gain-distance feature 계산
→ scan / CNN / AoA 처리
```

구분은 다음과 같다.

```text
clip / overload 판단:
- SDR raw IQ 기준
- raw_abs_max
- clip_count
- clip_ratio
- overload flag

gain-distance feature:
- DC offset 제거 후
- amplitude normalization 전 IQ 기준
- raw_abs_p99
- raw_rms
- frame_power_p99
```

CNN 입력용 normalization 이후 값으로 절대세기 feature를 계산하지 않는다.

---

### 2.5 Precision Mode 정책

Scan mode에서 후보 대역이 발견되면 해당 center frequency에 머물러 Precision Mode를 수행한다.

```text
Candidate center frequency 고정
→ IQ block 수집
→ raw safety check
→ DC offset 제거
→ STFT spectrogram 생성
→ CNN inference
→ CNN voting
```

예시 정책:

```text
precision_blocks = 5
confirm_votes = 3

5 block 중 3 block 이상 positive
→ Confirmed Uplink-side RF Activity

5 block 중 2 block 이하 positive
→ candidate rejected
→ Scan Mode 복귀
```

정확한 block 수와 voting 기준은 실험 후 조정한다.

---

### 2.6 AoA / Track Mode 정책

AoA는 항상 수행되는 branch가 아니다.  
AoA는 Precision Mode에서 CNN voting을 통과한 후보에 대해서만 수행한다.

```text
CNN voting confirmed
→ current_phase_offset.json 로드
→ gain_phase_table_2450.json 조회
→ 현재 gain에 맞는 phase_offset_to_apply 계산
→ RX1 위상 오프셋 보상
→ coherence check
→ phase difference estimation
→ AoA angle 계산
→ sector 출력
```

Coherence가 낮으면 즉시 포기하지 않고 일정 block 동안 hold한다.

```text
coherence low
→ hold block 동안 추가 관찰
→ coherence 회복 시 AoA 계속
→ 계속 낮으면 Scan Mode 복귀
```

초기 정책에서는 coherence low 상황에서 자동 gain up을 본 정책에 넣지 않는다.  
이는 future adaptive gain policy로 남긴다.

---

### 2.7 Signal Lost 정책

Precision 또는 Track/AoA mode에서 신호가 잠깐 사라질 수 있으므로 즉시 scan mode로 복귀하지 않는다.

```text
signal lost
→ lost_count 증가
→ hold block 동안 같은 center frequency 유지
→ 신호 회복 시 lost_count reset
→ 계속 lost 상태면 Scan Mode 복귀
```

예시:

```text
lost_hold_blocks = 10
```

---

## 3. 남은 핵심 작업

## Phase 1. 실험 기준값 제작

### 3.1 Noise Calibration

목적:

```text
- scan threshold 기준 확보
- background noise floor 확인
- scan mode 오탐/미탐 조정
```

결과물:

```text
noise calibration result
scan threshold 후보값
```

체크 사항:

```text
- 실내/실외 noise floor 차이
- Wi-Fi 주변 환경 영향
- threshold가 너무 낮아 candidate가 과다 발생하지 않는지
- threshold가 너무 높아 약한 신호를 놓치지 않는지
```

---

### 3.2 Robust Phase Calibration

목적:

```text
- reference gain에서 RX0/RX1 phase offset 측정
- current_phase_offset.json 생성
```

권장 조건:

```text
center frequency: 2450 MHz
signal generator: 2452 MHz
reference gain: 30 dB
total blocks: 200
discard blocks: 30
```

결과물:

```text
configs/calibration/current_phase_offset.json
```

품질 기준:

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

---

### 3.3 Gain-dependent Phase Table 제작

목적:

```text
- gain 변경 시 phase offset 변화량 보정
- current_phase_offset + gain_delta(current_gain) 구조 확보
```

권장 gain 후보:

```text
25 dB
30 dB
35 dB
```

필요하면 추후 확장:

```text
20 dB
25 dB
30 dB
35 dB
40 dB
```

결과물:

```text
configs/calibration/gain_phase_table_2450.json
```

주의:

```text
- phase table은 절대 phase offset 표가 아니다.
- reference gain 대비 delta table이다.
- 현장 시작 시 current_phase_offset.json을 먼저 만들고,
  runtime에서는 여기에 gain delta를 더해 사용한다.
```

---

### 3.4 Gain Feature Profile 제작

목적:

```text
- gain별 정상 raw feature 범위 확보
- 후보 신호가 너무 강한지/약한지 판단
- CNN 학습 조건과 runtime 수신 세기를 맞추기 위한 기준 확보
```

저장 대상 feature:

```text
raw_abs_p99
raw_rms
frame_power_p99
raw_abs_max 또는 overload flag
```

초기 gain 판단에 직접 쓰는 feature는 너무 많게 잡지 않는다.

```text
초기 판단 핵심:
1. raw_abs_p99
2. raw_rms
3. overload flag 또는 raw_abs_max
```

결과물:

```text
outputs/viewer/gain_feature_profiles.csv
outputs/viewer/gain_feature_profiles_latest.json
```

역할:

```text
CSV:
- 실험 기록
- 사후 분석
- 거리/gain/feature 관계 확인

JSON:
- CLI runtime과 viewer가 함께 참조하는 최신 gain profile table
```

---

## 4. Pipeline 연결 작업

### 4.1 Scan Runtime 연결

구현 목표:

```text
- 4 block discard
- 4 block scan
- 2-of-4 candidate 판단
- candidate 없으면 scan 지속
- candidate 있으면 precision mode 진입
```

체크 포인트:

```text
- retune 후 첫 측정이 튀지 않는지
- candidate가 너무 많이 뜨지 않는지
- candidate가 아예 안 뜨지 않는지
- sweep 속도가 실험에 충분한지
```

---

### 4.2 Precision CNN Voting 연결

구현 목표:

```text
- candidate center frequency 고정
- STFT spectrogram 생성
- CNN inference
- block 단위 positive vote 계산
- voting 기준 통과 시 confirmed
- 실패 시 scan mode 복귀
```

추천 초기값:

```text
precision_blocks = 5
confirm_votes = 3
```

추후 RF burst 특성에 따라 다음도 가능하다.

```text
precision_blocks = 3
confirm_votes = 2
```

---

### 4.3 AoA Runtime 연결

구현 목표:

```text
- confirmed 후보에 대해서만 AoA 수행
- current_phase_offset.json 로드
- gain_phase_table_2450.json 로드
- current gain 기준 phase offset 적용
- coherence 계산
- phase_diff 계산
- angle_deg 계산
```

주의:

```text
- AoA는 CNN voting 전에는 수행하지 않는다.
- coherence가 낮으면 angle을 확정하지 않는다.
- phase table에 없는 gain은 최종 AoA 기록용으로 사용하지 않는다.
```

---

### 4.4 Sector 출력 연결

처음부터 무거운 UI를 만들지 않는다.  
초기 구현은 문자열과 CSV 저장 중심으로 한다.

초기 sector 예시:

```text
LEFT_2
LEFT_1
CENTER
RIGHT_1
RIGHT_2
```

예시 기준:

```text
angle < -30°        → LEFT_2
-30° ~ -10°         → LEFT_1
-10° ~ +10°         → CENTER
+10° ~ +30°         → RIGHT_1
angle > +30°        → RIGHT_2
```

추후 확장:

```text
FAR_LEFT
LEFT_2
LEFT_1
CENTER
RIGHT_1
RIGHT_2
FAR_RIGHT
```

Angle 안정화:

```text
- 최근 K개 angle 저장
- coherence threshold 이상인 block만 사용
- 단순 mean/median으로 초기 구현
- 추후 coherence-weighted circular smoothing 적용
- angle 분산이 크면 UNSTABLE 또는 HOLD 표시
```

초기 overlay 예시:

```text
AoA: +15.7 deg | Sector: RIGHT_1 | coh=0.74
```

이 정도는 OpenCV text overlay만으로 가능하므로 렉 부담이 작다.

---

### 4.5 Logging 연결

저장 대상:

```text
timestamp
mode
center_freq
gain
scan_score
candidate 여부
cnn_raw_prediction
cnn_confidence
cnn_vote_count
confirmed 여부
coherence
phase_diff_deg
angle_deg
sector
raw_abs_p99
raw_rms
overload flag
state transition
```

결과 파일 예시:

```text
outputs/runtime/scan_events.csv
outputs/runtime/precision_events.csv
outputs/runtime/track_events.csv
outputs/runtime/latest_summary.json
```

---

## 5. 실험 검증 순서

### Step 1. 설정 확인

```bash
git status
git diff configs/receiver.yaml configs/scan.yaml
```

확인할 값:

```text
warmup_reads = 4
retune_warmup_reads = 4
scan_blocks = 4
min_pass_blocks = 2
```

---

### Step 2. Noise Calibration

```bash
PYTHONPATH=. python scripts/run_noise_calibration.py
```

확인:

```text
noise floor
threshold
false candidate rate
```

---

### Step 3. Phase Calibration

```bash
PYTHONPATH=. python scripts/calibrate_phase_offset_outdoor.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --num-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20
```

확인:

```text
quality = OK 또는 최소 WARNING
phase_std
valid_blocks
cluster_ratio
coherence_median
```

---

### Step 4. Gain Phase Table

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
  --output configs/calibration/gain_phase_table_2450.json
```

---

### Step 5. Gain Feature Profile

Viewer 또는 전용 script로 각 gain에서 profile 저장.

```text
gain 25
gain 30
gain 35
```

각 gain에서 저장:

```text
raw_abs_p99
raw_rms
frame_power_p99
overload flag
```

---

### Step 6. Scan Mode 단독 검증

```bash
PYTHONPATH=. python scripts/run_scan.py
```

확인:

```text
- scan loop 정상 동작
- retune 후 안정화
- candidate 발생 여부
- candidate 과다 여부
```

---

### Step 7. Precision Mode 검증

후보 대역에서 CNN voting이 기대대로 동작하는지 확인한다.

확인:

```text
- STFT shape 정상
- CNN confidence 정상
- voting 결과가 너무 흔들리지 않는지
- candidate rejected 시 scan 복귀
```

---

### Step 8. AoA / Track Mode 검증

신호발생기 또는 조종기 신호 기준으로 확인한다.

확인:

```text
- current_phase_offset 적용 여부
- gain phase table 적용 여부
- coherence
- angle_deg
- sector
- coherence low 시 hold 후 scan 복귀
```

---

## 6. 우선순위

### 1순위

```text
Gain phase table 제작
Gain feature profile 제작
Scan mode 4+4 정책 검증
```

### 2순위

```text
Precision CNN voting 연결
confirmed / rejected 상태 전이 구현
```

### 3순위

```text
AoA runtime 연결
coherence hold
signal lost hold
```

### 4순위

```text
sector 문자열 출력
CSV / JSON logging
```

### 5순위

```text
viewer overlay 개선
sector 그래픽 UI
adaptive gain up policy
```

---

## 7. 지금 당장 하지 않아도 되는 것

다음 기능은 후순위로 둔다.

```text
- 복잡한 sector 그래픽 UI
- matplotlib 기반 실시간 방향판
- coherence low에서 자동 gain up
- 1 dB 단위 adaptive gain control
- 7-sector 정밀 분류
- circular smoothing 고도화
```

초기에는 다음만 수행한다.

```text
- 5-sector 문자열 출력
- angle_deg 저장
- coherence 저장
- sector 저장
```

---

## 8. 최종 완료 기준

본 단계의 완료 기준은 다음과 같다.

```text
1. Noise calibration 결과 확보
2. current_phase_offset.json 확보
3. gain_phase_table_2450.json 확보
4. gain_feature_profiles_latest.json 확보
5. scan mode에서 candidate band 탐색 가능
6. candidate band에서 CNN voting 가능
7. voting 통과 시 AoA 계산 가능
8. coherence low 또는 signal lost 시 scan mode 복귀 가능
9. angle_deg와 sector를 CSV/JSON에 저장 가능
10. viewer는 보조 도구로 angle/sector 문자열만 표시 가능
```

---

## 9. 한 줄 요약

```text
실험 기준값 두 개,
즉 gain phase table과 gain feature profile을 만든 뒤,
CLI scan/runtime 상태머신에 CNN voting과 AoA sector 출력을 연결하면
본 파이프라인의 핵심 준비는 끝난다.
```
