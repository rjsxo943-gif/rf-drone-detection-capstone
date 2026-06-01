# 2026-06-01 RF 드론 탐지 실험 및 개발 보고서

## 1. 오늘 작업 개요

오늘 작업의 핵심은 **실시간 RF viewer 정리**, **STFT 표시 문제 수정**, **phase calibration 재현성 확인**, **gain phase table 제작**, 그리고 **내일 수행할 드론 조종기 기반 거리–gain 실험 계획 수립**이었다.

기존에는 신호발생기 기반 phase calibration과 gain phase table을 고정 보정값처럼 사용할 수 있을지 검토했으나, 실험 결과 gain 변화 자체보다는 receiver session 시작 시점과 주변 환경 변화가 phase offset에 더 큰 영향을 줄 수 있다는 점을 확인하였다.

따라서 최종 운영 방향은 다음과 같이 정리하였다.

```text
1. Runtime 시작 전 session phase calibration 수행
2. 사용할 gain 후보를 연속으로 calibration
3. Runtime 중 gain이 바뀌면 해당 gain의 phase offset을 적용
4. 거리별 gain 정책은 실제 드론 조종기 신호 기준으로 수립
5. 판단 기준은 raw feature + CNN confirmed + coherence를 함께 사용
```

---

## 2. 오늘 수정 및 확인한 주요 파일

## 2.1 `scripts/live_rf_viewer.py`

### 역할

OpenCV 기반 실시간 RF viewer 실행 스크립트이다.  
현재 프로젝트에서 현장 실험용 핵심 도구로 사용한다.

지원 모드:

```text
fast    : 스펙트로그램 빠른 확인
profile : raw feature profile 저장
cnn     : CNN inference 확인
aoa     : coherence / AoA 확인
full    : raw feature + CNN + AoA + profile + CSV log 통합
```

### 오늘 한 작업

기존 viewer의 STFT 기본 설정이 `nperseg=512`, `noverlap=384`, `nfft=512`로 되어 있어, 화면에 표시되는 spectrogram shape가 다음처럼 나왔다.

```text
raw shape = (512, 125)
```

하지만 현재 프로젝트에서 사용하려는 CNN/viewer 기준은 다음이다.

```text
frequency bins = 128
time bins      = 509
display shape  = (128, 509)
```

따라서 live viewer 기본 STFT 설정을 다음으로 맞추었다.

```text
nperseg = 128
noverlap = 96
nfft = 128
```

또한 shape 확인용 옵션을 추가하였다.

```text
--debug-shape
```

정상 출력 예시는 다음과 같다.

```text
VIEW SPEC SHAPE: raw= (128, 509) display= (128, 509)
```

### 사용 명령어

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode fast \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --display-scale 2 \
  --debug-shape
```

정상 확인 후에는 `--debug-shape`를 제거하고 사용한다.

---

## 2.2 `src/viewer/opencv_renderer.py`

### 역할

OpenCV 창에 spectrogram을 띄우는 renderer이다.

### 오늘 한 작업

화면이 너무 작게 보이는 문제를 해결하기 위해 display scale 기능을 적용하였다.

주요 기능:

```text
display_scale  : 화면 확대 배율
display_width  : 고정 표시 가로 크기
display_height : 고정 표시 세로 크기
```

중요한 정책:

```text
renderer는 spectrogram 축을 판단하거나 transpose하지 않는다.
축 보정은 live_rf_viewer.py에서 하고,
renderer는 받은 이미지를 그대로 확대해서 보여준다.
```

이렇게 정리한 이유는, frequency/time 축을 renderer가 임의로 판단하면 화면 방향이 다시 꼬일 수 있기 때문이다.

---

## 2.3 `scripts/build_gain_phase_table.py`

### 역할

gain별 phase offset을 측정하고 gain phase table을 생성하는 스크립트이다.

### 오늘 논의한 핵심

초기에는 다음 방식으로 생각했다.

```text
reference gain에서 current phase offset 측정
+ gain phase delta table 적용
```

그러나 실험 결과 같은 receiver session 안에서 gain을 바꾸면 phase offset 변화가 매우 작았다.

오늘 생성한 gain phase table 결과:

```text
reference_gain = 40 dB

gain40: +139.337 deg | delta +0.000 deg | OK
gain35: +138.739 deg | delta -0.598 deg | OK
gain30: +139.630 deg | delta +0.293 deg | OK
gain25: +139.966 deg | delta +0.629 deg | WARNING
```

즉 gain 변화에 따른 phase offset 변화는 대략 ±1도 이내였다.

### 결론

우리가 처음 걱정했던 문제:

```text
gain이 바뀌면 SDR 내부 경로가 바뀌어서 phase offset이 크게 달라질 수 있음
```

현재 실험 결과:

```text
같은 receiver session 안에서는 gain을 바꿔도 phase offset 변화가 거의 없음
```

따라서 더 중요한 문제는 gain 변화 자체가 아니라:

```text
receiver 재초기화
runtime session 변경
주변 핫스팟/블루투스/반사체
실험 배치 변화
```

에 따른 absolute phase offset 변화이다.

---

## 2.4 `configs/calibration/gain_phase_table_2450.json`

### 역할

2450 MHz 중심 주파수에서 gain별 phase offset과 reference gain 대비 delta를 저장한 calibration 결과 파일이다.

### 오늘 생성한 값

```text
reference_gain = 40 dB
reference_phase_deg = +139.337 deg
```

gain별 delta:

```text
gain40: +0.000 deg
gain35: -0.598 deg
gain30: +0.293 deg
gain25: +0.629 deg
```

### 사용처

AoA 계산 시 gain 변화에 따른 phase 보정 참고값으로 사용할 수 있다.

다만 최종 운영 정책에서는 이 파일을 영구 보정값이라기보다 다음 의미로 해석한다.

```text
gain 변화 자체가 phase offset을 크게 바꾸지 않는다는 검증 결과
```

runtime에서는 가능하면 실행 시작 시 사용할 gain들을 연속으로 calibration하여 `current_session_phase_table.json` 형태로 저장하고, 현재 gain에 해당하는 phase offset을 직접 적용하는 구조가 더 안전하다.

---

## 2.5 `configs/calibration/current_phase_offset.json`

### 역할

현재 세션에서 측정된 phase offset을 저장하는 파일이다.

### 주의점

이 파일은 실험할 때마다 계속 덮어써진다.  
따라서 Git에 고정값으로 커밋하는 것은 추천하지 않는다.

권장 정책:

```text
current_phase_offset.json
→ 현장 세션용 임시 calibration 값
→ 매 실험 시작 전 새로 생성
→ Git 커밋 비추천
```

---

## 2.6 `outputs/viewer/drone_controller_gain_distance_sweep.csv`

### 역할

full mode 실행 중 자동으로 쌓이는 원본 로그 CSV이다.

### 저장 방식

full mode를 켜고 있으면 `--log-csv`에 지정한 파일로 update/block 단위 raw log가 계속 쌓인다.

이 파일은 중앙값이 아니다.

```text
full mode log CSV = block/update별 원자료
```

포함되는 주요 항목:

```text
distance_m
gain
memo
raw_abs_p99
raw_rms
raw_abs_max
frame_power_p99
overload
CNN result
CNN confidence
CNN confirmed
coherence
AoA angle
```

---

## 2.7 `outputs/viewer/gain_feature_profiles.csv`

### 역할

viewer에서 `s` 키를 눌렀을 때 저장되는 gain feature profile CSV이다.

### 사용 방식

full mode 또는 profile mode에서 `s` 키를 누르면 현재 조건의 profile이 저장된다.

```text
full mode 실행
→ 화면 안정 확인
→ s 키 누름
→ profile-blocks 개수만큼 raw feature 수집
→ gain_feature_profiles.csv에 저장
```

현재 목적은 거리/gain별 raw feature 기준을 쌓는 것이다.

---

## 2.8 `outputs/viewer/gain_feature_profiles_latest.json`

### 역할

최신 gain feature profile을 JSON 형태로 저장하는 파일이다.

### 사용처

나중에 runtime gain policy에서 현재 raw feature가 reference 조건보다 너무 큰지, 너무 작은지 판단하는 기준으로 사용할 수 있다.

예상 사용 방식:

```text
현재 raw_abs_p99가 reference보다 과도하게 큼
→ gain down

현재 raw_abs_p99가 너무 낮고 CNN/coherence도 약함
→ gain 유지 또는 scan 재시도

포화 없음 + CNN confirmed + coherence good
→ AoA 수행
```

---

## 2.9 `scripts/summarize_gain_distance_sweep.py`

### 역할

full mode에서 쌓인 원본 로그 CSV를 거리/gain/memo별로 묶어서 중앙값 요약표를 만드는 스크립트이다.

### 입력

```text
outputs/viewer/drone_controller_gain_distance_sweep.csv
```

### 출력

```text
outputs/viewer/drone_controller_gain_distance_summary.csv
docs/experiments/drone_controller_gain_distance_summary.md
```

### 동작 방식

```text
1. full mode 원본 CSV 읽기
2. distance_m, gain, memo가 같은 row끼리 그룹화
3. raw_abs_p99, raw_rms, raw_abs_max, coherence, CNN confidence 등의 중앙값 계산
4. CNN confirmed ratio와 overload ratio 계산
5. decision 자동 분류
6. CSV와 Markdown summary 생성
```

### 사용 명령어

```bash
PYTHONPATH=. python scripts/summarize_gain_distance_sweep.py \
  --input outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --output-csv outputs/viewer/drone_controller_gain_distance_summary.csv \
  --output-md docs/experiments/drone_controller_gain_distance_summary.md \
  --reference-distance 2.0 \
  --reference-gain 30
```

---

## 3. 오늘 수행한 실험 정리

## 3.1 Phase calibration 반복 재현성 확인

처음에는 1.5m / gain30 조건에서 phase calibration을 수행했으나, valid block과 cluster block이 매우 적어 FAIL이 발생하였다.

이후 gain을 35로 올렸지만 1.5m 조건에서는 phase offset이 계속 흔들렸고, 반복 재현성이 부족했다.

정리:

```text
1.5m / gain30
→ FAIL, 신호 부족 또는 coherence 부족

1.5m / gain35
→ 일부 WARNING, 하지만 run 간 phase offset 재현성 부족
```

---

## 3.2 1.0m / gain35 조건 확인

거리 1.0m, gain35에서 phase calibration이 크게 개선되었다.

대표 결과:

```text
+59.725 deg
+45.173 deg
+60.737 deg
+61.305 deg
+63.943 deg
```

이 중 +45도 근처 값은 실험자가 왼쪽 뒤 45도 방향에서 핫스팟을 켜고 있었던 영향으로 의심되었다.

핫스팟을 끄고 다시 수행한 결과 gain40에서 훨씬 안정적인 결과가 나왔다.

---

## 3.3 1.0m / gain40 / hotspot OFF 조건 확인

핫스팟을 끄고 gain40에서 phase calibration을 반복한 결과가 매우 안정적이었다.

결과:

```text
run1: +122.217 deg | WARNING | coherence median 0.988 | cluster 108/170
run2: +118.190 deg | OK      | coherence median 0.976 | cluster 170/170
run3: +118.442 deg | OK      | coherence median 0.968 | cluster 170/170
run4: +119.664 deg | OK      | coherence median 0.982 | cluster 170/170
```

판단:

```text
1.0m / gain40 / hotspot OFF 조건은 phase calibration 기준 조건으로 사용 가능
```

---

## 3.4 Gain phase table 제작

이후 gain40을 reference로 하여 gain40, gain35, gain30, gain25에 대한 gain phase table을 생성하였다.

명령어:

```bash
PYTHONPATH=. python scripts/build_gain_phase_table.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 40,35,30,25 \
  --reference-gain 40 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output configs/calibration/gain_phase_table_2450.json \
  --memo "outdoor_0deg_1p0m_gain_phase_table_ref40_sig2452_hotspot_off"
```

생성 결과:

```text
gain=40 | phase=+139.337 deg | delta=+0.000 deg | OK
gain=35 | phase=+138.739 deg | delta=-0.598 deg | OK
gain=30 | phase=+139.630 deg | delta=+0.293 deg | OK
gain=25 | phase=+139.966 deg | delta=+0.629 deg | WARNING
```

결론:

```text
동일 receiver session에서 gain을 변경해도 phase offset 변화는 ±1도 이내였다.
따라서 gain 변경 자체보다는 session 시작 시 absolute phase calibration이 더 중요하다.
```

---

## 4. 오늘 정리한 운영 정책

## 4.1 Phase calibration 정책

기존 계획:

```text
reference phase offset 1개 측정
+ gain phase delta table 적용
```

수정된 정책:

```text
Runtime 시작 직후 사용할 gain들을 연속 calibration
→ gain별 phase offset을 current_session_phase_table로 저장
→ runtime 중 현재 gain에 해당하는 phase offset 직접 적용
```

즉:

```text
gain 40 사용 중 → phase_40 적용
gain 35 사용 중 → phase_35 적용
gain 30 사용 중 → phase_30 적용
gain 25 사용 중 → phase_25 적용
```

기존 `gain_phase_table_2450.json`은 다음 의미로 유지한다.

```text
gain 변화가 phase offset에 큰 영향을 주지 않는다는 검증 결과
```

---

## 4.2 거리–gain 정책

이제 거리별 gain 실험은 신호발생기가 아니라 실제 드론 조종기 신호로 수행한다.

핵심 reference 조건:

```text
2m / gain30 / drone controller
```

이유:

```text
CNN 학습 데이터가 1~2m / gain30 근처 조건에서 수집되었기 때문
```

따라서 내일 실험은 다음을 확인하는 방향으로 진행한다.

```text
3~7m 거리에서 gain35, gain40, gain45를 사용했을 때
2m gain30 reference와 비슷한 raw feature와 CNN/coherence 안정성이 나오는지 확인
```

---

## 5. 내일 실험 계획

## 5.1 목적

실제 드론 조종기 신호 기준으로 gain-distance sweep을 수행한다.

목표:

```text
1. 2m gain30 reference feature 저장
2. gain30에서 멀어지며 weak/lost 거리 확인
3. gain35로 올리고 포화가 사라지는 최소 거리와 usable 거리 확인
4. gain40, gain45까지 반복
5. raw feature + CNN confirmed + coherence 기준으로 decision 작성
```

---

## 5.2 실험 전 체크리스트

```text
[ ] 핫스팟 OFF
[ ] 블루투스 OFF
[ ] 주변 Wi-Fi 간섭 최소화
[ ] 드론 조종기 위치/방향 고정
[ ] RX 안테나 위치/간격 고정
[ ] 사람/폰/노트북은 RX 전방 경로에서 멀리
[ ] current_phase_offset 또는 session phase calibration 준비
[ ] gain_phase_table_2450.json 준비
[ ] viewer full mode 실행 확인
```

---

## 5.3 추천 거리/Gain 조합

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

## 5.4 Full mode 공통 명령어

아래 명령어에서 바꿀 부분은 두 개이다.

```text
--gain <GAIN>
--distance-m <DISTANCE>
--memo "dist<DISTANCE>_gain<GAIN>_drone_controller"
```

공통 명령어:

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

실행 후 절차:

```text
1. 화면 안정 확인
2. CNN confirmed 확인
3. raw feature / overload 확인
4. coherence 확인
5. s 키 눌러 profile 저장
6. 50 block 저장 대기
7. q로 종료
8. 다음 조건 실행
```

---

## 5.5 Reference 명령어

2m / gain30 reference:

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

---

## 5.6 기록용 표

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

## 5.7 내일 실험 후 요약 스크립트 실행

```bash
PYTHONPATH=. python scripts/summarize_gain_distance_sweep.py \
  --input outputs/viewer/drone_controller_gain_distance_sweep.csv \
  --output-csv outputs/viewer/drone_controller_gain_distance_summary.csv \
  --output-md docs/experiments/drone_controller_gain_distance_summary.md \
  --reference-distance 2.0 \
  --reference-gain 30
```

생성 결과:

```text
outputs/viewer/drone_controller_gain_distance_summary.csv
docs/experiments/drone_controller_gain_distance_summary.md
```

---

## 6. 남은 작업

## 6.1 코드 정리 및 커밋

현재 git status에서 정리해야 할 항목:

```text
삭제:
- scripts/live_rf_viewer.py*.bak
- src/viewer/opencv_renderer.py*.bak

되돌림 권장:
- configs/calibration/current_phase_offset.json

커밋 후보:
- configs/calibration/gain_phase_table_2450.json
- scripts/live_rf_viewer.py
- src/viewer/opencv_renderer.py
- scripts/summarize_gain_distance_sweep.py
```

권장 커밋 분리:

```bash
git add configs/calibration/gain_phase_table_2450.json
git commit -m "calibration: add gain phase table at 2450 MHz"

git add scripts/live_rf_viewer.py src/viewer/opencv_renderer.py
git commit -m "viewer: fix spectrogram display and STFT defaults"

git add scripts/summarize_gain_distance_sweep.py
git commit -m "tools: add gain distance sweep summarizer"
```

---

## 6.2 Session phase table 스크립트 제작

추후 만들 파일:

```text
scripts/calibrate_session_phase_table.py
```

목적:

```text
runtime 시작 전 사용할 gain들을 같은 receiver session에서 연속 calibration하고,
current_session_phase_table.json으로 저장한다.
```

예상 명령어:

```bash
PYTHONPATH=. python scripts/calibrate_session_phase_table.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 40,35,30,25 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output configs/calibration/current_session_phase_table.json \
  --memo "runtime_start_session_phase_table"
```

---

## 7. 최종 결론

오늘의 핵심 결론은 다음과 같다.

```text
1. live viewer의 STFT 표시 기준을 128 x 509로 정리하였다.
2. gain40 / hotspot OFF 조건에서 phase calibration이 안정적으로 나왔다.
3. 같은 receiver session 안에서 gain을 바꿔도 phase offset 변화는 ±1도 이내였다.
4. gain 변화 자체보다 receiver session 시작 시 absolute phase calibration이 더 중요하다.
5. gain_phase_table_2450.json은 보정 테이블이면서 동시에 gain 변화 영향이 작다는 검증 결과로 사용할 수 있다.
6. 내일은 실제 드론 조종기 신호로 2m gain30 reference를 잡고, gain30~45에서 거리별 usable range를 찾는다.
7. full mode 원본 로그를 쌓고, summarize 스크립트로 거리/gain별 중앙값 요약표를 만든다.
```
