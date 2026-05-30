# 2026-05-28 드론 RF 고정 주파수 조건 저장 캡처 명령어 정리

## 목적

오늘 목표는 `scan mode`가 아니라 **2450 MHz 고정 주파수에서 드론 RF burst를 거리/gain별로 수집**하는 것이다.

이번 수정본 기준으로 사용하는 핵심 스크립트는 다음이다.

```text
scripts/capture_rf4_fixed_freq_dataset.py
```

이 스크립트는 기존 `debug_rf4_live_capture.py`와 다르게 **조건 저장 전용**이다.

핵심 흐름:

```text
1. OpenCV viewer로 실시간 spectrogram 확인
2. signal_ratio / 화면 패턴 확인
3. 조건이 괜찮으면 capture_rf4_fixed_freq_dataset.py 실행
4. save_policy에 따라 필요한 block만 저장
5. Drone-like / Background / Controller-only 데이터를 분리 저장
```

---

## 0. 공통 환경 진입

프로젝트 루트에서 실행한다.

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

---

## 1. 수정본 스크립트 배치

다운로드/생성한 수정본을 프로젝트 `scripts/` 폴더로 복사한다.

```bash
cp /mnt/data/capture_rf4_fixed_freq_dataset.py \
  ~/projects/rf-drone-detection-capstone/scripts/capture_rf4_fixed_freq_dataset.py
```

실행 권한을 준다.

```bash
chmod +x scripts/capture_rf4_fixed_freq_dataset.py
```

문법 확인을 한다.

```bash
PYTHONPATH=. python -m compileall scripts/capture_rf4_fixed_freq_dataset.py
```

옵션 확인:

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py -h
```

---

## 2. 새 capture script 핵심 옵션

| 옵션 | 의미 |
|---|---|
| `--label` | 저장 세션 이름 |
| `--center-freq` | 고정 중심 주파수 |
| `--gain` | SDR 수신 gain |
| `--num-blocks` | 읽을 전체 block 수 |
| `--max-saved` | 저장할 최대 block 수 |
| `--save-policy` | 어떤 조건의 block을 저장할지 선택 |
| `--min-signal-ratio` | weak signal 판단 기준 |
| `--valid-signal-ratio` | valid signal 판단 기준 |
| `--max-background-signal-ratio` | background 판단 기준 |
| `--max-clip-ratio` | 포화 block 제외 기준 |
| `--skip-inference` | CNN 추론 없이 저장 조건만 보고 캡처 |

---

## 3. save-policy 설명

| save-policy | 저장 조건 | 사용 목적 |
|---|---|---|
| `all` | 모든 block 저장 | controller-only 비교용 |
| `weak_or_valid` | `WEAK_SIGNAL`, `VALID_SIGNAL` 저장 | Drone-like 수집용 |
| `valid_only` | `VALID_SIGNAL`만 저장 | 강한 드론 신호만 수집 |
| `no_signal_only` | `NO_SIGNAL`만 저장 | Background 수집 |
| `final_drone_only` | CNN 최종 결과가 `Drone-like`인 것만 저장 | 모델 확신 기반 수집 |

오늘은 기본적으로 아래처럼 쓴다.

```text
Drone-like      → weak_or_valid
Background      → no_signal_only
Controller-only → all
```

---

## 4. 저장 조건 파라미터 기준

### 4.1 Drone-like 저장 조건

```text
min_signal_ratio = 2.0
valid_signal_ratio = 5.0
max_clip_ratio = 0.001
save_policy = weak_or_valid
```

해석:

| signal_ratio | 판단 | 저장 여부 |
|---:|---|---|
| `< 2.0` | 배경에 가까움 | 저장 X |
| `2.0 ~ 5.0` | 약한 드론 burst | 저장 O |
| `5.0 ~ 50` | 좋은 드론 burst | 저장 O |
| `50+` | 강한 드론 burst | 화면이 뭉개지지 않으면 저장 O |

---

### 4.2 Background 저장 조건

```text
max_background_signal_ratio = 1.5
save_policy = no_signal_only
```

해석:

| signal_ratio | 판단 | 저장 여부 |
|---:|---|---|
| `< 1.5` | 깨끗한 background | 저장 O |
| `1.5 ~ 2.0` | 애매함 | 제외 |
| `2.0+` | burst 섞임 | Background로 저장 X |

---

### 4.3 Controller-only 저장 조건

```text
save_policy = all
```

조종기만 켰을 때 어떤 패턴이 나오는지 보는 비교용 데이터다.  
나중에 드론 신호와 조종기 신호가 섞였는지 확인하는 데 사용한다.

---

## 5. OpenCV viewer로 먼저 확인

데이터 저장 전에 반드시 viewer로 10~20초 정도 확인한다.

### 5.1 드론 1m / gain 15 확인

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_opencv_selectable.py \
  --display-backend opencv \
  --center-freq 2450000000 \
  --gain 15 \
  --blocks-per-update 1 \
  --overload-peak 1000 \
  --overload-clip-ratio 1.0 \
  --memo "check_drone_d1m_g15"
```

### 5.2 드론 1.5m / gain 20 확인

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_opencv_selectable.py \
  --display-backend opencv \
  --center-freq 2450000000 \
  --gain 20 \
  --blocks-per-update 1 \
  --overload-peak 1000 \
  --overload-clip-ratio 1.0 \
  --memo "check_drone_d15m_g20"
```

### 5.3 Background 확인

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_opencv_selectable.py \
  --display-backend opencv \
  --center-freq 2450000000 \
  --gain 15 \
  --blocks-per-update 1 \
  --overload-peak 1000 \
  --overload-clip-ratio 1.0 \
  --memo "check_background_g15"
```

---
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_opencv_selectable.py \
  --display-backend opencv \
  --center-freq 2437000000 \
  --gain 30 \
  --blocks-per-update 1 \
  --overload-peak 1000 \
  --overload-clip-ratio 1.0 \
  --memo "check_blue_moving_g30"


PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label notdrone_bluetooth_calling_cf2450_g30_att10_airpods \
  --out-root "$DATA_ROOT" \
  --center-freq 2450000000 \
  --gain 30 \
  --max-saved 200 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 10.0 \
  --clip-peak 25 \
  --max-clip-ratio 0.001 \
  --skip-inference
## 6. Drone-like 캡처 명령어

### 6.1 Drone-like: 0.5m / gain 10 / 50 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g10_d05m_controller_far \
  --center-freq 2450000000 \
  --gain 10 \
  --num-blocks 100 \
  --max-saved 50 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

### 6.2 Drone-like: 0.5m / gain 15 / 50 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g15_d05m_controller_far \
  --center-freq 2450000000 \
  --gain 15 \
  --num-blocks 100 \
  --max-saved 50 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

### 6.3 Drone-like: 1.0m / gain 10 / 100 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g10_d1m_controller_far \
  --center-freq 2450000000 \
  --gain 10 \
  --num-blocks 200 \
  --max-saved 100 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

### 6.4 Drone-like: 1.0m / gain 15 / 150 saved target

오늘 가장 중요한 주력 조건이다.

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g15_d1m_controller_far \
  --center-freq 2450000000 \
  --gain 15 \
  --num-blocks 300 \
  --max-saved 150 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

### 6.5 Drone-like: 1.5m / gain 15 / 100 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g15_d15m_controller_far \
  --center-freq 2450000000 \
  --gain 15 \
  --num-blocks 250 \
  --max-saved 100 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

### 6.6 Drone-like: 1.5m / gain 20 / 100 saved target

먼 거리/약한 신호 보강용이다.

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_cf2450_g20_d15m_controller_far \
  --center-freq 2450000000 \
  --gain 20 \
  --num-blocks 250 \
  --max-saved 100 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-clip-ratio 0.001
```

---

## 7. Background 캡처 명령어

Background는 드론 OFF, 조종기 OFF 상태에서 수집한다.

### 7.1 Background: gain 10 / 100 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_cf2450_g10_home_droneoff_controlleroff \
  --center-freq 2450000000 \
  --gain 10 \
  --num-blocks 150 \
  --max-saved 100 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-clip-ratio 0.001 \
  --skip-inference
```

---

### 7.2 Background: gain 15 / 100 saved target

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_cf2450_g15_home_droneoff_controlleroff \
  --center-freq 2450000000 \
  --gain 15 \
  --num-blocks 150 \
  --max-saved 100 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-clip-ratio 0.001 \
  --skip-inference
```

---

## 8. Controller-only 캡처 명령어

조종기만 켜고 드론은 꺼둔다.  
조종기 신호만으로 Drone-like 오판이 나는지 확인하기 위한 비교 데이터다.

### 8.1 Controller only: gain 10 / 50 blocks

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label controller_only_cf2450_g10_far_corner_droneoff \
  --center-freq 2450000000 \
  --gain 10 \
  --num-blocks 50 \
  --save-policy all
```

---

### 8.2 Controller only: gain 15 / 50 blocks

```bash
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label controller_only_cf2450_g15_far_corner_droneoff \
  --center-freq 2450000000 \
  --gain 15 \
  --num-blocks 50 \
  --save-policy all
```

---

## 9. 오늘 수집 우선순위

| 우선순위 | 조건 | 목표 저장 개수 |
|---|---|---:|
| 1 | Drone 1.0m / gain 15 | 150 |
| 2 | Drone 1.0m / gain 10 | 100 |
| 3 | Drone 1.5m / gain 15 | 100 |
| 4 | Drone 1.5m / gain 20 | 100 |
| 5 | Drone 0.5m / gain 10 | 50 |
| 6 | Drone 0.5m / gain 15 | 50 |
| 7 | Background gain 10 | 100 |
| 8 | Background gain 15 | 100 |
| 9 | Controller-only gain 10 | 50 |
| 10 | Controller-only gain 15 | 50 |

총 목표 저장량: 약 850 blocks

---

## 10. 현장 판단 기준

| 상태 | 판단 | 행동 |
|---|---|---|
| `signal_ratio 1.0~1.3` | 거의 background | Drone-like로 저장 X |
| `signal_ratio 2~5` | 약한 드론 burst | 저장 O |
| `signal_ratio 5~50` | 좋은 드론 burst | 저장 O |
| `signal_ratio 50+` | 강한 드론 burst | 화면 패턴이 안 뭉개지면 저장 O |
| 화면 전체가 밝게 뭉개짐 | gain 과함 | gain 낮추기 |
| `clip_ratio` 증가 | 포화 의심 | gain 낮추기 |

---

## 11. 저장 결과 확인

캡처가 끝나면 출력 경로가 다음처럼 나온다.

```text
outputs/datasets/rf4_fixed_capture/YYYYMMDD_HHMMSS_label/
```

해당 폴더에는 다음이 저장된다.

```text
*.npy
summary.csv
session_meta.txt
```

파일 개수 확인:

```bash
find outputs/datasets/rf4_fixed_capture -name "*.npy" | wc -l
```

최근 세션 확인:

```bash
ls -lt outputs/datasets/rf4_fixed_capture | head
```

최근 세션의 summary 확인:

```bash
tail -n 20 outputs/datasets/rf4_fixed_capture/*/summary.csv
```

---

## 12. 오늘 결론

오늘은 `scan mode`가 아니라 다음 방식으로 진행한다.

```text
2450 MHz 고정
OpenCV viewer로 먼저 확인
capture_rf4_fixed_freq_dataset.py로 조건 저장
Drone-like는 signal_ratio >= 2.0부터 저장
Background는 signal_ratio <= 1.5만 저장
Controller-only는 비교용으로 전체 저장
gain은 15 중심, 필요 시 10/20으로 보강
```

PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --display-backend opencv \
  --view-mode dual \
  --db-preset basic \
  --gain 30 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "linear_x_or_cp_test"