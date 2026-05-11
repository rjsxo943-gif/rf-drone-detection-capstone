# SDR 기반 2.4GHz RF 드론 탐지 캡스톤

Pluto+ SDR과 Python 신호처리 파이프라인을 이용해 2.4GHz 대역 RF 신호를 수집·분석하고, CNN 기반 신호 분류와 2채널 위상차 기반 AoA(Angle of Arrival) 추정을 목표로 하는 캡스톤 프로젝트입니다.

이 프로젝트는 고가의 완성형 대드론 장비를 구현하는 것이 아니라, 그중 **RF 탐지 계층**을 저비용 SDR 장비와 소프트웨어 파이프라인으로 구현하는 것을 목표로 합니다.

```text
RF IQ 수신 → 전처리 → Energy Detection → STFT Spectrogram 생성
→ CNN 학습/추론 데이터 구성 → 필요 시 Coherence 검사 → 위상차 기반 AoA 추정
→ 결과 저장 및 보고서화
```

---

## 1. 프로젝트 목표

### 1.1 최종 목표

2.4GHz 대역에서 발생하는 Wi-Fi, Bluetooth, Drone-like, Background 신호를 구분하고, 드론 의심 신호가 탐지될 경우 방향 정보를 함께 제공하는 RF 기반 드론 탐지 프로토타입을 구현합니다.

### 1.2 현재 개발 초점

현재 브랜치에서는 전체 시스템 완성보다 **CNN 학습용 RF spectrogram 데이터 수집 파이프라인 안정화**에 초점을 둡니다.

현재 우선순위는 다음과 같습니다.

1. Pluto+ SDR 또는 저장된 IQ 파일에서 block 단위 IQ 데이터를 읽기
2. DC offset 제거, gain/phase 보정, 정규화 등 기본 전처리 적용
3. Energy Detector로 의미 있는 RF 신호 block 선별
4. STFT 기반 CNN 입력 spectrogram 생성
5. Wi-Fi / Bluetooth / Drone-like / Background 데이터셋 구성
6. 이후 학습된 CNN checkpoint를 pipeline에 연결
7. 2채널 입력이 확보되면 Coherence + AoA branch 검증

---

## 2. 현재 기준 파라미터

### 2.1 RF 수신 기준

| 항목 | 현재 기준 |
|---|---:|
| 주파수 대역 | 2.4GHz ISM band |
| 기본 중심 주파수 | 2.437GHz 근처 |
| Sample rate | 5 MSPS |
| RF bandwidth | 5 MHz |
| Block size | 16,384 samples |
| 1 block 시간 | 약 3.28 ms |
| SDR | Pluto+ |
| 기본 SDR URI | `ip:192.168.2.1` |

### 2.2 입력 소스

`configs/receiver.yaml`의 `source_type`으로 입력 방식을 선택합니다.

| source_type | 설명 |
|---|---|
| `sim` | synthetic IQ 신호 생성 |
| `file` | 저장된 `.npy` IQ 파일 재생 |
| `sdr` | Pluto+ SDR 실측 입력 |

현재 실측 수집 시에는 `source_type: sdr`을 사용합니다.

```yaml
source_type: sdr
sample_rate: 5000000
center_freq: 2437000000
block_size: 16384
num_samples: 16384
```

---

## 3. 현재 STFT / CNN 입력 기준

현재 CNN 학습 데이터 생성 기준은 RF3 실험 기준에 맞춰 **128 × 509 spectrogram**을 사용합니다.

```yaml
stft:
  nperseg: 128
  noverlap: 96
  hop_length: 32
  nfft: 128
  window: hann

spectrogram:
  input_shape: [128, 509, 1]
  resize: false
  use_log_scale: true
  normalize: true
```

| 항목 | 값 |
|---|---:|
| STFT frequency bins | 128 |
| STFT time frames | 509 |
| CNN spectrogram shape | `(128, 509)` |
| CNN input shape | `(128, 509, 1)` |
| 기본 CNN 입력 채널 | RX0 |

> 참고: 이전 실험에서는 `512 × 125` STFT도 사용했으나, 현재 브랜치의 `run_pipeline.py`와 `configs/ml.yaml` 기준은 `128 × 509`입니다.

---

## 4. 전체 파이프라인 구조

```text
[Receiver]
    ↓
[DC Offset Removal]
    ↓
[Gain Correction / Phase Offset Correction]
    ↓
[IQ Normalization]
    ↓
[Energy Detector]
    ↓
[STFT Spectrogram]
    ↓
[CNN Dataset Capture or CNN Inference]
    ↓
[Coherence Gate]        # 2채널 입력일 때
    ↓
[Phase Difference]      # 2채널 입력일 때
    ↓
[AoA Estimation]        # 2채널 입력일 때
    ↓
[outputs/runs/latest 저장]
```

현재는 CNN 학습용 데이터 수집이 우선이므로, 단일 채널 수집에서는 AoA 관련 값이 `None`으로 남을 수 있습니다. AoA는 RX0/RX1 2채널 입력이 확보되었을 때 의미 있게 동작합니다.

---

## 5. 주요 기능

### 5.1 Energy Detection

block 내부를 작은 frame으로 나누어 RF 신호 존재 여부를 1차 판정합니다.

```yaml
energy_detector:
  mode: block_median
  method: time_power
  frame_size: 1024
  hop_size: 512
  window: hann
  threshold_multiplier: 5.0
  min_detection_ratio: 0.05
```

16,384 sample block 기준 energy frame 수는 다음과 같습니다.

```text
floor((16384 - 1024) / 512) + 1 = 31 frames
```

`min_detection_ratio: 0.05`는 31개 frame 중 약 2개 이상이 threshold를 넘으면 해당 block을 active block으로 판단하는 수준입니다.

### 5.2 CNN Dataset Capture

Runtime CLI의 `[s]` 메뉴는 현재 **CNN 학습용 spectrogram 데이터 수집**에 사용됩니다.

```text
[n] noise calibration
[p] phase/gain calibration
[s] start CNN dataset capture
[q] quit
```

수집 시 사용자가 label을 직접 입력합니다.

예시 label:

```text
wifi
bluetooth
drone_like
background
```

현재 수집 흐름은 다음과 같습니다.

```text
Runtime CLI 실행
    ↓
noise calibration 확인
    ↓
label 입력
    ↓
scan/capture loop 실행
    ↓
Energy Detection 통과 block 선별
    ↓
STFT spectrogram 저장
    ↓
CNN 학습 데이터셋 구성
```

### 5.3 Scan Mode

2.4GHz 전체 대역을 한 번에 모두 분석하기 어렵기 때문에, 가벼운 scan으로 후보 주파수를 먼저 찾고 후보에 대해서만 정밀 분석을 수행합니다.

```yaml
scan:
  start_freq: 2400000000
  stop_freq: 2485000000
  step_freq: 5000000
  num_samples: 16384
  scan_blocks: 3
  min_pass_blocks: 2
```

스캔 대상 예시는 다음과 같습니다.

```text
2.400 GHz → 2.405 GHz → ... → 2.485 GHz
```

### 5.4 AoA Estimation

2채널 IQ 입력이 있을 때 RX0/RX1의 위상차를 이용해 도래각을 추정합니다.

```text
phase difference → arcsin 기반 angle 변환 → angle_deg 출력
```

AoA 계산 전에는 coherence gate를 통해 두 채널이 같은 신호를 보고 있는지 확인합니다.

현재 AoA branch는 다음 순서로 동작합니다.

```text
Dual-channel STFT
    ↓
Coherence Gate
    ↓
Phase Difference Estimation
    ↓
Angle Estimation
```

---

## 6. 실행 방법

### 6.1 가상환경 활성화

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

정상적으로 프로젝트 가상환경이 켜졌는지 확인합니다.

```bash
which python
```

기대 예시:

```text
/home/<user>/projects/rf-drone-detection-capstone/.venv/bin/python
```

---

### 6.2 단일 block pipeline 실행

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

주요 출력 파일:

```text
outputs/runs/latest/
├── summary.json
├── frame_energies.npy
├── detections.npy
├── fft_mag.npy
├── energy_plot.png
└── stage1/
    ├── cnn_spectrogram.npy
    ├── cnn_spectrogram.png
    ├── complex_stft.npy              # 1채널 입력일 때
    ├── rx0_complex_stft.npy           # 2채널 입력일 때
    └── rx1_complex_stft.npy           # 2채널 입력일 때
```

---

### 6.3 Runtime CLI 실행

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

메뉴:

```text
[n] noise calibration
[p] phase/gain calibration
[s] start CNN dataset capture
[q] quit
```

권장 순서:

```text
1. n 입력 → 노이즈 캘리브레이션
2. 필요 시 p 입력 → phase/gain 캘리브레이션
3. s 입력 → CNN 학습 데이터 수집
4. label / max_saved / rx_index / raw IQ 저장 여부 입력
5. q 입력 → 종료
```

---

### 6.4 Scan mode 실행

```bash
PYTHONPATH=. python scripts/run_scan.py
```

주요 출력 파일:

```text
outputs/runs/latest/
├── scan_events.json
└── scan_precision/
    └── <center_freq>_cnn_spectrogram.npy
```

---

### 6.5 Calibration 실행

```bash
# 통합 calibration
PYTHONPATH=. python scripts/calibrate.py

# 노이즈 캘리브레이션
PYTHONPATH=. python scripts/run_noise_calibration.py

# Phase + Gain 캘리브레이션
PYTHONPATH=. python scripts/run_phase_gain_calibration.py
```

Calibration 결과는 기본적으로 아래 경로에 저장됩니다.

```text
outputs/calibration/
├── noise_latest.json
└── phase_gain_latest.json
```

---

## 7. 프로젝트 구조

```text
rf-drone-detection-capstone/
├── README.md
├── requirements.txt
├── configs/
│   ├── receiver.yaml
│   ├── detect.yaml
│   ├── ml.yaml
│   ├── aoa.yaml
│   ├── scan.yaml
│   ├── paths.yaml
│   └── ui.yaml
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_scan.py
│   ├── run_runtime_cli.py
│   ├── calibrate.py
│   ├── run_noise_calibration.py
│   └── run_phase_gain_calibration.py
│
├── src/
│   ├── receiver/       # sim/file/sdr 입력 소스
│   ├── preprocess/     # DC 제거, 정규화, gain/phase 보정, framing
│   ├── detect/         # Energy Detector
│   ├── features/       # FFT, STFT, window
│   ├── ml/             # CNN 입력 변환 및 inference wrapper
│   ├── aoa/            # coherence, phase diff, angle estimation
│   ├── scan/           # frequency scan, precision analyzer
│   ├── runtime/        # CLI action, capture loop
│   ├── calibration/    # noise / phase / gain calibration
│   ├── ui/             # plot, dashboard, event logger
│   └── core/           # config, paths, types, artifact store
│
├── data/
│   ├── raw_iq/
│   ├── processed/
│   └── external/
│
├── outputs/
│   ├── calibration/
│   └── runs/
│
├── docs/
│   ├── report/
│   ├── planning/
│   └── experiments/
│
└── tests/
```

---

## 8. 현재 개발 상태

| 영역 | 상태 | 비고 |
|---|---|---|
| 프로젝트 기본 구조 | 완료 | `src/` 패키지 분리 완료 |
| YAML 설정 구조 | 완료 | receiver/detect/ml/scan/aoa 등 분리 |
| SimReceiver | 완료 | synthetic IQ 검증 가능 |
| RawFileReceiver | 완료 | 저장된 `.npy` 재생 가능 |
| PlutoReceiver | 구현됨 | 실측 연결/안정화 진행 중 |
| DC offset 제거 | 완료 | block 단위 적용 |
| IQ normalization | 완료 | CNN 입력 안정화용 |
| Gain correction | 완료 | 2채널 보정용 |
| Phase offset correction | 완료 | 2채널 보정용 |
| Energy Detector | 완료 | block/frame 기반 1차 탐지 |
| STFT spectrogram | 완료 | 현재 기준 `128 × 509` |
| CNN 입력 생성 | 완료 | RX0 기준 spectrogram 저장 |
| CNN 학습 데이터 수집 | 진행 중 | Wi-Fi / Bluetooth / Drone-like / Background |
| CNN inference 연결 | 구조 구현 | 실제 checkpoint 연결 필요 |
| Scan mode | 구현됨 | 후보 주파수 탐색 + 정밀 분석 |
| AoA branch | 구조 구현 | 2채널 실측 검증 필요 |
| Runtime CLI | 구현됨 | calibration + capture 중심 |
| 보고서 문서화 | 진행 중 | `docs/report/` 활용 |

---

## 9. 데이터 수집 전략

### 9.1 클래스 구성

현재 CNN 학습용 클래스는 다음 4개를 기준으로 합니다.

| Class | 설명 |
|---|---|
| Background | 의도적인 송신원이 없는 배경 RF 환경 |
| WiFi | 2.4GHz Wi-Fi 신호 |
| Bluetooth | Bluetooth / BLE 계열 신호 |
| Drone-like | 드론 조종/영상/제어 신호와 유사한 RF 패턴 |

### 9.2 수집 원칙

- 같은 label 안에서도 거리, 방향, gain, 위치 조건을 다양하게 둡니다.
- 파일 이름에는 label, 시간, 중심 주파수, gain, 거리 등 조건을 포함합니다.
- Windows에서 WSL로 파일을 옮길 때 `Zone.Identifier` 파일이 생기면 학습 데이터에서 제외합니다.
- 의미 있는 신호가 없는 block은 잘못된 label로 섞이지 않도록 주의합니다.
- 데이터셋은 train/val/test로 분리하고, 최종 평가는 수집 조건이 다른 test set에서 수행합니다.

예시 파일명:

```text
0001__20260509_160250__home_wifihot_ch6_on_2437_gain10_1m.npy
```

---

## 10. Git 사용 흐름

현재 작업 브랜치 예시:

```bash
git branch
```

작업 상태 확인:

```bash
git status
```

README 수정 후 저장:

```bash
git add README.md docs/report/
git commit -m "docs: update README for current RF detection pipeline"
git push origin experiment/stft-128-hop32
```

main 브랜치에 바로 push하기보다, 실험 브랜치에서 README를 먼저 정리한 뒤 확인하는 방식을 권장합니다.

---

## 11. 시연 시나리오

### 11.1 단일 block 분석 시연

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

확인할 것:

- `summary.json` 생성 여부
- `cnn_spectrogram_shape`가 `[128, 509]`인지 확인
- `energy_plot.png` 생성 여부
- `stage1/cnn_spectrogram.png` 생성 여부

### 11.2 CNN 데이터 수집 시연

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

흐름:

```text
n → noise calibration
s → label 입력 → capture 시작
q → 종료
```

### 11.3 2.4GHz scan 시연

```bash
PYTHONPATH=. python scripts/run_scan.py
```

확인할 것:

- 후보 주파수 탐지 여부
- `scan_events.json` 저장 여부
- 후보 주파수별 spectrogram 저장 여부

---

## 12. 한계 및 향후 개선

### 12.1 현재 한계

- CNN 성능은 아직 데이터셋 품질과 수량에 크게 의존합니다.
- Drone-like class는 실제 드론 RF 데이터 확보 여부에 따라 정의가 달라질 수 있습니다.
- Wi-Fi/Bluetooth가 강한 환경에서는 drone-like 신호와 혼재될 수 있습니다.
- 단일 채널 수집에서는 AoA를 계산할 수 없습니다.
- 2채널 AoA는 안테나 간격, gain/phase calibration, coherence 조건에 민감합니다.
- Pluto+ 연결 상태와 IIO context 안정성이 실측 결과에 영향을 줍니다.

### 12.2 향후 개선 방향

| 개선 항목 | 설명 |
|---|---|
| CNN checkpoint 연결 | 학습된 PyTorch/Keras 모델을 inference pipeline에 연결 |
| 데이터셋 자동 정리 | label별 spectrogram 저장 구조 표준화 |
| 실측 성능 평가 | confusion matrix, accuracy, F1-score 계산 |
| 2채널 AoA 검증 | 보어사이트 기준 phase/gain calibration 후 각도 변화 확인 |
| UI dashboard | 실시간 탐지 결과 표시 |
| Edge 배포 | Raspberry Pi / TFLite / ONNX 경량화 검토 |
| 보고서 자동화 | 실험 결과를 `docs/report/`에 누적 |

---

## 13. 주의사항

### 13.1 실측 전 확인

```bash
iio_info -u ip:192.168.2.1
```

연결 실패 시 확인할 것:

- Pluto+ 전원 및 USB/Ethernet 연결 상태
- PC 네트워크 인터페이스 IP 설정
- `configs/receiver.yaml`의 `sdr.uri`
- WSL에서 네트워크 접근 가능 여부

### 13.2 데이터 파일 관리

- `.npy` 원본 IQ 데이터는 용량이 커질 수 있으므로 Git에 직접 올리지 않습니다.
- `outputs/`, `data/raw_iq/`, 대용량 dataset은 `.gitignore` 대상으로 관리합니다.
- 보고서용 그림이나 요약 결과만 필요한 경우 `docs/report/` 아래에 정리합니다.

---

## 14. 핵심 명령어 모음

```bash
# 프로젝트 이동 및 가상환경 활성화
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

# 단일 block pipeline
PYTHONPATH=. python scripts/run_pipeline.py

# Runtime CLI
PYTHONPATH=. python scripts/run_runtime_cli.py

# Scan mode
PYTHONPATH=. python scripts/run_scan.py

# Noise calibration
PYTHONPATH=. python scripts/run_noise_calibration.py

# Phase/Gain calibration
PYTHONPATH=. python scripts/run_phase_gain_calibration.py

# Python 문법 확인
PYTHONPATH=. python -m compileall src scripts

# Git 상태 확인
git status
```

---

## 15. 현재 상태 요약

현재 프로젝트는 단순 아이디어 단계가 아니라, RF 신호처리 pipeline의 기본 골격이 구현된 상태입니다.

```text
Receiver / Preprocess / Energy Detector / STFT / CNN input / Scan / Runtime CLI / AoA branch
```

다음 핵심 과제는 다음과 같습니다.

1. Wi-Fi / Bluetooth / Drone-like / Background 데이터셋을 조건별로 안정적으로 수집
2. CNN 학습 checkpoint를 생성하고 pipeline에 연결
3. 실측 데이터 기준 confusion matrix와 정확도 산출
4. 2채널 입력 기반 AoA 검증
5. 캡스톤 보고서와 발표자료에 맞게 결과 정리
