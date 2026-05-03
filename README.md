# SDR 기반 비인가 드론 RF 신호 탐지 및 도래각 추정 모듈

Pluto+ SDR 기반 2.4GHz RF 신호를 이용해 Wi-Fi / Bluetooth / Drone-like / Background 신호를 분석하고,  
2채널 IQ 데이터의 위상차를 이용해 도래각(AoA, Angle of Arrival)을 추정하는 캡스톤 프로젝트입니다.

본 프로젝트는 고가의 통합 대드론 장비 전체를 구현하는 것이 아니라, 그중 **RF 탐지 계층**에 해당하는 핵심 기능을  
저비용 SDR 장비와 소프트웨어 신호처리 파이프라인으로 구현하는 것을 목표로 합니다.

```text
2채널 IQ 수신 → 전처리 → Energy Detection → STFT Spectrogram 생성
→ 2D CNN 분류 입력 생성 → Coherence 검사 → 위상차 기반 AoA 추정
→ 결과 저장 → 2.4GHz 대역 Scan mode 확장
```

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [현재 구현 기준](#2-현재-구현-기준)
3. [전체 Pipeline 구조](#3-전체-pipeline-구조)
4. [CNN Classification](#4-cnn-classification)
5. [AoA 추정](#5-aoa-추정)
6. [Calibration](#6-calibration)
7. [Scan Mode](#7-scan-mode)
8. [실행 방법](#8-실행-방법)
9. [프로젝트 구조](#9-프로젝트-구조)
10. [현재 개발 현황](#10-현재-개발-현황)
11. [데이터 전략](#11-데이터-전략)
12. [시연 시나리오](#12-시연-시나리오)
13. [한계 및 확장 방향](#13-한계-및-확장-방향)
14. [정리 필요 파일](#14-정리-필요-파일)
15. [기본 명령어](#15-기본-명령어)
16. [현재 프로젝트 상태 요약](#16-현재-프로젝트-상태-요약)

---

## 1. 프로젝트 개요

### 1.1 목표

2.4GHz 대역 RF 신호를 수신하여 드론 의심 신호를 탐지하고,  
수신 신호의 종류와 방향 정보를 함께 제공하는 **RF 기반 드론 탐지 프로토타입**을 구현합니다.

주요 목표는 다음과 같습니다.

- Pluto+ SDR을 이용한 2.4GHz RF 신호 수신
- RX0/RX1 2채널 IQ 데이터 처리
- Energy Detector 기반 1차 신호 탐지
- STFT 기반 spectrogram 생성
- 2D CNN 기반 RF 신호 분류 구조 구현
- Coherence 기반 AoA 신뢰도 검증
- RX0/RX1 위상차 기반 도래각(AoA) 추정
- 2.400~2.485GHz 대역 주파수 스캔 모드 구현
- 추후 Raspberry Pi 등 엣지 장치 배포 가능성 검토

---

### 1.2 하드웨어 구성

| 부품 | 역할 |
|---|---|
| Pluto+ SDR | 2채널 IQ 수신 |
| 2.4GHz 안테나 ×2 | RX0/RX1 위상차 기반 AoA 추정 |
| 노트북 / Raspberry Pi 후보 | 신호처리, CNN 추론, 결과 출력 |
| Python 실행 환경 | 전체 pipeline 실행 및 결과 저장 |

현재 개발은 하드웨어 의존성을 줄이기 위해 `sim`, `file`, `sdr` 입력을 모두 지원하는 구조로 진행합니다.  
Pluto+가 연결되지 않은 상황에서도 synthetic IQ 또는 저장된 IQ 파일로 pipeline을 검증할 수 있습니다.

---

## 2. 현재 구현 기준

### 2.1 처리 단위

전체 pipeline의 기본 처리 단위는 **block**입니다.

| 항목 | 값 |
|---|---:|
| Sample rate | 5 MSPS |
| Center frequency | 2.4 GHz |
| Block size | 16,384 samples |
| Block time | 약 3.28 ms |
| Channel count | 2 channels |
| 기본 입력 | `sim` |

```yaml
# configs/receiver.yaml 기본값
source_type: sim
sample_rate: 5000000
center_freq: 2400000000
block_size: 16384
num_samples: 16384
num_channels: 2
```

---

### 2.2 입력 소스

입력 소스는 `configs/receiver.yaml`의 `source_type`으로 선택합니다.  
입력 장치가 바뀌어도 뒤쪽 신호처리 pipeline은 그대로 유지됩니다.

| source_type | 설명 |
|---|---|
| `sim` | synthetic IQ 신호 생성 |
| `file` | 저장된 IQ 파일 재생 |
| `sdr` | Pluto+ SDR 실측 입력 |

```text
SimReceiver / RawFileReceiver / PlutoReceiver
                    ↓
              공통 pipeline
```

---

## 3. 전체 Pipeline 구조

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
[Dual-channel STFT]
    ↓
[CNN Spectrogram 생성]
    ↓
[Coherence Gate]
    ↓
[Phase Difference Estimation]
    ↓
[AoA Estimation]
    ↓
[summary.json / numpy artifacts 저장]
```

### 3.1 전처리

| 처리 | 모듈 |
|---|---|
| DC offset 제거 | `src/preprocess/dc_blocker.py` |
| IQ amplitude 정규화 | `src/preprocess/iq_normalizer.py` |
| RX0/RX1 gain mismatch 보정 | `src/preprocess/gain_matcher.py` |
| RX0/RX1 phase offset 추정/보정 | `src/preprocess/phaseoffset.py` |
| block / frame 분할 | `src/preprocess/framing.py` |
| CNN 입력용 RX 채널 선택 | `src/preprocess/channel_filter.py` |

---

### 3.2 Energy Detection

block 내부를 더 작은 energy frame으로 나누어 신호 존재 여부를 판단합니다.

```yaml
# configs/detect.yaml 기본값
frame_size: 1024
hop_size: 512
window: hann
threshold_multiplier: 5.0
min_detection_ratio: 0.05
```

block 하나당 energy frame 수:

```text
floor((16384 - 1024) / 512) + 1 = 31 frames
```

threshold를 넘은 frame 비율이 `min_detection_ratio` 이상이면 active block으로 판단합니다.

| mode | 설명 |
|---|---|
| `block_median` | 현재 block의 median energy를 noise floor로 사용 |
| `initial_calibration` | 초기 calibration block을 기반으로 noise floor 고정 |

현재 기본 방향은 `block_median`으로 빠르게 검증하고, 실측 단계에서 `initial_calibration` 기반 고정 threshold 구조로 확장하는 것입니다.

---

### 3.3 STFT / Spectrogram

```yaml
# STFT 설정
nperseg: 512
noverlap: 384
hop_length: 128
nfft: 512
window: hann
```

block 크기 16,384 기준 결과 shape:

```text
complex STFT shape  = (512, 125)
CNN spectrogram     = (512, 125)
CNN input shape     = (512, 125, 1)
PyTorch input shape = (B, 1, 512, 125)
```

복소 IQ 신호를 사용하므로 양방향 주파수 성분을 포함하는 512개 frequency bin을 사용합니다.  
이는 real signal STFT의 257 bin 구조와 다릅니다.

| 모듈 | 역할 |
|---|---|
| `src/features/fft.py` | FFT magnitude / power 계산 |
| `src/features/spectrogram.py` | STFT spectrogram 생성 |
| `src/features/window.py` | Hann / Hamming / Rect window 생성 |

---

## 4. CNN Classification

### 4.1 분류 클래스 / 입력 형태

| 항목 | 내용 |
|---|---|
| 분류 클래스 | Background / WiFi / Bluetooth / Drone-like |
| 입력 shape | `(512, 125, 1)` |
| PyTorch 입력 shape | `(B, 1, 512, 125)` |

---

### 4.2 모델 구조 — `SpectrogramCNN`

| 항목 | 내용 |
|---|---|
| 입력 | `(B, 1, 512, 125)` |
| 출력 | 4-class logits |
| 구조 | ConvBNAct → DepthwiseSeparableConv ×4 → AdaptiveAvgPool → Linear classifier |
| 특징 | Depthwise Separable Convolution 기반 경량 구조 |

| 모듈 | 역할 |
|---|---|
| `src/ml/model_2dcnn.py` | SpectrogramCNN 모델 정의 |
| `src/ml/inference.py` | CNN 추론 래퍼 |
| `src/ml/transforms.py` | 입력 shape 변환 |
| `src/ml/dataset.py` | 학습 데이터셋 로더 |
| `src/ml/train_2dcnn.py` | 2D CNN 학습 루프 |
| `src/ml/evaluate.py` | 평가 지표 계산 |
| `src/ml/export_tflite.py` | TFLite export 관련 기능 |
| `src/ml/infer_tflite.py` | TFLite inference 관련 기능 |
| `src/ml/synthetic_generator.py` | synthetic 학습 데이터 생성 |

---

### 4.3 현재 CNN 상태

지원 inference backend:

| backend | 설명 |
|---|---|
| `dummy` | 실제 모델 없이 pipeline 연결 테스트 |
| `keras` | Keras `.keras` 모델 로드 |
| `torch` | PyTorch 모델 로드 |

`model_path: null`인 경우 모델 구조 연결 테스트는 가능하지만,  
출력 결과를 실제 분류 성능으로 해석하면 안 됩니다.

```text
CNN 모델 구조              ✅ 구현됨
CNN inference interface   ✅ 구현됨
학습 checkpoint 연결       🔄 진행 중
실측 데이터 기반 검증       🔄 진행 예정
```

---

## 5. AoA 추정

### 5.1 계산 개념

```yaml
# configs/aoa.yaml 기본값
carrier_freq: 2400000000
wavelength_m: 0.125
antenna_spacing_m: 0.0625
method: phase_difference
```

```text
theta = arcsin( (phase_diff × wavelength) / (2π × antenna_spacing) )
```

2채널 구조 기준 기본 추정 범위는 **-90° ~ +90°**입니다.

---

### 5.2 Coherence Gate

AoA 계산 전에 RX0/RX1이 같은 신호를 수신하고 있는지 coherence로 검증합니다.

```text
coherence threshold  = 0.60
energy_percentile    = 75.0
```

| 모듈 | 역할 |
|---|---|
| `src/aoa/coherence.py` | STFT coherence 계산 |
| `src/aoa/phase_diff.py` | 채널 간 위상차 계산 |
| `src/aoa/angle_estimator.py` | arcsin 기반 방위각 산출 |
| `src/aoa/aoa_gate.py` | AoA 진입 조건 판단 |

---

## 6. Calibration

전원 ON 또는 실행 조건에 따라 RX0/RX1 채널 간 gain / phase offset이 달라질 수 있습니다.  
실측 환경에서는 calibration block을 먼저 읽고 다음 값을 추정한 뒤 분석 block에 적용합니다.

- noise floor / threshold
- gain correction
- phase offset
- phase offset coherence-like score

### 6.1 Calibration 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/calibrate.py` | 통합 calibration 실행 스크립트 |
| `scripts/run_noise_calibration.py` | noise floor / threshold 캘리브레이션 실행 |
| `scripts/run_phase_gain_calibration.py` | RX0/RX1 phase offset + gain mismatch 캘리브레이션 실행 |
| `src/calibration/noise_calibration.py` | noise calibration 로직 |
| `src/calibration/phase_gain_calibration.py` | phase / gain calibration 로직 |
| `src/calibration/params.py` | calibration parameter 구조 |
| `src/preprocess/gain_matcher.py` | 채널 간 gain 보정 |
| `src/preprocess/phaseoffset.py` | 위상 오프셋 추정 및 제거 |

### 6.2 Calibration 실행

```bash
# 통합 calibration
PYTHONPATH=. python scripts/calibrate.py

# 노이즈 캘리브레이션
PYTHONPATH=. python scripts/run_noise_calibration.py

# Phase + Gain 캘리브레이션
PYTHONPATH=. python scripts/run_phase_gain_calibration.py
```

---

## 7. Scan Mode

### 7.1 목적

SDR 수신 대역폭과 연산량 한계로 2.4GHz 대역 전체를 한 번에 분석하기 어렵습니다.  
따라서 가벼운 FFT scan으로 후보 주파수를 먼저 찾고, trigger된 주파수에서만 정밀 분석을 수행합니다.

| 항목 | 값 |
|---|---:|
| Start frequency | 2.400 GHz |
| Stop frequency | 2.485 GHz |
| Step frequency | 5 MHz |
| Samples per block | 16,384 |
| Scan blocks | 3 |
| Min pass blocks | 2 |

---

### 7.2 실행 흐름

```text
[2.400 ~ 2.485 GHz, 5 MHz step 주파수 목록 생성]
    ↓
[각 center frequency로 이동 → block 수집]
    ↓
[FFT max power 계산 → threshold 초과 여부 판단]
    ↓
[scan_blocks 중 min_pass_blocks 이상 통과 시 trigger]
    ↓
[trigger된 주파수만 PrecisionAnalyzer로 정밀 분석]
    ↓
[STFT / CNN / Coherence / AoA]
    ↓
[scan_events.json 저장]
```

| 모듈 | 역할 |
|---|---|
| `src/scan/scan_policy.py` | 스캔 주파수 생성 및 후보 판정 정책 |
| `src/scan/scanner.py` | FrequencyScanner / scan score 계산 |
| `src/scan/precision_analyzer.py` | trigger된 주파수 정밀 분석 |
| `scripts/run_scan.py` | Scan mode 실행 진입점 |

---

## 8. 실행 방법

### 8.1 가상환경 활성화

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

---

### 8.2 단일 block pipeline 실행

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

출력 결과:

```text
outputs/runs/latest/
├── summary.json
├── frame_energies.npy
├── detections.npy
├── fft_mag.npy
├── energy_plot.png
└── stage1/
    ├── cnn_spectrogram.npy
    ├── rx0_complex_stft.npy
    └── rx1_complex_stft.npy
```

---

### 8.3 Scan mode 실행

```bash
PYTHONPATH=. python scripts/run_scan.py
```

출력 결과:

```text
outputs/runs/latest/
├── scan_events.json
└── scan_precision/
    └── <center_freq>_cnn_spectrogram.npy
```

---

### 8.4 Runtime CLI 실행

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

Runtime 관련 모듈:

| 모듈 | 역할 |
|---|---|
| `src/runtime/cli.py` | Runtime CLI 진입점 |
| `src/runtime/calibration_actions.py` | calibration action |
| `src/runtime/scan_actions.py` | scan action |
| `src/runtime/scan_loop.py` | scan runtime loop |

---

### 8.5 Pluto+ 실측 입력으로 변경

`configs/receiver.yaml`에서 `source_type: sdr`로 변경 후 아래 설정을 확인합니다.

```yaml
sdr:
  uri: ip:192.168.2.1
  channels: [0, 1]
  center_freq: 2400000000
  sample_rate: 5000000
  rf_bandwidth: 5000000
  gain_control_mode: manual
  gain: 20
  num_samples: 16384
  block_size: 16384
```

---

## 9. 프로젝트 구조

아래 구조는 현재 `git ls-files` 기준으로 정리한 주요 추적 파일 구조입니다.

```text
rf-drone-detection-capstone/
├── README.md
├── requirements.txt
├── pipeline.md
├── .gitignore
│
├── configs/
│   ├── aoa.yaml
│   ├── detect.yaml
│   ├── ml.yaml
│   ├── paths.yaml
│   ├── receiver.yaml
│   ├── scan.yaml
│   └── ui.yaml
│
├── scripts/
│   ├── calibrate.py
│   ├── check_raw_iq_store.py
│   ├── check_stage1_branch.py
│   ├── check_stage1_coherence.py
│   ├── export_model.py
│   ├── preprocess_dataset.py
│   ├── run_noise_calibration.py
│   ├── run_phase_gain_calibration.py
│   ├── run_pipeline.py
│   ├── run_runtime_cli.py
│   ├── run_scan.py
│   └── train_model.py
│
├── src/
│   ├── aoa/
│   │   ├── angle_estimator.py
│   │   ├── aoa_gate.py
│   │   ├── coherence.py
│   │   └── phase_diff.py
│   │
│   ├── calibration/
│   │   ├── noise_calibration.py
│   │   ├── params.py
│   │   └── phase_gain_calibration.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── paths.py
│   │   ├── pipeline.py
│   │   ├── raw_iq_store.py
│   │   ├── stage1_artifact_store.py
│   │   ├── types.py
│   │   └── utils.py
│   │
│   ├── detect/
│   │   └── energy_detector.py
│   │
│   ├── features/
│   │   ├── fft.py
│   │   ├── spectrogram.py
│   │   └── window.py
│   │
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── export_tflite.py
│   │   ├── infer_tflite.py
│   │   ├── inference.py
│   │   ├── model_2dcnn.py
│   │   ├── synthetic_generator.py
│   │   ├── train_2dcnn.py
│   │   └── transforms.py
│   │
│   ├── preprocess/
│   │   ├── channel_filter.py
│   │   ├── dc_blocker.py
│   │   ├── framing.py
│   │   ├── gain_matcher.py
│   │   ├── iq_normalizer.py
│   │   └── phaseoffset.py
│   │
│   ├── receiver/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── pluto_receiver.py
│   │   ├── raw_file_receiver.py
│   │   └── sim_receiver.py
│   │
│   ├── runtime/
│   │   ├── calibration_actions.py
│   │   ├── cli.py
│   │   ├── scan_actions.py
│   │   └── scan_loop.py
│   │
│   ├── scan/
│   │   ├── precision_analyzer.py
│   │   ├── scan_policy.py
│   │   └── scanner.py
│   │
│   └── ui/
│       ├── dashboard.py
│       ├── event_logger.py
│       └── result_plotter.py
│
├── docs/
│   ├── hardware/
│   │   └── requirement_hw.md
│   ├── planning/
│   │   ├── COMMANDS.md
│   │   ├── PIPELINE_FILE_MAP.md
│   │   ├── STATUS.md
│   │   └── priority_table.html
│   ├── experiments/
│   └── presentation/
│
├── tests/
│   ├── test_aoa.py
│   ├── test_detector.py
│   ├── test_fft.py
│   ├── test_pipeline.py
│   ├── test_preprocess.py
│   └── fixtures/
│
├── data/
│   ├── external/
│   ├── processed/
│   └── synthetic/
│
├── models/
│   ├── checkpoints/
│   └── tflite/
│
├── outputs/
├── logs/
├── gnu_radio/
│   └── sim_rx.grc
│
└── root-level experiment files
    ├── asd.py
    ├── capture_fixed_channel.py
    ├── findwifi.py
    ├── pluto+exp.py
    └── iq_ch13.npy
```

---

## 10. 현재 개발 현황

| 모듈 | 상태 |
|---|---|
| 프로젝트 기본 구조 | 완료 |
| YAML 기반 설정 구조 | 완료 |
| SimReceiver | 완료 |
| RawFileReceiver | 완료 |
| PlutoReceiver 기본 구조 | 구현됨, 실측 검증 필요 |
| DC offset 제거 | 완료 |
| IQ normalization | 완료 |
| Gain mismatch correction | 완료 |
| Phase offset estimation/correction | 완료 |
| Noise calibration runner | 완료 |
| Phase/Gain calibration runner | 완료 |
| Energy Detector | 완료 |
| FFT feature 계산 | 완료 |
| STFT spectrogram 생성 | 완료 |
| Dual-channel STFT branch | 완료 |
| Coherence gate | 완료 |
| Phase difference 계산 | 완료 |
| AoA 계산 | 완료 |
| 2D CNN 모델 구조 | 완료 |
| CNN inference interface | 완료 |
| 단일 block pipeline | 완료 |
| Scan mode 기본 구조 | 완료 |
| Triggered precision analysis | 완료 |
| Runtime CLI 기본 구조 | 구현됨 |
| 실제 학습 checkpoint 연결 | 진행 중 |
| UI dashboard | 추후 확장 |
| TFLite / Raspberry Pi 배포 | 추후 확장 |
| Pluto+ 실측 통합 테스트 | 진행 예정 |
| 단위 테스트 | 기본 파일 존재, 지속 보강 필요 |

---

## 11. 데이터 전략

| 단계 | 데이터 | 용도 |
|---|---|---|
| 1차 학습 | 공개 RF 데이터셋 | 모델 초기 학습 |
| 합성 데이터 | synthetic spectrogram / synthetic IQ | 클래스 불균형 보완 및 pipeline 검증 |
| 전이학습 | Pluto+ 직접 수집 데이터 | 실제 환경 파인튜닝 |
| 실측 검증 | Wi-Fi / Bluetooth / Drone-like 환경 수집 | 최종 성능 평가 |

현재 단계에서는 **데이터셋 구성과 학습 checkpoint 연결**이 중요한 후속 과제입니다.

---

## 12. 시연 시나리오

| 단계 | 행동 | 기대 결과 |
|---|---|---|
| 1단계: 신호 탐지 | 2.4GHz 대역 신호 발생 | Energy Detection trigger |
| 2단계: 신호 분류 | Wi-Fi / Bluetooth / Drone-like 신호 입력 | CNN class / confidence 출력 |
| 3단계: 방향 추정 | 송신원 위치를 좌우로 이동 | AoA angle 변화 |
| 4단계: scan mode | 2.4GHz 대역 sweep | 후보 주파수 탐지 및 정밀 분석 |
| 5단계: 결과 저장 | pipeline 실행 | `summary.json` / `scan_events.json` 저장 |

---

## 13. 한계 및 확장 방향

### 13.1 현재 한계

- 2채널 위상차 기반 AoA는 -90° ~ +90° 정면 반구 추정에 적합합니다.
- Wi-Fi / Bluetooth 신호는 환경에 따라 드론 신호와 혼재될 수 있습니다.
- CNN 성능은 데이터셋 품질과 실측 데이터 다양성에 크게 의존합니다.
- Pluto+ 실측 단계에서는 채널 동기성, gain, phase offset, antenna spacing 검증이 필요합니다.
- CNN 구조는 준비되어 있으나, 실제 성능은 학습 checkpoint와 검증 데이터가 연결되어야 평가할 수 있습니다.

### 13.2 확장 방향

| 확장 항목 | 내용 |
|---|---|
| 전방위 탐지 | 다중 노드 또는 다중 안테나 구조로 확장 |
| 고각 추정 | 2D AoA 또는 팬틸트 구조 검토 |
| 정밀 AoA | MUSIC/ESPRIT 등 고해상도 AoA 알고리즘 검토 |
| 광학 융합 | 카메라 + YOLO 기반 시각 탐지와 결합 |
| 엣지 배포 | TFLite / ONNX / Raspberry Pi 배포 |
| 실시간 UI | dashboard 기반 탐지 결과 시각화 |

---

## 14. 정리 필요 파일

현재 `git ls-files` 기준으로 root에 실험용 또는 정리 검토가 필요한 파일이 일부 존재합니다.

| 파일 | 현재 판단 |
|---|---|
| `asd.py` | 임시 실험 파일로 보이며 정리 필요 |
| `capture_fixed_channel.py` | Pluto+ 고정 채널 수집 스크립트로 보이며 `scripts/` 이동 검토 |
| `findwifi.py` | Wi-Fi 탐색 실험 스크립트로 보이며 `scripts/` 이동 검토 |
| `pluto+exp.py` | Pluto+ 실험 파일로 보이며 파일명 정리 및 `scripts/` 이동 검토 |
| `iq_ch13.npy` | IQ 데이터 파일이므로 git 추적 제외 검토 |
| `-` | 의도치 않은 파일일 가능성이 높아 삭제 검토 |
| `작업용` | 작업 메모 파일이면 `docs/planning/` 이동 또는 삭제 검토 |
| `파이프라인순서` | pipeline 문서라면 확장자 `.md` 부여 후 `docs/planning/` 이동 검토 |

정리 전에는 삭제하지 말고, 각 파일 내용을 확인한 뒤 필요한 것은 적절한 폴더로 이동하는 것이 좋습니다.

---

## 15. 기본 명령어

```bash
# 가상환경 활성화
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

# 단일 block pipeline 실행
PYTHONPATH=. python scripts/run_pipeline.py

# Scan mode 실행
PYTHONPATH=. python scripts/run_scan.py

# Runtime CLI 실행
PYTHONPATH=. python scripts/run_runtime_cli.py

# 통합 calibration
PYTHONPATH=. python scripts/calibrate.py

# 노이즈 캘리브레이션
PYTHONPATH=. python scripts/run_noise_calibration.py

# Phase + Gain 캘리브레이션
PYTHONPATH=. python scripts/run_phase_gain_calibration.py

# requirements 저장
pip freeze > requirements.txt

# Git 저장
git add .
git commit -m "docs: update README for current RF detection pipeline"
git push origin main
```

---

## 16. 현재 프로젝트 상태 요약

현재 프로젝트는 단순 아이디어 단계가 아니라, **RF 신호처리 pipeline의 기본 골격이 구현된 상태**입니다.

```text
2채널 IQ 입력
→ DC offset 제거
→ Gain / Phase 보정
→ IQ 정규화
→ Energy Detection
→ STFT Spectrogram
→ CNN 입력 생성
→ Coherence 검사
→ 위상차 계산
→ AoA 계산
→ 결과 저장
```

2.4GHz 전체 대역을 5MHz step으로 훑는 **scan mode**도 구현되어 있어,  
추후 Pluto+ 실측 입력과 학습된 CNN checkpoint를 연결하면 실제 RF 드론 탐지 실험으로 확장할 수 있습니다.
