# SDR 기반 비인가 드론 RF 신호 탐지 및 도래각 추정 모듈

Pluto+ SDR 기반 2.4GHz RF 신호를 이용해 Wi-Fi / Bluetooth / Drone-like / Background 신호를 분석하고,  
2채널 IQ 데이터의 위상차를 이용해 도래각(AoA, Angle of Arrival)을 추정하는 캡스톤 프로젝트입니다.

본 프로젝트는 고가의 통합 대드론 장비 전체를 구현하는 것이 아니라, 그중 **RF 탐지 계층**에 해당하는 핵심 기능을  
저비용 SDR 장비와 소프트웨어 신호처리 파이프라인으로 구현하는 것을 목표로 합니다.

```text
2채널 IQ 수신 → 전처리 → Energy Detection → STFT Spectrogram 생성
→ 2D CNN 분류 → Coherence 검사 → 위상차 기반 AoA 추정 → 결과 저장
→ 2.4GHz 대역 Scan mode 확장
```

-----

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
1. [현재 구현 기준](#2-현재-구현-기준)
1. [전체 Pipeline 구조](#3-전체-pipeline-구조)
1. [CNN Classification](#4-cnn-classification)
1. [AoA 추정](#5-aoa-추정)
1. [Scan Mode](#6-scan-mode)
1. [실행 방법](#7-실행-방법)
1. [프로젝트 구조](#8-프로젝트-구조)
1. [현재 개발 현황](#9-현재-개발-현황)
1. [데이터 전략](#10-데이터-전략)
1. [시연 시나리오](#11-시연-시나리오)
1. [한계 및 확장 방향](#12-한계-및-확장-방향)
1. [개발 원칙](#13-개발-원칙)
1. [기본 명령어](#14-기본-명령어)
1. [현재 프로젝트 상태 요약](#15-현재-프로젝트-상태-요약)

-----

## 1. 프로젝트 개요

### 1.1 목표

2.4GHz 대역 RF 신호를 수신하여 드론 의심 신호를 탐지하고,  
수신 신호의 종류와 방향 정보를 함께 제공하는 **RF 기반 드론 탐지 프로토타입**을 구현합니다.

- Pluto+ SDR을 이용한 2.4GHz RF 신호 수신
- RX0/RX1 2채널 IQ 데이터 처리
- Energy Detector 기반 1차 신호 탐지
- STFT 기반 spectrogram 생성
- 2D CNN 기반 RF 신호 분류 구조 구현
- Coherence 기반 AoA 신뢰도 검증
- RX0/RX1 위상차 기반 도래각(AoA) 추정
- 2.400~2.485GHz 대역 주파수 스캔 모드 구현
- 추후 Raspberry Pi 등 엣지 장치 배포 가능성 검토

### 1.2 하드웨어 구성

|부품                   |역할                    |
|---------------------|----------------------|
|Pluto+ SDR           |2채널 IQ 수신             |
|2.4GHz 안테나 ×2        |RX0/RX1 위상차 기반 AoA 추정 |
|노트북 / Raspberry Pi 후보|신호처리, CNN 추론, 결과 출력   |
|Python 실행 환경         |전체 pipeline 실행 및 결과 저장|

현재 개발은 하드웨어 의존성을 줄이기 위해 `sim`, `file`, `sdr` 입력을 모두 지원하는 구조로 진행합니다.  
Pluto+가 연결되지 않은 상황에서도 synthetic IQ 또는 저장된 IQ 파일로 pipeline을 검증할 수 있습니다.

-----

## 2. 현재 구현 기준

### 2.1 처리 단위

전체 pipeline의 기본 처리 단위는 **block**입니다.

|항목              |값             |
|----------------|-------------:|
|Sample rate     |5 MSPS        |
|Center frequency|2.4 GHz       |
|Block size      |16,384 samples|
|Block time      |약 3.28 ms     |
|Channel count   |2 channels    |
|기본 입력           |`sim`         |

```yaml
# configs/receiver.yaml 기본값
source_type: sim
sample_rate: 5000000
center_freq: 2400000000
block_size: 16384
num_samples: 16384
num_channels: 2
```

### 2.2 입력 소스

입력 소스는 `configs/receiver.yaml`의 `source_type`으로 선택합니다.  
입력 장치가 바뀌어도 뒤쪽 신호처리 pipeline은 그대로 유지됩니다.

|source_type|설명                |
|-----------|------------------|
|`sim`      |synthetic IQ 신호 생성|
|`file`     |저장된 IQ 파일 재생      |
|`sdr`      |Pluto+ SDR 실측 입력  |

```text
SimReceiver / RawFileReceiver / PlutoReceiver
                    ↓
              공통 pipeline
```

-----

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
[Energy Detector]  ──── 신호 없음 → skip
    ↓ 신호 있음
[Dual-channel STFT]
    ↓
[CNN Spectrogram 생성]
    ↓
[Coherence Gate]  ──── coherence 부족 → AoA skip
    ↓
[Phase Difference Estimation]
    ↓
[AoA Estimation]
    ↓
[summary.json / numpy artifacts 저장]
```

### 3.1 전처리

|처리                        |모듈                               |
|--------------------------|---------------------------------|
|DC offset 제거              |`src/preprocess/dc_blocker.py`   |
|IQ amplitude 정규화          |`src/preprocess/iq_normalizer.py`|
|RX0/RX1 gain mismatch 보정  |`src/preprocess/gain_matcher.py` |
|RX0/RX1 phase offset 추정/보정|`src/preprocess/phaseoffset.py`  |
|block / frame 분할          |`src/preprocess/framing.py`      |

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

|mode                 |설명                                       |
|---------------------|-----------------------------------------|
|`block_median`       |현재 block의 median energy를 noise floor로 사용 |
|`initial_calibration`|초기 calibration block을 기반으로 noise floor 고정|

현재 기본 방향: `block_median`으로 빠른 검증 → 실측 단계에서 `initial_calibration`으로 확장

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
CNN input shape     = (512, 125, 1)  →  PyTorch: (B, 1, 512, 125)
```

> 복소 IQ 신호를 사용하므로 양방향 주파수 성분을 포함하는 512개 frequency bin을 사용합니다.  
> (real signal STFT의 257 bin과 다름)

|모듈                           |역할                    |
|-----------------------------|----------------------|
|`src/features/fft.py`        |FFT 변환, 로그 스케일 변환     |
|`src/features/spectrogram.py`|STFT spectrogram 생성   |
|`src/features/window.py`     |Hanning/Hamming 윈도우 함수|

-----

## 4. CNN Classification

### 4.1 분류 클래스 / 입력 형태

|항목                |내용                                        |
|------------------|------------------------------------------|
|분류 클래스            |Background / WiFi / Bluetooth / Drone-like|
|입력 shape (외부)     |`(512, 125, 1)`                           |
|입력 shape (PyTorch)|`(B, 1, 512, 125)`                        |

### 4.2 모델 구조 — `SpectrogramCNN`

|항목|내용                                                                         |
|--|---------------------------------------------------------------------------|
|입력|`(B, 1, 512, 125)`                                                         |
|출력|4-class logits                                                             |
|구조|ConvBNAct → DepthwiseSeparableConv ×4 → AdaptiveAvgPool → Linear classifier|
|특징|Depthwise Separable Convolution 기반 경량 구조                                   |

|모듈                     |역할                  |
|-----------------------|--------------------|
|`src/ml/model_2dcnn.py`|SpectrogramCNN 모델 정의|
|`src/ml/inference.py`  |CNN 추론 래퍼           |
|`src/ml/transforms.py` |입력 전처리 변환           |

### 4.3 현재 CNN 상태

지원 inference backend:

|backend|설명                      |
|-------|------------------------|
|`dummy`|실제 모델 없이 pipeline 연결 테스트|
|`keras`|Keras `.keras` 모델 로드    |
|`torch`|PyTorch 모델 로드           |


> `model_path: null`인 경우 모델 구조 연결 테스트는 가능하지만,  
> 출력 결과를 실제 분류 성능으로 해석하면 안 됩니다.

```text
CNN 모델 구조          ✅ 구현됨
CNN inference interface ✅ 구현됨
학습 checkpoint 연결    🔄 진행 예정 또는 진행 중
실측 데이터 기반 검증   🔄 진행 예정
```

-----

## 5. AoA 추정

### 5.1 계산 개념

```yaml
# configs/aoa.yaml 기본값
carrier_freq: 2400000000
wavelength_m: 0.125
antenna_spacing_m: 0.0625   # λ/2
method: phase_difference
```

```text
theta = arcsin( (phase_diff × wavelength) / (2π × antenna_spacing) )
```

2채널 구조 기준 추정 범위: **-90° ~ +90°**

### 5.2 Coherence Gate

AoA 계산 전에 RX0/RX1이 같은 신호를 수신하고 있는지 coherence로 검증합니다.

```text
coherence threshold  = 0.60
energy_percentile    = 75.0   (상위 25% 에너지 bin만 사용)
```

|모듈                          |역할               |
|----------------------------|-----------------|
|`src/aoa/coherence.py`      |STFT coherence 계산|
|`src/aoa/phase_diff.py`     |채널 간 위상차 계산      |
|`src/aoa/angle_estimator.py`|arcsin 기반 방위각 산출 |
|`src/aoa/aoa_gate.py`       |AoA 진입 조건 판단     |

### 5.3 Calibration

전원 ON 또는 실행 조건에 따라 채널 간 gain/phase offset이 달라질 수 있습니다.  
실측 환경에서는 calibration block을 먼저 읽고 다음 값을 추정한 뒤 분석 block에 적용합니다.

- gain correction
- phase offset
- phase offset coherence-like score

```bash
PYTHONPATH=. python scripts/calibrate.py
```

|모듈                              |역할            |
|--------------------------------|--------------|
|`src/preprocess/gain_matcher.py`|채널 간 gain 보정  |
|`src/preprocess/phaseoffset.py` |위상 오프셋 추정 및 제거|

-----

## 6. Scan Mode

### 6.1 목적

SDR 수신 대역폭과 연산량 한계로 2.4GHz 대역 전체를 한 번에 분석하기 어렵습니다.  
가벼운 FFT scan으로 후보 주파수를 먼저 찾고, trigger된 주파수에서만 정밀 분석을 수행합니다.

|항목               |값        |
|-----------------|--------:|
|Start frequency  |2.400 GHz|
|Stop frequency   |2.485 GHz|
|Step frequency   |5 MHz    |
|Samples per block|16,384   |
|Scan blocks      |3        |
|Min pass blocks  |2        |

### 6.2 실행 흐름

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

|모듈                       |역할                                  |
|-------------------------|------------------------------------|
|`src/scan/scan_policy.py`|스캔 주파수 생성 및 후보 판정 정책                |
|`src/scan/scanner.py`              | FrequencyScanner|
|`src/scan/precision_analyzer.py`   | PrecisionAnalyzer|
|`scripts/run_scan.py`    |Scan mode 실행 진입점                    |

-----

## 7. 실행 방법

### 7.1 가상환경 활성화

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

### 7.2 단일 block pipeline 실행

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

### 7.3 Scan mode 실행

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

### 7.4 Pluto+ 실측 입력으로 변경

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

-----

## 8. 프로젝트 구조

```text
rf-drone-detection-capstone/
├── README.md
├── requirements.txt
│
├── configs/
│   ├── receiver.yaml       # 입력 소스, sample rate, block size, SDR 설정
│   ├── detect.yaml         # Energy Detector 설정
│   ├── ml.yaml             # STFT, CNN, inference 설정
│   ├── aoa.yaml            # AoA, antenna spacing, coherence 설정
│   ├── scan.yaml           # 2.4GHz scan mode 설정
│   ├── paths.yaml          # 데이터/출력 경로 설정
│   └── ui.yaml
│
├── scripts/
│   ├── run_pipeline.py     # 단일 block 분석 pipeline
│   ├── run_scan.py         # 2.4GHz 대역 scan mode
│   ├── calibrate.py        # calibration 실행
│   ├── check_*.py          # 단계별 검증용 스크립트
│   ├── train_model.py      # CNN 학습
│   ├── preprocess_dataset.py
│   └── export_model.py
│
├── src/
│   ├── receiver/           # Sim / File / Pluto receiver
│   ├── preprocess/         # DC 제거, 정규화, gain/phase 보정, framing
│   ├── features/           # FFT, STFT, spectrogram, window
│   ├── detect/             # Energy Detector
│   ├── aoa/                # Coherence, phase difference, angle estimation
│   ├── ml/                 # 2D CNN, inference, transforms
│   ├── scan/               # Frequency scanner, precision analyzer
│   ├── core/               # config, paths, types, utility
│   ├── ui/                 # 결과 출력 / plot
│   ├── calibration/
│   └── runtime/
│
├── data/
│   ├── raw_iq/
│   ├── processed/
│   ├── external/
│   ├── labels/
│   └── synthetic/
│
├── outputs/
│   ├── runs/
│   ├── figures/
│   ├── predictions/
│   └── exports/
│
├── models/
│   ├── checkpoints/
│   └── tflite/
│
└── docs/
    ├── planning/
    ├── experiments/
    ├── hardware/
    └── presentation/
```

-----

## 9. 현재 개발 현황

|모듈                                |상태              |
|----------------------------------|----------------|
|프로젝트 기본 구조                        |✅ 완료            |
|YAML 기반 설정 구조                     |✅ 완료            |
|SimReceiver                       |✅ 완료            |
|RawFileReceiver                   |✅ 완료            |
|PlutoReceiver 기본 구조               |✅ 구현됨 (실측 검증 필요)|
|DC offset 제거                      |✅ 완료            |
|IQ normalization                  |✅ 완료            |
|Gain mismatch correction          |✅ 완료            |
|Phase offset estimation/correction|✅ 완료            |
|Energy Detector                   |✅ 완료            |
|FFT feature 계산                    |✅ 완료            |
|STFT spectrogram 생성               |✅ 완료            |
|Dual-channel STFT branch          |✅ 완료            |
|Coherence gate                    |✅ 완료            |
|Phase difference 계산               |✅ 완료            |
|AoA 계산                            |✅ 완료            |
|2D CNN 모델 구조                      |✅ 완료            |
|CNN inference interface           |✅ 완료            |
|단일 block pipeline                 |✅ 완료            |
|Scan mode 기본 구조                   |✅ 완료            |
|Triggered precision analysis      |✅ 완료            |
|실제 학습 checkpoint 연결               |🔄 진행 중          |
|Runtime state machine             |🔄 설계 및 확장 예정    |
|UI dashboard                      |🔜 추후 확장         |
|TFLite / Raspberry Pi 배포          |🔜 추후 확장         |
|Pluto+ 실측 통합 테스트                  |🔜 진행 예정         |

-----

## 10. 데이터 전략

|단계    |데이터                                 |용도                      |
|------|------------------------------------|------------------------|
|1차 학습 |공개 RF 데이터셋                          |모델 초기 학습                |
|합성 데이터|synthetic spectrogram / synthetic IQ|클래스 불균형 보완 및 pipeline 검증|
|전이학습  |Pluto+ 직접 수집 데이터                    |실제 환경 파인튜닝              |
|실측 검증 |Wi-Fi / Bluetooth / Drone-like 환경 수집|최종 성능 평가                |

현재 단계에서는 **데이터셋 구성과 학습 checkpoint 연결**이 중요한 후속 과제입니다.

-----

## 11. 시연 시나리오

|단계            |행동                                  |기대 결과                                 |
|--------------|------------------------------------|--------------------------------------|
|1단계: 신호 탐지    |2.4GHz 대역 신호 발생                     |Energy Detection trigger              |
|2단계: 신호 분류    |Wi-Fi / Bluetooth / Drone-like 신호 입력|CNN class / confidence 출력             |
|3단계: 방향 추정    |송신원 위치를 좌우로 이동                      |AoA angle 변화                          |
|4단계: scan mode|2.4GHz 대역 sweep                     |후보 주파수 탐지 및 정밀 분석                     |
|5단계: 결과 저장    |pipeline 실행                         |`summary.json` / `scan_events.json` 저장|

-----

## 12. 한계 및 확장 방향

### 12.1 현재 한계

- 2채널 위상차 기반 AoA는 -90° ~ +90° 정면 반구 추정에 적합합니다.
- Wi-Fi / Bluetooth 신호는 환경에 따라 드론 신호와 혼재될 수 있습니다.
- CNN 성능은 데이터셋 품질과 실측 데이터 다양성에 크게 의존합니다.
- Pluto+ 실측 단계에서는 채널 동기성, gain, phase offset, antenna spacing 검증이 필요합니다.
- CNN 구조는 준비되어 있으나, 실제 성능은 학습 checkpoint와 검증 데이터가 연결되어야 평가할 수 있습니다.

### 12.2 확장 방향

|확장 항목 |내용                             |
|------|-------------------------------|
|전방위 탐지|다중 노드 또는 다중 안테나 구조로 확장         |
|고각 추정 |2D AoA 또는 팬틸트 구조 검토            |
|정밀 AoA|MUSIC/ESPRIT 등 고해상도 AoA 알고리즘 검토|
|광학 융합 |카메라 + YOLO 기반 시각 탐지와 결합        |
|엣지 배포 |TFLite / ONNX / Raspberry Pi 배포|
|실시간 UI|dashboard 기반 탐지 결과 시각화         |

-----

## 13. 개발 원칙

1. 하드웨어 입력과 신호처리 pipeline을 분리한다.
1. 모든 주요 파라미터는 YAML 설정 파일로 관리한다.
1. 전체 처리 단위는 block으로 통일한다.
1. Energy Detection은 빠른 1차 gate 역할을 한다.
1. STFT / CNN / AoA는 trigger 이후 정밀 분석 단계에서 수행한다.
1. CNN branch와 AoA branch는 독립적으로 검증 가능하게 유지한다.
1. 실제 장비 연결 전에도 `sim` 입력으로 전체 흐름을 검증할 수 있게 한다.
1. 결과는 `outputs/runs/latest/`에 저장하여 반복 실험을 쉽게 한다.
1. 구현된 기능과 계획 중인 기능을 README에서 명확히 구분한다.

-----

## 14. 기본 명령어

```bash
# 가상환경 활성화
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

# 단일 block pipeline 실행
PYTHONPATH=. python scripts/run_pipeline.py

# Scan mode 실행
PYTHONPATH=. python scripts/run_scan.py

# requirements 저장
pip freeze > requirements.txt

# Git 저장
git add .
git commit -m "docs: update README for current RF detection pipeline"
git push origin main
```

-----

## 15. 현재 프로젝트 상태 요약

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