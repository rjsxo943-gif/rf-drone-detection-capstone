# SDR 기반 비인가 드론 RF 신호 탐지 및 도래각 추정 모듈

2.4GHz 대역의 드론 제어 신호를 PLUTO+ SDR 2채널로 동시 수신하고,  
FFT 기반 신호처리와 에너지 검출, 1D CNN 분류, 2채널 위상차 기반 도래각(AoA) 추정을 통해  
드론 신호를 탐지·분류·방향 추정·시각화하는 캡스톤 프로젝트입니다.

-----

## 1. 프로젝트 개요

본 프로젝트는 **저비용 RF 기반 드론 탐지 노드**를 목표로 합니다.  
고가의 통합 대드론 장비 전체를 구현하는 것이 아니라, 그중 **RF 탐지 계층**에 해당하는 핵심 기능을  
소형/저예산 환경에서 프로토타입 형태로 구현하는 것을 목표로 합니다.

### 핵심 기능

- PLUTO+ SDR 2채널 동시 수신 (AD9363 기반)
- GNU Radio 기반 실시간 수신 파이프라인
- DC 오프셋 제거 및 대역 필터링
- Hanning/Hamming 윈도우 적용 후 FFT 스펙트럼 분석
- 에너지 기반 1차 탐지 (CFAR 유사 임계치)
- 1D CNN 기반 신호 분류 (드론 / Wi-Fi / 블루투스 / 배경)
- 소프트웨어 캘리브레이션 기반 2채널 위상차 AoA 추정 (±90°)
- 라즈베리파이4B 기반 TFLite 엣지 AI 추론
- 실시간 PyQt5 대시보드 시각화

### 하드웨어 구성

|부품                        |역할                 |
|--------------------------|-------------------|
|PLUTO+ SDR (AD9363 2RX2TX)|2채널 동시 IQ 수신       |
|2.4GHz 패치 안테나 ×2          |위상차 AoA 듀얼 채널      |
|라즈베리파이4B 4GB              |TFLite 추론 + 대시보드 실행|

-----

## 2. 개발 전략

본 프로젝트는 **하드웨어 의존 부분과 소프트웨어 파이프라인을 분리**하여 개발합니다.

초기에 하드웨어가 없더라도 다음 기능들을 먼저 구현합니다.

- 시뮬레이션 IQ 신호 생성 (`sim_receiver.py`)
- Raw IQ 파일 기반 재생 (`raw_file_receiver.py`)
- FFT, 윈도우 함수, 스펙트로그램 처리
- 에너지 검출기
- 공개 데이터셋(DroneDetect, RFUAV 등) 기반 1D CNN 학습
- AoA 알고리즘 시뮬레이션 검증
- GUI 대시보드

이후 PLUTO+가 준비되면, 기존 파이프라인은 유지한 채  
**입력 소스만 `pluto_receiver.py`로 교체**하는 방식으로 확장합니다.

### AoA 캘리브레이션

PLUTO+는 전원 ON 시마다 채널 간 위상 오프셋이 랜덤 초기화됩니다.  
매 실행 전 `scripts/calibrate.py`를 실행하여 소프트웨어로 오프셋을 측정하고  
`configs/aoa.yaml`에 자동 저장합니다. 소요 시간은 약 1분입니다.

```bash
python scripts/calibrate.py
```

-----

## 3. 현재 목표

**1차 목표: Sim Receiver → Framing → Window → FFT → Energy Detector**

하드웨어 없이 돌아가는 최소 파이프라인을 먼저 완성한 뒤,  
점진적으로 CNN 분류기, AoA, GUI를 추가하는 방식으로 개발합니다.

-----

## 4. 데이터 전략

CNN 학습을 위해 공개 데이터셋과 자체 수집 데이터를 병행합니다.

|단계   |데이터                        |용도                |
|-----  |---------------------------  |------------------|
|1차 학습|DroneDetect, RFUAV, DroneRF  |공개 데이터셋으로 모델 초기 학습|
|전이학습 |직접 수집 (PLUTO+)           |실제 환경 파인튜닝        |

공개 데이터셋은 `data/datasets/`에,  
직접 수집한 원시 IQ는 `data/raw_iq/pluto/`에 저장합니다.

-----

## 5. 프로젝트 구조

```text
rf-drone-detection-capstone/
├── README.md
├── .gitignore
├── requirements.txt
│
├── configs/                         # 설정 파일 (샘플레이트, FFT 크기, 임계치 등)
│   ├── aoa.yaml                     # AoA 설정 + 캘리브레이션 오프셋 저장
│   ├── detect.yaml
│   ├── ml.yaml
│   ├── receiver.yaml
│   └── ui.yaml
│
├── data/
│   ├── datasets/                    # 공개 데이터셋 (git 제외)
│   │   ├── drone_detect/
│   │   ├── drone_rf/
│   │   └── cardrf/
│   ├── raw_iq/                      # 직접 수집한 원시 IQ (git 제외)
│   │   ├── pluto/
│   │   └── sim/
│   ├── processed/                   # 전처리 완료본 (git 제외)
│   └── labels/                      # 라벨 / 메타데이터
│
├── docs/
│   ├── experiments/                 # 실험 맥락 기록 (Markdown)
│   └── presentation/                # 발표자료
│
├── gnu_radio/
│   └── drone_detect.grc             # GNU Radio 플로우그래프
│
├── logs/                            # 실행 로그 (git 제외)
│
├── models/
│   ├── checkpoints/                 # 학습 중간 저장본 (git 제외)
│   ├── artifacts/                   # confusion matrix, F1 리포트 등
│   └── tflite/                      # 배포용 경량화 모델 (git 제외)
│
├── notebooks/
│   ├── 01_fft_window_experiment.ipynb
│   ├── 02_detector_tuning.ipynb
│   ├── 03_cnn_training.ipynb
│   ├── 04_tflite_validation.ipynb
│   └── 05_aoa_simulation.ipynb
│
├── outputs/
│   ├── figures/
│   ├── runs/
│   ├── predictions/
│   └── exports/
│
├── scripts/
│   ├── calibrate.py                 # PLUTO+ 위상 오프셋 캘리브레이션 자동화
│   ├── preprocess_dataset.py        # 공개 데이터셋 raw → processed 변환
│   └── run_pipeline.py              # 전체 파이프라인 실행 진입점
│
├── src/
│   ├── core/
│   │   ├── config.py                # YAML 로더
│   │   ├── logger.py                # 공통 로깅
│   │   ├── pipeline.py              # 전체 모듈 연결
│   │   └── types.py                 # IQFrame, DetectionResult 등 공통 타입
│   ├── receiver/
│   │   ├── pluto_receiver.py        # PLUTO+ 2채널 실측 수신
│   │   ├── sim_receiver.py          # 시뮬레이션 가상 IQ 생성
│   │   └── raw_file_receiver.py     # IQ 파일 재생 (테스트용)
│   ├── preprocess/
│   │   ├── framing.py               # 연속 IQ 스트림 → 고정길이 프레임 분할
│   │   ├── channel_filter.py        # 2.4GHz 관심 대역 필터
│   │   ├── dc_blocker.py            # DC 오프셋 제거
│   │   └── iq_normalizer.py         # IQ 진폭 정규화
│   ├── features/
│   │   ├── window.py                # Hanning/Hamming 윈도우 함수
│   │   ├── fft.py                   # FFT 변환, 로그 스케일 변환
│   │   └── spectrogram.py           # 시각화용 스펙트로그램 생성
│   ├── detect/
│   │   └── energy_detector.py       # 에너지 기반 1차 검출기
│   ├── ml/
│   │   ├── train_1dcnn.py           # 1D CNN 설계 및 학습
│   │   ├── export_tflite.py         # 학습 모델 → TFLite 변환
│   │   └── infer_tflite.py          # TFLite 추론
│   ├── aoa/
│   │   ├── phase_diff.py            # 2채널 위상차 계산
│   │   └── angle_estimator.py       # arcsin 기반 방위각 산출 + 오프셋 보정
│   └── ui/
│       └── dashboard.py             # PyQt5 실시간 대시보드
│
└── tests/
    ├── test_fft.py
    ├── test_detector.py
    ├── test_aoa.py
    ├── test_preprocess.py
    └── test_pipeline.py
```

-----

## 6. 시연 시나리오

|단계           |행동                 |화면 반응                        |
|---------------|--------------------|---------------------------------|
|1단계: 탐지     |드론 호버링          |스펙트로그램 신호 출현 + 탐지 알람 |
|2단계: 분류     |드론 끄고 Wi-Fi 켜기 |분류 결과 전환 (드론 → Wi-Fi)     |
|3단계: 방향 추정|드론 좌/우 이동      |방위각 표시 변화 (±90° 범위)       |

-----

## 6. AoA 한계 및 확장 방향

본 시스템은 수평면 기준 **±90° 정면 반구**만 탐지합니다.  
전방위 탐지가 필요한 경우 다수 노드 분산 배치로 확장 가능합니다.

|확장 항목  |내용                      |
|------    |--------------------------|
|전방위 탐지|노드 3개 이상 삼각측량      |
|고각 탐지  |팬틸트 2축 서보 추가       |
|정밀 AoA   |KrakenSDR + MUSIC 알고리즘|
|광학 융합  |RPi Camera + YOLO 결합    |