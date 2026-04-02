# SDR 기반 비인가 드론 RF 신호 탐지 및 도래각 추정 모듈

2.4GHz 대역의 드론 제어 신호를 SDR(Software Defined Radio)로 수신하고,  
FFT 기반 신호처리와 에너지 검출, 1D CNN 분류, 위상차 기반 도래각(AoA) 추정을 통해  
드론 신호를 탐지·분류·시각화하는 캡스톤 프로젝트입니다.

---

## 1. 프로젝트 개요

본 프로젝트는 **저비용 RF 기반 드론 탐지 노드**를 목표로 합니다.  
고가의 통합 대드론 장비 전체를 구현하는 것이 아니라, 그중 **RF 탐지 계층**에 해당하는 핵심 기능을  
소형/저예산 환경에서 프로토타입 형태로 구현하는 것을 목표로 합니다.

프로젝트의 핵심 기능은 다음과 같습니다.

- SDR 기반 2.4GHz RF 신호 수신
- FFT 기반 스펙트럼 분석
- 에너지 기반 초기 탐지
- 1D CNN 기반 신호 분류
- 2채널 위상차 기반 도래각(AoA) 추정
- 실시간 대시보드 시각화

---

## 2. 개발 전략

본 프로젝트는 **하드웨어 의존 부분과 소프트웨어 파이프라인을 분리**하여 개발합니다.

즉, 초기에 하드웨어가 없더라도 다음 기능들을 먼저 구현합니다.

- 시뮬레이션 IQ 신호 생성
- Raw IQ 파일 기반 재생
- FFT 및 스펙트로그램 처리
- 에너지 검출기
- 1D CNN 분류기
- GUI 대시보드

이후 실제 HackRF 및 안테나가 준비되면,  
기존 파이프라인은 유지한 채 **입력 소스만 실제 RF 수신기로 교체**하는 방식으로 확장합니다.

---

## 3. 현재 목표

현재 1차 목표는 다음과 같습니다.

**Sim Receiver → FFT → Energy Detector**

즉, 하드웨어 없이도 돌아가는 최소 파이프라인을 먼저 완성한 뒤,  
점진적으로 분류기, AoA, GUI를 추가하는 방식으로 개발합니다.

---

## 4. 프로젝트 구조

```text
rf-drone-detection-capstone/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ configs/                  # 샘플레이트, FFT 크기, threshold 등 설정 파일
├─ docs/                     # 제안서, 발표자료, 실험 기록 문서
│  ├─ experiments/
│  └─ presentation/
├─ data/
│  ├─ sim/                   # 합성 데이터
│  ├─ raw_iq/                # 실측 원본 IQ 데이터
│  ├─ processed/             # 전처리된 데이터
│  └─ labels/                # 라벨 정보 또는 라벨링 결과
├─ models/
│  ├─ checkpoints/           # 학습 중간 저장 모델
│  ├─ reports/               # 평가 결과 (정확도, F1-score 등)
│  └─ tflite/                # 경량화된 배포용 모델
├─ notebooks/                # 실험용 노트북
├─ outputs/                  # 결과 이미지, 시각화 산출물
├─ logs/                     # 실행 로그
├─ scripts/                  # 실행 스크립트
├─ src/
│  ├─ receiver/              # 시뮬레이션/실측 IQ 입력
│  ├─ preprocess/            # IQ 정제, 필터링
│  ├─ features/              # FFT, 스펙트로그램
│  ├─ detect/                # 에너지 검출기
│  ├─ ml/                    # 1D CNN, Autoencoder, TFLite
│  ├─ aoa/                   # 도래각 추정
│  ├─ ui/                    # 대시보드
│  └─ core/                  # 공통 유틸, 파이프라인 연결
└─ tests/                    # 테스트 코드