# ══════════════════════════════════════════

# rf-drone-detection-capstone

# Python 패키지 목록

# ══════════════════════════════════════════

# 

# [설치]

# pip install -r requirements.txt

# 

# [시스템 패키지 - apt 별도 설치]

# sudo apt install gnuradio

# sudo apt install libiio-dev python3-libiio

# 

# [라즈베리파이4B]

# tensorflow 주석 처리 후 tflite-runtime 주석 해제

# 

# ══════════════════════════════════════════

# ── SDR / PLUTO+ ──────────────────────────

pyadi-iio           # PLUTO+ 2채널 수신

# ── 신호처리 ───────────────────────────────

numpy               # FFT, 위상차 계산
scipy               # 대역 필터 설계

# ── 설정 파일 ──────────────────────────────

pyyaml              # configs/*.yaml 로더

# ── ML / 학습 (PC 환경) ────────────────────

tensorflow          # 1D CNN 학습 + TFLite 변환
scikit-learn        # confusion matrix, F1-score

# tflite-runtime    # ← 라즈베리파이4B에서만 사용

# ── 데이터 처리 ────────────────────────────

pandas              # 라벨, 메타데이터
h5py                # 데이터셋 파일 I/O

# ── 시각화 / 대시보드 ──────────────────────

matplotlib          # 스펙트로그램, 학습 곡선
PyQt5               # 실시간 대시보드
pyqtgraph           # 실시간 플롯 (matplotlib보다 빠름)

# ── 기타 ───────────────────────────────────

tqdm                # 진행 표시
jupyter             # 노트북
ipykernel           # 노트북 커널