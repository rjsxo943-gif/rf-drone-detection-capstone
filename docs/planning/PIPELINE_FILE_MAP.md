# RF 드론 탐지 프로젝트 파이프라인 기준 파일 정리

## 0. 표시 기준

| 표시 | 의미 |
|---|---|
| 🟦 | 공통 설정 / 실행 제어 |
| 🟩 | 공통 전처리 |
| 🟨 | CNN / STFT 관련 |
| 🟥 | AoA 관련 |
| 🟪 | 저장 / 결과 / UI |
| 🧪 | 실험 / 검증 / 학습용 |
| ❌ | 정리 또는 삭제 후보 |

---

# 1. 전체 실시간 파이프라인 파일 맵

| 순서 | 파이프라인 단계 | 주요 역할 | 관련 파일 | 사용 조건 |
|---:|---|---|---|---|
| 0 | 설정 로드 / 실행 시작 | YAML 설정을 읽고 전체 실행 시작 | 🟦 `configs/receiver.yaml`<br>🟦 `configs/detect.yaml`<br>🟦 `configs/ml.yaml`<br>🟦 `configs/aoa.yaml`<br>🟦 `configs/paths.yaml`<br>🟦 `configs/ui.yaml`<br>🟦 `src/core/config.py`<br>🟦 `scripts/run_pipeline.py` | 실행 시작 시 항상 |
| 1 | Receiver 선택 | sim/file/sdr 중 입력 소스 선택 | 🟦 `src/receiver/base.py`<br>🟦 `src/receiver/factory.py` | 실행 시작 시 항상 |
| 2 | IQ 수신 | 16,384 sample block 수신 | `src/receiver/pluto_receiver.py`<br>`src/receiver/raw_file_receiver.py`<br>`src/receiver/sim_receiver.py` | `source_type`에 따라 하나만 사용 |
| 3 | 공통 전처리 | block 단위 DC offset 제거 | 🟩 `src/preprocess/dc_blocker.py` | 매 block마다 사용 |
| 4 | 선택 전처리 | IQ amplitude 정규화 | 🟩 `src/preprocess/iq_normalizer.py` | CNN 실험 옵션 |
| 5 | Energy Gate | 신호가 있는 block인지 1차 판단 | `configs/detect.yaml`<br>`src/detect/energy_detector.py`<br>`src/preprocess/framing.py`<br>`src/features/fft.py`<br>`src/features/window.py` | 신호 탐색용 |
| 6 | Energy 결과 분기 | 신호 없으면 CNN/AoA skip | `src/detect/energy_detector.py` | `energy_detected=False`면 skip |
| 7 | CNN 입력 채널 선택 | RX0 또는 RX1 중 CNN용 채널 선택 | 🟨 `src/preprocess/channel_filter.py` | 신호 있음일 때 |
| 8 | STFT / Spectrogram 생성 | IQ block을 CNN 입력 spectrogram으로 변환 | 🟨 `src/features/spectrogram.py`<br>🟨 `src/features/window.py` | 신호 있음일 때 |
| 9 | CNN 분류 | Background/WiFi/Bluetooth/Drone-like 분류 | 🟨 `configs/ml.yaml`<br>🟨 `src/ml/dataset.py`<br>🟨 추후 `src/ml/inference.py` 필요 | 신호 있음일 때 |
| 10 | AoA Gate | Drone-like이고 confidence ≥ 0.85인지 판단 | 🟥 `src/aoa/aoa_gate.py`<br>🟥 `configs/aoa.yaml` | CNN 결과 이후 |
| 11 | AoA Branch 시작 | gate pass일 때만 AoA 계산 | 🟥 `src/preprocess/phaseoffset.py` | Drone-like ≥ 0.85일 때만 |
| 12 | Coherence 확인 | RX0/RX1 위상 관계 안정성 확인 | 🟥 `src/aoa/coherence.py` | AoA gate pass 후 |
| 13 | 위상차 계산 | RX0/RX1 phase difference 계산 | 🟥 `src/aoa/phase_diff.py` | AoA gate pass 후 |
| 14 | 각도 변환 | 위상차를 AoA 각도로 변환 | 🟥 `src/aoa/angle_estimator.py` | AoA gate pass 후 |
| 15 | 각도 smoothing | AoA 각도 흔들림 완화 | 🟥 `src/aoa/aoa_smoother.py` | 후반부 선택 기능 |
| 16 | 결과 구조화 | block_index 기준 결과 포맷 정리 | 🟪 `src/core/types.py` | 결과 저장 시 |
| 17 | Stage1 산출물 저장 | spectrogram/STFT/log magnitude 저장 | 🟪 `src/core/stage1_artifact_store.py` | 옵션 저장 |
| 18 | Raw IQ 저장 | 원본 IQ 저장 | 🟪 `src/core/raw_iq_store.py` | 수집/검증 시 |
| 19 | 결과 시각화 / UI | 그래프, 이벤트, 대시보드 출력 | 🟪 `src/ui/result_plotter.py`<br>🟪 `src/ui/event_logger.py`<br>🟪 `src/ui/dashboard.py` | 후반부 |
| 20 | 최종 출력 | block 단위 결과 저장 | 🟪 `outputs/runs/latest/` | 매 실행 결과 |

---

# 2. 실시간 실행 순서 요약

```text
[0] 설정 로드
configs/*.yaml
src/core/config.py
scripts/run_pipeline.py

        ↓

[1] Receiver 생성
src/receiver/factory.py

        ↓

[2] IQ block 수신
src/receiver/pluto_receiver.py
또는 raw_file_receiver.py
또는 sim_receiver.py

출력:
raw_block shape = (2, 16384)

        ↓

[3] 공통 전처리
src/preprocess/dc_blocker.py

출력:
block_dc shape = (2, 16384)

        ↓

[4] Energy Gate
src/detect/energy_detector.py

신호 없음:
    CNN skip
    AoA skip
    결과 저장

신호 있음:
    CNN branch 진행

        ↓

[5] CNN / STFT Branch
src/preprocess/channel_filter.py
src/features/spectrogram.py
src/features/window.py

출력:
class_name
confidence

        ↓

[6] AoA Gate
src/aoa/aoa_gate.py

조건:
class_name == Drone-like
confidence >= 0.85

Fail:
    AoA 계산 안 함

Pass:
    AoA Branch 실행

        ↓

[7] AoA Branch
src/preprocess/phaseoffset.py
src/aoa/coherence.py
src/aoa/phase_diff.py
src/aoa/angle_estimator.py
src/aoa/aoa_smoother.py

출력:
aoa_deg

        ↓

[8] 결과 저장 / UI
src/core/types.py
src/core/stage1_artifact_store.py
src/core/raw_iq_store.py
src/ui/result_plotter.py
src/ui/event_logger.py
src/ui/dashboard.py
outputs/runs/latest/
```

---

# 3. Branch별 파일 정리

## 3.1 공통 설정 / 실행

| 파일 | 역할 | 사용 여부 |
|---|---|---|
| 🟦 `configs/receiver.yaml` | 입력 소스, sample rate, block size 설정 | 항상 |
| 🟦 `configs/detect.yaml` | Energy detector 설정 | 항상 |
| 🟦 `configs/ml.yaml` | STFT/CNN 설정 | CNN branch |
| 🟦 `configs/aoa.yaml` | AoA/gate/smoothing 설정 | AoA branch |
| 🟦 `configs/paths.yaml` | 경로 설정 | 항상 |
| 🟦 `configs/ui.yaml` | UI 설정 | 후반부 |
| 🟦 `src/core/config.py` | YAML 설정 로드 | 항상 |
| 🟦 `src/core/pipeline.py` | 전체 파이프라인 연결 | 핵심 |
| 🟦 `scripts/run_pipeline.py` | 실행 진입점 | 핵심 |

---

## 3.2 IQ 수신 Branch

| 파일 | 역할 | 사용 조건 |
|---|---|---|
| `src/receiver/base.py` | Receiver 공통 인터페이스 | 항상 |
| `src/receiver/factory.py` | receiver 생성 | 항상 |
| `src/receiver/pluto_receiver.py` | Pluto+ 실시간 수신 | `source_type: sdr` |
| `src/receiver/raw_file_receiver.py` | 저장된 IQ 파일 읽기 | `source_type: file` |
| `src/receiver/sim_receiver.py` | synthetic IQ 생성 | `source_type: sim` |

---

## 3.3 공통 전처리 Branch

| 파일 | 역할 | 데이터 값 변경 여부 |
|---|---|---:|
| 🟩 `src/preprocess/dc_blocker.py` | block별 DC offset 제거 | ✅ |
| 🟩 `src/preprocess/iq_normalizer.py` | IQ amplitude 정규화 | ✅ |
| 🟩 `src/preprocess/framing.py` | block 분할 / energy frame 분할 | ❌/부분 |
| 🟩 `src/preprocess/channel_filter.py` | RX0/RX1 중 CNN용 채널 선택 | ❌ |
| 🟩 `src/preprocess/phaseoffset.py` | RX0/RX1 phase offset 보정 | ✅ |

---

## 3.4 Energy Gate Branch

| 파일 | 역할 | 비고 |
|---|---|---|
| `configs/detect.yaml` | frame_size, hop_size, threshold 설정 | 설정 |
| `src/detect/energy_detector.py` | block 내부 energy 계산 및 gate 판단 | 핵심 |
| `src/preprocess/framing.py` | block 내부 energy frame 생성 | 중복 사용 |
| `src/features/fft.py` | FFT 기반 energy 실험용 | 선택 |
| `src/features/window.py` | Hann/Hamming window 생성 | 중복 사용 |

---

## 3.5 CNN / STFT Branch

| 파일 | 역할 | 비고 |
|---|---|---|
| 🟨 `configs/ml.yaml` | STFT/CNN 입력 설정 | 핵심 |
| 🟨 `src/preprocess/channel_filter.py` | RX0 또는 RX1 선택 | 기본 RX0 |
| 🟨 `src/features/spectrogram.py` | STFT spectrogram 생성 | 핵심 |
| 🟨 `src/features/window.py` | STFT window 생성 | 공통 |
| 🟨 `src/ml/dataset.py` | CNN dataset loader | 학습/추론 |
| 🟨 추후 `src/ml/inference.py` | CNN 추론 | 아직 없음 |
| 🟨 `models/checkpoints/` | 학습된 모델 저장 | 추론 시 사용 |
| 🟨 `models/tflite/` | TFLite 모델 저장 | 후반부 |

---

## 3.6 AoA Gate / AoA Branch

| 파일 | 역할 | 실행 조건 |
|---|---|---|
| 🟥 `src/aoa/aoa_gate.py` | AoA 계산 여부 판단 | CNN 후 |
| 🟥 `src/preprocess/phaseoffset.py` | phase offset 보정 | gate pass 후 |
| 🟥 `src/aoa/coherence.py` | coherence 확인 | gate pass 후 |
| 🟥 `src/aoa/phase_diff.py` | RX0/RX1 위상차 계산 | gate pass 후 |
| 🟥 `src/aoa/angle_estimator.py` | 위상차 → 각도 변환 | gate pass 후 |
| 🟥 `src/aoa/aoa_smoother.py` | 각도 흔들림 완화 | 선택 |
| 🟥 `configs/aoa.yaml` | AoA 설정 | AoA branch |

---

## 3.7 저장 / UI Branch

| 파일/폴더 | 역할 | 비고 |
|---|---|---|
| 🟪 `src/core/types.py` | block 결과 데이터 구조 정의 | 중요 |
| 🟪 `src/core/stage1_artifact_store.py` | spectrogram/STFT 중간 산출물 저장 | Stage1 |
| 🟪 `src/core/raw_iq_store.py` | raw IQ 저장 | 수집/검증 |
| 🟪 `src/ui/result_plotter.py` | 결과 그래프 저장 | 유지 |
| 🟪 `src/ui/event_logger.py` | 이벤트 로그 저장 | 후반부 |
| 🟪 `src/ui/dashboard.py` | 대시보드 | 후반부 |
| 🟪 `outputs/runs/latest/` | 최신 실행 결과 | 항상 |
| 🟪 `outputs/figures/` | 그림 저장 | 보고서용 |
| 🟪 `outputs/predictions/` | 예측 결과 저장 | 후반부 |

---

# 4. 오프라인 학습 / 데이터셋 파이프라인

실시간 파이프라인과 별도로, CNN 학습용 데이터셋을 만들고 학습하는 흐름이다.

| 순서 | 단계 | 관련 파일/폴더 | 역할 |
|---:|---|---|---|
| 0 | 원본 데이터 | `data/raw_iq/pluto/`<br>`data/external/drone_rf/`<br>`data/external/drone_detect/`<br>`data/external/cardrf/`<br>`data/synthetic/` | 원본 IQ / 외부 데이터 |
| 1 | block 분할 | `src/preprocess/framing.py` | 16,384 sample block 생성 |
| 2 | DC 제거 | `src/preprocess/dc_blocker.py` | 공통 전처리 |
| 3 | spectrogram 생성 | `src/features/spectrogram.py` | CNN 입력 생성 |
| 4 | dataset 저장 | `data/processed/`<br>`data/labels/`<br>`data/splits/` | 학습용 데이터셋 |
| 5 | 전처리 실행 | `scripts/preprocess_dataset.py` | 데이터셋 생성 스크립트 |
| 6 | 모델 학습 | `scripts/train_model.py`<br>`src/ml/train_1dcnn.py` | 현재 1D CNN 파일 있음 |
| 7 | 평가 | `src/ml/evaluate.py`<br>`models/reports/` | 정확도/혼동행렬 |
| 8 | 모델 저장 | `models/checkpoints/` | `.keras` 등 |
| 9 | export | `scripts/export_model.py`<br>`src/ml/export_tflite.py`<br>`models/tflite/` | TFLite 변환 |

---

# 5. 검증 / 실험용 파일

| 목적 | 파일 | 현재 판단 |
|---|---|---|
| raw IQ 저장 확인 | 🧪 `scripts/check_raw_iq_store.py` | 유지 |
| Stage1 branch 확인 | 🧪 `scripts/check_stage1_branch.py` | 중요 |
| coherence 확인 | 🧪 `scripts/check_stage1_coherence.py` | ✅ 최근 통과 |
| 보정 실험 | 🧪 `scripts/calibrate.py` | AoA 보정과 연결 필요 |
| 테스트 | 🧪 `tests/test_aoa.py`<br>🧪 `tests/test_detector.py`<br>🧪 `tests/test_fft.py`<br>🧪 `tests/test_pipeline.py`<br>🧪 `tests/test_preprocess.py` | block 기준 확인 필요 |
| 노트북 실험 | 🧪 `notebooks/` | 선택 |
| GNU Radio 실험 | 🧪 `gnu_radio/sim_rx.grc` | 보조 |
| HackRF GNU Radio | ❌ `gnu_radio/hackrf_rx.grc` | HackRF 미사용이면 보관/삭제 |

---

# 6. 정리 후보 파일

| 파일/폴더 | 판단 |
|---|---|
| ❌ `-` | 실수로 생성된 파일 가능성 높음 |
| ❌ `asd.py` | 임시 파일로 보임 |
| `capture_fixed_channel.py` | `scripts/capture_fixed_channel.py`로 이동 후보 |
| `findwifi.py` | `scripts/findwifi.py`로 이동 후보 |
| `pluto+exp.py` | `scripts/pluto_exp.py`로 이동 후보 |
| `iq_ch13.npy` | `data/raw_iq/pluto/iq_ch13.npy`로 이동 후보 |
| `작업용` | 내용 확인 후 정리 |
| `파이프라인순서` | 문서라면 `docs/planning/`로 이동 후보 |
| `src/**/__pycache__/` | `.gitignore` 대상 |

---

# 7. 핵심 결론

현재 프로젝트 파일은 모두 매번 실행되는 것이 아니다.

## 항상 실행에 가까운 핵심 흐름

```text
config
→ receiver
→ dc_blocker
→ energy_detector
→ channel_filter
→ spectrogram
→ CNN inference
→ aoa_gate
→ result 저장
```

## 조건부 실행

```text
AoA 관련 파일들은 CNN이 Drone-like이고 confidence >= 0.85일 때만 실행한다.

phaseoffset.py
coherence.py
phase_diff.py
angle_estimator.py
aoa_smoother.py
```

## 실험/학습 전용

```text
check_stage1_*.py
train_model.py
preprocess_dataset.py
tests/
notebooks/
docs/
```