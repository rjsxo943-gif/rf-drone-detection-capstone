# RF Drone Detection Capstone 현재 파일/기능 진행상황

## 0. 현재 기준

이 프로젝트는 앞으로 `segment` 단위가 아니라 **block 단위**로 처리한다.

| 항목 | 기준 |
|---|---:|
| 처리 단위 이름 | block |
| block size | 16,384 samples |
| sample rate | 5 MSPS 기준 |
| block 시간 길이 | 약 3.28 ms |
| 결과 인덱스 | `block_index` |
| STFT 기준 입력 | IQ block 1개 |
| CNN 기준 입력 | block spectrogram 1개 |
| AoA 기준 입력 | RX0/RX1의 같은 `block_index` |

---

## 1. 전체 파이프라인 기준

```text
Pluto+ RX0/RX1 IQ 수신
        ↓
16,384 samples 단위 block 분할
        ↓
block 단위 DC offset 제거
        ↓
block 단위 IQ normalization
        ↓
Branch A: block spectrogram 생성 → CNN 분류
Branch B: RX0/RX1 위상차 계산 → AoA 추정
        ↓
block_index 기준으로 CNN 결과 + AoA 결과 결합
        ↓
결과 저장 / 시각화 / 대시보드 출력
```

---

## 2. 상태 표시 기준

| 표시 | 의미 |
|---|---|
| ✅ | 이미 존재하고 유지 |
| 🟡 | 이미 존재하지만 수정/점검 필요 |
| ⬜ | 아직 만들어야 함 |
| ⏳ | 후반부에 만들 예정 |
| ❌ | 삭제 후보 / 현재 사용하지 않음 |

---

## 3. 최상위 파일/폴더 현황

| 파일/폴더 | 상태 | 판단 |
|---|---:|---|
| `README.md` | ✅ | 프로젝트 설명 파일 |
| `requirements.txt` | 🟡 | `scipy` 포함 여부 확인 필요 |
| `configs/` | ✅ | 설정 파일 폴더 |
| `data/` | ✅ | 데이터 저장 폴더 |
| `docs/` | ✅ | 문서 폴더 |
| `gnu_radio/` | 🟡 | GNU Radio 실험 파일 |
| `models/` | ✅ | 모델 저장 폴더 |
| `notebooks/` | ✅ | 실험용 노트북 폴더 |
| `outputs/` | ✅ | 실행 결과 저장 폴더 |
| `scripts/` | ✅ | 실행 스크립트 폴더 |
| `src/` | ✅ | 핵심 소스 코드 |
| `tests/` | ✅ | 테스트 코드 |
| `logs/` | ✅ | 로그 저장 폴더 |
| `capture_fixed_channel.py` | 🟡 | Pluto+ 단일 채널 실험용. `scripts/`로 이동 고려 |
| `findwifi.py` | 🟡 | WiFi 탐색 실험용. `scripts/`로 이동 고려 |
| `pluto+exp.py` | 🟡 | Pluto+ 실험용. 파일명 정리 필요 |
| `iq_ch13.npy` | 🟡 | 실험 데이터. `data/raw_iq/pluto/`로 이동 고려 |
| `asd.py` | ❌ | 임시 파일로 보임. 삭제 후보 |
| `-` | ❌ | 실수로 생성된 파일 가능성 높음. 삭제 후보 |
| `원칙` | 🟡 | 내용 확인 필요 |
| `작업용` | 🟡 | 내용 확인 필요 |

---

## 4. 설정 파일 현황

| 파일 | 상태 | 역할 | 수정 필요 |
|---|---:|---|---|
| `configs/receiver.yaml` | ✅ | SDR/file/sim 입력 설정 | Pluto+ 기준 확인 | 완료
| `configs/detect.yaml` | ✅ | 에너지 탐지 설정 | 유지 | 완료
| `configs/ml.yaml` | 🟡 | ML/STFT 설정 | `block_size: 16384` 명시 필요 | 완료
| `configs/aoa.yaml` | 🟡 | AoA 설정 | 안테나 간격, 위상 보정값 확인 |
| `configs/paths.yaml` | ✅ | 경로 설정 | 유지 |
| `configs/ui.yaml` | ✅ | UI 설정 | 후반부 사용 |

### `configs/ml.yaml`에 넣을 기준값 예시

```yaml
block:
  block_size: 16384

stft:
  nperseg: 512
  noverlap: 384
  hop_length: 128
  image_size: [64, 64]
```

---

## 5. 데이터 폴더 현황

| 경로 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `data/external/cardrf/` | 🟡 | 외부 RF 데이터셋 | 구조 확인 필요 |
| `data/external/drone_detect/` | 🟡 | 외부 드론 데이터셋 | 사용 가능성 확인 |
| `data/external/drone_rf/` | 🟡 | DroneRF 데이터셋 | 용량/형식 확인 |
| `data/interim/` | ✅ | 중간 처리 데이터 | 유지 |
| `data/labels/` | 🟡 | 라벨 저장 | label format 확정 필요 |
| `data/processed/` | 🟡 | CNN 입력 데이터 저장 | block spectrogram 저장 예정 |
| `data/raw_iq/pluto/` | ✅ | Pluto+ 원본 IQ 저장 | 핵심 |
| `data/raw_iq/hackrf/` | ❌ | HackRF 데이터 | 현재 Pluto+만 쓰므로 삭제 후보 |
| `data/splits/` | 🟡 | train/val/test split | CNN 학습 전 필요 |
| `data/synthetic/` | ✅ | 합성 데이터 | 테스트용 유지 |

---

## 6. 문서 폴더 현황

| 파일/폴더 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `docs/planning/COMMANDS.md` | ✅ | 자주 쓰는 명령어 정리 | 유지 |
| `docs/planning/STATUS.md` | ✅ | 현재 상태 정리 | 계속 갱신 |
| `docs/planning/priority_table.html` | 🟡 | 우선순위 표 | md 버전도 있으면 좋음 |
| `docs/hardware/requirement_hw.md` | ✅ | 하드웨어 요구사항 | 유지 |
| `docs/experiments/` | 🟡 | 실험 기록 | stage별 md 파일 추가 필요 |
| `docs/presentation/` | ✅ | 발표자료 저장 | 후반부 사용 |

### 추가하면 좋은 문서

| 파일 | 상태 | 역할 |
|---|---:|---|
| `docs/planning/FILE_STATUS.md` | ⬜ | 전체 파일 현황표 |
| `docs/planning/PIPELINE_BLOCK_OVERVIEW.md` | ⬜ | block 기반 전체 파이프라인 설명 |
| `docs/planning/CNN_HANDOFF.md` | ⬜ | CNN 담당 팀원 전달 문서 |
| `docs/experiments/stage1_branch.md` | ⬜ | Stage 1 branch 실험 기록 |
| `docs/experiments/stage1_coherence.md` | ⬜ | coherence 실험 기록 |
| `docs/experiments/stft_block_params.md` | ⬜ | block-STFT 파라미터 기록 |

---

## 7. 실행 스크립트 현황

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `scripts/run_pipeline.py` | ✅ | 전체 파이프라인 실행 | 핵심 유지 |
| `scripts/check_stage1_branch.py` | 🟡 | Stage 1 branch 확인 | block 기준 점검 필요 |
| `scripts/check_stage1_coherence.py` | 🟡 | RX0/RX1 coherence 확인 | `scipy` 문제 해결 필요 |
| `scripts/check_raw_iq_store.py` | ✅/🟡 | raw IQ 저장 확인 | block 저장 방식 확인 |
| `scripts/calibrate.py` | 🟡 | 보정 스크립트 | AoA 위상 보정과 연결 필요 |
| `scripts/preprocess_dataset.py` | 🟡 | 데이터 전처리 실행 | block spectrogram dataset 생성용으로 수정 |
| `scripts/train_model.py` | 🟡 | 모델 학습 실행 | 현재 1D/2D CNN 방향 확인 필요 |
| `scripts/export_model.py` | ⏳ | 모델 export | 후반부 사용 |

### 추가하면 좋은 스크립트

| 파일 | 상태 | 역할 |
|---|---:|---|
| `scripts/capture_pluto_iq.py` | ⬜ | Pluto+ IQ 수집 전용 |
| `scripts/make_block_spectrogram_dataset.py` | ⬜ | raw IQ → block spectrogram dataset 생성 |
| `scripts/evaluate_model.py` | ⬜ | 모델 평가 실행 |
| `scripts/run_realtime_demo.py` | ⏳ | 최종 실시간 데모 |

---

## 8. 핵심 소스 코드 현황

### 8.1 Core

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/core/config.py` | ✅ | YAML 설정 로드 | 유지 | 완료
| `src/core/pipeline.py` | 🟡 | 전체 파이프라인 로직 | block 기준 확인 필요 |
| `src/core/raw_iq_store.py` | ✅/🟡 | raw IQ 저장 | block 저장 옵션 확인 |
| `src/core/stage1_artifact_store.py` | ✅/🟡 | Stage 1 산출물 저장 | spectrogram 저장 on/off 구조에 적합 |
| `src/core/types.py` | 🟡 | 데이터 타입 정의 | `segment` 표현 있으면 `block`으로 수정 |
| `src/core/utils.py` | ✅ | 유틸 함수 | 유지 |

---

### 8.2 Receiver

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/receiver/base.py` | ✅ | receiver 공통 인터페이스 | 유지 | 완료
| `src/receiver/factory.py` | ✅ | receiver 선택 | 유지 | 완료  
| `src/receiver/pluto_receiver.py` | 🟡 | Pluto+ 수신 | 핵심 점검 대상 | 완료
| `src/receiver/raw_file_receiver.py` | ✅ | 저장된 IQ 읽기 | 유지 | 완료
| `src/receiver/sim_receiver.py` | ✅ | 시뮬레이션 입력 | 유지 | 완료
| `src/receiver/hackrf_receiver.py` | ❌ | HackRF 수신 | 현재 사용 안 함. 삭제 후보 | 삭제완료

---

### 8.3 Preprocess

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/preprocess/dc_blocker.py` | ✅ | DC offset 제거 | block 단위 적용 확인 | 완료
| `src/preprocess/iq_normalizer.py` | ✅ | IQ 정규화 | block 단위 적용 확인 | 완료
| `src/preprocess/framing.py` | 🟡 | IQ 분할 | segment가 아니라 block 기준으로 수정 | 완료
| `src/preprocess/channel_filter.py` | 🟡 | 채널 필터링 | 필요성 확인 | 완료 근데 쓸진모름

### Preprocess 핵심 수정

| 파일 | 수정 내용 |
|---|---|
| `framing.py` | `split_into_segments()` 같은 표현이 있으면 `split_into_blocks()`로 변경 |
| `dc_blocker.py` | block마다 평균 제거하는지 확인 |
| `iq_normalizer.py` | block마다 정규화하는지 확인 |

---

### 8.4 Features

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/features/fft.py` | ✅ | FFT 계산 | 유지 | 완료
| `src/features/spectrogram.py` | 🟡 | STFT spectrogram 생성 | block 입력 기준 확인 | 완료
| `src/features/window.py` | ✅/🟡 | window 함수 | STFT와 연결 확인 | 완료

### block 기준 STFT 예상 크기

| 항목 | 값 |
|---|---:|
| block size | 16,384 |
| `nperseg` | 512 |
| `noverlap` | 384 |
| hop length | 128 |
| frequency bins | 257 |
| time frames | 125 |
| 원본 spectrogram 크기 | 257 × 125 |
| CNN 입력 크기 | 64 × 64 × 1 예정 |

---

### 8.5 Detect

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/detect/energy_detector.py` | ✅ | 에너지 기반 탐지 | 현재 안정적인 부분 |

### 추가하면 좋은 파일

| 파일 | 상태 | 역할 |
|---|---:|---|
| `src/detect/event_detector.py` | ⬜ | 연속 block 탐지를 이벤트로 묶기 |
| `src/detect/noise_floor.py` | ⬜ | noise floor 계산 분리 |

---

### 8.6 ML

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/ml/dataset.py` | 🟡 | 데이터셋 로더 | block spectrogram 기준 확인 |
| `src/ml/evaluate.py` | 🟡 | 모델 평가 | 유지 |
| `src/ml/export_tflite.py` | ⏳ | TFLite 변환 | 후반부 |
| `src/ml/infer_tflite.py` | ⏳ | TFLite 추론 | 후반부 |
| `src/ml/synthetic_generator.py` | 🟡 | 합성 데이터 생성 | 보조용 |
| `src/ml/train_1dcnn.py` | 🟡 | 1D CNN 학습 | 현재 spectrogram 기반 2D CNN 방향과 맞는지 확인 필요 |

### ML 방향 판단

현재 목표는 아래 흐름이다.

```text
IQ block → STFT spectrogram → 2D CNN
```

그런데 현재 파일은 `train_1dcnn.py`가 있으므로, 2D CNN으로 갈 경우 아래 파일을 추가하는 것이 좋다.

| 파일 | 상태 | 역할 |
|---|---:|---|
| `src/ml/model_2dcnn.py` | ⬜ | spectrogram 입력용 2D CNN |
| `src/ml/train_2dcnn.py` | ⬜ | 2D CNN 학습 |
| `src/ml/inference.py` | ⬜ | 일반 모델 추론 |
| `src/ml/transforms.py` | ⬜ | spectrogram resize/normalize |

---

### 8.7 AoA

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/aoa/phase_diff.py` | ✅/🟡 | RX0/RX1 위상차 계산 | block 기준 확인 | 위상차만 계산 
| `src/aoa/angle_estimator.py` | ✅/🟡 | 위상차 → 각도 변환 | 보정값 필요 | 위상차를 각도로 계산
| `src/aoa/coherence.py` | 🟡 | coherence 계산 | Stage 1 핵심 |
| `src/aoa/__init__.py` | ✅ | 패키지 파일 | 유지 |

### 추가하면 좋은 파일

| 파일 | 상태 | 역할 |
|---|---:|---|
| `src/aoa/calibration.py` | ⬜ | 채널 위상 offset 보정 |
preprocessing에 넣음 phaseoffset.py로
| `src/aoa/aoa_gate.py` | ⬜ | Drone-like일 때만 AoA 활성화 | 드론 확률 85퍼 이상이면 aoa 계산 통과
| `src/aoa/aoa_smoother.py` | ⏳ | 각도 흔들림 완화 |

---

### 8.8 UI

| 파일 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `src/ui/dashboard.py` | 🟡 | 대시보드 | 후반부 연결 필요 |
| `src/ui/event_logger.py` | 🟡 | 이벤트 기록 | block 결과 저장과 연결 가능 |
| `src/ui/result_plotter.py` | ✅ | 결과 그래프 저장 | 유지 |

### 추가하면 좋은 파일

| 파일 | 상태 | 역할 |
|---|---:|---|
| `src/ui/spectrogram_plotter.py` | ⬜ | block spectrogram 확인용 이미지 저장 |
| `src/ui/aoa_display.py` | ⏳ | 방위각 표시 |

---

## 9. 테스트 코드 현황

| 파일/폴더 | 상태 | 역할 | 판단 |
|---|---:|---|---|
| `tests/test_aoa.py` | ✅/🟡 | AoA 테스트 | block 기준 확인 |
| `tests/test_detector.py` | ✅ | detector 테스트 | 유지 |
| `tests/test_fft.py` | ✅ | FFT 테스트 | 유지 |
| `tests/test_pipeline.py` | 🟡 | 전체 파이프라인 테스트 | block 기준 수정 |
| `tests/test_preprocess.py` | 🟡 | 전처리 테스트 | block size 16384 테스트 추가 |
| `tests/fixtures/` | ✅ | 테스트 데이터 | 유지 |

### 추가하면 좋은 테스트

| 파일 | 상태 | 역할 |
|---|---:|---|
| `tests/test_spectrogram.py` | ⬜ | block → spectrogram 크기 검증 |
| `tests/test_block_framing.py` | ⬜ | 16,384 단위 분할 검증 |
| `tests/test_raw_iq_store.py` | ⬜ | raw IQ 저장/로드 검증 |

---

## 10. 기능별 진행상황

| 기능 | 상태 | 판단 |
|---|---:|---|
| 프로젝트 폴더 구조 | ✅ | 충분히 구성됨 |
| 설정 파일 관리 | ✅/🟡 | `block_size` 추가 필요 |
| Pluto+ 수신 구조 | 🟡 | 파일 있음, 점검 필요 |
| raw IQ 저장 구조 | ✅/🟡 | 구조 있음 |
| block 분할 | 🟡 | segment → block 수정 필요 |
| DC offset 제거 | ✅/🟡 | block 적용 확인 |
| IQ normalization | ✅/🟡 | block 적용 확인 |
| FFT / 에너지 탐지 | ✅ | 안정적인 편 |
| STFT spectrogram | 🟡 | `scipy` 설치 후 확인 |
| Stage1 artifact 저장 | ✅/🟡 | 구조 있음 |
| coherence 확인 | 🟡 | 지금 막힌 핵심 |
| AoA 계산 | 🟡 | 기본 파일 있음 |
| CNN dataset | 🟡 | 파일은 있으나 방향 정리 필요 |
| CNN 학습 | 🟡 | 1D/2D 방향 결정 필요 |
| CNN + AoA fusion | ⬜ | 아직 없음 |
| UI dashboard | 🟡/⏳ | 파일은 있으나 후반부 |
| 테스트 코드 | ✅/🟡 | 있음, block 기준 수정 필요 |
| 문서화 | 🟡 | STATUS 있음, 실험 문서 추가 필요 |

---

## 11. 삭제 또는 정리 후보

### 삭제 후보

| 파일/폴더 | 판단 |
|---|---|
| `-` | 실수로 생성된 파일 가능성 높음 |
| `asd.py` | 임시 파일로 보임 |
| `data/raw_iq/hackrf/` | Pluto+만 사용할 경우 삭제 후보 |
| `src/receiver/hackrf_receiver.py` | HackRF 미사용 시 삭제 후보 |
| `gnu_radio/hackrf_rx.grc` | HackRF 미사용 시 보관 또는 삭제 |
| `__pycache__/` 전체 | Git에 올리면 안 됨 |

### 이동 후보

| 현재 위치 | 추천 위치 |
|---|---|
| `capture_fixed_channel.py` | `scripts/capture_fixed_channel.py` |
| `findwifi.py` | `scripts/findwifi.py` |
| `pluto+exp.py` | `scripts/pluto_exp.py` |
| `iq_ch13.npy` | `data/raw_iq/pluto/iq_ch13.npy` |

---

## 12. 지금 바로 우선순위

### 1순위: block 기준 통일 완료

| 작업 | 관련 파일 |
|---|---|
| `segment` 표현 제거 | 전체 코드/문서 |
| `block_size = 16384` 설정 추가 | `configs/ml.yaml` |
| block 분할 함수 확인 | `src/preprocess/framing.py` |
| `block_index` 기준 저장 | `src/core/types.py`, `src/core/stage1_artifact_store.py` | 완료

---

### 2순위: `scipy` 문제 해결

| 작업 | 관련 파일 |
|---|---|
| `scipy` 설치 | `.venv` |
| requirements 반영 | `requirements.txt` |
| coherence 스크립트 재실행 | `scripts/check_stage1_coherence.py` |

명령어:

```bash
pip install scipy
pip freeze > requirements.txt
PYTHONPATH=. python scripts/check_stage1_coherence.py
```

---

### 3순위: Stage 1 branch 확인

| 작업 | 관련 파일 |
|---|---|
| raw IQ block 입력 | `src/core/raw_iq_store.py` |
| block DC 제거 | `src/preprocess/dc_blocker.py` |
| block spectrogram 생성 | `src/features/spectrogram.py` |
| 산출물 저장 | `src/core/stage1_artifact_store.py` |
| 실행 확인 | `scripts/check_stage1_branch.py` |

---

### 4순위: CNN 방향 정리

| 선택지 | 판단 |
|---|---|
| 1D CNN | 현재 파일 존재: `src/ml/train_1dcnn.py` |
| 2D CNN | 현재 목표인 spectrogram 기반 분류와 더 잘 맞음 | 채택

추천 방향:

```text
src/ml/train_1dcnn.py는 유지하되,
spectrogram CNN용으로 아래 파일을 새로 만드는 것이 좋음.

src/ml/model_2dcnn.py
src/ml/train_2dcnn.py
src/ml/transforms.py
```

---

## 13. 현재 가장 중요한 파일 5개

| 우선순위 | 파일 | 이유 |
|---:|---|---|
| 1 | `configs/ml.yaml` | `block_size: 16384` 기준 설정 |
| 2 | `src/preprocess/framing.py` | IQ를 block으로 나누는 핵심 |
| 3 | `src/features/spectrogram.py` | block을 CNN 입력 spectrogram으로 변환 |
| 4 | `scripts/check_stage1_coherence.py` | RX0/RX1 coherence 확인 |
| 5 | `src/core/stage1_artifact_store.py` | block spectrogram 저장 on/off 구조 관리 |

---

## 14. 현재 결론

현재 프로젝트는 파일 기준으로 보면 꽤 많이 만들어져 있다.

다만 지금부터는 새 파일을 계속 늘리는 것보다 아래 4개를 먼저 해야 한다.

```text
1. segment 표현을 block으로 통일
2. block_size = 16384 기준으로 framing.py 수정
3. spectrogram.py가 block 하나를 입력받는지 확인
4. check_stage1_coherence.py 실행 성공시키기
```

현재 상태는 다음과 같이 볼 수 있다.

```text
[완료에 가까움]
- 프로젝트 폴더 구조
- 설정 파일 기본 구조
- 시뮬레이션 입력
- 파일 입력
- FFT
- 에너지 탐지
- 결과 저장 구조
- 테스트 폴더 기본 구조

[지금 진행 중]
- Pluto+ RX0/RX1 수신
- 16,384 block 처리 기준 통일
- STFT spectrogram 생성
- coherence 확인
- AoA 기본 계산

[아직 해야 함]
- block spectrogram dataset 생성
- 2D CNN 학습 구조 확정
- CNN inference
- CNN + AoA fusion
- 실시간 dashboard
- Raspberry Pi 배포
```

1. 처음 20 block 수집
2. 각 block마다:
   - DC offset 제거
   - gain ratio 계산
   - phase offset 계산
   - energy 계산

3. 20개 결과에서:
   - noise_floor = median(energy)
   - gain_correction = median(gain_correction_list)
   - phase_offset_rad = circular mean 또는 median

4. 이후 실제 detection block부터:
   - DC offset 제거
   - 고정 gain correction 적용
   - 고정 phase offset 적용
   - normalization
   - energy detection / STFT / AoA