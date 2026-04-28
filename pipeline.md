# 전체 파이프라인 기준 파일 정리

## 중복 표시 기준

| 표시 | 의미 |
|---|---|
| 🟦 | 여러 단계에서 참조되는 공통 설정/유틸 |
| 🟩 | Energy/CNN/AoA branch가 함께 쓰는 공통 전처리 |
| 🟨 | STFT/CNN/Stage1 저장에서 중복 사용 |
| 🟥 | AoA gate 이후 AoA branch에서 중복 사용 |
| 🟪 | 결과 저장, UI, 문서화에서 중복 사용 |
| ❌ | 삭제 후보 또는 현재 파이프라인에서 제외 |

---

## 1. 실시간 탐지 파이프라인

| 0. 설정/실행 진입 | 1. IQ 수신 | 2. block 전처리 | 3. Energy Gate | 4. CNN/STFT 분류 | 5. AoA Gate | 6. AoA 계산 | 7. 결과 저장/UI |
|---|---|---|---|---|---|---|---|
| `configs/receiver.yaml` 🟦 | `src/receiver/pluto_receiver.py` | `src/preprocess/dc_blocker.py` 🟩 | `configs/detect.yaml` 🟦 | `configs/ml.yaml` 🟦 | `src/aoa/aoa_gate.py` 🟥 | `src/preprocess/phaseoffset.py` 🟥 | `src/core/types.py` 🟪 |
| `configs/ml.yaml` 🟦 | `src/receiver/raw_file_receiver.py` | `src/preprocess/framing.py` 🟩 | `src/detect/energy_detector.py` | `src/preprocess/channel_filter.py` | `configs/aoa.yaml` 🟦 | `src/aoa/coherence.py` 🟥 | `src/core/stage1_artifact_store.py` 🟨🟪 |
| `configs/aoa.yaml` 🟦 | `src/receiver/sim_receiver.py` | `src/preprocess/iq_normalizer.py` 🟩 | `src/preprocess/framing.py` 🟩 | `src/features/spectrogram.py` 🟨 |  | `src/aoa/phase_diff.py` 🟥 | `src/core/raw_iq_store.py` 🟪 |
| `configs/paths.yaml` 🟦 | `src/receiver/base.py` | `src/preprocess/channel_filter.py` | `src/features/fft.py` | `src/features/window.py` 🟨 |  | `src/aoa/angle_estimator.py` 🟥 | `src/ui/result_plotter.py` |
| `src/core/config.py` 🟦 | `src/receiver/factory.py` |  |  | `src/ml/dataset.py` |  | `src/aoa/aoa_smoother.py` 🟥 | `src/ui/event_logger.py` |
| `src/core/pipeline.py` 🟦 |  |  |  | `src/ml/evaluate.py` |  |  | `src/ui/dashboard.py` |
| `scripts/run_pipeline.py` |  |  |  | 추후 `src/ml/inference.py` 필요 |  |  | `outputs/runs/latest/` |

---

## 2. 실시간 파이프라인 실행 순서

```text
[0] 설정 로드
configs/*.yaml
src/core/config.py
scripts/run_pipeline.py

        ↓

[1] IQ 수신
src/receiver/factory.py
src/receiver/pluto_receiver.py / raw_file_receiver.py / sim_receiver.py
출력: raw_block shape = (2, 16384)

        ↓

[2] 공통 전처리
src/preprocess/dc_blocker.py
출력: block_dc shape = (2, 16384)

        ↓

[3] Energy Gate
src/detect/energy_detector.py
신호 없음 → CNN/AoA skip
신호 있음 → CNN branch 진행

        ↓

[4] CNN/STFT Branch
src/preprocess/channel_filter.py
src/features/spectrogram.py
src/features/window.py
CNN 결과: class_name, confidence

        ↓

[5] AoA Gate
src/aoa/aoa_gate.py
Drone-like confidence >= 0.85 이면 pass
fail이면 AoA 계산 안 함

        ↓

[6] AoA Branch
src/preprocess/phaseoffset.py
src/aoa/coherence.py
src/aoa/phase_diff.py
src/aoa/angle_estimator.py
src/aoa/aoa_smoother.py

        ↓

[7] 결과 저장/UI
src/core/types.py
src/core/stage1_artifact_store.py
src/core/raw_iq_store.py
src/ui/result_plotter.py
outputs/runs/latest/