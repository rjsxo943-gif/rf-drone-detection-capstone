# 2026-05-31 RF Drone Detection 실험 명령어 모음집

이 문서는 실험 중 바로 복사해서 사용할 수 있도록 명령어만 따로 정리한 파일이다.

---

## 1. 기본 경로 이동

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

필요 시:

```bash
PYTHONPATH=. python -V
```

---

## 2. 문법 검사

오늘 수정/추가한 주요 파일을 검사한다.

```bash
python -m py_compile src/calibration/gain_phase_table.py
python -m py_compile src/calibration/__init__.py
python -m py_compile src/runtime/phase_calibration_runtime.py
python -m py_compile src/runtime/__init__.py
python -m py_compile scripts/calibrate_phase_offset_outdoor.py
python -m py_compile scripts/build_gain_phase_table.py
python -m py_compile scripts/run_pipeline.py
python -m py_compile scripts/live_siggen_aoa_viewer.py
```

전체 import 확인:

```bash
PYTHONPATH=. python - <<'PY'
from src.calibration import dominant_cluster_phase, build_gain_phase_table
from src.runtime import resolve_phase_offset_to_apply, apply_phase_offset_to_iq
print("phase calibration modules OK")
PY
```

---

## 3. 현재 phase calibration 파일 로드 테스트

```bash
PYTHONPATH=. python - <<'PY'
from src.runtime import resolve_phase_offset_to_apply, print_phase_calibration_state

state = resolve_phase_offset_to_apply(
    current_phase_path="configs/calibration/current_phase_offset.json",
)

print_phase_calibration_state(state)
PY
```

기대 출력 예시:

```text
=== Phase Calibration Runtime ===
enabled        : True
source         : current_phase_offset_only
quality        : OK
ref_offset     : +44.xxx deg
apply_offset   : +44.xxx deg
```

---

## 4. Robust Phase Calibration 실행

조건:
- SDR center frequency: 2450 MHz
- 신호발생기: 2452 MHz
- 거리: 약 1.5~2 m
- gain: 30
- 총 200 block 수집
- 앞 30 block discard
- 170 block 사용

```bash
PYTHONPATH=. python scripts/calibrate_phase_offset_outdoor.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --num-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --memo "outdoor_0deg_1p5m_gain30_sig2452_robust"
```

성공 기준:

```text
quality           : OK
phase_offset      : +44도 근처
phase_std         : 3도 이하
coherence median  : 0.70 이상
cluster blocks    : 80% 이상
```

출력 파일:

```text
configs/calibration/current_phase_offset.json
outputs/calibration/<timestamp>_phase_offset.../phase_offset_calibration.json
```

---

## 5. Gain Phase Table 제작

처음 테스트용으로 gain 25, 30, 35만 측정한다.

```bash
PYTHONPATH=. python scripts/build_gain_phase_table.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 25,30,35 \
  --reference-gain 30 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output configs/calibration/gain_phase_table_2450.json \
  --memo "outdoor_0deg_1p5m_sig2452_gain_phase_table"
```

확장 버전:

```bash
PYTHONPATH=. python scripts/build_gain_phase_table.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 20,22,25,27,30,32,35 \
  --reference-gain 30 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output configs/calibration/gain_phase_table_2450.json \
  --memo "outdoor_0deg_1p5m_sig2452_gain_phase_table_full"
```

출력 파일:

```text
configs/calibration/gain_phase_table_2450.json
```

---

## 6. Gain Table 적용 테스트

```bash
PYTHONPATH=. python - <<'PY'
from src.runtime import resolve_phase_offset_to_apply, print_phase_calibration_state

state = resolve_phase_offset_to_apply(
    current_phase_path="configs/calibration/current_phase_offset.json",
    gain_table_path="configs/calibration/gain_phase_table_2450.json",
    current_gain=30,
)

print_phase_calibration_state(state)
PY
```

gain 25 테스트:

```bash
PYTHONPATH=. python - <<'PY'
from src.runtime import resolve_phase_offset_to_apply, print_phase_calibration_state

state = resolve_phase_offset_to_apply(
    current_phase_path="configs/calibration/current_phase_offset.json",
    gain_table_path="configs/calibration/gain_phase_table_2450.json",
    current_gain=25,
)

print_phase_calibration_state(state)
PY
```

---

## 7. run_pipeline.py 적용 확인

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

확인할 출력:

```text
=== Phase Calibration Runtime ===
source         : current_phase_offset_only
apply_offset   : +XX.XXX deg
```

summary 확인:

```bash
cat outputs/runs/latest/summary.json
```

또는 필요한 항목만:

```bash
PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/runs/latest/summary.json")
data = json.loads(p.read_text(encoding="utf-8"))

keys = [
    "runtime_phase_calibration_enabled",
    "runtime_phase_calibration_source",
    "runtime_phase_calibration_quality",
    "runtime_phase_offset_to_apply_deg",
    "runtime_phase_uncertainty_deg",
    "phase_diff_deg",
    "angle_deg",
    "angle_valid",
]

for k in keys:
    print(f"{k}: {data.get(k)}")
PY
```

---

## 8. live_siggen_aoa_viewer.py 실행

현재 phase calibration만 적용:

```bash
PYTHONPATH=. python scripts/live_siggen_aoa_viewer.py \
  --enable-aoa \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json
```

gain table까지 적용:

```bash
PYTHONPATH=. python scripts/live_siggen_aoa_viewer.py \
  --enable-aoa \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json \
  --gain 30
```

수동 phase offset 적용 방식:

```bash
PYTHONPATH=. python scripts/live_siggen_aoa_viewer.py \
  --enable-aoa \
  --aoa-phase-offset-rad 0.7714
```

---

## 9. 경량 Spectrogram Viewer 실행

경량 viewer 파일명이 다음이라고 가정한다.

```text
scripts/live_spectrogram_only_viewer.py
```

실행 예시:

```bash
PYTHONPATH=. python scripts/live_spectrogram_only_viewer.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30
```

기본 조작:

```text
q : 종료
p : pause/resume
[ : gain 감소
] : gain 증가
```

---

## 10. AoA 관련 파일 검색

```bash
grep -R "estimate_phase_diff\|phase_diff_to_angle\|estimate_angle\|AoA\|aoa" -n scripts src | head -100
```

pycache 제외:

```bash
grep -R --exclude="*.pyc" "estimate_phase_diff\|phase_diff_to_angle\|estimate_angle\|AoA\|aoa" -n scripts src | head -100
```

---

## 11. Calibration 결과 파일 확인

최근 calibration 결과 보기:

```bash
find outputs/calibration -name "phase_offset_calibration.json" | sort | tail -10
```

특정 결과 보기:

```bash
cat configs/calibration/current_phase_offset.json
```

요약 출력:

```bash
PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path

p = Path("configs/calibration/current_phase_offset.json")
d = json.loads(p.read_text(encoding="utf-8"))

for k in [
    "quality",
    "phase_offset_deg",
    "phase_std_deg",
    "coherence_median",
    "cluster_blocks",
    "valid_blocks",
    "cluster_ratio",
    "gain",
    "center_freq",
    "signal_freq",
]:
    print(f"{k}: {d.get(k)}")
PY
```

---

## 12. Git 상태 확인

```bash
git status
```

수정 파일 확인:

```bash
git diff --stat
```

상세 diff:

```bash
git diff
```

추천 add:

```bash
git add src/calibration/gain_phase_table.py
git add src/calibration/__init__.py
git add src/runtime/phase_calibration_runtime.py
git add src/runtime/__init__.py
git add scripts/calibrate_phase_offset_outdoor.py
git add scripts/build_gain_phase_table.py
git add scripts/run_pipeline.py
git add scripts/live_siggen_aoa_viewer.py
```

commit 예시:

```bash
git commit -m "Add robust phase calibration and gain-dependent phase table"
```

---

## 13. 실험 전 체크리스트

```text
1. 신호발생기 주파수 확인: 2452 MHz
2. SDR center 확인: 2450 MHz
3. gain 확인: reference gain 30
4. 핫스팟/블루투스 가능하면 OFF
5. 폰/사람/노트북은 RX 근처에서 멀리
6. 송신원은 RX0/RX1 배열 정면 0도
7. 송신원-RX 거리 1.5~2 m
8. 안테나/케이블/RX 포트/간격 고정
9. robust calibration 먼저 수행
10. quality OK 확인 후 AoA 측정
```

---

## 14. 실험 후 기록할 값

```text
phase_offset_deg
phase_std_deg
coherence_median
cluster_ratio
quality
gain
center_freq
signal_freq
distance
memo
AoA angle_deg
angle_valid
coherence_passed
```
