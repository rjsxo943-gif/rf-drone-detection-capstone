# 2026-05-25 RF 드론 탐지 프로젝트 작업 보고서

## 1. 작업 개요

본 보고서는 2026년 5월 25일 수행한 RF 기반 드론 탐지 캡스톤 프로젝트 작업 내용을 정리한 문서이다.
오늘의 주요 목표는 Pluto+ SDR 수신 안정화, Wi-Fi 및 드론 RF 데이터 수집 검증, CNN 학습용 데이터 캡처 파이프라인 개선, 그리고 AoA 방향 추정 결과를 8개 sector로 양자화하는 기능을 추가하는 것이었다.

오늘 작업을 통해 SDR 수신 안정성 문제를 상당 부분 해결했고, Wi-Fi 및 Drone-like 신호에 대한 실제 `.npz` 데이터 저장을 성공적으로 확인하였다. 또한 기존 연속 각도 기반 AoA 출력의 불안정성을 줄이기 위해 sector 기반 방향 출력 구조를 추가하였다.

---

## 2. 실험 및 개발 환경

| 항목 | 내용 |
|---|---|
| 프로젝트 | RF 기반 드론 탐지 시스템 |
| 브랜치 | `experiment/stft-128-hop32` |
| SDR | Pluto+ SDR |
| 주요 대역 | 2.4 GHz ISM band |
| 주요 중심 주파수 | Wi-Fi: 2437 MHz / Drone-like: 2450~2465 MHz 근처 |
| 샘플링 레이트 | 5 MSPS |
| block size | 16384 samples |
| CNN 입력 형태 | `(128, 509, 1)` spectrogram |
| 실행 환경 | WSL Ubuntu + Python 가상환경 |

---

## 3. 오늘 수행한 주요 작업

### 3.1 SDR 전원 안정화

기존에는 SDR과 노트북 간 연결이 자주 불안정하게 끊기는 문제가 있었다. 오늘 외부 5V 전원을 추가로 연결한 뒤 SDR 수신이 안정화되었고, 장시간 CLI 기반 수집이 가능해졌다.

```text
이전 상태:
- SDR 연결이 자주 끊김
- 장시간 capture 불안정

개선 후:
- 외부 5V 전원 추가
- Data USB + 외부 전원 동시 연결
- Wi-Fi 및 Drone-like 데이터 수집 가능
```

결론적으로 Pluto+ SDR 사용 시 외부 5V 전원 공급이 안정성에 큰 영향을 주는 것으로 확인되었다.

---

### 3.2 Wi-Fi 신호 검출 및 데이터 수집 성공

2437 MHz Wi-Fi 채널을 대상으로 CNN Dataset Capture를 수행하였다. 초기에는 scan 설정이 제대로 반영되지 않아 전체 2.400~2.485 GHz 대역을 모두 스캔하는 문제가 있었지만, 이후 `configs/scan.yaml` 값을 정상 반영하도록 수정하였다.

Wi-Fi 수집 결과는 다음과 같다.

```text
label        : wifi
center_freq  : 2437 MHz
saved_count  : 30
file format  : .npz
```

업로드된 Wi-Fi 샘플을 확인한 결과, 모든 샘플이 정상적인 구조를 가지고 있었다.

```text
spectrogram shape : (128, 509)
cnn_input shape   : (128, 509, 1)
label             : wifi
```

스펙트로그램상에서도 Wi-Fi 특유의 burst 형태가 확인되었으며, 학습용 데이터로 사용할 수 있는 품질로 판단하였다.

---

### 3.3 Drone-like 신호 수집 성공

Wi-Fi 수집 파이프라인 검증 후, 실제 드론 신호를 대상으로 Drone-like 라벨 데이터 수집을 수행하였다. 업로드된 Drone-like 샘플들을 확인한 결과, Wi-Fi와 구분되는 강한 block/burst 형태가 관찰되었다.

확인된 주요 특성은 다음과 같다.

```text
label             : drone_like
spectrogram shape : (128, 509)
cnn_input shape   : (128, 509, 1)
주요 주파수        : 2450 / 2460 / 2465 MHz 근처
```

현재 판단:

```text
- Drone-like RF 수집 성공
- CNN 학습용 데이터로 사용 가능
- Wi-Fi보다 강한 구간형 burst 패턴이 관찰됨
```

다만 Drone-like 데이터가 특정 주파수 대역에 몰릴 경우 모델이 “드론 신호의 패턴”이 아니라 “특정 주파수”를 외울 위험이 있으므로, 향후 거리·방향·위치·환경을 바꿔 추가 수집할 필요가 있다.

---

### 3.4 CNN Dataset Capture 구조 개선

기존 문제는 다음과 같았다.

```text
CNN Dataset Capture 모드에서 configs/scan.yaml의 주파수 설정을 따르지 않고,
코드 내부 고정값으로 2400~2485 MHz 전체 대역을 스캔함.
```

수정 후에는 다음 항목들이 `configs/scan.yaml`에서 반영되도록 개선하였다.

```text
start_freq
stop_freq
step_freq
scan_blocks
min_pass_blocks
scan_score_db_threshold
precision_blocks_per_candidate
settle_sec
```

수정 후 정상 동작 예시는 다음과 같다.

```text
scan freqs : 1 freqs
[scan] f=2.4370 GHz block=1/5 ...
```

또는 드론 대역 확인 시:

```text
scan range: 2.435 GHz ~ 2.465 GHz
scan freqs MHz: [2435.0, 2440.0, 2445.0, 2450.0, 2455.0, 2460.0, 2465.0]
```

---

### 3.5 FFT score 기반 candidate 판정 추가

기존 데이터 수집에서는 `detection_ratio`가 계속 0으로 나와 candidate로 인정되지 않는 문제가 있었다. 실제 로그에서는 FFT score가 강하게 튀는 구간이 존재했음에도 저장이 되지 않았다.

이를 해결하기 위해 다음과 같은 조건을 추가하였다.

```text
기존:
- detection_ratio >= min_detection_ratio 인 경우만 통과

개선:
- ratio_passed OR score_passed
```

즉, `detection_ratio`가 낮더라도 FFT score가 특정 threshold 이상이면 candidate로 인정하도록 수정하였다.

관련 설정 예시는 다음과 같다.

```yaml
scan_score_db_threshold: 76.0
precision_blocks_per_candidate: 10
settle_sec: 0.0
```

이 수정 이후 Wi-Fi 및 Drone-like 샘플 저장이 실제로 가능해졌다.

---

### 3.6 AoA 8-sector quantizer 추가

실험 중 사람의 몸 움직임, 반사파, 굴절파, 실내 다중경로 영향으로 AoA 연속 각도값이 크게 흔들리는 것을 확인하였다. 따라서 연속적인 각도값을 그대로 출력하기보다는 전방 영역을 8개 sector로 양자화하는 방식이 더 현실적이라고 판단하였다.

새로 추가한 파일:

```text
src/aoa/sector_quantizer.py
```

기능:

```text
angle_deg → sector index
sector index → sector label
sector voting 구조 준비
```

8-sector 구조는 다음과 같다.

| Sector | 각도 범위 | Label |
|---|---:|---|
| 0 | -90.0° ~ -67.5° | Far Left |
| 1 | -67.5° ~ -45.0° | Left 2 |
| 2 | -45.0° ~ -22.5° | Left 1 |
| 3 | -22.5° ~ 0.0° | Center Left |
| 4 | 0.0° ~ 22.5° | Center Right |
| 5 | 22.5° ~ 45.0° | Right 1 |
| 6 | 45.0° ~ 67.5° | Right 2 |
| 7 | 67.5° ~ 90.0° | Far Right |

---

### 3.7 CLI에 AoA sector scan 메뉴 추가

기존 `[s]` 메뉴는 CNN Dataset Capture 용도로 사용되었기 때문에 AoA 방향 출력과 분리할 필요가 있었다. 이에 따라 CLI에 새로운 메뉴를 추가하였다.

```text
[a] AoA sector scan
```

역할:

```text
- PrecisionAnalyzer 기반 scan loop 실행
- angle_deg 출력
- 8-sector 방향 출력
- coherence 기반 sector_valid 출력
```

실행 예시:

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

CLI에서:

```text
select> a
```

---

### 3.8 Phase calibration 실제 적용

기존 문제:

```text
phase/gain calibration 결과는 CLI에 출력되지만,
실제 AoA 계산에서는 phase_offset_rad=0.0이 사용됨.
```

오늘 수정 후:

```text
[AoA CAL] phase_offset_rad loaded: -3.0816128165 rad
```

위와 같이 calibration 파일에서 phase offset을 읽어 `PrecisionAnalyzer`에 전달하도록 수정하였다.

수정된 흐름:

```text
scan_loop.py
→ load_calibration_params()
→ phase_offset_rad 읽기
→ PrecisionAnalyzer(..., phase_offset_rad=phase_offset_rad)
→ phase_diff_to_angle(..., phase_offset_rad=self.phase_offset_rad)
```

이제 AoA 계산은 phase/gain calibration 결과를 반영한다.

---

### 3.9 Coherence 기반 sector valid 처리

기존에는 coherence가 낮아도 sector가 계속 출력되어 Left/Right가 무작위로 튀는 문제가 있었다. 이를 개선하기 위해 coherence threshold를 통과한 경우에만 sector를 valid로 인정하도록 수정하였다.

수정 후 동작:

```text
coherence 낮음:
sector=None(Unknown)
sector_valid=False

coherence threshold 통과:
sector=4(Center Right)
sector_valid=True
```

현재 디버그용 threshold는 다음과 같이 설정하였다.

```yaml
coherence_threshold: 0.001
```

이 값은 매우 낮은 디버그용 기준이며, 실제 시연 또는 안정화 단계에서는 더 높은 threshold를 검토해야 한다.

---

## 4. 주요 실험 결과

### 4.1 Wi-Fi 데이터 수집 결과

```text
label       : wifi
saved_count : 30
center_freq : 2437 MHz
```

판단:

```text
- Wi-Fi 수집 성공
- spectrogram 구조 정상
- CNN 학습용 데이터로 사용 가능
```

---

### 4.2 Drone-like 데이터 수집 결과

```text
label       : drone_like
주요 대역    : 2450~2465 MHz
```

판단:

```text
- Drone-like RF 신호 수집 성공
- Wi-Fi와 다른 강한 burst/block 패턴 확인
- 학습용 데이터로 사용 가능
```

---

### 4.3 AoA sector scan 결과

실행 결과:

```text
[AoA CAL] phase_offset_rad loaded: -3.0816128165 rad
coherence_threshold: 0.001
```

일부 결과에서 sector가 출력되었다.

```text
sector=4(Center Right) | sector_valid=True
sector=2(Left 1)       | sector_valid=True
```

대부분의 경우는 coherence가 낮아 Unknown으로 처리되었다.

```text
sector=None(Unknown) | sector_valid=False
```

판단:

```text
- sector 출력 기능 자체는 성공
- phase calibration 적용 성공
- coherence 기반 Unknown 처리 성공
- 실제 방향 안정성은 아직 낮음
```

---

## 5. 현재 한계 및 문제점

### 5.1 AoA coherence가 매우 낮음

현재 로그에서 coherence는 대부분 다음 수준이다.

```text
0.0001 ~ 0.002 정도
```

이는 기존 이상적인 threshold인 0.6에 비해 매우 낮다. 따라서 현재 환경에서는 연속 각도 또는 단일 sector 출력을 그대로 신뢰하기 어렵다.

가능한 원인:

```text
- 실내 반사파 및 다중경로
- 사람 몸 움직임에 따른 흡수/반사 변화
- 안테나 간 위상 정렬 불안정
- phase/gain calibration 환경 부정확
- 2채널 동기 또는 수신 품질 문제
```

---

### 5.2 모든 주파수에서 trigger가 발생함

현재 scan range 내 7개 주파수에서 대부분 trigger가 발생하고 있다.

```text
2435 / 2440 / 2445 / 2450 / 2455 / 2460 / 2465 MHz
```

이는 드론 신호가 넓은 대역 또는 hopping 형태로 관찰되는 것일 수 있으나, threshold가 낮아 잡음 또는 주변 RF 활동까지 trigger될 가능성도 있다.

---

### 5.3 AoA sector가 아직 안정적이지 않음

현재 sector는 일부 출력되지만 주파수와 cycle에 따라 값이 튄다.

```text
Center Right
Center Left
Left 1
Unknown
```

따라서 최종 시연용으로 사용하려면 단일 결과가 아니라 voting 기반 안정화가 필요하다.

---

## 6. 변경된 주요 파일

현재 `git status` 기준 변경 파일은 다음과 같다.

```text
modified: configs/aoa.yaml
modified: configs/ml.yaml
modified: configs/receiver.yaml
modified: configs/scan.yaml
modified: src/aoa/__init__.py
modified: src/runtime/calibration_actions.py
modified: src/runtime/cli.py
modified: src/runtime/cnn_capture_actions.py
modified: src/runtime/scan_loop.py
modified: src/scan/precision_analyzer.py
modified: src/scan/scanner.py

untracked: scripts/aoa_stream_tone.py
untracked: scripts/debug_aoa_tone_interactive.py
untracked: scripts/debug_aoa_tone_live.py
untracked: scripts/ml/make_rf4_canonical01_dataset.py
untracked: scripts/ml/make_rf4_norm01_dataset.py
untracked: src/aoa/sector_quantizer.py
```

핵심 변경 파일:

| 파일 | 내용 |
|---|---|
| `src/runtime/cnn_capture_actions.py` | scan.yaml 기반 CNN Dataset Capture, FFT score 기반 저장 조건 |
| `src/runtime/scan_loop.py` | phase calibration 로드, AoA sector 출력 |
| `src/runtime/cli.py` | `[a] AoA sector scan` 메뉴 추가 |
| `src/scan/precision_analyzer.py` | phase offset 적용, sector result 추가, coherence 기반 sector valid |
| `src/aoa/sector_quantizer.py` | 8-sector 양자화 모듈 추가 |
| `src/scan/scanner.py` | 주파수 scan 디버그 및 실제 중심주파수 변경 관련 수정 |
| `configs/scan.yaml` | scan range, threshold, coherence threshold 등 실험 설정 |

---

## 7. 현재 추천 설정

### 7.1 Wi-Fi 수집용 예시

```yaml
scan:
  start_freq: 2437000000
  stop_freq: 2437000000
  step_freq: 5000000

  num_samples: 16384
  threshold: 1000000.0
  scan_blocks: 5
  min_pass_blocks: 1

  scan_score_db_threshold: 76.0
  precision_blocks_per_candidate: 10
  settle_sec: 0.0

  save_spectrogram: true
  save_stft: false
  cnn_enabled: true
```

### 7.2 Drone-like 수집용 예시

```yaml
scan:
  start_freq: 2435000000
  stop_freq: 2465000000
  step_freq: 5000000

  num_samples: 16384
  threshold: 1000000.0
  scan_blocks: 5
  min_pass_blocks: 1

  scan_score_db_threshold: 76.0
  precision_blocks_per_candidate: 10
  settle_sec: 0.0

  save_spectrogram: true
  save_stft: false
  cnn_enabled: false
```

### 7.3 AoA sector scan 디버그용 예시

```yaml
scan:
  start_freq: 2450000000
  stop_freq: 2460000000
  step_freq: 5000000

  num_samples: 16384
  threshold: 1000000.0
  scan_blocks: 5
  min_pass_blocks: 1

  coherence_threshold: 0.001

  save_spectrogram: true
  save_stft: false
  cnn_enabled: false
```

---

## 8. 다음 작업 계획

### 8.1 데이터셋 추가 수집

우선 목표:

```text
Wi-Fi       : 최소 100장 이상
Bluetooth  : 최소 100장 이상
Background : 최소 100장 이상
Drone-like : 최소 100장 이상
```

조건별로 분리 수집 권장:

```text
clean_front
body_move
side_reflection
near
far
```

---

### 8.2 RF4 모델 재학습

오늘 수집한 Wi-Fi 및 Drone-like 데이터를 포함하여 RF4 모델을 재학습해야 한다.

예상 단계:

```text
1. 새 수집 데이터 manifest 생성
2. train / val / test split
3. RF4 CNN 재학습
4. confusion matrix 확인
5. live inference 재검증
```

---

### 8.3 AoA sector voting 추가

현재는 cycle별 또는 frequency별 sector가 흔들리므로 voting이 필요하다.

추천 구조:

```text
1. coherence threshold 통과한 sector만 후보로 사용
2. 각 cycle에서 coherence가 가장 높은 sector 선택
3. 최근 5 cycle 중 3회 이상 같은 sector이면 최종 방향 출력
4. 조건 불충족 시 Unknown 또는 Unstable 출력
```

최종 목표 출력 예시:

```text
Direction: Center Right
Sector: 4 / 8
Status: Stable
```

또는:

```text
Direction: Unknown
Status: Low coherence / Multipath unstable
```

---

### 8.4 Phase/Gain calibration 재수행

현재 phase/gain calibration의 coherence_like가 낮은 편이다.

```text
coherence_like: 0.011496
```

따라서 다음 조건에서 calibration을 다시 수행하는 것이 좋다.

```text
- 신호원을 두 안테나의 정면 0도 방향에 배치
- 사람 움직임 최소화
- 안테나 간 거리 고정
- 주변 금속/벽 반사 최소화
- 가능하면 신호원과 안테나 사이 LOS 확보
```

---

## 9. Git 커밋 제안

핵심 기능 커밋:

```bash
git add configs/aoa.yaml configs/ml.yaml configs/receiver.yaml configs/scan.yaml \
  src/aoa/__init__.py src/aoa/sector_quantizer.py \
  src/runtime/cli.py src/runtime/cnn_capture_actions.py src/runtime/scan_loop.py \
  src/scan/precision_analyzer.py src/scan/scanner.py

git commit -m "feat: add configurable RF capture and AoA sector scan"
```

실험 스크립트 별도 커밋:

```bash
git add scripts/aoa_stream_tone.py \
  scripts/debug_aoa_tone_interactive.py \
  scripts/debug_aoa_tone_live.py \
  scripts/ml/make_rf4_canonical01_dataset.py \
  scripts/ml/make_rf4_norm01_dataset.py

git commit -m "chore: add RF and AoA debug scripts"
```

`src/runtime/calibration_actions.py`는 실제 의도한 수정인지 diff 확인 후 커밋 여부를 결정하는 것이 좋다.

```bash
git diff src/runtime/calibration_actions.py
```

---

## 10. 결론

오늘 작업을 통해 프로젝트는 다음 상태까지 진전되었다.

```text
SDR 수신 안정화 성공
Wi-Fi 데이터 수집 성공
Drone-like 데이터 수집 성공
CNN Dataset Capture 구조 개선
scan.yaml 기반 주파수 제어 성공
FFT score 기반 저장 조건 추가
AoA 8-sector quantizer 추가
CLI AoA sector scan 메뉴 추가
phase calibration 실제 적용
coherence 기반 Unknown 처리 적용
```

현재 CNN 데이터 수집 파이프라인은 실사용 가능한 수준에 도달했다. AoA 방향 추정은 기능적으로 연결되었으나, 실내 다중경로와 낮은 coherence 문제로 인해 안정화가 더 필요하다. 따라서 다음 단계는 데이터셋 추가 수집 및 RF4 모델 재학습, 그리고 AoA sector voting 안정화 구현이다.
