# 2026-06-03 개발 보고서
## Gain-wise Raw Noise Gate 기반 Scan 후보 생성 및 Representative Block 기반 CNN/AoA 정밀검사 개선

## 1. 개발 배경

RF 드론 탐지 프로젝트에서 기존 CLI scan/runtime pipeline은 scan 단계에서 후보 주파수를 찾은 뒤, 후보 주파수에서 CNN screening 및 precision hold를 수행하는 구조였다. 그러나 실제 실험 중 다음 문제가 확인되었다.

1. scan 단계에서 background 또는 주변 RF 신호가 후보로 올라올 수 있었다.
2. live OpenCV viewer에서는 드론 신호가 잘 보이지만, CLI 정밀검사에서는 CNN 판정이 안정적으로 이어지지 않았다.
3. 드론 신호가 2.450 GHz뿐 아니라 2.460 GHz, 2.465 GHz에서도 실제로 관측되었지만, 해당 대역은 학습 데이터에 충분히 포함되지 않아 CNN confidence가 낮게 나왔다.
4. representative block 구조를 적용한 뒤에는 `analyze()` 1회가 CNN vote 1개만 생성하게 되었고, 기존 `entry_screening.require_confirmed: true` 설정 때문에 precision hold에 진입하지 못했다.

오늘의 핵심 목표는 다음과 같았다.

```text
1. Scan 후보 생성은 gain-wise raw noise calibration 기반으로 단순화한다.
2. 후보 주파수 정밀검사는 여러 block 중 대표 block을 선택하는 방식으로 안정화한다.
3. CNN이 confirmed되기 전이라도 raw gate 통과 + CNN raw drone hit가 있으면 precision hold에 진입하도록 한다.
4. OpenCV live viewer에서 검증된 representative block 구조를 CLI runtime에도 반영한다.
```

---

## 2. 오늘 수정한 핵심 구조

### 2.1 Live RF Viewer 개선

대상 파일:

```text
scripts/live_rf_viewer_drone_aoa.py
configs/ui.yaml
```

기존 `live_rf_viewer_drone_aoa.py`는 한 번에 한 block을 읽고, 그 block을 바로 CNN voting과 AoA 판단에 사용했다. 이 방식은 업데이트가 빠르다는 장점이 있지만, 드론 burst 사이의 background block이 voting history에 들어가 CNN confirmed 상태가 쉽게 끊기는 문제가 있었다.

개선 후 구조는 다음과 같다.

```text
N개 block 수집
→ 각 block raw feature 계산
→ 각 block RawNoiseGate 평가
→ raw gate 통과 block 중 score_max가 가장 큰 대표 block 선택
→ selected block 하나만 CNN voting에 입력
→ CNN confirmed이면 selected block으로 AoA 계산
```

`configs/ui.yaml`에는 다음 설정을 추가하였다.

```yaml
live_rf_viewer:
  blocks_per_update: 20
  select_policy: raw_gate_pass_score_max
  cli_log_every_n: 1
  disable_cli_log: false
```

실험 결과 `blocks_per_update: 20`에서 CNN Drone 판정이 끊기지 않고 안정적으로 유지되는 것이 확인되었다.

---

### 2.2 CLI Scan 후보 생성 방식 변경

대상 파일:

```text
src/runtime/scan_loop.py
configs/detect.yaml
```

기존 scan 후보 생성은 `FrequencyScanner.scan_once()` 기반 threshold / pass count 구조에 의존했다. 오늘 수정 후 scan 후보는 `RawNoiseGate` 통과 여부만으로 단순화하였다.

변경된 scan stage:

```text
각 center frequency로 retune
→ 8 block 읽기
→ 앞 4 block discard
→ 뒤 4 block에 RawNoiseGate 적용
→ usable 4 block 중 1개 이상 raw gate 통과 시 후보 저장
```

`configs/detect.yaml` 추가 설정:

```yaml
scan_candidate:
  enabled: true
  blocks_per_freq: 8
  discard_blocks_after_tune: 4
  min_raw_gate_pass_count: 1
  max_candidates: 5
```

의미:

```text
pass_count=2/4
= usable 4 block 중 raw energy gate를 통과한 block이 2개
= CNN 통과 횟수가 아니라 raw gate 통과 횟수
```

즉 scan stage는 “드론인지”를 판단하지 않고, “현재 gain의 noise floor보다 충분히 큰 RF 신호가 있는지”만 판단한다.

---

### 2.3 CLI Candidate Verify 대표 block 구조 추가

대상 파일:

```text
src/scan/precision_analyzer.py
configs/detect.yaml
```

기존 `PrecisionAnalyzer.analyze()`는 `precision_blocks` 개수만큼 block을 읽고, 각 block마다 CNN을 수행하여 temporal voting을 업데이트했다. 이 구조에서는 드론 burst 사이의 background block이 CNN voting에 섞여 confirmed 상태가 만들어지기 어려웠다.

개선 후 구조:

```text
후보 주파수 진입
→ blocks_per_decision 개수만큼 block 읽기
→ 각 block RawNoiseGate 평가
→ raw gate pass block 중 score_max 최대 block 선택
→ selected block 하나만 STFT/CNN 수행
→ CNN voting 1회 업데이트
→ confirmed 상태이면 selected block으로 AoA 계산
```

`configs/detect.yaml` 추가 설정:

```yaml
candidate_verify:
  enabled: true
  representative_selection: true
  blocks_per_decision: 20
  select_policy: raw_gate_pass_score_max
  block_cnn_on_raw_gate_fail: true
  reset_temporal_on_raw_gate_fail: false
```

주의할 점은 representative 방식에서는 `analyze()` 1회가 CNN vote 1개만 만든다는 것이다. 따라서 `confirmed=True`를 entry screening 단계에서 요구하면 hold 진입이 매우 어려워진다.

---

### 2.4 Precision Hold 진입 조건 수정

대상 파일:

```text
src/runtime/scan_loop.py
configs/scan.yaml
```

문제 로그 예시:

```text
[CNN_SCREEN] cf=2.450000 GHz | cnn=Drone prob=1.0 thr=0.35 votes=1/5 candidate=False confirmed=False accepted=False reason=not_drone_rejected
```

CNN은 Drone 확률을 1.0으로 강하게 출력했지만, representative 방식에서는 `analyze()` 1회당 vote가 1개뿐이므로 `confirmed=False`가 된다. 기존 설정은 `require_confirmed: true`였기 때문에 precision hold에 진입하지 못했다.

개선 후 entry screening 조건:

```text
raw_gate_passed == True
and drone_probability >= entry_probability_threshold
→ precision hold 진입 허용
```

`configs/scan.yaml` 변경:

```yaml
precision_hold:
  entry_screening:
    enabled: true
    precision_blocks: 5

    require_confirmed: false
    allow_candidate: false

    accept_raw_drone_hit: true
    entry_probability_threshold: 0.35
    require_raw_gate_passed: true

    reject_not_drone: true
```

현재 `entry_probability_threshold: 0.35`는 개발 단계용 값이다. 2.460 GHz, 2.465 GHz 드론 데이터를 아직 CNN 학습에 충분히 넣지 않았기 때문에 낮게 열어둔 상태다. 데이터 보강 후에는 다음처럼 올릴 수 있다.

```text
데이터 보강 전: 0.35 ~ 0.50
데이터 보강 후: 0.65 ~ 0.80
```

---

## 3. 최종 Runtime 흐름

오늘 작업 후 CLI runtime은 다음 구조가 되었다.

```text
[Scan]
center frequency sweep
→ 각 주파수당 8 block read
→ 앞 4 block discard
→ 뒤 4 block RawNoiseGate 평가
→ raw gate pass_count >= 1이면 candidate 저장

[Candidate Verify]
candidate frequency로 retune
→ 여러 block read
→ RawNoiseGate 기반 대표 block 선택
→ selected block으로 CNN inference
→ CNN raw drone hit이면 precision hold 진입

[Precision Hold]
같은 후보 주파수에서 반복 analyze
→ representative block 기반 CNN vote 누적
→ confirmed=True 생성
→ confirmed 이후 AoA / sector 계산

[Return to Scan]
hold 종료 또는 signal lost 시 scan mode 복귀
```

---

## 4. 현재 확인된 실험 결과

### 4.1 Live Viewer

`configs/ui.yaml`에서 `blocks_per_update: 20`으로 설정했을 때 다음이 확인되었다.

```text
20개 block 중 대표 block 선택
→ CNN Drone 판정이 끊기지 않음
→ AoA 계산 조건에 더 안정적으로 도달
```

### 4.2 CLI Scan

RawNoiseGate 기반 scan 후보 생성은 성공하였다.

예시 로그:

```text
[RAW_SCAN] f=2.450000 GHz pass=1/4 trigger=True score=947.8987 thr=8.8124 label=RAW_GATE_PASS
[RAW_GATE_TRIGGER] 2.450 GHz | score=947.8987 | pass_count=1/4
```

### 4.3 Candidate Verify / Hold

기존에는 CNN이 Drone을 출력해도 `confirmed=False` 때문에 hold에 진입하지 못했다.

```text
cnn=Drone prob=1.0 thr=0.35 votes=1/5 confirmed=False accepted=False
```

entry screening 조건 수정 후에는 raw gate pass + CNN raw hit 조건으로 precision hold에 진입하는 데 성공하였다.

---

## 5. 코드 변경 요약

### 5.1 `configs/ui.yaml`

```yaml
live_rf_viewer:
  blocks_per_update: 20
  select_policy: raw_gate_pass_score_max
  cli_log_every_n: 1
  disable_cli_log: false
```

### 5.2 `configs/detect.yaml`

```yaml
scan_candidate:
  enabled: true
  blocks_per_freq: 8
  discard_blocks_after_tune: 4
  min_raw_gate_pass_count: 1
  max_candidates: 5

candidate_verify:
  enabled: true
  representative_selection: true
  blocks_per_decision: 20
  select_policy: raw_gate_pass_score_max
  block_cnn_on_raw_gate_fail: true
  reset_temporal_on_raw_gate_fail: false
```

### 5.3 `configs/scan.yaml`

```yaml
precision_hold:
  entry_screening:
    enabled: true
    precision_blocks: 5
    require_confirmed: false
    allow_candidate: false
    accept_raw_drone_hit: true
    entry_probability_threshold: 0.35
    require_raw_gate_passed: true
    reject_not_drone: true
```

### 5.4 핵심 Python 로직 요약

#### RawNoiseGate scan 후보 생성

```python
for center_freq in scan_freqs:
    set_receiver_center_freq(center_freq)

    gate_results = []
    for block_idx in range(blocks_per_freq):
        iq_block = read_block()

        if block_idx < discard_blocks_after_tune:
            continue

        result = raw_gate.evaluate(iq_block, gain=current_gain)
        gate_results.append(result)

    pass_count = sum(result.passed for result in gate_results)
    triggered = pass_count >= min_raw_gate_pass_count
```

#### Candidate verify 대표 block 선택

```python
blocks = []
raw_gate_results = []

for _ in range(blocks_per_decision):
    iq_block = read_block()
    result = raw_gate.evaluate(iq_block, gain=current_gain)
    blocks.append(iq_block)
    raw_gate_results.append(result)

passed_indices = [
    i for i, result in enumerate(raw_gate_results)
    if (not result.enabled) or result.passed
]

if passed_indices:
    selected_idx = max(
        passed_indices,
        key=lambda i: raw_gate_results[i].score_max,
    )
else:
    selected_idx = max(
        range(len(raw_gate_results)),
        key=lambda i: raw_gate_results[i].score_max,
    )

selected_block = blocks[selected_idx]
```

#### Entry screening hold 진입 조건

```python
raw_drone_hit = drone_probability >= entry_probability_threshold
raw_gate_ok = raw_gate_passed if require_raw_gate_passed else True
strong_entry_hit = raw_drone_hit and raw_gate_ok

if require_confirmed:
    accepted = confirmed_status
elif accept_raw_drone_hit:
    accepted = strong_entry_hit
elif allow_candidate:
    accepted = confirmed_status or candidate_status
else:
    accepted = confirmed_status
```

---

## 6. 현재 한계 및 원인 분석

### 6.1 2.460 / 2.465 GHz CNN confidence 문제

실제 viewer로 확인한 결과, 드론 신호는 2.450 GHz뿐 아니라 2.460 GHz, 2.465 GHz에서도 관측되었다. 그러나 해당 대역의 드론 spectrogram 패턴이 CNN 학습 데이터에 충분히 포함되지 않아 confidence가 낮게 나오는 문제가 있다.

예시:

```text
2.450 GHz → Drone prob ≈ 1.0
2.465 GHz → Drone prob ≈ 0.66
2.460 GHz → Drone probability가 threshold 근처에서 흔들림
```

이는 raw gate가 실패한 것이 아니라, CNN 학습 데이터 분포가 실제 scan 관측 범위를 충분히 포함하지 못한 문제로 해석된다.

### 6.2 Scan 후보는 넓게, Hold 진입은 점진적으로 보수화

현재는 데이터 보강 전이므로 hold 진입 기준을 낮게 열어두었다.

```text
entry_probability_threshold: 0.35
```

향후 2.460 / 2.465 GHz 드론 positive 데이터를 추가 학습한 뒤에는 기준을 상향한다.

```text
1차 상향: 0.50
2차 상향: 0.65
최종 안정화: 0.80 근처 검토
```

---

## 7. 다음 작업

### 7.1 데이터 보강

추가 수집이 필요한 positive 데이터:

```text
Drone ON:
- 2.450 GHz
- 2.455 GHz
- 2.460 GHz
- 2.465 GHz
- 가능하면 2.440 / 2.445 GHz 일부
```

추가 수집이 필요한 negative 데이터:

```text
Drone OFF:
- 동일 center frequency들
- Background
- Wi-Fi / Bluetooth / 기타 2.4 GHz 간섭
```

### 7.2 CNN 재학습

데이터 보강 후 다음을 수행한다.

```text
1. 2.460 / 2.465 GHz Drone positive 추가
2. 같은 대역 NotDrone negative 추가
3. CNN fine-tuning
4. entry_probability_threshold를 0.35에서 0.65~0.80으로 상향
5. CLI scan/runtime에서 재검증
```

### 7.3 코드 안정화 후 공통화

현재는 빠른 실험 안정화를 위해 `live_rf_viewer_drone_aoa.py`와 `PrecisionAnalyzer`에 각각 representative selection 로직을 넣었다. 향후 안정화가 끝나면 다음 공통 모듈로 분리한다.

```text
src/runtime/representative_block_selector.py
```

공통화 대상:

```text
- live_rf_viewer_drone_aoa.py
- src/scan/precision_analyzer.py
- 향후 CLI / 자동 수집 / dataset capture pipeline
```

---

## 8. 결론

오늘 작업으로 CLI runtime의 핵심 병목이 해결되었다.

```text
Before:
scan trigger는 되지만 CNN confirmed가 만들어지지 않아 precision hold 진입 실패

After:
RawNoiseGate scan 후보 생성
→ representative block 기반 candidate verify
→ raw gate pass + CNN raw hit 조건으로 precision hold 진입 성공
```

이제 프로젝트는 단순 viewer 실험 단계를 넘어, 실제 CLI scan/runtime pipeline에서 후보 탐색과 정밀 추적을 연결하는 구조로 발전했다. 다음 핵심 과제는 2.460 / 2.465 GHz 등 실제 드론 신호가 관측되는 인접 대역을 학습 데이터에 포함하여 CNN 일반화 성능을 높이는 것이다.
