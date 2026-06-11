# SDR 기반 비인가 드론 RF 신호 탐지 및 AoA 추정 모듈

Pluto+ SDR 기반 2.4GHz RF 신호를 이용해 Wi-Fi / Bluetooth / Background와 구분되는 **조종기 기원 uplink-side RF activity**를 탐지하고, 2채널 IQ 데이터의 위상차를 이용해 도래각(AoA, Angle of Arrival)을 추정하는 캡스톤 프로젝트입니다.

본 프로젝트는 고가의 통합 대드론 장비 전체를 구현하는 것이 아니라, 그중 **RF 탐지 계층**에 해당하는 핵심 기능을 저비용 SDR 장비와 소프트웨어 신호처리 파이프라인으로 구현하는 것을 목표로 합니다.

본 시스템의 주 실행 흐름은 live viewer가 아니라 **CLI 기반 scan/runtime pipeline**입니다. Live viewer는 RF 패턴 확인, gain profile 저장, CNN/AoA 디버깅을 위한 보조 실험 도구로 사용합니다.

```text
사전 준비
Gain-wise Noise Calibration → Gain-wise Phase/Gain Calibration → Gain/Distance Strength Profile

본 실행
CLI Scan Runtime → FFT Sweep → Candidate Band → Precision CNN Voting
→ Coherence Check → AoA Sector Estimation → Result Logging
```

---

## 1. 프로젝트 개요

### 1.1 목표

2.4GHz 대역 RF 신호를 수신하여 드론 운용과 관련된 **조종기 기원 uplink-side RF activity**를 탐지하고, 수신 신호의 방향 정보를 함께 제공하는 RF 기반 탐지 프로토타입을 구현합니다.

주요 목표는 다음과 같습니다.

- Pluto+ SDR을 이용한 2.4GHz RF 신호 수신
- RX0/RX1 2채널 IQ 데이터 처리
- FFT 기반 scan mode로 신호가 존재하는 후보 대역 탐색
- 후보 대역에서 STFT spectrogram 생성
- CNN voting 기반 uplink-side RF activity / NonDrone 판정
- 조종기 기원 RF activity를 통한 드론 운용 가능성 탐지
- Coherence 기반 AoA 신뢰도 검증
- RX0/RX1 위상차 기반 도래각(AoA) 추정
- Gain-wise robust phase/gain calibration profile 구축
- Gain-wise noise profile 및 gain/distance strength profile 기반 수신 상태 관리
- OpenCV 기반 real-time viewer를 통한 RF 패턴, CNN, AoA sector 확인
- 향후 Raspberry Pi 등 엣지 장치 배포 가능성 검토

### 1.2 하드웨어 구성

| 부품 | 역할 |
|---|---|
| Pluto+ SDR | 2채널 IQ 수신 |
| 2.4GHz 안테나 ×2 | RX0/RX1 위상차 기반 AoA 추정 |
| 신호발생기 | AoA phase/gain calibration 및 각도 검증 |
| 노트북 | 신호처리, CNN 추론, CLI runtime / viewer 실행 |
| 드론 / 조종기 | 실측 RF 데이터 수집 대상 |
| Python 실행 환경 | 전체 pipeline 실행 및 결과 저장 |

---

## 2. 현재 구현 기준

### 2.1 처리 단위

전체 pipeline의 기본 처리 단위는 **block**입니다.

| 항목 | 값 |
|---|---:|
| Sample rate | 5 MSPS |
| 기본 center frequency | 2.45 GHz 실험 중심 |
| Block size | 16,384 samples |
| Block time | 약 3.28 ms |
| Channel count | 2 channels |
| SDR input | Pluto+ SDR |
| AoA calibration tone | 2452 MHz 권장 |
| Gain-wise calibration range | 20 / 25 / 30 / 35 / 40 dB |

```yaml
# configs/receiver.yaml 예시
source_type: sdr
sample_rate: 5000000
center_freq: 2450000000
block_size: 16384
num_samples: 16384
num_channels: 2

sdr:
  uri: ip:192.168.2.1
  channels: [0, 1]
  center_freq: 2450000000
  sample_rate: 5000000
  rf_bandwidth: 5000000
  gain_control_mode: manual
  gain: 30
  warmup_reads: 4
  retune_warmup_reads: 4

# configs/scan.yaml 핵심값
scan:
  scan_blocks: 4
  min_pass_blocks: 2
```

### 2.2 입력 소스

입력 소스는 `configs/receiver.yaml`의 `source_type`으로 선택합니다.

| source_type | 설명 |
|---|---|
| `sim` | synthetic IQ 신호 생성 |
| `file` | 저장된 IQ 파일 재생 |
| `sdr` | Pluto+ SDR 실측 입력 |

---

## 3. 전체 Pipeline 구조

본 프로젝트의 pipeline은 단순히 `Receiver → 전처리 → CNN → AoA`로 한 번에 흐르는 구조가 아니라, **사전 준비 단계**와 **runtime 상태 전이 단계**로 나뉩니다.

가장 중요한 원칙은 다음과 같습니다.

```text
1. Gain-wise noise calibration과 gain-wise phase/gain calibration은 runtime 중간이 아니라 실행 전 준비 단계에서 수행한다.
2. Scan mode는 FFT 기반으로 후보 대역을 찾는 단계이며, CNN/AoA를 바로 수행하지 않는다.
3. AoA는 독립적으로 항상 도는 branch가 아니라, 후보 대역이 CNN voting을 통과한 뒤에만 진입하는 정밀 추정 단계이다.
4. 절대세기 feature는 CNN 입력용 normalization 이후 값에서 계산하지 않는다.
5. Runtime에서는 현재 gain을 기준으로 noise threshold와 phase/gain calibration profile을 lookup한다.
```

```mermaid
flowchart TD
    A[Before Runtime] --> A1[Gain-wise Noise Calibration]
    A1 --> A2[Gain-wise Phase/Gain Calibration]
    A2 --> A3[Gain/Distance Strength Profile 준비]

    A3 --> B[SCAN Mode]

    B --> B0[Center Frequency Retune]
    B0 --> B0a[Retune Warmup<br/>4 Blocks Discard]
    B0a --> B1[4 Scan Blocks 수집]
    B1 --> B2[Raw IQ Safety Check<br/>clip / overload]
    B2 --> B3[DC Offset Removal]
    B3 --> B4[Gain-wise Noise Threshold Lookup]
    B4 --> B5[FFT Scan Score 계산]
    B5 --> B6{2 / 4 Blocks Passed?}

    B6 -- No --> B
    B6 -- Yes --> C[PRECISION Mode]

    C --> C1[Candidate Center Frequency 유지 또는 재설정]
    C1 --> C2[IQ Block 수집]
    C2 --> C3[Raw IQ Safety Check<br/>clip / overload]
    C3 --> C4{Too Strong?}
    C4 -- Yes --> C5[Gain 후보에서 한 단계 Down]
    C5 --> C6[Warmup Block Discard]
    C6 --> C2

    C4 -- No --> C7[DC Offset Removal]
    C7 --> C8[Raw Strength Feature 계산]
    C8 --> C9[STFT Spectrogram]
    C9 --> C10[Log Magnitude]
    C10 --> C11[Spectrogram Normalization]
    C11 --> C12[CNN Inference]
    C12 --> C13[CNN Voting]
    C13 --> C14{Voting Confirmed?}

    C14 -- No --> B
    C14 -- Yes --> D[TRACK / AOA Mode]

    D --> D1[현재 Gain 확인]
    D1 --> D2[phase_gain_by_gain_latest.json Lookup]
    D2 --> D3[RX1 Gain/Phase 보상]
    D3 --> D4[Coherence Check]
    D4 --> D5{Coherence OK?}

    D5 -- Yes --> D6[Phase Difference Estimation]
    D6 --> D7[AoA Angle 계산]
    D7 --> D8[Angle Smoothing / Sector Output]
    D8 --> E[CSV / JSON / summary 저장]

    D5 -- No --> D9[Coherence Hold]
    D9 --> D10{Recovered?}
    D10 -- Yes --> D4
    D10 -- No --> B
```

### 3.1 단계별 역할

| 단계 | 목적 | 주요 처리 |
|---|---|---|
| Before Runtime | 실행 전 기준값 확보 | gain-wise noise calibration, gain-wise phase/gain calibration, gain/distance strength profile |
| Scan Mode | 신호가 있는 후보 대역 탐색 | raw IQ safety, gain별 threshold, FFT scan score, candidate detection |
| Precision Mode | 후보 대역 정밀 판정 | raw feature 확인, STFT, CNN inference, voting |
| Track / AoA Mode | 확정 후보의 방향 추정 | gain/phase 보상, coherence check, phase difference, angle / sector |
| Viewer / Logging | 실험 확인 및 기록 | OpenCV viewer, CSV log, JSON, summary artifacts |

### 3.2 절대세기 feature 계산 위치

절대세기 feature는 CNN 입력용 normalization 이후 값에서 계산하지 않습니다.  
목적에 따라 다음처럼 구분합니다.

| 목적 | 계산 위치 | 예시 feature |
|---|---|---|
| Clip / overload 판단 | SDR에서 받은 raw IQ 기준 | `raw_abs_max`, overload flag, saturation ratio |
| Gain별 noise 기준 | gain별 background IQ block | `noise_floor`, `threshold`, safety summary |
| Gain-distance 비교 | DC offset 제거 후, amplitude normalization 전 IQ | `raw_abs_p99`, `raw_rms` |
| CNN 입력 | STFT 이후 log magnitude를 spectrogram normalization | `cnn_spectrogram` |
| AoA 계산 | RX0/RX1 위상 관계가 보존된 IQ | phase difference, coherence |

즉, `raw_abs_p99`, `raw_rms`는 CNN용 0~1 spectrogram에서 계산하지 않습니다. 이 값들은 gain과 거리 변화에 따른 수신 세기 상태를 보기 위한 값이므로, **DC offset 제거 후 amplitude normalization 전 IQ**를 기준으로 계산합니다.

### 3.3 주요 모듈

| 처리 | 모듈 |
|---|---|
| DC offset 제거 | `src/preprocess/dc_blocker.py` |
| Clip / raw feature 계산 | `src/viewer/raw_features.py`, `src/calibration/raw_iq_safety.py` |
| IQ amplitude 정규화 | `src/preprocess/iq_normalizer.py` |
| RX0/RX1 gain mismatch 보정 | `src/preprocess/gain_matcher.py` |
| RX0/RX1 phase offset 추정/보정 | `src/preprocess/phaseoffset.py` |
| Gain-wise noise calibration | `src/calibration/gain_noise_calibration.py`, `scripts/run_noise_calibration_by_gain.py` |
| Gain-wise phase/gain calibration | `src/calibration/phase_gain_calibration.py`, `src/calibration/phase_gain_by_gain_calibration.py` |
| Runtime gain-wise noise lookup | `src/runtime/gain_noise_runtime.py` |
| Runtime RX1 gain/phase 보상 | `src/runtime/phase_calibration_runtime.py`, `src/viewer/aoa_runtime.py` |
| FFT / Scan score 계산 | `src/features/fft.py`, `src/scan/` |
| STFT spectrogram 생성 | `src/features/spectrogram.py` |
| Coherence 계산 | `src/aoa/coherence.py` |
| 위상차 계산 | `src/aoa/phase_diff.py` |
| AoA 변환 | `src/aoa/angle_estimator.py` |
| AoA sector 변환 | `src/aoa/sector_quantizer.py` |
| Pluto+ SDR 수신 | `src/receiver/pluto_receiver.py` |
| Real-time viewer runtime | `src/viewer/`, `scripts/live_rf_viewer.py` |

---

## 4. 본 실행 흐름: CLI Scan / Runtime Pipeline

본 프로젝트의 주 실행 흐름은 CLI 기반 scan/runtime pipeline입니다. Viewer는 이 흐름을 대체하는 프로그램이 아니라, 각 단계의 RF 패턴과 CNN/AoA 상태를 확인하기 위한 보조 도구입니다.

### 4.1 Runtime 상태 전이

```mermaid
stateDiagram-v2
    [*] --> PREPARE

    PREPARE --> SCAN: gain-wise calibration/profile ready

    SCAN --> SCAN: no RF candidate
    SCAN --> PRECISION: FFT candidate found

    PRECISION --> PRECISION_GAIN_DOWN: clip/overload or too strong
    PRECISION_GAIN_DOWN --> PRECISION: lower gain / warmup discard

    PRECISION --> SCAN: CNN voting failed
    PRECISION --> TRACK_AOA: CNN voting confirmed

    TRACK_AOA --> TRACK_AOA: coherence OK and signal active
    TRACK_AOA --> COHERENCE_HOLD: coherence low
    TRACK_AOA --> SIGNAL_HOLD: signal lost

    COHERENCE_HOLD --> TRACK_AOA: coherence recovered
    COHERENCE_HOLD --> SCAN: low coherence timeout

    SIGNAL_HOLD --> TRACK_AOA: signal recovered
    SIGNAL_HOLD --> SCAN: signal lost timeout
```

### 4.2 전체 운영 순서

```text
[Before Runtime]
1. Gain-wise noise calibration 수행
2. Gain-wise phase/gain calibration 수행
3. Gain/distance strength profile 준비
4. calibration/profile JSON quality 확인

[Runtime]
1. Scan mode에서 비교적 강한 초기 gain으로 2.4GHz 대역 sweep
2. 각 대역으로 retune한 뒤 4 block을 warmup/discard
3. 같은 대역에서 4 block을 실제 scan 판정용으로 수집
4. 각 block은 raw IQ safety check 후 DC offset을 제거하고 FFT score를 계산
5. 현재 gain의 noise profile에서 threshold를 조회
6. 4 block 중 2 block 이상 threshold를 넘으면 후보 대역으로 판단
7. 후보 대역 발견 시 precision mode 진입
8. 후보 신호의 raw_abs_max / raw_abs_p99 / overload flag 확인
9. 입력이 너무 강하면 gain 후보에서 한 단계 낮추고 같은 대역 재확인
10. 적정 gain 조건에서 STFT spectrogram 생성
11. CNN voting으로 uplink-side RF activity 여부 판단
12. voting 실패 시 scan mode 복귀
13. voting 통과 시 Track / AoA mode 진입
14. 현재 gain에 맞는 phase/gain profile을 lookup
15. RX1 gain correction 및 phase offset compensation 적용
16. coherence가 충분하면 AoA angle 및 sector 계산
17. coherence가 낮으면 hold block 동안 추가 관찰
18. signal lost가 지속되면 scan mode 복귀
```

### 4.3 Runtime의 출력

| 출력 | 목적 |
|---|---|
| `summary.json` | block 또는 run 단위 결과 요약 |
| `scan_events.json` | scan 후보 대역 및 trigger 기록 |
| CSV log | 시간 순서별 feature, CNN, AoA, mode 상태 기록 |
| numpy artifacts | spectrogram, STFT, frame energy 등 분석용 중간 결과 |

---

## 5. Calibration 흐름

Calibration은 runtime 중간에 수행되는 branch가 아니라, **본 실행 전에 기준값을 확보하는 준비 단계**입니다. Runtime에서는 calibration을 새로 수행하는 것이 아니라, 저장된 calibration 결과를 로드하여 현재 gain 조건에 맞게 적용합니다.

### 5.1 Gain-wise Noise Calibration

Noise calibration은 단일 noise threshold를 만드는 방식이 아니라, **gain별 noise profile을 생성하여 JSON으로 저장하고 runtime에서 현재 gain에 맞는 profile을 조회하는 방식**으로 수행합니다.

SDR gain이 바뀌면 배경 noise floor와 energy threshold도 함께 달라집니다. 따라서 하나의 threshold로 모든 gain을 처리하면 낮은 gain에서는 신호를 놓칠 수 있고, 높은 gain에서는 noise floor가 과대평가되거나 오탐 기준이 불안정해질 수 있습니다.

```text
[사전 작업]
gain 20 / 25 / 30 / 35 / 40에서 각각 noise block 수집
→ DC offset 제거
→ EnergyDetector 기준 frame energy 계산
→ gain별 noise_floor / noise_mean / noise_std / threshold 계산
→ Raw IQ safety 요약 저장
→ noise_by_gain_latest.json 저장

[실시간]
현재 SDR gain 확인
→ noise_by_gain_latest.json에서 현재 gain profile 선택
→ 해당 gain의 threshold 적용
→ raw IQ safety check와 함께 scan / precision / CNN / AoA runtime에 사용
```

기본 gain sweep 범위는 다음과 같습니다.

```text
20 dB, 25 dB, 30 dB, 35 dB, 40 dB
```

기본 저장 경로는 다음과 같습니다.

```text
outputs/calibration/noise_by_gain_latest.json
```

각 gain profile에는 다음 정보가 포함됩니다.

```text
gain
noise_floor
noise_mean
noise_std
noise_min
noise_max
threshold
dc_offset
safety.status
safety.max_abs_max
safety.rms_mean
safety.saturation_ratio_max
safety.near_saturation_ratio_max
```

이 구조를 사용하면 integrated CLI, live viewer, OpenCV viewer가 모두 같은 noise profile을 공유할 수 있습니다. 또한 자동 gain 조절 정책에서 현재 gain의 noise 기준선과 raw IQ saturation 상태를 함께 사용할 수 있습니다.

### 5.2 Gain-wise Phase/Gain Calibration

RX0/RX1은 같은 정면 0도 신호를 받아도 SDR 내부 경로, 케이블 길이, 안테나 배치 차이 때문에 위상차가 0이 아닐 수 있습니다. 또한 SDR gain 값이 바뀌면 RX0/RX1의 상대 gain mismatch와 phase offset도 함께 달라질 수 있습니다.

따라서 본 프로젝트에서는 reference gain에서 측정한 phase offset과 별도 gain delta table을 조합하는 방식이 아니라, **각 gain에서 직접 phase/gain calibration을 수행하여 gain별 calibration profile을 JSON으로 저장하는 방식**을 사용합니다.

```text
[사전 작업]
gain 20 / 25 / 30 / 35 / 40에서 각각 calibration block 수집
→ DC offset 제거
→ RX0/RX1 gain mismatch 추정
→ RX1 gain correction 계산
→ RX1-RX0 phase offset 추정
→ coherence-like 품질 지표 계산
→ phase_gain_by_gain_latest.json 저장

[실시간]
현재 SDR gain 확인
→ phase_gain_by_gain_latest.json에서 현재 gain profile 선택
→ RX1 gain correction 적용
→ RX1 phase offset 보상 적용
→ coherence check
→ AoA angle 계산
→ sector output 생성
```

기본 저장 경로는 다음과 같습니다.

```text
outputs/calibration/phase_gain_by_gain_latest.json
```

Runtime 보정은 다음처럼 단순화됩니다.

```python
rx1_gain_corrected = rx1 * gain_correction
rx1_compensated = rx1_gain_corrected * np.exp(-1j * phase_offset_rad)
```

### 5.3 Robust Calibration 방식

기존 단순 평균 방식은 멀티패스 또는 Wi-Fi 간섭 block이 섞이면 phase offset이 중간값으로 왜곡될 수 있습니다. 이를 개선하기 위해 다음 robust calibration 방식을 사용합니다.

```text
1. gain별로 총 N block 수집
2. 초기 안정화용 앞 discard block 제거
3. block별 RX1-RX0 phase와 coherence-like 계산
4. coherence 기준 미달 block 제거
5. phase 분포에서 dominant cluster 선택
6. dominant cluster 내부 block만 coherence-weighted circular mean
7. phase_std, coherence_median, cluster_ratio 기반 quality 판정
8. gain별 phase/gain profile JSON 저장
```

### 5.4 Calibration Quality

```text
OK:
- phase_std < 3°
- valid_blocks >= 100
- cluster_ratio >= 0.80
- coherence_median >= 0.70

WARNING:
- phase_std < 7°
- valid_blocks >= 50
- cluster_ratio >= 0.60
- coherence_median >= 0.55

FAIL:
- 위 조건 미달
```

`OK`는 AoA 보정값으로 사용할 수 있는 상태입니다. `WARNING`은 방향성 확인용으로는 사용할 수 있지만 정밀 각도값을 강하게 신뢰하면 안 됩니다. `FAIL`은 calibration을 다시 수행해야 하는 상태입니다.

### 5.5 Legacy 구조 정리: Gain-dependent Phase Table 제거

기존에는 다음 구조를 고려했습니다.

```text
reference gain에서 current_phase_offset.json 생성
+ gain_phase_table_2450.json에서 gain_delta lookup
= phase_offset_to_apply
```

그러나 최종 구조에서는 별도의 gain-dependent phase delta table을 유지하지 않습니다. 대신 gain별로 phase/gain calibration을 직접 수행하고, runtime에서는 현재 gain에 해당하는 profile을 바로 조회합니다.

```text
기존:
current_phase_offset.json + gain_phase_table_2450.json

수정:
phase_gain_by_gain_latest.json profile lookup
```

이 방식의 장점은 다음과 같습니다.

```text
1. reference gain 기준 delta table을 별도로 유지하지 않아도 된다.
2. 현재 gain에서 실제 측정된 phase/gain 보정값을 직접 사용할 수 있다.
3. gain 자동 조절 정책과 AoA 보정값 선택을 자연스럽게 연결할 수 있다.
4. integrated CLI, live viewer, OpenCV viewer가 동일한 JSON profile을 공유할 수 있다.
```

Legacy 파일과 스크립트는 당장 삭제하지 않고 보존합니다. 다만 최종 runtime 설계와 문서에서는 `phase_gain_by_gain_latest.json` 기반 구조를 기준으로 합니다.

---

## 6. Scan Mode: FFT 기반 후보 대역 탐색

Scan mode는 본 runtime pipeline의 시작점입니다. 이 단계에서는 CNN이나 AoA를 바로 수행하지 않고, 2.4GHz 대역을 sweep하면서 신호가 있는 후보 대역을 먼저 찾습니다.

```text
2.400 GHz → 2.405 GHz → 2.410 GHz → ... → 2.485 GHz
```

### 6.1 Scan 정책

현재 scan mode는 대역마다 다음 순서로 동작합니다.

```text
1. 각 center frequency로 SDR LO 설정
2. retune 직후 4 block을 warmup/discard
3. 같은 center frequency에서 4 block을 실제 scan 판정용으로 수집
4. 각 block에 대해 raw IQ safety check 수행
5. DC offset 제거 후 현재 gain의 noise threshold를 조회
6. FFT score 또는 energy score 계산
7. threshold를 넘은 block 수 계산
8. 4 block 중 2 block 이상 통과하면 후보 대역으로 등록
9. 후보가 없으면 scan 계속
10. 후보가 있으면 우선순위가 높은 후보로 precision mode 진입
```

현재 권장 설정은 다음과 같습니다.

```yaml
# configs/receiver.yaml
sdr:
  warmup_reads: 4
  retune_warmup_reads: 4

# configs/scan.yaml
scan:
  scan_blocks: 4
  min_pass_blocks: 2
```

### 6.2 Retune warmup 정책

SDR 중심 주파수를 바꾸면 LO와 내부 RX buffer가 바로 안정된다고 가정하지 않습니다. 따라서 scan mode에서는 center frequency를 변경한 직후 몇 block을 읽고 버린 다음 scan score를 계산합니다.

현재 운영 정책은 단순하게 **4 block discard + 4 block scan**으로 통일합니다.

```text
center frequency 변경
→ 4 block discard
→ 4 block scan
→ 2 block 이상 통과 시 candidate
```

1 block은 5 MSPS, 16,384 samples 기준 약 3.28 ms이므로, 4 block discard는 약 13.1 ms입니다. 현재 단계에서는 scan 속도보다 retune 직후 안정성을 우선합니다.

### 6.3 Scan mode의 역할

| 항목 | 설명 |
|---|---|
| 목적 | 신호가 있는 후보 대역 탐색 |
| 입력 | RX0/RX1 IQ block |
| 주요 feature | FFT max power, FFT median, energy score |
| 판정 기준 | 현재 gain의 threshold 기준 4 block 중 2 block 이상 통과 |
| 출력 | candidate center frequency |
| 실패 시 | 다음 frequency로 계속 sweep |

Scan mode의 목적은 분류가 아니라 후보 탐색입니다. CNN inference와 AoA 계산은 후보 대역이 발견된 뒤 precision mode에서 수행합니다.

---

## 7. Precision Mode: STFT + CNN Voting 정책

Scan mode에서 후보 대역이 발견되면 해당 center frequency에 잠시 머물러 precision mode를 수행합니다.

### 7.1 Precision mode 흐름

```text
후보 center frequency 고정
→ precision_blocks 수집
→ raw IQ safety 및 gain별 threshold 확인
→ 각 block을 STFT spectrogram으로 변환
→ CNN inference
→ positive vote 수 계산
→ 과반 이상이면 confirmed
→ 실패하면 scan mode 복귀
```

### 7.2 Voting 정책

RF burst는 순간적으로 들어왔다가 사라질 수 있으므로 1개 block 결과만으로 최종 판정하지 않습니다. 후보 대역에서 여러 block을 보고 voting으로 판단합니다.

```text
예시:
precision_blocks = 5
confirm_votes = 3

5 block 중 3 block 이상 positive
→ Confirmed uplink-side RF activity

5 block 중 2 block 이하 positive
→ 후보 기각
→ Scan Mode 복귀
```

### 7.3 Positive / Negative 해석

| 결과 | 해석 |
|---|---|
| Positive | 조종기 기원 uplink-side RF activity 가능성 있음 |
| Negative | Background / Wi-Fi / Bluetooth / 기타 NonDrone 가능성 높음 |
| Voting failed | 후보 대역으로 보기 어려우므로 scan mode 복귀 |

---

## 8. Track / AoA Mode: Coherence, Sector, 복귀 정책

CNN voting을 통과한 후보만 Track / AoA mode로 진입합니다. AoA는 두 채널이 같은 신호를 안정적으로 보고 있다는 전제가 있어야 하므로, phase difference를 계산하기 전에 coherence를 확인합니다.

### 8.1 Coherence 정책

```text
coherence >= threshold
→ 현재 gain에 맞는 phase/gain profile lookup
→ RX1 gain correction 및 phase offset 보상 적용
→ phase difference 계산
→ AoA angle 계산
→ angle smoothing
→ sector 변환

coherence < threshold
→ 즉시 포기하지 않고 hold block 동안 추가 관찰
→ hold 기간 안에 coherence가 회복되면 Track / AoA 유지
→ 계속 낮으면 AoA 신뢰 불가로 판단하고 scan mode 복귀
```

초기 README 정책에서는 coherence가 낮을 때 자동 gain up을 본 정책으로 넣지 않습니다. Coherence low의 원인이 신호 약함이 아니라 멀티패스, 간섭, 포화, 채널 정렬 문제일 수 있기 때문입니다. Adaptive gain recovery는 향후 실험 옵션으로만 검토합니다.

### 8.2 Signal lost 정책

Precision 또는 Track / AoA mode에서 신호가 잠깐 약해질 수 있으므로, 신호가 사라졌다고 즉시 scan mode로 복귀하지 않습니다.

```text
signal lost
→ lost_count 증가
→ hold 기간 동안 같은 center frequency 유지
→ 신호 회복 시 lost_count reset
→ lost_count가 threshold 이상이면 scan mode 복귀
```

### 8.3 Sector 표현

AoA 결과는 순간 angle 하나로만 해석하지 않고 sector로 표현합니다. 현재 구현은 전방 -90° ~ +90° 범위를 기본 8-sector로 나눕니다.

| Sector | 각도 범위 |
|---|---:|
| S0 Far Left | -90.0° ~ -67.5° |
| S1 Left | -67.5° ~ -45.0° |
| S2 Left-Center | -45.0° ~ -22.5° |
| S3 Center-Left | -22.5° ~ 0.0° |
| S4 Center-Right | 0.0° ~ 22.5° |
| S5 Right-Center | 22.5° ~ 45.0° |
| S6 Right | 45.0° ~ 67.5° |
| S7 Far Right | 67.5° ~ 90.0° |

위 각도 범위는 고정값이 아니라, 실측 AoA 오차 분포와 안테나 배치에 따라 조정할 수 있습니다.

### 8.4 Angle 안정화 정책

AoA angle은 block 단위 순간값을 그대로 사용하지 않고, 최근 K개 block의 angle 또는 sector 결과를 이용해 안정화합니다.

```text
1. coherence threshold를 통과한 block만 angle / sector buffer에 추가
2. 최근 K개 angle을 coherence-weighted circular smoothing으로 안정화
3. 또는 최근 N개 sector를 majority voting으로 안정화
4. angle variance 또는 sector vote가 기준 이상일 때 sector 확정
5. 분산이 크면 sector를 HOLD 또는 UNSTABLE로 표시
```

출력 예시:

```text
AoA angle        : +18.4°
Smoothed angle   : +15.7°
Sector           : S4 Center-Right
Stability        : OK
```

---

## 9. Gain Control Policy 및 Feature Profile 운용

### 9.1 Gain을 실험 조건으로 관리하는 이유

본 프로젝트에서 gain은 단순한 수신 세기 조절값이 아니라, scan 거리, CNN 입력 안정성, raw feature 비교, overload 판단, noise threshold lookup, phase/gain calibration profile 적용에 영향을 주는 중요한 runtime 조건입니다.

Scan mode 시작 전에는 아직 어떤 신호가 들어올지 모르기 때문에 gain 적정성을 완벽히 판단하기 어렵습니다. 따라서 본 프로젝트는 낮은 gain에서 시작하지 않고, **실험적으로 안전하다고 확인한 비교적 강한 gain**에서 scan을 시작합니다.

단, 이는 SDR의 최대 gain을 무조건 사용하는 것을 의미하지 않습니다. 후보 신호가 관측된 뒤 raw IQ safety와 절대세기 feature를 확인하고, 너무 강하게 들어오면 gain 후보에서 한 단계 낮춰 같은 후보 대역을 다시 확인합니다.

```text
핵심 정책:
적당히 강한 gain으로 후보 신호를 먼저 잡고,
후보 신호가 너무 강하면 raw feature와 safety를 보고 즉시 gain을 낮춘다.
```

### 9.2 Gain/Distance Strength Profile 준비

Gain/distance strength profile은 선택 사항이 아니라, 본 runtime pipeline과 viewer가 함께 참조하는 기준값입니다.

```text
Set gain = 20 / 25 / 30 / 35 / 40
→ 거리별 또는 조건별 N-block 수집
→ raw feature 계산
→ median / mean / std / p25 / p75 계산
→ gain/거리별 대표 profile 저장
→ CSV와 JSON으로 기록
```

Gain/distance strength profile은 scan 시작 전에 gain을 완벽히 결정하기 위한 것이 아닙니다. 실제로 의미 있는 비교는 후보 신호가 관측된 이후부터 가능합니다. 후보 신호가 잡힌 뒤 현재 raw feature가 기준 profile보다 과도하게 큰지, 너무 약한지, clip 위험이 있는지를 판단하는 기준표로 사용합니다.

초기 gain 판단의 핵심 feature는 다음 세 가지로 제한합니다.

```text
raw_abs_p99
raw_rms
overload flag 또는 raw_abs_max
```

`frame_power_p99` 등 추가 feature는 저장과 사후 분석에는 활용하되, 초기 gain up/down 조건에는 직접 포함하지 않습니다.

### 9.3 저장 파일

```text
outputs/calibration/noise_by_gain_latest.json
outputs/calibration/phase_gain_by_gain_latest.json
outputs/viewer/gain_feature_profiles.csv
outputs/viewer/gain_feature_profiles_latest.json
```

CSV 파일은 실험 기록 및 사후 분석용입니다. 여러 번 저장한 gain profile이 누적되므로, 나중에 거리/gain/feature 관계를 그래프로 분석하거나 실험 조건을 비교할 때 사용합니다.

JSON 파일은 최신 gain별 profile table입니다. 이 파일은 viewer 전용 파일이 아니라, CLI 기반 scan/runtime pipeline과 live viewer가 함께 참조할 수 있는 runtime reference입니다.

### 9.4 Gain 후보 step 운용

Runtime에서 gain은 임의의 1 dB 단위로 계속 조절하지 않고, 사전에 noise profile과 phase/gain calibration profile이 준비된 gain 후보 안에서 이동합니다.

예시:

```text
사용 gain 후보: 20 / 25 / 30 / 35 / 40 dB
초기 scan gain: 안전하다고 확인한 비교적 강한 gain
Gain down: 40 → 35 → 30 → 25 → 20
```

이 방식은 CNN이 학습한 수집 조건의 raw feature 분포와 runtime 후보 신호의 raw feature를 맞추기 위한 것입니다. 즉 gain 후보 table은 거리 자체를 정확히 추정하기 위한 표가 아니라, CNN 입력이 학습 조건과 유사한 수신 세기 영역에 들어오도록 조정하기 위한 운영 기준입니다.

### 9.5 Gain down 조건

후보 신호가 관측된 이후 다음 조건이면 gain을 한 단계 낮춥니다.

```text
- raw IQ safety status가 CLIPPED 또는 WARNING
- overload flag 발생
- raw_abs_max가 clip threshold 근처
- raw_abs_p99가 현재 gain profile 기준보다 과도하게 큼
- spectrogram이 포화되어 burst pattern이 뭉개짐
```

Gain down 이후에는 현재 후보 대역을 바로 버리지 않습니다.

```text
Gain down
→ SDR gain 적용
→ warmup/discard
→ 같은 center frequency에서 raw feature 재확인
→ 적정 gain이면 STFT / CNN voting 수행
```

Clip threshold는 SDR 출력 scale과 실측 raw feature 분포에 따라 달라지므로 README에서 고정 수치로 확정하지 않습니다. 실제 runtime에서는 config 값 또는 calibration/profile 기반 threshold로 관리합니다.

### 9.6 Gain up 조건

초기 runtime 정책에서는 gain up을 자동으로 강하게 사용하지 않습니다. 특히 coherence가 낮다는 이유만으로 gain을 올리지는 않습니다.

다만 향후 adaptive gain policy로 다음 조건을 검토할 수 있습니다.

```text
- 후보 신호는 존재함
- raw_abs_p99 / raw_rms가 기준 profile보다 낮음
- raw IQ safety가 SAFE
- noise profile과 phase/gain calibration profile에 해당 gain이 존재함
```

현재 기본 정책은 다음과 같습니다.

```text
Scan mode:
비교적 강한 gain으로 후보 신호를 먼저 찾는다.

Precision mode:
후보 신호가 너무 강하면 gain 후보 step 단위로 낮춘다.

AoA mode:
phase/gain profile이 준비된 gain에서만 신뢰 가능한 AoA 결과를 기록한다.
```

### 9.7 Reference gain 복귀 조건

기존 reference gain 복귀 정책은 phase delta table 구조에서는 중요했지만, gain-wise phase/gain profile 구조에서는 필수 조건이 아닙니다. 다만 다음 상황에서는 안정성을 위해 기준 gain 또는 가장 신뢰도 높은 gain으로 복귀할 수 있습니다.

```text
- 정밀 AoA 기록 전
- gain 변경 후 coherence 또는 phase 안정성이 나빠졌을 때
- 후보 추적 종료 후 scan mode로 복귀할 때
- calibration 기준 조건으로 다시 맞출 때
```

AoA 최종 기록은 가능하면 `phase_gain_by_gain_latest.json`에 포함되고 quality가 OK인 gain에서 수행합니다.

### 9.8 Gain 변경과 RX1 gain/phase 보상

Gain을 변경하면 현재 gain에 맞는 phase/gain profile을 다시 적용해야 합니다.

```text
[Gain Change Requested]
→ Set SDR Gain
→ Warmup Blocks Discard
→ Lookup phase_gain_by_gain profile
→ Apply RX1 gain correction
→ Apply RX1 phase offset compensation
→ Continue CNN / AoA Runtime
```

실시간 적용값은 다음과 같습니다.

```text
current_gain
→ phase_gain_by_gain_latest.json profile lookup
→ gain_correction_mean 적용
→ phase_offset_rad_mean 적용
```

---

## 10. CNN 기반 RF 신호 분류

### 10.1 현재 분류 목적

현재 CNN은 STFT spectrogram을 입력으로 받아 **조종기 기원 uplink-side RF activity**와 NonDrone 계열 신호를 구분하는 실험 구조입니다.

초기에는 드론 기체 자체의 RF 신호 탐지를 목표로 접근했으나, 실험 과정에서 다음 현상이 확인되었습니다.

```text
드론 ON + 조종기 ON + 링크 상태
→ Drone-like 또는 Confirmed Drone

드론 OFF + 조종기 ON
→ 일정 시간 Drone-like로 탐지

드론 OFF + 조종기 OFF
→ NonDrone으로 전환
```

이는 현재 모델이 드론 기체 자체에서 방사되는 별도 downlink 신호보다, 조종기에서 발생하는 제어 신호, 탐색 burst, 페어링/재연결 burst 등 **controller-originated uplink-side RF activity**에 더 민감하게 반응한다는 것을 의미합니다.

### 10.2 탐지 대상

```text
탐지 대상:
드론 기체 자체의 downlink 신호가 아니라,
조종기에서 발생하는 uplink-side RF activity

포함되는 신호:
- 조종기 제어/명령 신호
- 드론 탐색 또는 페어링 시도 burst
- 재연결 또는 idle 상태에서 발생하는 controller-originated burst
```

### 10.3 데이터 전략

```text
Positive / Uplink-side RF activity:
- 조종기 ON
- 드론 탐색 또는 페어링 시도 상태
- 드론 ON + 조종기 ON + linked 상태
- 조종기 조작 또는 재연결 과정에서 발생하는 RF burst

Negative / NonDrone:
- 조종기 OFF + 드론 OFF
- Wi-Fi
- Bluetooth
- Background
- 기타 2.4GHz 주변 RF activity
```

즉, 현재 단계에서는 controller-only를 무조건 제거해야 할 오탐으로 보기보다, **드론 운용과 관련된 조종기 기원 RF activity**로 묶어 탐지하는 것이 더 현실적입니다.

---

## 11. Real-time Viewer: 보조 실험 도구

Real-time viewer는 본 runtime pipeline을 대체하는 메인 실행 흐름이 아니라, RF 패턴 확인, gain profile 저장, CNN/AoA 디버깅을 위한 보조 실험 도구입니다.

### 11.1 통합 viewer

최종 viewer 실행은 하나의 script에서 처리합니다.

```text
scripts/live_rf_viewer.py
```

지원 mode는 다음과 같습니다.

| mode | 목적 | 주요 기능 |
|---|---|---|
| `fast` | 고속 spectrogram 확인 | OpenCV 표시, CNN/AoA 없음 |
| `profile` | gain별 feature profile 저장 | N-block feature 수집, CSV/JSON 저장 |
| `cnn` | CNN 디버깅 | spectrogram + CNN inference + temporal smoothing |
| `aoa` | AoA 검증 | phase/gain calibration, coherence, phase difference, angle |
| `full` | 통합 실험 | CNN + AoA + gain profile + logging |

OpenCV viewer는 계산을 담당하지 않고, CNN 판정 결과, AoA sector, gain 상태, raw IQ safety, noise threshold 상태를 빠르게 표시하는 역할을 합니다.

### 11.2 기존 viewer 처리 방침

기존 viewer들은 새 `live_rf_viewer.py`가 안정화될 때까지 삭제하지 않습니다.

```text
scripts/live_cnn_spectrogram_viewer.py
→ legacy 유지, 기능 추가 중단

scripts/live_siggen_aoa_viewer.py
→ 신호발생기 검증용 legacy 유지 또는 live_rf_viewer --mode aoa로 대체

scripts/live_spectrogram_only_viewer.py
→ fast mode로 흡수
```

---

## 12. 실행 방법

### 12.1 가상환경 활성화

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

### 12.2 단일 block pipeline 실행

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

### 12.3 Gain-wise Noise Calibration

```bash
PYTHONPATH=. python scripts/run_noise_calibration_by_gain.py \
  --gains 20 25 30 35 40 \
  --num-blocks-per-gain 50 \
  --output outputs/calibration/noise_by_gain_latest.json
```

### 12.4 Gain-wise Phase/Gain Calibration

```bash
PYTHONPATH=. python scripts/run_phase_gain_calibration_by_gain.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 20 25 30 35 40 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output outputs/calibration/phase_gain_by_gain_latest.json \
  --memo "outdoor_0deg_1p5m_sig2452_phase_gain_by_gain"
```

### 12.5 Scan Mode 실행

```bash
PYTHONPATH=. python scripts/run_scan.py
```

### 12.6 통합 Live Viewer 실행 예시

Fast mode:

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode fast \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 10
```

Profile mode:

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode profile \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 10 \
  --profile-blocks 20
```

Full debug mode:

```bash
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 5 \
  --profile-blocks 20 \
  --cnn-backend dummy \
  --noise-profile outputs/calibration/noise_by_gain_latest.json \
  --phase-gain-profile outputs/calibration/phase_gain_by_gain_latest.json
```

---

## 13. 프로젝트 구조

주요 구조는 다음과 같습니다.

```text
rf-drone-detection-capstone/
├── README.md
├── requirements.txt
├── configs/
│   ├── aoa.yaml
│   ├── detect.yaml
│   ├── ml.yaml
│   ├── receiver.yaml
│   ├── scan.yaml
│   ├── ui.yaml
│   └── calibration/
│       ├── legacy_current_phase_offset.json
│       └── legacy_gain_phase_table_2450.json
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_scan.py
│   ├── run_noise_calibration_by_gain.py
│   ├── run_phase_gain_calibration_by_gain.py
│   ├── live_rf_viewer.py
│   ├── live_cnn_spectrogram_viewer.py
│   ├── live_siggen_aoa_viewer.py
│   ├── live_spectrogram_only_viewer.py
│   └── train_model.py
│
├── src/
│   ├── aoa/
│   │   └── sector_quantizer.py
│   ├── calibration/
│   │   ├── noise_calibration.py
│   │   ├── gain_noise_calibration.py
│   │   ├── raw_iq_safety.py
│   │   ├── phase_gain_calibration.py
│   │   └── phase_gain_by_gain_calibration.py
│   ├── runtime/
│   │   ├── gain_noise_runtime.py
│   │   └── phase_calibration_runtime.py
│   ├── receiver/
│   ├── features/
│   ├── preprocess/
│   ├── detect/
│   ├── ml/
│   ├── scan/
│   ├── viewer/
│   │   ├── state.py
│   │   ├── raw_features.py
│   │   ├── gain_profile_runtime.py
│   │   ├── cnn_runtime.py
│   │   ├── aoa_runtime.py
│   │   ├── opencv_renderer.py
│   │   └── logging.py
│   └── ui/
│
├── docs/
│   ├── command/
│   ├── planning/
│   ├── report/
│   └── experiments/
│
├── data/
├── models/
├── outputs/
│   └── calibration/
│       ├── noise_by_gain_latest.json
│       └── phase_gain_by_gain_latest.json
└── tests/
```

---

## 14. 현재 개발 현황

| 모듈 | 상태 |
|---|---|
| 프로젝트 기본 구조 | 완료 |
| YAML 기반 설정 구조 | 완료 |
| SimReceiver / RawFileReceiver | 완료 |
| PlutoReceiver | 실측 연동 및 gain runtime control 구현 |
| DC offset 제거 | 완료 |
| Raw feature / overload branch | 구현 |
| Raw IQ safety check | 구현 |
| IQ normalization | 완료 |
| Gain mismatch correction | 완료 |
| Phase offset estimation/correction | 완료 |
| 단일 phase/gain calibration | 구현 |
| Gain-wise noise calibration | 구현 |
| Gain-wise phase/gain calibration | 설계 반영, wrapper 구현 예정 |
| Gain/distance strength profile | 구현/보강 예정 |
| Runtime gain-wise noise lookup | 구현 |
| Runtime RX1 gain/phase 보상 | 구현/전환 예정 |
| FFT / Scan mode | 4 block warmup + 4 block scan 정책 적용 |
| Precision CNN voting policy | 정책 정리, 실측 검증 필요 |
| Coherence hold / signal lost return policy | 정책 정리, 구현 보강 필요 |
| FFT / STFT feature 계산 | 완료 |
| Dual-channel STFT branch | 완료 |
| Coherence gate | 완료 |
| Phase difference 계산 | 완료 |
| AoA 계산 | 구현 |
| AoA sector quantizer | 구현 |
| AoA smoothing / sector voting | 구현 보강 예정 |
| Integrated live RF viewer | 리팩토링 진행 |
| CNN inference interface | 구현 |
| Uplink-side RF activity 데이터셋 | 보강 예정 |
| TFLite / Raspberry Pi 배포 | 추후 확장 |

---

## 15. 실험 운영 순서

권장 실험 순서는 다음과 같습니다.

```text
1. py_compile / import 검사
2. 신호발생기 2452 MHz 설정
3. gain 20/25/30/35/40 noise calibration 수행
4. noise_by_gain_latest.json quality 확인
5. gain 20/25/30/35/40 phase/gain calibration 수행
6. phase_gain_by_gain_latest.json quality 확인
7. gain/distance strength profile 제작
8. 신호발생기 정면 0도 AoA 검증
9. 좌우 ±10°, ±20° AoA 방향성 검증
10. scan mode에서 retune 후 4 block discard / 4 block scan 동작 확인
11. 4 block 중 2 block 이상 통과 시 후보 대역으로 잡히는지 확인
12. 후보 대역 precision CNN voting 확인
13. coherence hold / signal lost 복귀 정책 확인
14. 조종기 ON/OFF 조건에서 uplink-side RF activity 재현성 확인
15. NonDrone background / Wi-Fi / Bluetooth 데이터 보강
```

---

## 16. 현재 한계

### 16.1 CNN 모델 한계

현재 모델은 드론 기체 자체의 downlink 신호보다 조종기에서 발생하는 uplink-side RF activity에 반응하는 것으로 해석됩니다. 따라서 본 프로젝트의 결과는 드론 기체 단독 탐지가 아니라, 조종기 기원 RF activity를 통한 드론 운용 가능성 탐지로 설명하는 것이 적절합니다.

### 16.2 AoA 한계

2채널 위상차 기반 AoA는 다음 요소에 민감합니다.

- RX0/RX1 하드웨어 phase offset
- 케이블 길이 및 안테나 간격
- 신호원 정렬 오차
- 실내 멀티패스
- 주변 2.4GHz Wi-Fi / Bluetooth 간섭
- SDR gain 변경에 따른 phase/gain response 변화

따라서 AoA는 먼저 신호발생기 기반으로 검증한 뒤, 드론 또는 조종기 신호에 적용해야 합니다.

### 16.3 Gain-wise Calibration 한계

Gain-wise noise profile과 phase/gain calibration profile은 모든 환경에서 절대 보정을 보장하지 않습니다. 실외에서 측정한 profile이라도 주변 RF 환경, 안테나 방향, 케이블 상태, 신호원 위치에 따라 달라질 수 있습니다.

따라서 중요한 실험 전에는 현재 환경에서 gain-wise noise calibration과 phase/gain calibration을 다시 수행하는 것이 좋습니다.

### 16.4 자동 Gain 정책 한계

초기 runtime에서는 coherence low 상황에서 자동 gain up을 본 정책으로 사용하지 않습니다. Gain 변경은 phase offset, coherence, CNN 입력 형태를 동시에 바꿀 수 있으므로, 실측 검증 전에는 보수적으로 적용합니다.

---

## 17. 다음 작업 계획

다음 작업은 다음과 같습니다.

1. Gain-wise phase/gain calibration wrapper 구현
2. `phase_gain_by_gain_latest.json` loader 및 runtime adapter 구현
3. Live viewer에 gain-wise noise profile과 phase/gain profile 연결
4. OpenCV overlay에 CNN 판정, raw IQ safety, AoA sector 표시
5. AoA sector range 및 sector voting 보강
6. Gain/distance strength profile JSON 구조 확정
7. Scan mode 4 block warmup / 4 block scan 정책 실측 검증
8. Precision mode CNN voting 정책 구현/검증
9. Coherence hold 및 signal lost 복귀 정책 구현/검증
10. 조종기 ON/OFF 조건에서 uplink-side RF activity 재현성 확인
11. Background / Wi-Fi / Bluetooth 등 NonDrone 데이터 보강
12. 조종기 기원 RF activity 기준으로 CNN 데이터셋 재정리
13. 실험 결과를 `docs/report/`에 정리

---

## 18. 기본 명령어

```bash
# 가상환경 활성화
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

# 단일 block pipeline 실행
PYTHONPATH=. python scripts/run_pipeline.py

# Scan mode 실행
PYTHONPATH=. python scripts/run_scan.py

# gain-wise noise calibration
PYTHONPATH=. python scripts/run_noise_calibration_by_gain.py \
  --gains 20 25 30 35 40 \
  --num-blocks-per-gain 50 \
  --output outputs/calibration/noise_by_gain_latest.json

# gain-wise phase/gain calibration
PYTHONPATH=. python scripts/run_phase_gain_calibration_by_gain.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --signal-freq 2452000000 \
  --sample-rate 5000000 \
  --gains 20 25 30 35 40 \
  --total-blocks 200 \
  --discard-blocks 30 \
  --warmup-reads 20 \
  --coherence-threshold 0.50 \
  --cluster-window-deg 5.0 \
  --output outputs/calibration/phase_gain_by_gain_latest.json

# integrated viewer fast mode
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode fast \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 10

# Git 저장
git status
git add README.md
git commit -m "docs: update README for gain-wise calibration policy"
git push
```

---

## 19. 현재 프로젝트 상태 요약

현재 프로젝트는 단순 아이디어 단계가 아니라, **RF 신호처리 pipeline, scan mode, 실시간 viewer, AoA 실험 도구, robust calibration 구조가 구현된 상태**입니다.

```text
2채널 IQ 입력
→ Gain-wise Noise Calibration / Gain-wise Phase-Gain Calibration / Gain-Distance Profile 준비
→ FFT 기반 scan mode
  - retune 후 4 block discard
  - 4 block scan / 2 block 이상 통과 시 후보
→ 후보 대역 precision CNN voting
→ coherence 기반 AoA 진입 판단
→ 현재 gain의 phase/gain profile lookup
→ RX1 gain correction + phase offset compensation
→ AoA angle 계산
→ angle smoothing / sector 표현
→ 결과 저장 또는 scan mode 복귀
```

현재 CNN은 조종기 단독 조건에서도 RF activity를 탐지하므로, 프로젝트의 핵심 탐지 대상은 드론 기체 자체 downlink가 아니라 조종기 기원 uplink-side RF activity로 정리합니다.

다음 핵심 과제는 CLI scan/runtime 상태머신을 안정화하고, 신호발생기 기반 AoA 안정성 검증과 함께 실제 조종기 RF source의 방향 추정을 수행하는 것입니다.
