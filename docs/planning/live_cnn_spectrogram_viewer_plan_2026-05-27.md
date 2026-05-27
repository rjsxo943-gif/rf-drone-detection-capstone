# Live CNN Spectrogram Viewer 구현 계획 보고서

작성일: 2026-05-27  
프로젝트: RF 기반 드론 탐지 캡스톤 프로젝트  
문서 목적: `Live CNN Spectrogram Viewer` 구현 전 요구사항, 설계 방향, 판단 기준, 구현 체크리스트를 상세히 정리한다.

---

## 1. 문서 작성 배경

현재 RF 드론 탐지 프로젝트는 Pluto+ SDR을 이용하여 2.4GHz 대역 신호를 수집하고, STFT 기반 spectrogram을 CNN 입력으로 변환하여 `Background / WiFi / Bluetooth / Drone-like`를 분류하는 구조로 진행되고 있다.

기존 실험에서는 실제 드론 신호를 수집하여 `live gain15` 기반 RF4 모델을 재학습하였고, 오프라인 테스트 기준으로 `Drone-like` recall이 상당히 개선되었다. 그러나 현재 수집 및 검증 조건은 주로 다음 조건에 묶여 있다.

```text
거리: 약 70cm 근처
receiver gain: 15
center frequency: 2.4GHz 대역 일부
실내 환경
```

따라서 실제 완성도를 높이기 위해서는 다음 문제가 남아 있다.

```text
1. 거리가 멀어지면 신호 세기와 spectrogram 분포가 달라질 수 있다.
2. gain을 올리면 먼 신호는 잘 보이지만, 가까운 신호는 포화될 수 있다.
3. CNN 입력은 정규화 과정을 거치므로 절대적인 세기 정보를 일부 잃는다.
4. trigger는 정규화 전 raw IQ 기반으로 판단해야 한다.
5. 거리와 gain 변화에 따른 적정 조건을 실험적으로 기록할 필요가 있다.
```

이 문서는 위 문제를 해결하기 위한 실험 도구인 `Live CNN Spectrogram Viewer`의 구현 계획을 정리한다.

---

## 2. Viewer의 핵심 목적

이 viewer는 일반적인 RF spectrum analyzer가 아니다. 또한 실시간 CNN 추론기나 AoA viewer도 아니다.

본 viewer의 목적은 다음과 같다.

```text
CNN이 실제로 입력받는 spectrogram과 동일한 형식의 이미지를 실시간으로 확인하고,
동시에 raw IQ 기반 세기 feature를 기록하여
거리별 / gain별 적정 수신 조건을 실험적으로 결정하는 것
```

즉, 이 viewer는 다음 질문에 답하기 위한 도구이다.

```text
1. 현재 안테나가 보는 신호가 CNN 입력 형식에서는 어떻게 보이는가?
2. 현재 거리와 gain 조건에서 spectrogram 분포가 학습 데이터와 비슷한가?
3. 신호가 너무 약해서 trigger가 어려운 상태인가?
4. gain이 너무 높아서 포화가 발생하는 상태인가?
5. 거리별로 어떤 gain profile이 적절한가?
6. 나중에 gain-distance 근사식 또는 gain profile 선택 기준을 만들 수 있는가?
```

---

## 3. Viewer에서 하지 않을 것

초기 구현에서는 기능을 의도적으로 제한한다. 이유는 디버깅과 실험 해석을 명확히 하기 위해서이다.

초기 viewer에서 하지 않을 기능은 다음과 같다.

```text
1. CNN inference 수행하지 않음
2. AoA 계산하지 않음
3. 주파수 scan 기능 넣지 않음
4. 자동 gain 변경하지 않음
5. yaml hot reload 기본 적용하지 않음
6. 모든 IQ block을 CNN spectrogram으로 변환하지 않음
```

특히 CNN inference를 붙이지 않는 이유는 다음과 같다.

```text
현재 목적은 모델의 정답 여부가 아니라,
모델에 들어가기 전 입력이 제대로 보이는지 확인하는 것이다.

CNN을 붙이면 문제가 생겼을 때
- RF 신호가 안 보이는 문제인지
- 전처리 문제인지
- 모델 분류 문제인지
구분하기 어려워진다.
```

따라서 초기 viewer는 철저히 `CNN 입력 품질 검사 도구`로 설계한다.

---

## 4. 전체 설계 철학

본 viewer는 기존 탐지 파이프라인과 같은 철학을 따른다.

기존 프로젝트에서는 다음과 같은 구조를 사용한다.

```text
Raw IQ
  ↓
Energy detector 또는 scan gate
  ↓
의미 있는 후보만 CNN/AoA로 전달
```

viewer도 동일하게 구성한다.

```text
Raw IQ 수신
  ↓
가벼운 raw feature 계산
  ↓
raw gate로 의미 있는 block 선택
  ↓
선택된 block만 CNN 입력 spectrogram으로 변환
  ↓
화면 표시 + CSV 기록
```

핵심은 다음이다.

```text
무거운 CNN 입력 spectrogram 생성 작업을 모든 block에 수행하지 않는다.
먼저 raw IQ 기반으로 볼 가치가 있는 block을 고른다.
```

---

## 5. Raw branch와 CNN branch 분리

Viewer는 내부적으로 두 개의 branch를 가진다.

### 5.1 Raw branch

Raw branch는 trigger, gain 판단, 포화 판단을 위한 branch이다.

입력:

```text
SDR에서 읽은 complex IQ block
```

역할:

```text
1. 신호가 noise floor 대비 의미 있게 올라왔는지 판단
2. 현재 gain이 너무 낮은지 판단
3. 현재 gain이 너무 높아 포화가 의심되는지 판단
4. CNN spectrogram을 만들 block을 선택
```

계산 feature:

```text
raw_rms
raw_peak
raw_abs_mean
raw_abs_median
raw_abs_p95
raw_abs_p99
clip_ratio
noise_floor
signal_ratio
frame_power_median
frame_power_p95
frame_power_p99
```

### 5.2 CNN input branch

CNN input branch는 사람이 볼 spectrogram을 생성하는 branch이다.

이 branch는 반드시 기존 CNN 학습 데이터 생성 방식과 동일해야 한다.

기준 pipeline:

```text
remove_dc_offset
  ↓
get_cnn_input_iq
  ↓
peak normalize
  ↓
_compute_cnn_spectrogram_numpy 또는 동일 로직
  ↓
128 x 509 CNN input spectrogram
```

이 branch에서 계산하는 feature:

```text
cnn_spec_mean
cnn_spec_std
cnn_spec_min
cnn_spec_p50
cnn_spec_p95
cnn_spec_p99
cnn_spec_max
```

이 값들은 학습 데이터 분포와 비교하기 위한 값이다.

---

## 6. 왜 정규화된 CNN 입력만 보면 안 되는가

CNN 입력 spectrogram은 보통 정규화 과정을 거친다. 이 정규화는 모델이 거리나 수신 세기 변화에 덜 민감하게 패턴을 보도록 도와준다.

그러나 그 결과 절대적인 세기 정보는 일부 사라질 수 있다.

예를 들면 다음 두 조건이 CNN 입력에서는 비슷하게 보일 수 있다.

```text
70cm, gain15
2m, gain25
```

CNN 분류 입장에서는 이것이 장점일 수 있다.

```text
거리 변화에도 유사한 패턴으로 인식 가능
```

하지만 trigger와 gain 제어 입장에서는 단점이다.

```text
실제로 신호가 약한지
노이즈만 증폭된 것인지
포화 직전인지
판단하기 어려움
```

따라서 viewer는 반드시 다음 두 정보를 동시에 보여줘야 한다.

```text
1. CNN이 보는 정규화 spectrogram
2. 정규화 전 raw IQ 기반 세기 feature
```

---

## 7. Trigger와 CNN 입력의 역할 분리

Trigger는 정규화 전 raw IQ에서 판단한다.

CNN 분류 또는 CNN 입력 확인은 정규화된 spectrogram에서 수행한다.

구조는 다음과 같다.

```text
Trigger 판단:
  raw IQ power / noise floor / signal ratio 기반

CNN 입력 확인:
  정규화된 CNN spectrogram 기반
```

절대 하지 말아야 할 구조는 다음과 같다.

```text
정규화된 spectrogram만 보고 trigger 판단
```

이렇게 하면 약한 잡음도 정규화되어 그럴듯하게 보일 수 있고, false positive가 증가할 수 있다.

---

## 8. Gain과 threshold에 대한 설계 방향

### 8.1 Gain은 연속변수로 두지 않는다

Gain을 다음처럼 연속적으로 무한히 바꾸는 방식은 피한다.

```text
gain = 13.2
gain = 17.8
gain = 22.4
```

대신 몇 개의 정해진 discrete profile로 둔다.

초기 gain profile:

```text
gain_profiles = [10, 15, 20, 25]
```

각 gain의 의미는 다음과 같다.

| Gain | 목적 |
|---:|---|
| 10 | 근거리 포화 회피용 |
| 15 | 현재 학습/검증 기준에 가까운 기본 viewer gain |
| 20 | 중거리 확장 확인용 |
| 25 | 3m 근처 원거리 확인용 |

### 8.2 Viewer 초기 gain

Viewer는 자동 탐지기가 아니라 실험 도구이므로 `gain15`에서 시작하는 것이 좋다.

이유:

```text
1. 현재 학습 데이터 조건과 가장 가깝다.
2. 가까운 거리에서 포화 위험이 gain20/25보다 낮다.
3. baseline으로 삼기 좋다.
```

초기값:

```text
initial_gain = 15
auto_gain = false
```

### 8.3 Runtime 탐지 초기 gain과는 다르게 생각한다

나중에 실제 runtime 탐지에 gain profile을 붙인다면 초기 scan gain은 `20`이 더 적절할 수 있다.

이유:

```text
runtime에서는 드론이 가까이에 있다고 보장할 수 없기 때문이다.
```

따라서 구분은 다음과 같다.

```text
viewer 초기 gain = 15
runtime scan 초기 gain = 20 후보
```

---

## 9. Threshold는 고정값보다 관계식으로 둔다

기존처럼 절대 threshold 하나만 두면 gain 변화에 취약하다.

예:

```text
threshold = 4000000.0
```

이 값은 gain15에서는 적절할 수 있지만 gain20/25에서는 너무 낮거나 높을 수 있다.

따라서 threshold는 다음과 같은 관계식으로 두는 것이 더 적절하다.

```text
threshold = noise_floor × threshold_multiplier
```

초기 개념:

```text
noise_floor = median(frame_power)
threshold_multiplier = 5.0
trigger_threshold = noise_floor × 5.0
```

이 방식은 gain이 올라가서 전체 noise floor가 함께 올라가는 상황에도 어느 정도 대응할 수 있다.

---

## 10. Viewer의 상태 분기

Viewer는 raw feature를 기반으로 각 update마다 상태를 판단한다.

초기 상태는 다음 네 가지로 둔다.

```text
NO_SIGNAL
WEAK_SIGNAL
VALID_SIGNAL
OVERLOAD
```

### 10.1 NO_SIGNAL

조건 예시:

```text
signal_ratio < no_signal_ratio
```

초기값:

```text
no_signal_ratio = 2.0
```

의미:

```text
noise floor 근처이며 의미 있는 신호가 거의 없음
```

동작:

```text
CNN spectrogram 생성 생략 가능
화면에는 NO_SIGNAL 표시
CSV에는 raw feature만 기록
gain 변경 추천 없음
```

### 10.2 WEAK_SIGNAL

조건 예시:

```text
2.0 <= signal_ratio < 5.0
```

의미:

```text
뭔가 튀는 신호는 있지만 CNN 입력으로 보기에는 애매함
```

동작:

```text
기본적으로 CNN spectrogram 생성 생략 또는 참고용 표시
여러 번 반복되면 higher gain 추천
자동 gain 변경은 하지 않음
```

추천 메시지 예시:

```text
status=WEAK_SIGNAL
suggestion=TRY_HIGHER_GAIN_IF_REPEATED
```

### 10.3 VALID_SIGNAL

조건 예시:

```text
signal_ratio >= 5.0
and clip_ratio < 0.001
and raw_peak < 0.95
```

초기값:

```text
valid_signal_ratio = 5.0
overload_clip_ratio = 0.001
overload_peak = 0.95
```

의미:

```text
noise 대비 충분히 의미 있는 신호가 있고 포화는 없음
```

동작:

```text
선택된 block을 CNN 입력 spectrogram으로 변환
화면 표시
CSV에 raw feature + CNN feature 모두 기록
gain 유지
```

### 10.4 OVERLOAD

조건 예시:

```text
raw_peak >= 0.95
or clip_ratio >= 0.001
```

의미:

```text
현재 gain이 너무 높거나 SDR 입력단 포화가 의심됨
```

동작:

```text
CNN spectrogram은 만들더라도 참고용으로만 표시
화면에 OVERLOAD 경고 표시
CSV에 OVERLOAD 기록
lower gain 추천
```

추천 메시지 예시:

```text
status=OVERLOAD
suggestion=TRY_LOWER_GAIN
```

---

## 11. Gain 변경 분기 원칙

중요 원칙은 다음과 같다.

```text
gain down은 빠르게,
gain up은 신중하게.
```

이유:

```text
포화는 현재 gain이 과하다는 강한 증거이다.
하지만 약한 신호는 드론이 멀다는 뜻일 수도 있고,
그냥 아무 신호도 없다는 뜻일 수도 있다.
```

따라서 viewer에서는 다음 원칙을 따른다.

```text
NO_SIGNAL:
  gain 유지
  자동으로 올리지 않음

WEAK_SIGNAL:
  gain 유지
  반복될 때만 higher gain 추천

VALID_SIGNAL:
  gain 유지

OVERLOAD:
  lower gain 추천
```

초기 viewer에서는 자동 gain 변경을 하지 않는다.

이유:

```text
1. 실험 로그 해석이 명확해야 한다.
2. 거리 2m gain15 실험 중 자동으로 gain20으로 바뀌면 데이터가 섞인다.
3. viewer는 자동 탐지기가 아니라 실험 조건 기록 도구이다.
```

따라서 초기 구현에서는 다음을 권장한다.

```text
한 번 실행 = 하나의 gain 조건
조건 변경 = 프로그램 종료 후 재실행
```

---

## 12. YAML 변경과 hot reload 정책

기본적으로 Python 프로그램은 실행 시작 시 yaml 파일을 한 번 읽고, 그 값을 메모리에 저장한 뒤 계속 사용한다.

따라서 실행 중에 yaml 값을 수정해도 자동으로 반영되지 않는다.

예:

```text
프로그램 실행
→ configs/receiver.yaml 읽음
→ gain=15로 SDR 설정
→ 실행 중 yaml에서 gain=20으로 수정
→ 실행 중인 프로그램에는 자동 반영 안 됨
```

특히 다음 값들은 yaml만 바꿔서는 실제 SDR에 반영되지 않는다.

```text
gain
center_freq
sample_rate
rf_bandwidth
rx_index
block_size
```

해당 값들은 코드에서 SDR 객체에 다시 설정해야 한다.

초기 viewer에서는 hot reload를 넣지 않는다.

정책:

```text
1. yaml은 프로그램 시작 시 한 번만 읽는다.
2. 실행 중 yaml 변경은 반영하지 않는다.
3. 조건을 바꾸려면 프로그램을 종료하고 다시 실행한다.
4. 나중에 필요하면 수동 reload 키를 추가할 수 있다.
```

향후 확장 후보:

```text
r 키 입력 시 yaml 수동 reload
reload 후 SDR 재설정
LO/gain 변경 후 몇 block discard
CSV에 config_reload event 기록
```

그러나 초기 구현에서는 제외한다.

---

## 13. Pipeline 밀림과 latency에 대한 설계

### 13.1 문제

현재 block 설정을 예로 들면:

```text
sample_rate = 5 MSPS
block_size = 16384
```

한 block의 시간 길이는 약 3.3ms이다.

```text
16384 / 5,000,000 ≈ 0.0032768 sec
```

이론적으로 1초 동안 약 300개 block이 들어올 수 있다.

모든 block을 CNN spectrogram으로 변환하면 연산량이 과도해질 수 있다.

### 13.2 Viewer에서는 모든 block을 처리하지 않는다

Viewer 목적은 모든 데이터를 빠짐없이 저장하는 것이 아니다.

목적은 다음이다.

```text
현재 거리/gain 조건에서 CNN 입력이 어떻게 보이는지 대표적으로 확인하는 것
```

따라서 다음 정책을 사용한다.

```text
1초마다 N개 block만 읽는다.
각 block에 대해 raw feature만 계산한다.
그중 가장 의미 있는 block 하나를 선택한다.
선택된 block만 CNN spectrogram으로 변환한다.
```

초기값:

```text
update_interval_sec = 1.0
blocks_per_update = 20
select_policy = max_signal_ratio
```

### 13.3 오래된 block을 쌓아두지 않는다

Viewer는 backlog를 쌓지 않는다.

원칙:

```text
최신 상태 우선
오래된 데이터 폐기 가능
모든 block 처리하지 않음
```

이후 thread 구조를 도입한다면 queue는 작게 둔다.

```text
queue_size = 1~3
latest-frame policy
```

그러나 초기 구현은 단일 루프 방식으로 충분하다.

---

## 14. Block 선택 정책

각 화면 update마다 여러 block을 읽고, 그중 하나를 선택한다.

초기 정책:

```text
select_policy = max_signal_ratio
```

절차:

```text
1. blocks_per_update 개수만큼 IQ block 수집
2. 각 block에서 raw feature 계산
3. signal_ratio가 가장 큰 block 선택
4. 선택된 block의 상태 판단
5. VALID_SIGNAL 또는 참고 표시 대상이면 CNN spectrogram 생성
```

대안 정책:

```text
max_raw_p99
max_frame_power_p99
max_raw_rms
```

초기에는 `max_signal_ratio`가 가장 적절하다.

이유:

```text
단순 세기보다 noise floor 대비 얼마나 튀었는지가 중요하기 때문이다.
```

---

## 15. Viewer 화면 구성

초기 viewer 화면에는 다음을 표시한다.

### 15.1 메인 이미지

```text
CNN 입력과 동일한 spectrogram
shape: 128 x 509
값 범위: 학습 입력과 동일한 정규화 범위
```

### 15.2 제목 또는 side text

표시 항목:

```text
center_freq
sample_rate
rf_bandwidth
gain
rx_index
distance_m
memo
status
suggestion
```

### 15.3 Raw feature

```text
raw_rms
raw_peak
raw_abs_p95
raw_abs_p99
clip_ratio
noise_floor
signal_ratio
frame_power_p99
```

### 15.4 CNN spectrogram feature

```text
cnn_spec_mean
cnn_spec_std
cnn_spec_p95
cnn_spec_p99
cnn_spec_max
```

### 15.5 경고 표시

상태별 화면 표시:

```text
NO_SIGNAL:
  title에 NO_SIGNAL 표시

WEAK_SIGNAL:
  title에 WEAK_SIGNAL 표시
  suggestion=TRY_HIGHER_GAIN_IF_REPEATED

VALID_SIGNAL:
  title에 VALID_SIGNAL 표시
  suggestion=KEEP_GAIN

OVERLOAD:
  title에 OVERLOAD 표시
  suggestion=TRY_LOWER_GAIN
```

---

## 16. CSV logging 설계

Viewer는 화면 표시뿐 아니라 CSV 로그를 남겨야 한다.  
이 로그는 거리별 gain 조건을 분석하고, 나중에 gain-distance 근사식을 만들기 위한 기반 데이터이다.

### 16.1 로그 저장 위치

초기 저장 위치:

```text
outputs/live_viewer/logs/
```

파일명 예시:

```text
20260527_173000_live_cnn_viewer_log.csv
```

### 16.2 CSV column

필수 column:

```text
timestamp
session_id
update_index

center_freq
sample_rate
rf_bandwidth
block_size
rx_index
gain
distance_m
memo

blocks_per_update
selected_block_index
select_policy

status
suggestion

raw_rms
raw_peak
raw_abs_mean
raw_abs_median
raw_abs_p95
raw_abs_p99
clip_ratio

frame_power_median
frame_power_p95
frame_power_p99
noise_floor
signal_ratio

cnn_spec_mean
cnn_spec_std
cnn_spec_min
cnn_spec_p50
cnn_spec_p95
cnn_spec_p99
cnn_spec_max

latency_sec
processing_time_sec
```

### 16.3 CNN spectrogram을 생성하지 않은 경우

`NO_SIGNAL` 상태에서 CNN spectrogram을 생성하지 않았다면 CNN feature column은 비워두거나 `NaN`으로 기록한다.

예:

```text
cnn_spec_mean = NaN
cnn_spec_std = NaN
```

---

## 17. 이미지 저장 옵션

초기 viewer는 화면 표시를 기본으로 한다.

다만 WSL 또는 원격 환경에서 matplotlib 창이 제대로 뜨지 않을 수 있으므로 이미지 저장 옵션을 둔다.

옵션:

```text
--save-latest
```

저장 위치:

```text
outputs/live_viewer/latest/live_cnn_spectrogram_latest.png
```

또는 세션별 저장:

```text
outputs/live_viewer/images/<session_id>/
```

초기 구현에서는 `latest png`만 저장해도 충분하다.

---

## 18. 설정 입력 방식

Viewer는 기본적으로 yaml을 읽는다.

읽을 파일 후보:

```text
configs/receiver.yaml
configs/ml.yaml
```

사용할 설정:

```text
receiver:
  uri
  center_freq
  sample_rate
  rf_bandwidth
  block_size
  gain
  channels 또는 rx_index

ml:
  stft.nperseg
  stft.noverlap
  stft.nfft
  expected_freq_bins
  expected_time_frames
```

실험 편의를 위해 CLI override를 허용한다.

예:

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py   --gain 20   --distance-m 2.0   --memo drone_front_gain20_d2m
```

초기 CLI 인자 후보:

```text
--config-dir
--gain
--center-freq
--distance-m
--memo
--rx-index
--update-interval-sec
--blocks-per-update
--save-latest
--no-display
```

---

## 19. 초기 실행 예시

### 19.1 기본 yaml 기준 실행

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py
```

### 19.2 gain15, 70cm, 정면 드론

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py   --gain 15   --distance-m 0.7   --memo drone_front_g15_d070
```

### 19.3 gain20, 2m, 정면 드론

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py   --gain 20   --distance-m 2.0   --memo drone_front_g20_d200
```

### 19.4 창 없이 이미지와 CSV만 저장

```bash
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py   --gain 20   --distance-m 2.0   --memo drone_front_g20_d200   --no-display   --save-latest
```

---

## 20. 거리/gain 실험 계획

초기 실험은 모든 조합을 무작정 돌리지 않는다.

### 20.1 1차 실험: gain15 baseline

```text
0.7m gain15
1.0m gain15
1.5m gain15
2.0m gain15
```

목적:

```text
현재 학습 조건과 가까운 gain15에서 어느 거리까지 유효한지 확인
```

### 20.2 2차 실험: 약한 구간만 gain20

```text
1.5m gain20
2.0m gain20
3.0m gain20
```

목적:

```text
gain15에서 약한 구간이 gain20에서 개선되는지 확인
```

### 20.3 3차 실험: 필요 시 gain25

```text
2.0m gain25
3.0m gain25
```

목적:

```text
3m 근처 원거리에서 적정 gain 후보 확인
```

### 20.4 포화 확인

가까운 거리에서 높은 gain도 일부 확인한다.

```text
0.7m gain20
0.7m gain25
```

목적:

```text
가까운 거리에서 포화가 발생하는 gain 조건 확인
```

---

## 21. 적정 gain 판단 기준

적정 gain은 단순히 신호가 강하게 보이는 gain이 아니다.

적정 gain 조건:

```text
1. signal_ratio가 valid threshold 이상
2. clip_ratio가 거의 0
3. raw_peak가 1.0 근처에 붙어 있지 않음
4. CNN spectrogram feature가 학습 데이터 분포와 크게 다르지 않음
5. 사람이 보기에 패턴이 구분 가능함
6. WiFi/Bluetooth/Background 조건에서도 Drone-like로 오해될 만큼 과도하게 증폭되지 않음
```

초기 기준:

```text
clip_ratio < 0.001
raw_peak < 0.95
signal_ratio >= 5.0
cnn_spec_mean이 학습 데이터 평균 근처
cnn_spec_std가 학습 데이터 표준편차 근처
```

기존 live gain15 학습 데이터 기준 참고값:

```text
mean ≈ 0.618
std ≈ 0.145
```

단, 이 값은 전체 학습 데이터 기준이므로 절대 기준이 아니라 참고 기준으로 사용한다.

---

## 22. Gain-distance 근사식에 대한 계획

거리와 gain의 관계는 단순 선형식보다 로그 관계가 더 자연스럽다.

단순 후보:

```text
gain = a × distance + b
```

더 자연스러운 후보:

```text
gain_dB ≈ a × log10(distance_m) + b
```

그러나 초기 단계에서는 근사식을 바로 만들기보다, 먼저 다음 표를 만든다.

| 거리 | 적정 gain | 근거 |
|---:|---:|---|
| 0.7m | 15 | clip 없음, 학습분포 근처 |
| 1.5m | 15 또는 20 | signal_ratio와 cnn feature 비교 |
| 2.0m | 20 | gain15에서 약하면 보정 |
| 3.0m | 20 또는 25 | signal_ratio와 포화 여부 기준 |

이 표가 충분히 쌓이면 나중에 근사식을 만든다.

---

## 23. 구현 파일 계획

초기 파일명:

```text
scripts/live_cnn_spectrogram_viewer.py
```

향후 분리 가능 모듈:

```text
src/runtime/live_cnn_viewer.py
src/runtime/viewer_features.py
src/runtime/viewer_logging.py
```

하지만 초기에는 빠른 실험을 위해 script 하나로 구현해도 된다.  
다만 함수는 분리해서 작성한다.

함수 후보:

```text
parse_args()
load_viewer_configs()
build_receiver_from_configs()
apply_gain_override()
read_blocks()
compute_raw_features()
estimate_noise_floor()
compute_signal_ratio()
classify_signal_status()
select_representative_block()
compute_cnn_input_spectrogram()
compute_cnn_spec_features()
update_display()
append_csv_log()
main()
```

---

## 24. 구현 시 반드시 지킬 점

### 24.1 CNN 입력 생성 방식 일치

가장 중요하다.

```text
viewer에서 보는 spectrogram = 학습 데이터 생성 spectrogram = runtime CNN 입력 spectrogram
```

이 원칙이 깨지면 viewer 의미가 없어진다.

따라서 가능하면 기존 `_compute_cnn_spectrogram_numpy()` 함수를 그대로 import해서 사용한다.

만약 private 함수라서 import가 애매하면, 동일 로직을 공용 함수로 분리하는 것이 좋다.

추천 방향:

```text
src/features/cnn_spectrogram.py
```

또는 기존 `cnn_capture_actions.py`의 함수를 공용화한다.

### 24.2 Raw feature는 정규화 전 기준

포화와 gain 판단은 정규화 전 IQ에서 계산한다.

주의:

```text
peak normalize 이후 raw_peak를 계산하면 안 됨
```

### 24.3 CSV에는 실제 사용 gain 기록

CLI override가 있으면 override된 gain을 기록한다.

yaml 값과 실행 값이 다를 수 있으므로 다음을 구분해도 좋다.

```text
yaml_gain
effective_gain
```

초기에는 `gain` 하나만 기록해도 된다.

### 24.4 실행 중 yaml 변경은 반영하지 않음

초기 viewer에서는 hot reload 없음.

문서화:

```text
조건 변경 시 Ctrl+C로 종료 후 재실행
```

---

## 25. 초기 parameter 제안

```yaml
live_cnn_viewer:
  update_interval_sec: 1.0
  blocks_per_update: 20
  select_policy: max_signal_ratio

  auto_gain: false
  gain_profiles: [10, 15, 20, 25]
  initial_gain: 15

  thresholds:
    no_signal_ratio: 2.0
    valid_signal_ratio: 5.0
    overload_peak: 0.95
    overload_clip_ratio: 0.001

  logging:
    enabled: true
    log_dir: outputs/live_viewer/logs
    save_latest_image: true
    latest_image_dir: outputs/live_viewer/latest
```

초기에는 별도 yaml을 만들지 않고 script 내부 default + CLI override로 시작해도 된다.

나중에 안정화되면:

```text
configs/live_viewer.yaml
```

로 분리한다.

---

## 26. Acceptance Criteria

Viewer가 완성되었다고 판단하는 기준은 다음과 같다.

```text
1. yaml 또는 CLI 기준으로 Pluto+ SDR에서 IQ block을 읽을 수 있다.
2. 1초마다 N개 block을 읽고 raw feature를 계산한다.
3. signal_ratio가 가장 큰 block을 선택한다.
4. 선택된 block을 기존 CNN 입력과 동일한 spectrogram으로 변환한다.
5. 화면에 128 x 509 spectrogram이 표시된다.
6. status가 NO_SIGNAL / WEAK_SIGNAL / VALID_SIGNAL / OVERLOAD 중 하나로 표시된다.
7. raw feature와 CNN feature가 화면에 표시된다.
8. CSV 로그가 생성된다.
9. gain, distance_m, memo가 CSV에 기록된다.
10. 실행 중 yaml을 수정해도 자동 반영되지 않는다는 점이 명확히 문서화된다.
```

---

## 27. 향후 확장 계획

초기 viewer가 안정화되면 다음 기능을 추가할 수 있다.

### 27.1 Runtime 탐지와 연결

Viewer에서 검증한 raw gate 기준을 실제 detection pipeline에 반영한다.

```text
raw gate
  ↓
CNN candidate
  ↓
CNN classification
  ↓
AoA
```

### 27.2 제한적 adaptive gain

Viewer에서는 자동 gain 변경을 하지 않지만, runtime에서는 제한적으로 사용할 수 있다.

정책 후보:

```text
OVERLOAD:
  gain 한 단계 낮춰 precision 재측정

WEAK_SIGNAL 반복:
  한 단계 높은 gain으로 짧게 재확인

NO_SIGNAL:
  gain 유지

VALID_SIGNAL:
  gain 유지
```

### 27.3 수동 reload 기능

나중에 필요하면 `r` 키 입력 시 config를 다시 읽을 수 있다.

주의:

```text
center_freq 또는 gain 변경 후 SDR 재설정 필요
몇 block discard 필요
CSV에 reload event 기록 필요
```

### 27.4 실험 분석 스크립트

CSV 로그를 기반으로 거리/gain별 통계를 분석하는 script를 만든다.

후보 파일:

```text
scripts/analyze_live_viewer_logs.py
```

출력:

```text
거리별 적정 gain 표
gain별 clip ratio 그래프
distance vs signal_ratio 그래프
distance vs cnn_mean 그래프
```

---

## 28. 최종 요약

본 viewer는 단순히 화면에 spectrogram을 띄우는 도구가 아니다.

핵심은 다음이다.

```text
1. CNN 입력과 동일한 spectrogram을 실시간으로 사람이 확인한다.
2. 정규화된 CNN 입력만 보지 않고 raw IQ feature도 함께 기록한다.
3. raw feature를 이용해 CNN spectrogram을 만들 가치가 있는 block을 선별한다.
4. gain은 연속변수가 아니라 제한된 profile로 다룬다.
5. threshold는 고정 상수가 아니라 noise floor 기반 관계식으로 다룬다.
6. viewer에서는 자동 gain 변경을 하지 않고 추천만 표시한다.
7. CSV 로그를 통해 거리별 / gain별 적정 조건을 실험적으로 결정한다.
```

한 문장으로 정리하면 다음과 같다.

```text
Live CNN Spectrogram Viewer는 RF 드론 탐지 모델의 CNN 입력 품질을
거리와 gain 조건별로 검증하기 위한 실험용 시각화 및 feature logging 도구이다.
```
