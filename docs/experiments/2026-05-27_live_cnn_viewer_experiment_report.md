# 2026-05-27 Live CNN Spectrogram Viewer 실험 보고서

작성일: 2026-05-27  
프로젝트: RF 기반 드론 탐지 캡스톤 프로젝트  
브랜치: `experiment/runtime-integrated-pipeline`  
작성 목적: Live CNN Spectrogram Viewer 구현 및 2.45GHz 드론 RF 링크 관측 실험 결과를 정리하고, 기존 데이터셋 가정의 문제점과 다음 재학습 방향을 정의한다.

---

## 1. 오늘 작업 요약

오늘은 `scripts/live_cnn_spectrogram_viewer.py`를 추가하여 Pluto+ SDR에서 실시간 IQ block을 읽고, CNN 입력과 동일한 spectrogram을 생성해 확인하는 viewer를 구현하였다.

초기 viewer의 목적은 CNN 추론이 아니라 다음을 확인하는 것이다.

```text
1. SDR에서 실제 IQ block이 정상 수신되는가
2. CNN 학습 입력과 동일한 128 x 509 spectrogram이 생성되는가
3. 정규화된 CNN spectrogram과 정규화 전 raw feature를 동시에 확인할 수 있는가
4. gain / 거리 / 방향 / 조종기 위치 변화에 따른 signal_ratio 변화를 기록할 수 있는가
5. 향후 데이터 수집 기준을 정할 수 있는가
```

오늘 실험을 통해 viewer는 정상 동작함을 확인하였다. 또한 기존의 중요한 가정 하나가 잘못되었을 수 있음을 발견하였다.

기존 가정은 다음과 같았다.

```text
드론이 안테나 근처로 접근하면 드론 자체 또는 드론 쪽 RF 신호가 강하게 잡힐 것이다.
```

그러나 오늘 관찰 결과는 다음에 가까웠다.

```text
강한 2.45GHz 신호는 드론 단독 또는 조종기 단독 상태에서는 거의 나타나지 않았다.
드론과 조종기가 연결되었을 때 강하게 나타났고,
특히 조종기 위치와 방향에 따라 수신 세기가 크게 바뀌었다.
```

따라서 현재까지의 결과만 보면, 기존 모델은 순수한 “드론 접근 탐지”라기보다 “드론-조종기 링크 RF 감지”에 가까운 데이터를 학습했을 가능성이 있다.

---

## 2. 구현한 파일

추가된 파일:

```text
scripts/live_cnn_spectrogram_viewer.py
```

현재 `git status` 기준 상태:

```text
Untracked files:
  scripts/live_cnn_spectrogram_viewer.py
```

커밋 전 확인 명령어:

```bash
PYTHONPATH=. python -m py_compile scripts/live_cnn_spectrogram_viewer.py
```

커밋 후보:

```bash
git add scripts/live_cnn_spectrogram_viewer.py
git commit -m "feat: add live CNN spectrogram viewer"
git push
```

---

## 3. Viewer v1 기능

현재 viewer v1은 다음 기능을 수행한다.

```text
1. configs/receiver.yaml, configs/ml.yaml 로드
2. CLI 인자로 center_freq, gain, distance_m, memo 등 override
3. SDR에서 blocks_per_update 개수만큼 IQ block 수집
4. 각 block의 raw feature 계산
5. signal_ratio가 가장 큰 block 선택
6. 선택된 block의 상태 판단
   - NO_SIGNAL
   - WEAK_SIGNAL
   - VALID_SIGNAL
   - OVERLOAD
7. VALID_SIGNAL 또는 참고 대상 block에 대해 CNN input spectrogram 생성
8. latest PNG 저장
9. CSV 로그 저장
```

중요한 점은 viewer가 CNN inference를 수행하지 않는다는 것이다. 현재 목적은 분류 결과 확인이 아니라, CNN에 들어가기 전 입력 품질을 확인하는 것이다.

---

## 4. 실험 공통 설정

오늘 주로 사용한 조건은 다음과 같다.

```text
SDR: Pluto+ SDR
center_freq: 2450000000 Hz
sample_rate: 5000000 Hz
rf_bandwidth: 5000000 Hz
block_size: 16384
rx_index: 0
STFT nperseg: 128
STFT noverlap: 96
STFT nfft: 128
CNN spectrogram shape: 128 x 509
gain: 10 또는 15
valid_signal_ratio: 5 또는 50
```

최종적으로 의미 있는 기준으로 판단한 값은 다음이다.

```text
center_freq = 2450000000
gain = 10
valid_signal_ratio = 50
```

---

## 5. Overload 판단 기준 문제

초기 viewer에서는 `raw_peak >= 0.95` 또는 `clip_ratio >= 0.001`이면 `OVERLOAD`로 판단하도록 설정하였다.

하지만 실제 PlutoReceiver가 넘겨주는 raw IQ 값은 `[-1, 1]` 범위로 정규화된 값이 아니었다. 따라서 아무 신호가 없는 상태에서도 `raw_peak`가 3 이상으로 나타났고, 기존 기준을 그대로 적용하면 대부분의 block이 `OVERLOAD`로 잘못 판단되었다.

따라서 오늘 실험에서는 임시로 다음 옵션을 사용하여 overload 판단을 사실상 비활성화하였다.

```bash
--overload-peak 1000000000
--overload-clip-ratio 1.1
```

향후 수정 방향:

```text
1. raw_peak 기준 0.95를 기본값으로 사용하지 않는다.
2. Pluto raw scale에 맞는 포화 판단 기준을 새로 정의한다.
3. clip_ratio 계산도 raw IQ scale에 의존하지 않도록 재설계한다.
4. viewer에는 raw scale mode 또는 calibrated overload mode를 추가한다.
```

---

## 6. 2.437GHz 실험 결과

처음에는 `center_freq = 2437000000 Hz`에서 확인하였다.

이 주파수는 Wi-Fi 2.4GHz 채널 6 중심에 해당한다. 드론을 켠 상태에서도 강한 신호가 잘 잡히지 않았고, 주변 Wi-Fi 또는 Bluetooth로 보이는 간헐적인 background burst만 일부 관측되었다.

결론:

```text
2.437GHz는 현재 드론 링크 관측 중심주파수로 적합하지 않았다.
```

---

## 7. 2.450GHz 실험 결과

`center_freq = 2450000000 Hz`로 변경하자 드론 연결 상태에서 신호가 매우 강하게 잡혔다.

### 7.1 0.7m, gain15

조건:

```text
center_freq = 2450MHz
gain = 15
distance = 0.7m
drone ON + controller connected
```

결과:

```text
30/30 update가 VALID_SIGNAL
signal_ratio가 수백~수천 단위로 나타남
일부 구간에서 signal_ratio 5000 이상 관측
```

결론:

```text
2450MHz에서 드론 연결 신호는 매우 강하게 잡힌다.
0.7m에서는 gain15까지 올릴 필요가 없을 가능성이 높다.
```

### 7.2 0.7m, gain10

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 0.7m
drone ON + controller connected
```

결과:

```text
10/10 update가 VALID_SIGNAL
signal_ratio 대략 1380 ~ 1818
clip_ratio = 0
```

결론:

```text
0.7m에서는 gain10으로도 충분히 안정적으로 잡힌다.
```

### 7.3 1.2m, gain10

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
drone ON + controller connected
```

결과:

```text
30/30 update가 VALID_SIGNAL
signal_ratio 대략 359 ~ 1207
clip_ratio = 0
```

결론:

```text
집 실험 기준 최대 거리인 1.2m에서도 gain10으로 충분히 강하게 잡힌다.
```

### 7.4 1.2m, gain10, valid_signal_ratio 50

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
valid_signal_ratio = 50
drone ON + controller connected
```

결과:

```text
30/30 update가 VALID_SIGNAL
signal_ratio 최저 약 431
signal_ratio 최고 약 1147
```

결론:

```text
valid_signal_ratio를 50까지 올려도 드론 연결 상태는 안정적으로 통과한다.
```

---

## 8. Background 실험 결과

조건:

```text
center_freq = 2450MHz
gain = 10
drone OFF
valid_signal_ratio = 50
```

결과:

```text
VALID_SIGNAL = 0회
대부분 NO_SIGNAL
일부 WEAK_SIGNAL
signal_ratio 대부분 1.1 ~ 1.4
약한 burst는 3 ~ 6 정도
```

결론:

```text
2450MHz / gain10 / threshold50 조건에서는 background와 드론 연결 상태가 매우 잘 분리된다.
```

비교:

```text
background 최대 signal_ratio ≈ 6.4
drone ON 1.2m 최저 signal_ratio ≈ 431
```

따라서 현재 집 실험 기준 raw gate는 다음 값이 적절하다.

```text
signal_ratio >= 50
```

---

## 9. 방향 실험 결과

### 9.1 오른쪽 20도, 1.2m, gain10

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
angle = 오른쪽 20도
valid_signal_ratio = 50
```

결과:

```text
30개 update 중 29개 VALID_SIGNAL
1개 WEAK_SIGNAL
VALID 구간 signal_ratio 대략 123 ~ 1844
WEAK 구간 signal_ratio 약 7.3
```

결론:

```text
오른쪽 20도까지는 gain10으로 대부분 안정적으로 잡힌다.
다만 순간적으로 신호가 약해지는 block이 발생할 수 있다.
```

이 결과는 runtime에서 단일 block 기준 판단보다 multi-block voting이 필요하다는 근거가 된다.

---

## 10. 상태별 분리 실험

### 10.1 조종기만 ON, 드론 OFF

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
controller ON
drone OFF
valid_signal_ratio = 50
```

결과:

```text
VALID_SIGNAL = 0회
대부분 NO_SIGNAL 또는 WEAK_SIGNAL
signal_ratio 대략 1.1 ~ 3.4
```

결론:

```text
조종기 단독 상태에서는 2450MHz에서 강한 신호가 지속적으로 잡히지 않았다.
```

### 10.2 드론만 ON, 조종기 OFF

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
drone ON
controller OFF
valid_signal_ratio = 50
```

결과:

```text
VALID_SIGNAL = 0회
대부분 NO_SIGNAL
일부 WEAK_SIGNAL
signal_ratio 대략 1.1 ~ 4.1
```

결론:

```text
드론 본체 단독 상태에서도 2450MHz에서 강한 신호는 거의 잡히지 않았다.
```

### 10.3 드론 ON + 조종기 ON + 연결 완료 + 모터 OFF

조건:

```text
center_freq = 2450MHz
gain = 10
distance = 1.2m
drone ON
controller ON
connection established
motor OFF
valid_signal_ratio = 50
```

결과:

```text
강한 구간에서는 signal_ratio 수백~수천
약한 방향/위치에서는 signal_ratio 십 단위까지 감소
조종기와 안테나/드론의 상대 위치에 따라 값이 크게 바뀜
```

결론:

```text
강한 RF 신호는 모터 동작 자체보다는 드론-조종기 연결 링크에서 주로 발생하는 것으로 보인다.
```

---

## 11. 조종기 위치에 따른 신호 변화

마지막 실험에서 조종기 위치를 여러 곳으로 이동시키며 관찰하였다.

사용자 관찰 기록:

```text
0~15 update: 조종기가 드론 뒤
16~33 update: 조종기가 안테나 뒤
34~39 update: 안테나 앞, 드론과 안테나 약 60도, 거리 약 1m
그 뒤 큰 값: 조종기를 드론과 안테나 사이에 둔 상태
```

관찰 결과:

```text
조종기가 드론 뒤에 있을 때도 일부 강한 신호가 잡힘
조종기가 안테나와 수직이거나 안테나 뒤쪽에 있을 때 신호가 십 단위로 약해짐
조종기를 드론과 안테나 사이에 두면 signal_ratio가 수천 단위로 급증
```

중요 결론:

```text
현재 2450MHz에서 강하게 잡히는 신호는 드론 본체가 전방위로 균일하게 방사하는 신호가 아니다.
조종기-드론 링크의 방향성, 조종기 위치, 드론 자세, 실내 반사 환경에 의해 수신 세기가 크게 달라진다.
```

---

## 12. 오늘 발견한 핵심 문제

기존 프로젝트의 목표는 다음과 같았다.

```text
조종자는 멀리 있고,
드론이 안테나 근처로 접근할 때,
안테나가 드론 쪽 RF를 탐지한다.
```

그러나 오늘 실험 결과, 가까운 실내 조건에서는 조종기 위치가 수신 세기에 큰 영향을 미쳤다.

따라서 기존 학습 데이터가 다음 조건을 많이 포함했다면 문제가 될 수 있다.

```text
조종기가 안테나 근처에 있음
조종기가 안테나 방향을 향함
조종기가 드론과 안테나 사이에 있음
```

이 경우 모델은 “드론 자체”가 아니라 “조종기 또는 조종기-드론 링크가 가까운 조건”을 Drone-like로 학습했을 가능성이 있다.

따라서 현재 모델을 그대로 최종 모델로 쓰면 다음 위험이 있다.

```text
1. 조종기가 가까운 상황을 Drone-like로 강하게 판단할 수 있음
2. 실제 운용 조건인 controller far / drone near 조건에서 성능이 떨어질 수 있음
3. 드론 접근 탐지 모델이 아니라 조종기 근접 탐지 모델처럼 동작할 수 있음
```

---

## 13. 기존 데이터 처리 방향

기존 데이터를 모두 버릴 필요는 없다. 하지만 라벨과 의미를 재분류해야 한다.

### 13.1 유지 가능 데이터

```text
Background
WiFi
Bluetooth
일부 Drone-connected 링크 실험 데이터
```

### 13.2 주의가 필요한 데이터

```text
controller 위치가 안테나 근처였던 Drone-like 데이터
controller가 안테나 방향을 향했던 Drone-like 데이터
controller가 드론과 안테나 사이에 있었던 데이터
```

이 데이터는 순수한 `drone_near_controller_far` 양성 데이터로 쓰면 위험하다.

### 13.3 새로 필요한 데이터

```text
1. controller_near_negative
2. controller_only_negative
3. drone_only_negative
4. drone_near_controller_far_positive
5. outdoor_background
6. drone_connected_controller_far_motor_off
7. drone_connected_controller_far_motor_on
```

---

## 14. 재학습 방향

기존 4-class 구조는 다음과 같았다.

```text
Background
WiFi
Bluetooth
Drone-like
```

하지만 오늘 결과를 반영하면 최소한 다음 hard negative가 필요하다.

```text
Controller-only / Controller-near
```

추천 클래스 구조:

```text
Background
WiFi
Bluetooth
Controller-near
Drone-near-controller-far
```

혹은 단순 binary 모델을 유지하더라도 negative에 반드시 다음을 포함해야 한다.

```text
Controller-only
Controller-near
Controller-facing-antenna
Controller-between-drone-and-antenna
```

양성 데이터는 다음 조건으로 다시 수집해야 한다.

```text
Drone-near-controller-far
```

정의:

```text
안테나와 드론은 가까움
조종기는 안테나와 충분히 멂
조종기가 안테나를 직접 향하지 않음
드론과 조종기 연결은 유지됨
```

---

## 15. 다음 실험 계획

오늘은 실내 공간 제약 때문에 1.2m까지만 실험하였다. 다음 실험은 실외에서 진행해야 한다.

### 15.1 실외 background

```text
center_freq = 2450MHz
gain = 10
valid_signal_ratio = 50
drone OFF
controller OFF
```

### 15.2 조종기 멀리 두기 실험

핵심 실험:

```text
드론-안테나 거리: 1.2m 또는 2m
조종기-안테나 거리: 3m, 5m, 10m
조종기 방향: 안테나를 직접 향하지 않게
모터: OFF / ON 각각
```

확인할 질문:

```text
조종기가 멀어져도 드론 근처 안테나에서 2450MHz 링크 신호가 충분히 잡히는가?
```

### 15.3 controller-near negative 수집

다음 조건을 negative로 수집해야 한다.

```text
조종기만 ON, 드론 OFF
조종기 가까움
조종기가 안테나를 향함
조종기가 안테나 앞/뒤/측면에 있음
```

이 데이터는 모델이 조종기 신호를 Drone-like로 오인하지 않도록 하는 hard negative 역할을 한다.

---

## 16. Runtime 설계 반영점

오늘 실험에서 방향성에 따라 signal_ratio가 크게 변동함을 확인하였다. 따라서 runtime에서 단일 block 하나만 보고 판단하면 안 된다.

추천 방식:

```text
최근 N개 update 중 K개 이상 signal_ratio >= threshold이면 후보로 판단
```

예시:

```text
최근 10개 update 중 3개 이상 VALID_SIGNAL
또는 최근 30개 update 중 5개 이상 VALID_SIGNAL
```

이 방식이 필요한 이유:

```text
1. 드론 링크 신호는 hopping/방향성/자세/반사 영향으로 순간적으로 약해질 수 있음
2. 단일 block 기준이면 false negative가 커짐
3. 시간 window 기반 voting은 순간 약화를 완화할 수 있음
```

---

## 17. 다음 코드 작업

### 17.1 viewer 저장 기능 추가

현재 viewer는 CSV와 latest PNG만 저장한다. 다음 기능을 추가해야 한다.

```text
--save-selected-spec
--save-selected-iq
--save-only-valid
```

저장 구조 후보:

```text
outputs/live_viewer/selected_spec/<session_id>/
outputs/live_viewer/selected_iq/<session_id>/
```

CSV에는 저장된 파일 경로를 함께 기록한다.

### 17.2 overload 판단 재설계

현재 임시로 overload 판단을 꺼두었다. Pluto raw scale에 맞는 기준을 새로 만들어야 한다.

### 17.3 실험 분석 스크립트

CSV 로그를 기반으로 조건별 signal_ratio 통계를 정리하는 스크립트가 필요하다.

후보 파일:

```text
scripts/analyze_live_viewer_logs.py
```

출력:

```text
조건별 VALID 비율
signal_ratio min/median/max
raw_peak min/median/max
memo별 요약표
```

---

## 18. 오늘 최종 결론

오늘 실험의 최종 결론은 다음과 같다.

```text
1. Live CNN Spectrogram Viewer는 정상 동작한다.
2. 드론 연결 링크는 2.450GHz에서 매우 강하게 관측된다.
3. 1.2m 이내 실내 조건에서는 gain10으로도 충분하다.
4. Background와 드론 연결 상태는 signal_ratio 기준으로 매우 잘 분리된다.
5. valid_signal_ratio=50은 현재 집 실험에서 적절한 raw gate 후보이다.
6. 조종기 단독, 드론 단독 상태는 강한 신호를 만들지 않았다.
7. 드론과 조종기가 연결되면 강한 신호가 나타난다.
8. 하지만 조종기 위치와 방향이 수신 세기에 큰 영향을 준다.
9. 기존 데이터가 조종기 근접 조건에 치우쳤다면 재학습이 필요하다.
10. 앞으로는 controller-near hard negative와 drone-near-controller-far positive를 새로 수집해야 한다.
```

한 문장으로 정리하면 다음과 같다.

```text
오늘 실험은 단순한 viewer 검증을 넘어, 기존 Drone-like 데이터셋이 조종기-드론 링크 방향성에 영향을 받을 수 있음을 확인한 실험이다.
따라서 최종 드론 접근 탐지 모델을 만들기 위해서는 조종기 위치를 분리한 새 데이터셋 설계와 재학습이 필요하다.
```
