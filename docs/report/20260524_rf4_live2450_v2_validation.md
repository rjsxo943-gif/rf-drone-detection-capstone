# RF4 live2450 v2 실증 결과 정리

## 1. 목적

RF4 모델에서 WiFi 및 Background가 Drone-like로 오탐되는 문제를 줄이기 위해, live 환경에서 수집한 hard negative 데이터와 Drone-like selected burst 데이터를 추가하였다.

이후 rf4_cnn_live2450_v2 모델을 학습하고, 실제 실시간 수신 조건에서 Background, WiFi, Drone-like 조건을 각각 검증하였다.

## 2. 대표 모델

모델 경로:

outputs/ml/rf4_cnn_live2450_v2/best_model.pt

## 3. 학습 데이터 구성

이번 v2 학습에는 각 클래스당 100장씩 사용하였다.

- Background: 2450 MHz, gain 25, selected background block 100장
- WiFi: 2437 MHz, gain 25, WiFi channel 6 hard negative 100장
- Drone-like: 2450 MHz, gain 25, drone/control selected burst 100장
- Bluetooth: 기존 canonical01 데이터 100장

## 4. 학습 결과

전체 테스트 정확도:

test_acc = 0.8667

클래스별 결과:

- Background precision=0.7222, recall=0.8667, f1=0.7879
- WiFi precision=1.0000, recall=0.6667, f1=0.8000
- Bluetooth precision=1.0000, recall=0.9333, f1=0.9655
- Drone-like precision=0.8333, recall=1.0000, f1=0.9091

Confusion matrix:

Background:
13 Background, 0 WiFi, 0 Bluetooth, 2 Drone-like

WiFi:
4 Background, 10 WiFi, 0 Bluetooth, 1 Drone-like

Bluetooth:
1 Background, 0 WiFi, 14 Bluetooth, 0 Drone-like

Drone-like:
0 Background, 0 WiFi, 0 Bluetooth, 15 Drone-like

오프라인 테스트에서는 Background와 WiFi 일부가 Drone-like로 오분류되었다. 따라서 CNN 단독 판정보다는 p99/max 기반 strong gate와 multi-block voting을 함께 사용하는 방식이 필요하다.

## 5. Live 검증 결과

### 5.1 Drone @2450 MHz, gain 25

조건:

- 드론 ON
- 조종기 ON
- 중심주파수 2450 MHz
- gain 25
- 50 blocks 테스트

결과:

final Drone-like = 13 / 50

드론 신호는 연속 신호가 아니라 burst 형태로 나타나므로 모든 block이 Drone-like로 나올 필요는 없다. 50 block 중 13개가 strong gate를 통과하여 Drone-like로 판정되었다.

### 5.2 WiFi channel 6 @2437 MHz, gain 25

조건:

- 드론 OFF
- WiFi channel 6 트래픽 발생
- 중심주파수 2437 MHz
- gain 25
- 50 blocks 테스트

결과:

final Drone-like = 0 / 50

WiFi hard negative 추가 이후, 강한 WiFi burst가 WiFi 또는 Background/Unknown으로 처리되었고 Drone-like 오탐은 발생하지 않았다.

### 5.3 Background @2450 MHz, gain 25

조건:

- 드론 OFF
- 조종기 OFF
- 신호발생기 OFF
- 중심주파수 2450 MHz
- gain 25
- 50 blocks 테스트

결과:

final Drone-like = 0 / 50
final Background = 48 / 50
final Unknown = 2 / 50

Background 조건에서 CNN raw 결과가 일부 Drone-like로 흔들리는 경우가 있었으나, p99/max strong gate를 통과하지 못해 최종 판정은 Background로 내려갔다.

## 6. 현재 판정 규칙

현재 실시간 판정은 다음 흐름을 사용한다.

1. RF block 수신
2. canonical01 spectrogram 생성
3. CNN 추론
4. CNN raw 결과가 Drone-like여도 p99 >= 0.65 and max >= 0.80 조건을 만족해야 최종 Drone-like로 인정
5. 여러 block 중 final Drone-like 개수가 일정 개수 이상이면 Drone candidate로 판단

초기 voting 기준:

50 blocks 중 final Drone-like >= 3 이면 Drone candidate로 판단

## 7. 결론

rf4_cnn_live2450_v2 모델과 strong gate를 결합한 결과, 현재 실험 조건에서 다음 결과를 얻었다.

- Drone @2450 MHz: final Drone-like 13 / 50
- WiFi channel 6 @2437 MHz: final Drone-like 0 / 50
- Background @2450 MHz: final Drone-like 0 / 50

따라서 현재 기준에서는 rf4_cnn_live2450_v2 모델과 p99/max strong gate를 결합한 방식이 가장 안정적이다.

## 8. 다음 단계

현재 모델은 각 클래스당 100장 기준의 검증용 모델이다. 최종 안정화를 위해서는 다음 데이터 확장이 필요하다.

- Background: 300장 이상
- WiFi: 300장 이상
- Bluetooth: 300장 이상
- Drone-like: 300장 이상

특히 같은 조건에서 연속으로 300장을 모으는 것보다, 거리, 방향, gain, 시간대, 주변 간섭 조건을 나누어 여러 세션으로 수집하는 것이 더 중요하다.

추천 확장 조건:

Drone-like:
- 2450 MHz, gain 25, 50 cm, control move
- 2450 MHz, gain 25, 70 cm, control move
- 2450 MHz, gain 25, 1 m, motor/control

WiFi:
- 2437 MHz, gain 25, channel 6 strong traffic
- 2437 MHz, gain 20, channel 6 normal traffic
- 2450 MHz 근처 WiFi leakage 확인용

Background:
- 2450 MHz, gain 25, all devices OFF
- 2450 MHz, gain 20, all devices OFF
- 시간대를 바꾼 주변 간섭 포함 조건

## 9. 현재 대표 조합

대표 모델:

outputs/ml/rf4_cnn_live2450_v2/best_model.pt

대표 판정 방식:

CNN + p99/max strong gate + multi-block voting

현재 기준 판정:

50 blocks 중 final Drone-like >= 3 이면 Drone candidate
