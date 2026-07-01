# Project Story for Interview

이 문서는 RF 드론 탐지 졸업작품 경험을 자기소개서, 포트폴리오, 면접 답변에 활용하기 위해 정리한 문서이다.

기술 문서는 시스템의 구조를 설명하는 것이 목적이고, 이 문서는 “내가 어떤 문제를 어떻게 해결했는가”를 말로 설명하기 위한 자료이다.

---

## 1. One-line Summary

2.4GHz RF 신호를 SDR로 수집하고, CNN 기반 드론 신호 분류와 2채널 위상차 기반 AoA 추정을 결합하여 드론 관련 RF activity의 존재와 방향을 실시간으로 추정하는 시스템을 구현했다.

---

## 2. Project in One Paragraph

본 프로젝트는 Pluto+ SDR을 이용해 2.4GHz 대역의 RF 신호를 수집하고, STFT spectrogram 기반 CNN으로 드론 관련 RF activity를 판별한 뒤, RX0/RX1 두 수신 채널의 위상차를 이용해 신호 도래각을 추정하는 RF sensing prototype이다.

단순히 CNN으로 Drone / NotDrone을 분류하는 데서 끝내지 않고, RawNoiseGate, CNN Top5 vote, immediate recheck, coherence gate, sector voting, experimental range indication, OpenCV runtime dashboard를 결합하여 실제 시연 가능한 통합 pipeline으로 구성했다.

---

## 3. Problem

2.4GHz 대역은 드론 조종 신호뿐 아니라 Wi-Fi, Bluetooth 등 다양한 간섭 신호가 함께 존재하는 환경이다.

따라서 단순히 RF energy가 강하다는 이유만으로 드론 신호라고 판단하면 오탐이 발생할 수 있다.

또한 드론 신호를 탐지하는 것뿐 아니라, 신호가 어느 방향에서 들어오는지 추정하는 기능도 필요했다.

핵심 문제는 다음과 같았다.

```text
1. 복잡한 2.4GHz RF 환경에서 드론 관련 신호 후보를 어떻게 찾을 것인가?
2. 단일 CNN 결과의 오탐을 어떻게 줄일 것인가?
3. 두 수신 채널의 위상차를 이용해 방향 정보를 어떻게 얻을 것인가?
4. 실시간으로 확인 가능한 runtime UI를 어떻게 구성할 것인가?
5. 야외 실험에서 흔들리는 RF 신호를 어떻게 안정적으로 다룰 것인가?
```

---

## 4. My Contribution

내가 중점적으로 기여한 부분은 SDR 수신 이후의 RF signal processing pipeline과 실시간 추정 구조를 시스템으로 연결한 것이다.

```text
- Pluto+ SDR 기반 2채널 IQ 수신 구조 구성
- DC offset 제거 및 gain/phase calibration 흐름 정리
- STFT spectrogram 기반 CNN 추론 pipeline 구현
- RawNoiseGate 기반 background block 차단 구조 적용
- CNN Top5 vote 및 immediate recheck로 오탐 억제
- RX0/RX1 phase difference 기반 AoA 계산 구조 구현
- coherence 기반 AoA 신뢰도 검증
- sector voting 및 experimental range indication 구성
- OpenCV 기반 실시간 scan/precision dashboard 구현
- 야외 실험 및 CSV replay 기반 검증 수행
```

---

## 5. Technical Keywords

```text
SDR
Pluto+
IQ signal
2.4GHz ISM band
DC offset removal
STFT
Spectrogram
CNN
RawNoiseGate
Top5 vote
Immediate recheck
Phase difference
AoA
Coherence
Sector estimation
OpenCV dashboard
CSV replay
```

---

## 6. STAR Answer Draft

### Situation

졸업작품으로 비인가 드론 탐지 시스템을 개발해야 했고, 카메라 기반 탐지가 아니라 RF 신호 기반 탐지 방식을 선택했다.

2.4GHz 대역에는 드론 조종 신호뿐 아니라 Wi-Fi, Bluetooth 등 다양한 간섭 신호가 존재하기 때문에, 실제 환경에서 안정적으로 동작하는 탐지 구조가 필요했다.

### Task

목표는 Pluto+ SDR을 이용해 2.4GHz RF 신호를 수집하고, 드론 관련 RF activity를 구분하며, 가능하면 신호가 들어오는 방향까지 추정하는 시스템을 구현하는 것이었다.

단순 탐지뿐 아니라, 실시간 시연이 가능하도록 scan mode, precision mode, AoA/Sector 표시, runtime dashboard까지 연결해야 했다.

### Action

먼저 SDR에서 수집한 IQ 데이터를 block 단위로 처리하고, DC offset 제거와 raw energy 기반 gate를 적용했다.

이후 STFT spectrogram을 CNN 입력으로 만들어 Drone / NotDrone 분류를 수행했다.

CNN 단일 block 결과만 사용하면 순간 오탐에 취약했기 때문에, RawNoiseGate, Top5 vote, immediate recheck를 결합하여 후보 신호를 검증했다.

방향 추정을 위해 RX0/RX1 두 채널의 phase difference를 계산하고, wavelength와 antenna spacing을 이용해 AoA를 추정했다.

또한 coherence gate를 적용해 신뢰도가 낮은 AoA 후보를 제외하고, sector voting으로 방향 출력을 안정화했다.

마지막으로 OpenCV dashboard를 구성하여 scan 상태와 precision tracking 상태를 실시간으로 확인할 수 있게 했다.

### Result

최종적으로 RF activity 탐지, CNN 기반 후보 검증, AoA/Sector 추정, experimental range indication, scan/precision dashboard까지 연결된 실시간 RF sensing prototype을 구현했다.

이 경험을 통해 단순 모델 학습보다 실제 시스템에서는 신호처리, calibration, threshold policy, 상태 관리, UI, 실험 검증이 함께 중요하다는 점을 배웠다.

---

## 7. Strong Interview Version

이 프로젝트에서 저는 단순히 CNN 모델을 학습시키는 데서 끝내지 않고, SDR 수신부터 실시간 dashboard까지 이어지는 전체 RF sensing pipeline을 구성했습니다.

2.4GHz 대역은 Wi-Fi나 Bluetooth 간섭이 많기 때문에 RF energy만으로는 드론 신호를 판단하기 어려웠습니다. 그래서 RawNoiseGate로 background block을 먼저 걸러내고, CNN Top5 vote와 immediate recheck를 적용해 순간 오탐을 줄였습니다.

또한 2채널 IQ 데이터의 위상차를 이용해 AoA를 계산하고, coherence gate와 sector voting을 붙여 방향 추정 결과를 안정화했습니다.

최종적으로 scan mode에서 후보 주파수를 찾고, precision mode에서 AoA/Sector/Range 정보를 OpenCV dashboard로 표시하는 구조까지 구현했습니다.

이 경험을 통해 이론적인 신호처리와 머신러닝 모델을 실제 RF 환경에서 동작하는 runtime system으로 연결하는 경험을 얻었습니다.

---

## 8. Short Resume Version

Pluto+ SDR 기반 2.4GHz RF 드론 탐지 시스템을 개발하였다. IQ 신호 수집, STFT spectrogram 생성, CNN 기반 Drone/NotDrone 분류, RawNoiseGate 기반 후보 제한, RX0/RX1 위상차 기반 AoA 추정, coherence 기반 신뢰도 검증, sector voting, OpenCV 실시간 dashboard를 통합하여 RF activity 탐지 및 방향 추정 prototype을 구현하였다.

---

## 9. Self-introduction Version

전자공학 전공 지식을 바탕으로 실제 RF 신호를 수집하고 분석하는 졸업작품을 수행했습니다.

Pluto+ SDR을 이용해 2.4GHz 대역 IQ 데이터를 수집했고, STFT spectrogram과 CNN을 활용해 드론 관련 RF 신호를 분류했습니다.

또한 두 수신 채널의 위상차를 이용해 신호 도래 방향을 추정하고, OpenCV dashboard로 실시간 시각화하는 시스템을 구현했습니다.

이 과정에서 단순 알고리즘 구현뿐 아니라 calibration, threshold 설정, 오탐 억제, 실험 검증, runtime UI 구성까지 경험했습니다.

---

## 10. Interview Questions and Answers

### Q1. 이 프로젝트에서 본인이 가장 많이 기여한 부분은?

A. SDR로 수신한 IQ 데이터를 실제로 탐지와 방향 추정까지 연결하는 pipeline을 구성한 부분입니다. STFT spectrogram 기반 CNN inference, RawNoiseGate, Top5 vote, AoA 계산, coherence gate, sector voting, OpenCV dashboard를 연결해 실시간으로 동작하는 구조를 만들었습니다.

---

### Q2. CNN만 사용하지 않은 이유는?

A. 실제 2.4GHz RF 환경에서는 Wi-Fi, Bluetooth, spur, noise 등 다양한 간섭이 존재합니다. CNN 단일 block 결과만 사용하면 순간적인 오탐이 발생할 수 있기 때문에, RawNoiseGate로 후보를 먼저 제한하고 Top5 vote와 immediate recheck를 붙여 안정성을 높였습니다.

---

### Q3. AoA는 어떤 방식으로 구했나요?

A. RX0/RX1 두 채널의 IQ 신호에서 phase difference를 계산하고, 안테나 간격과 wavelength를 이용해 arcsin 관계로 AoA를 추정했습니다. 다만 실제 환경에서는 멀티패스나 위상 흔들림이 있기 때문에 coherence gate와 sector voting을 함께 적용했습니다.

---

### Q4. 가장 어려웠던 점은 무엇인가요?

A. 실제 RF 신호가 이론처럼 안정적으로 들어오지 않는다는 점이 가장 어려웠습니다. SDR 온도 변화, 안테나 편차, gain mismatch, multipath, 주변 2.4GHz 간섭 때문에 결과가 흔들렸습니다. 이를 줄이기 위해 calibration, raw gate, CNN vote, recheck, coherence gate 같은 여러 방어 구조를 추가했습니다.

---

### Q5. 이 프로젝트가 단순한 CNN 분류 프로젝트와 다른 점은?

A. CNN은 전체 pipeline의 한 부분일 뿐입니다. 실제 시스템은 SDR 수신, IQ 전처리, raw signal gate, CNN 검증, AoA 추정, sector 안정화, range indication, auto-return, OpenCV dashboard까지 포함합니다. 즉 모델 하나가 아니라 RF sensing runtime system을 구현한 프로젝트입니다.

---

### Q6. 결과가 완벽했나요?

A. 완벽한 제품 수준은 아닙니다. Range class는 정확한 거리 추정이 아니라 실험적 coarse indication이고, 다양한 환경에서 일반화하려면 데이터가 더 필요합니다. 하지만 졸업작품 범위에서는 RF activity 탐지, CNN 검증, 방향 추정, 실시간 시각화까지 연결된 prototype을 구현했다는 데 의의가 있습니다.

---

### Q7. 프로젝트에서 배운 점은?

A. 실제 시스템에서는 알고리즘 하나보다 pipeline 전체 설계가 중요하다는 점을 배웠습니다. 모델 정확도만으로는 부족하고, calibration, 데이터 품질, threshold policy, 신호처리, UI, 예외 상태 처리까지 함께 설계해야 실제 환경에서 동작한다는 것을 경험했습니다.

---

## 11. Things to Be Careful About in Interview

면접에서 과장하면 안 되는 부분은 다음과 같다.

```text
- 정확한 거리 추정기라고 말하지 않기
- 상용 대드론 시스템 전체를 구현했다고 말하지 않기
- 모든 드론을 탐지할 수 있다고 말하지 않기
- CNN이 모든 판단을 한다고 말하지 않기
- 완전한 장시간 운용 안정성을 확보했다고 말하지 않기
```

추천 표현은 다음과 같다.

```text
- RF 탐지 계층 prototype
- 2.4GHz RF activity 기반 드론 의심 신호 탐지
- CNN 기반 후보 검증
- 2채널 위상차 기반 AoA 추정
- experimental coarse range indication
- 실시간 dashboard 기반 검증 시스템
```

---

## 12. Final Positioning

이 프로젝트는 상용 대드론 장비 전체를 만든 것이 아니라, RF 기반 탐지 계층의 핵심 기능을 저비용 SDR과 소프트웨어 신호처리 pipeline으로 구현한 졸업작품이다.

핵심 가치는 다음과 같다.

```text
1. 전자공학 기반 RF 신호처리 경험
2. 실제 SDR 장비 운용 경험
3. 머신러닝을 신호처리 pipeline에 결합한 경험
4. 실험 데이터를 기반으로 threshold와 gate를 조정한 경험
5. 실시간 runtime dashboard를 구현한 경험
6. 한계와 future work를 명확히 구분한 경험
```

따라서 이 경험은 RF, 임베디드, 신호처리, 머신러닝 응용, 스마트팩토리 센싱, 산업용 모니터링 시스템 직무에서 “실제 데이터를 받아 시스템으로 연결해본 경험”으로 설명할 수 있다.
