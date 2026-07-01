# Future Work: RF-centered Hybrid Drone Detection

본 문서는 현재 졸업작품으로 구현한 RF 기반 드론 탐지 시스템을 향후 RF + 비RF 센서융합 구조로 확장하는 방향을 정리한다.

현재 시스템은 2.4GHz 대역의 RF activity를 기반으로 드론 관련 신호를 탐지하고, 2채널 IQ 위상차를 이용해 신호 도래 방향을 추정하는 구조이다.

향후에는 RF 업링크 / 다운링크 탐지 범위를 확장하고, RF로 탐지하기 어려운 드론을 보완하기 위해 음향, 카메라, 레이더, LiDAR 등의 비RF 센서를 함께 사용하는 하이브리드 탐지 시스템으로 발전시킬 수 있다.

---

## 1. Why RF Remains the Core

RF 신호 기반 탐지는 단순히 드론의 물리적 형상을 보는 것이 아니라, 조종기와 드론 사이의 통신 행위 자체를 검출한다는 점에서 차별성이 있다.

특히 본 프로젝트에서 사용한 AoA 기반 방향 추정은 신호가 어느 방향에서 들어오는지를 분석할 수 있기 때문에, 조종기 신호가 탐지되는 경우 조종자 방향 추정에 활용될 수 있다.

따라서 향후 확장 구조에서도 RF는 시스템의 중심축으로 유지하는 것이 적절하다.

```text
RF sensing core
→ communication activity detection
→ CNN-based signal verification
→ AoA-based direction estimation
→ operator / drone direction inference
```

---

## 2. Uplink and Downlink Expansion

현재 시스템은 주로 컨트롤러에서 드론으로 송신되는 2.4GHz 업링크 신호를 중심으로 검증하였다.

향후에는 드론에서 송신되는 다운링크 신호까지 학습 데이터에 포함하여 RF 기반 탐지 범위를 확장할 수 있다.

| RF Signal Type | Source | Possible Use |
|---|---|---|
| Uplink | Controller → Drone | 조종자 방향 추정 |
| Downlink | Drone → Controller | 드론 방향 추정 |
| Telemetry-like signal | Drone / controller | 상태 정보 기반 탐지 |
| Video transmission signal | Drone → controller/display | 드론 자체 RF activity 탐지 |

업링크 신호는 조종자 방향 추정에 유리하고, 다운링크 신호는 드론 자체 방향 추정에 유리하다.

---

## 3. Limitation of RF-only Detection

RF 기반 탐지는 통신 신호가 존재할 때 강점을 가진다. 하지만 다음과 같은 경우에는 RF 단독 탐지가 어려워질 수 있다.

```text
- 자율비행 드론
- 통신 신호가 약한 드론
- 순간적으로 RF activity가 줄어드는 경우
- 주파수 hopping 또는 비정형 신호 사용
- RF 환경 잡음이 매우 강한 장소
- 무선침묵형 또는 사전 경로 기반 무인기
```

따라서 실제 운용 시스템에서는 RF 단독 구조보다, 비RF 센서를 함께 사용하는 병렬 탐지 구조가 더 안정적일 수 있다.

---

## 4. Non-RF Sensor Complement

비RF 센서는 RF가 놓칠 수 있는 상황을 보완하는 역할로 사용할 수 있다.

| Sensor | Main Role | Strength | Limitation |
|---|---|---|---|
| Acoustic | 프로펠러 / 모터 소리 탐지 | 저비용, 근거리 탐지 가능 | 도심 소음에 취약 |
| Camera | 드론 형상 식별 | 직관적 객체 확인 | 야간, 안개, 시야 가림에 취약 |
| Radar | 거리 / 속도 / 이동 궤적 추정 | 악천후와 야간에 강함 | 비용과 시스템 복잡도 증가 |
| LiDAR | 근거리 정밀 위치 / 형상 추정 | 정밀한 공간 정보 | 원거리 및 악천후 한계 |

비RF 센서는 RF를 대체하는 것이 아니라, RF로 탐지되지 않는 영역을 보완하는 방향이 적절하다.

---

## 5. Parallel First-stage Detection

향후 시스템은 RF 탐지 후 비RF를 사용하는 순차 구조보다, RF와 비RF 센서를 1차 탐지 단계에서 병렬 운용하는 구조가 더 적합하다.

```text
                ┌─ RF uplink detection
Input stream ───┼─ RF downlink detection
                ├─ Acoustic candidate detection
                ├─ Camera candidate detection
                └─ Radar / LiDAR candidate detection

→ candidate fusion
→ sensor-specific verification
→ final drone decision
```

이 구조는 통신 신호가 존재하는 드론은 RF로 빠르게 탐지하고, RF 신호가 약하거나 없는 드론은 비RF 센서로 후보를 생성할 수 있다는 장점이 있다.

---

## 6. Hierarchical Fusion Concept

최종 판단은 단일 센서 결과만으로 내리지 않고, 각 센서의 후보와 신뢰도를 종합하는 계층적 구조로 확장할 수 있다.

```text
Stage 1. Candidate Detection
- RF uplink candidate
- RF downlink candidate
- acoustic candidate
- vision candidate
- radar / LiDAR candidate

Stage 2. Sensor-specific Verification
- CNN-based RF signal verification
- AoA direction estimation
- visual object detection
- acoustic pattern classification
- radar velocity / range estimation

Stage 3. Fusion Decision
- drone probability
- direction
- estimated position
- threat level
- tracking state
```

---

## 7. Possible Final Architecture

```text
RF Receiver
  ├─ Uplink CNN / AoA branch
  └─ Downlink CNN / AoA branch

Non-RF Sensors
  ├─ Camera detector
  ├─ Acoustic detector
  ├─ Radar tracker
  └─ LiDAR localizer

Fusion Layer
  ├─ candidate association
  ├─ confidence weighting
  ├─ direction / position estimation
  └─ final drone decision

Runtime UI
  ├─ RF activity map
  ├─ direction / sector display
  ├─ sensor confidence
  └─ tracking state
```

---

## 8. Relation to Current Project

현재 졸업작품은 위 구조 전체를 구현한 것이 아니라, 그중 RF sensing core에 해당하는 부분을 구현한 프로토타입이다.

현재 구현된 범위는 다음과 같다.

```text
- Pluto+ SDR 기반 2채널 IQ 수신
- 2.4GHz RF activity 탐지
- RawNoiseGate 기반 후보 제한
- STFT spectrogram 기반 CNN 검증
- RX0/RX1 phase difference 기반 AoA 추정
- coherence 기반 신뢰도 검증
- sector-level 방향 안정화
- experimental coarse range indication
- OpenCV runtime dashboard
```

즉, 향후 하이브리드 탐지 시스템에서 RF branch의 핵심 기반을 구현한 경험으로 정리할 수 있다.

---

## 9. Long-term Direction

장기적으로는 현재 시스템을 다음 방향으로 확장할 수 있다.

```text
1. RF 업링크 데이터셋 확장
2. RF 다운링크 데이터셋 추가
3. 다양한 드론 모델 / 거리 / 각도 / gain 조건 검증
4. 장시간 운용 안정성 개선
5. embedded deployment 최적화
6. 비RF 센서 후보 탐지 branch 추가
7. RF + 비RF fusion decision layer 구현
```

최종 목표는 RF를 중심축으로 유지하되, RF로 탐지되지 않는 대상을 비RF 센서가 보완하는 계층적 드론 탐지 시스템이다.

---

## 10. Summary

본 프로젝트의 후속 연구 방향은 RF 기반 탐지 시스템을 단순 주파수 탐색기에서 끝내는 것이 아니라, 드론 통신 신호 탐지와 방향 추정을 담당하는 RF sensing core로 발전시키는 것이다.

RF 업링크는 조종자 방향 추정에, RF 다운링크는 드론 방향 추정에 활용할 수 있다. 여기에 음향, 카메라, 레이더, LiDAR 등의 비RF 센서를 병렬적으로 결합하면 다양한 운용 방식의 드론에 대응하는 하이브리드 탐지 시스템으로 확장할 수 있다.
