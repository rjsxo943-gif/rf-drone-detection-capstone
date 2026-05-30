# 2026-06-01 RF 드론 탐지 실험 계획 보고서

## 1. 실험 목적

내일 실험의 목적은 RF 드론 탐지 시스템의 실시간 운용 조건을 더 구체화하는 것이다.
특히 거리 변화에 따라 적절한 수신 gain을 어떻게 설정할지 확인하고, 실외에서 얻은 거리-gain 기준이 실내 시연 장소에서도 적용 가능한지 검증한다.

최종 목표는 다음과 같다.

* 실외 환경에서 거리별 적정 gain table 도출
* 거리 증가에 따른 gain 보정 관계식 또는 경험적 기준 확보
* 실내 시연 장소에서 실외 기준의 일반화 가능성 확인
* 실내에서 성능 저하가 발생할 경우 domain adaptation용 데이터 수집 방향 결정

---

## 2. 실험 우선순위

내일 실험은 다음 순서로 진행한다.

1. 실외 기존 실험 장소에서 거리별 적정 gain 측정
2. 거리-gain 기준 table 작성
3. 실외 기준을 실내 시연 장소에 그대로 적용하여 일반화 테스트
4. 실내에서 실패하거나 오탐이 많을 경우 실내 domain adaptation 데이터 수집
5. 필요 시 gain-aware threshold 또는 temporal voting 파라미터 재조정

---

## 3. 1단계: 실외 거리별 적정 Gain 측정

### 3.1 목적

실외 환경에서 거리 변화에 따라 드론 RF 신호가 안정적으로 탐지되는 gain 값을 찾는다.
이때 단순히 드론이 잘 잡히는 gain만 찾는 것이 아니라, NonDrone 오탐이 과도하게 증가하지 않는 gain을 찾는 것이 중요하다.

### 3.2 추천 거리 조건

다음 거리 조건을 우선 측정한다.

|  거리 | 목적             |
| --: | -------------- |
| 1 m | 기준 거리          |
| 2 m | 근거리 확장         |
| 3 m | 실제 시연 가능 거리 후보 |
| 5 m | 탐지 한계 확인용      |

상황이 가능하면 4 m 또는 6 m 조건도 추가한다.

### 3.3 추천 Gain 후보

각 거리에서 다음 gain 후보를 비교한다.

| Gain | 역할         |
| ---: | ---------- |
|  g20 | 낮은 gain 기준 |
|  g25 | 안정 기준 후보   |
|  g30 | 현재 주요 후보   |
|  g35 | 먼 거리 보정 후보 |

단, g30 이상에서는 NonDrone 오탐 가능성이 증가할 수 있으므로 background 상태도 함께 확인한다.

### 3.4 기록할 지표

각 거리/gain 조건에서 다음 값을 기록한다.

* raw_peak
* raw_abs_p99
* frame_power_p99
* signal_ratio
* clip_ratio
* CNN Drone probability
* raw CNN decision
* gain-aware decision
* temporal voting final decision
* Candidate / Confirmed 상태
* 드론 있음/없음 조건에서의 오탐 여부

---

## 4. 2단계: 거리-Gain 기준 Table 작성

실외 측정 결과를 바탕으로 거리별 권장 gain table을 만든다.

예상 형식은 다음과 같다.

|  거리 | 권장 Gain | 판단 근거           |
| --: | ------: | --------------- |
| 1 m |     g25 | 기준 신호 안정, 오탐 낮음 |
| 2 m | g25~g30 | signal_ratio 유지 |
| 3 m |     g30 | 탐지 안정성 확보       |
| 5 m | g30~g35 | 먼 거리 보정 필요      |

가능하면 다음과 같은 경험적 관계식도 검토한다.

```text
G(d) = G0 + k log10(d / d0)
```

여기서 `G(d)`는 거리 d에서의 권장 gain, `G0`는 기준 거리 1 m에서의 gain, `k`는 실험적으로 조정되는 보정 계수이다.

이 관계식은 이론적 정답이 아니라, 실험 기반의 empirical gain rule로 사용한다.

---

## 5. 3단계: 실내 시연 장소 일반화 테스트

### 5.1 목적

실외에서 얻은 거리별 gain 기준이 실내 시연 장소에서도 동작하는지 확인한다.

실내는 반사, 벽, 사람, WiFi, Bluetooth, multipath 영향이 있기 때문에 실외보다 RF 환경이 복잡할 수 있다.

### 5.2 테스트 방법

실외에서 얻은 거리별 권장 gain을 그대로 실내에 적용한다.

예시:

| 실외 기준         | 실내 적용               |
| ------------- | ------------------- |
| 1 m → g25     | 실내 1 m에서 g25 적용     |
| 3 m → g30     | 실내 3 m에서 g30 적용     |
| 5 m → g30~g35 | 실내 5 m에서 동일 gain 적용 |

### 5.3 확인할 항목

실내 테스트에서는 다음을 확인한다.

* 드론이 있을 때 Candidate 상태가 빠르게 뜨는가
* 드론이 있을 때 Confirmed Drone까지 올라가는가
* 드론이 없을 때 False Drone이 발생하는가
* WiFi / Bluetooth 환경에서 오탐이 증가하는가
* g30 조건에서 background가 Drone-like로 뜨는가
* temporal voting이 오탐과 미탐을 안정화하는가

---

## 6. 4단계: 실내 성능 저하 시 Domain Adaptation 데이터 수집

실내에서 실외 기준이 잘 적용되지 않을 경우, 실내 domain adaptation을 위한 데이터를 수집한다.

### 6.1 수집할 NonDrone 데이터

* indoor background
* indoor WiFi
* indoor Bluetooth
* indoor mixed WiFi + Bluetooth
* 사람 이동이 있는 실내 background

### 6.2 수집할 Drone 데이터

* drone motor off
* drone motor on no stick
* drone linked 상태
* 거리 1 m
* 거리 3 m
* 거리 5 m

### 6.3 Gain 조건

실외 기준 gain을 중심으로 주변 gain도 함께 수집한다.

예시:

| 기준 Gain | 추가 수집 Gain    |
| ------: | ------------- |
|     g25 | g20, g25, g30 |
|     g30 | g25, g30, g35 |

이렇게 수집하면 실내 환경에서 gain 변화에 따른 CNN confidence 분포를 더 잘 학습할 수 있다.

---

## 7. 내일 사용할 기본 Decision 설정

현재까지의 외부 검증 결과를 바탕으로, 실시간 탐지 기본 설정은 다음을 사용한다.

| 항목                |      값 |
| ----------------- | -----: |
| decision mode     | hybrid |
| g25 threshold     |   0.35 |
| g30 threshold     |   0.80 |
| default threshold |   0.50 |
| temporal window   |      5 |
| candidate vote k  |      2 |
| confirmed vote k  |      3 |

실시간 판단 기준은 다음과 같다.

* 최근 5 block 중 2개 이상 Drone이면 Drone-like Candidate
* 최근 5 block 중 3개 이상 Drone이면 Confirmed Drone

---

## 8. 내일 실험 성공 기준

내일 실험의 성공 기준은 다음과 같다.

1. 실외에서 거리별 권장 gain table을 작성할 수 있다.
2. 실외 기준 gain을 적용했을 때 드론 탐지가 안정적으로 동작한다.
3. 실내 시연 장소에서 실외 기준 gain이 어느 정도 통하는지 확인한다.
4. 실내에서 실패할 경우, 실패 원인을 기록하고 domain adaptation용 데이터 수집 조건을 정한다.
5. 실시간 viewer에서 Candidate / Confirmed 상태가 실제 드론 유무와 일치하는지 확인한다.

---

## 9. 예상 결론 시나리오

### 시나리오 A: 실외 기준이 실내에서도 잘 동작

이 경우 실외에서 도출한 거리별 gain rule이 실내 시연 환경에도 어느 정도 일반화된다고 볼 수 있다.
보고서에는 “실외 기준 기반 gain preset이 실내에서도 동작 가능함을 확인하였다”고 정리한다.

### 시나리오 B: 실내에서 오탐이 증가

이 경우 실내 multipath, WiFi, Bluetooth, background RF 환경 차이로 인해 NonDrone이 Drone-like로 오인되는 것으로 해석한다.
이때는 실내 NonDrone 데이터를 추가 수집하고, threshold 또는 fine-tuning을 적용한다.

### 시나리오 C: 실내에서 드론 미탐이 증가

이 경우 실내 반사 또는 안테나 방향 문제로 드론 RF 패턴이 학습 데이터와 달라졌을 가능성이 있다.
거리별 gain을 재조정하고, 실내 Drone 데이터를 추가 수집한다.

---

## 10. 내일 최종 산출물

내일 실험 후 남겨야 할 산출물은 다음과 같다.

* 거리별 gain table
* 거리별 signal_ratio / raw_abs_p99 / CNN probability 기록
* 실외 기준 gain의 실내 적용 결과
* 실내 실패 조건 목록
* 필요 시 실내 domain adaptation 데이터셋
* 실험 결과 요약 보고서
* live viewer 실행 명령어 기록
