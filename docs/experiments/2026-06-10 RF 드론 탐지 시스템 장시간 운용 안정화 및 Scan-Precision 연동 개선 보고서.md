# 2026-06-10 RF 드론 탐지 시스템 장시간 운용 안정화 및 Scan-Precision 연동 개선 보고서

## 1. 작업 목적

본 작업의 목적은 RF 드론 탐지 시스템의 `sf` 모드, 즉 scan 모드에서 RF 후보를 탐색한 뒤 fixed 2.450GHz precision 모드로 전환되는 흐름을 장시간 운용 환경에서도 안정적으로 사용할 수 있도록 점검하고 개선하는 것이다.

기존 구조에서는 scan 단계와 precision 단계가 서로 다른 OpenCV 창에서 실행되었다. 이로 인해 scan에서 후보가 검출되면 scan 창이 종료되고 precision 창이 새로 열리는 방식으로 동작하였다. 발표 및 시연 상황에서는 이러한 창 전환이 흐름을 끊어 보이게 만들 수 있으므로, scan과 precision을 하나의 화면에서 이어지는 것처럼 보이게 하는 개선이 필요하였다.

또한 장시간 구동 시 Pluto SDR의 온도 상승 및 내부 상태 누적으로 인해 false positive가 증가하는 현상이 관찰되었다. 따라서 본 작업에서는 하드웨어 냉각 조건, 장시간 구동 안정성, precision 단계에서의 자동 scan 복귀 정책을 함께 검토하였다.

---

## 2. 장시간 구동 안정성 관찰

초기에는 Pluto SDR에 냉각을 충분히 제공하면 false positive가 크게 줄어드는 경향이 있었다. 특히 에어컨을 직접적으로 맞추는 조건에서는 감쇠기를 제거한 상태에서도 scan 오탐이 줄어드는 것을 확인하였다. 이는 감쇠기 삽입 여부뿐 아니라 SDR의 온도 안정성이 RF 수신 품질과 CNN 입력 분포에 영향을 줄 수 있음을 의미한다.

그러나 냉각을 충분히 하더라도 장시간 구동을 지속하면 다시 오탐이 증가하는 현상이 발생하였다. 이는 단순한 순간 온도 상승 문제라기보다, 장시간 수신 과정에서 SDR front-end 상태, LO/IQ 안정성, spur, thermal drift, 내부 버퍼 상태 등이 누적적으로 변할 가능성을 시사한다.

따라서 냉각은 필수적인 안정화 조건이지만, 장시간 운용 안정성을 완전히 보장하지는 못한다. 최종 운용에서는 냉각과 함께 소프트웨어 차원의 auto-return, handoff 조건 조정, watchdog 구조가 필요하다.

---

## 3. 현재 `sf` 모드 동작 구조

현재 `sf` 모드는 다음과 같은 흐름으로 동작한다.

```text
SCAN
→ 여러 중심 주파수 sweep
→ RawNoiseGate로 RF activity 후보 탐색
→ candidate_top_k 후보 선정
→ 후보 주파수별 CNN Top5 vote 수행
→ 같은 주파수 즉시 recheck 수행
→ 통과 시 fixed 2.450GHz precision 모드 진입
→ precision에서 AoA/sector 추적
→ AoA 또는 coherence 유실이 누적되면 scan으로 자동 복귀
```

scan 단계에서는 단순히 RF 세기가 큰 후보만을 사용하는 것이 아니라, 후보 블록에 대해 CNN 기반 Drone/NotDrone 판단을 수행한다. 또한 1차 검증 이후 같은 주파수에서 즉시 recheck를 수행하여 일시적인 오탐을 줄이도록 구성하였다.

precision 단계에서는 fixed 2.450GHz에서 raw pass, CNN drone vote, AoA candidate, sector consensus, coherence 등을 이용하여 현재 방향 및 sector 상태를 추적한다.

---

## 4. Precision 단계 자동 복귀 확인

precision 모드에서 조종기 신호가 사라졌을 때, 시스템은 즉시 scan으로 복귀하지 않고 일정 기간 hold 상태를 유지한다. 이때 `trusted` 상태가 유지되지 않고 `hold_no_valid_aoa` 또는 `hold_no_consensus` 상태가 연속적으로 누적되면 `[AUTO-RETURN]`이 발생하며 scan 모드로 복귀한다.

실험 중 조종기를 끈 이후에도 일부 구간에서 일시적으로 `trusted` 상태가 나타나는 현상이 있었다. 이 때문에 auto-return이 즉시 발생하지 않고 어느 정도 버티는 모습이 관찰되었다. 이는 잔류 신호, 주변 간섭, 또는 SDR 상태 변화에 의해 일시적인 sector consensus가 생성되었기 때문으로 볼 수 있다.

그러나 최종적으로 AoA/coherence lost가 연속적으로 누적되면 scan으로 복귀하는 구조는 정상적으로 동작하였다. 따라서 현재 auto-return은 false handoff 이후에도 precision 단계에서 실제 추적이 유지되지 않을 경우 다시 scan으로 돌아오게 하는 안전장치 역할을 한다.

---

## 5. Same-window 운용 개선

기존에는 scan 창과 precision 창이 분리되어 있어 화면 전환이 발생하였다. 이를 개선하기 위해 scan runtime과 fixed precision dashboard가 동일한 OpenCV window name을 사용하도록 조정하였다.

공통 창 이름은 다음과 같이 설정하였다.

```text
RF Drone Detection Runtime
```

적용된 구조는 다음과 같다.

```text
src/runtime/scan_activity_cnn_runtime.py
- scan renderer window_name을 공통 이름으로 변경
- sf handoff 시 scan renderer.close()를 호출하지 않도록 조건부 처리
- fixed precision 진입 시 창 유지 관련 환경변수 전달

scripts/experimental/live_aoa_sector_dashboard.py
- fixed precision renderer window_name을 공통 이름으로 변경
- auto-return 시 renderer.close()를 skip하도록 guard 추가
- SystemExit(20) 기반 auto-return 상황을 감지하여 창 유지

src/viewer/opencv_renderer.py
- scan OpenCV 창의 위치 및 크기 조정 로직 적용
```

이 구조는 내부적으로 scan과 precision이 완전히 하나의 while-loop 안에서 동작하는 구조는 아니다. 하지만 OpenCV window name과 close 조건을 맞춤으로써, 사용자가 보기에는 scan과 precision이 하나의 창에서 이어지는 것처럼 보이도록 개선하였다.

---

## 6. OpenCV 창 위치 및 크기 개선

scan 창이 화면 구석이나 아래쪽에 열려 일부 내용이 잘리는 문제가 있었다. 이를 해결하기 위해 OpenCV 창을 화면 중앙에 배치하고, 표시 frame이 화면보다 클 경우 축소하여 전체 화면이 보이도록 하는 방향을 적용하였다.

특히 scan 창은 `src/viewer/opencv_renderer.py`에서 생성되며, `cv2.imshow()` 전에 frame 크기를 조정하고 `cv2.moveWindow()`를 이용해 창 위치를 조정하는 방식으로 개선하였다.

이 작업은 탐지 알고리즘이나 CNN 판정 정책에는 영향을 주지 않고, 시연 편의성과 화면 가독성을 개선하기 위한 UI 수정이다.

---

## 7. 현재 최종 운용 정책

현재 권장 운용 정책은 다음과 같다.

```text
[s]
Clean scan observe only
- scan 후보 탐색 및 CNN 검증 상태 확인
- precision 진입 없음

[sf]
Scan-fixed handoff mode
- scan에서 후보 탐색
- CNN Top5 vote 및 recheck 통과 시 fixed 2.450GHz precision 진입
- precision에서 AoA/coherence lost 누적 시 scan 복귀
- scan과 precision을 동일 OpenCV 창 이름으로 운용

[f]
Fixed 2.450GHz precision only
- scan 없이 정밀 AoA/sector dashboard 직접 실행
- scan 복귀 없음
```

---

## 8. 향후 개선 방향

장시간 운용 시 false positive가 다시 증가할 수 있으므로, 다음 개선이 필요하다.

1. SDR 냉각 유지
   Pluto SDR에 직접적인 냉각을 제공하여 RF front-end 안정성을 확보한다.

2. calibration 재수행
   냉각 조건이나 감쇠기 삽입 여부가 바뀌면 noise calibration 및 phase/gain calibration을 다시 수행한다.

3. 장시간 watchdog 도입
   일정 시간 이상 scan을 지속하면 receiver를 close/reopen하고, warm-up block을 버린 뒤 scan을 재개하는 구조를 검토한다.

4. `sf` handoff 조건 강화 검토
   발표용 자동전환 모드에서는 3/5 + recheck 3/5보다 4/5 + recheck 4/5와 같은 보수적인 조건을 고려할 수 있다.

5. 완전한 단일 상태머신 구조 검토
   현재는 같은 창 이름을 이용해 scan과 precision을 이어 보이게 만든 구조이다. 향후에는 scan runtime 내부에 precision branch를 직접 포함하여 `SCAN → TRACK_AOA → HOLD → SCAN`을 하나의 while-loop 안에서 처리하는 구조로 발전시킬 수 있다.

---

## 9. 결론

오늘 작업을 통해 RF 드론 탐지 시스템의 `sf` 모드는 scan과 precision을 더 자연스럽게 연결할 수 있게 되었다. 특히 precision 단계에서 신호 또는 AoA consensus가 사라졌을 때 scan으로 자동 복귀하는 흐름을 확인하였고, OpenCV 창 이름과 close 조건을 조정하여 scan과 precision이 하나의 화면에서 이어지는 형태로 개선하였다.

장시간 구동 시 SDR 상태 변화에 따른 false positive 가능성은 여전히 존재한다. 따라서 최종 시연에서는 냉각을 필수 조건으로 두고, auto-return 구조를 유지하며, 향후 watchdog과 handoff 조건 강화 정책을 추가하는 것이 바람직하다.

핵심 결론은 다음과 같다.

```text
냉각은 필수 안정화 조건이다.
하지만 장시간 안정성은 냉각만으로 충분하지 않다.
sf 모드는 scan → precision → lost 시 scan 복귀 구조로 안정성을 확보한다.
same-window 운용으로 발표 시 화면 전환이 자연스러워졌다.
향후 watchdog과 handoff 조건 강화가 필요하다.
```
