# Final Validation Summary

이 문서는 졸업작품 최종 단계에서 수행한 RF 드론 탐지 시스템의 검증 결과를 요약한다.

## 1. Validation Scope

최종 검증 범위는 다음과 같다.

```text
- RF activity clean scan
- CNN Top5 vote verification
- Same-frequency immediate recheck
- Fixed 2.450GHz precision tracking
- AoA / Sector estimation
- Experimental coarse range indication
- OpenCV same-window dashboard
- Signal/AoA lost auto-return
```

## 2. Sector Capture Validation

Sector capture 검증에서는 실제 각도 라벨과 추정 sector가 저장되고, CSV replay를 통해 결과를 다시 확인할 수 있는 구조를 만들었다.

확인한 내용은 다음과 같다.

```text
1. trusted-only capture 기능이 정상 동작하였다.
2. true_angle_deg 라벨이 CSV에 저장되었다.
3. sector name을 각도 범위 기반으로 정리하여 해석이 쉬워졌다.
4. 오른쪽 방향에서는 CENTER → RIGHT 계열 sector로 자연스럽게 이동하였다.
5. 왼쪽 방향에서도 LEFT 계열 sector가 나타나는 것을 확인하였다.
6. coherence가 높은 구간을 중심으로 AoA 후보를 사용하였다.
```

## 3. Range Dashboard / CSV Replay Validation

Range dashboard는 정확한 거리 회귀 모델이 아니라, sector별 raw feature profile을 기반으로 한 실험적 coarse range indication이다.

확인한 내용은 다음과 같다.

```text
1. sector profile CSV 기반으로 WITHIN_9M / RANGE_9_TO_15M profile JSON을 생성하였다.
2. SectorRangeEstimator를 통해 sector별 feature 조합 기반 range class를 계산하였다.
3. 거리 구분이 불안정한 경우 SECTOR_ONLY로 처리하여 방향 정보는 유지하였다.
4. CSV replay dashboard를 통해 저장된 distance_m / true_angle_deg 라벨과 추정 결과를 함께 확인하였다.
5. fan dashboard에서 sector grid와 range cell이 같은 좌표계에서 표시되도록 구성하였다.
```

## 4. Scan + Precision UI Validation

OpenCV 기반 UI에서는 scan 상태와 precision 상태를 하나의 runtime 화면처럼 보여주는 구조를 검증하였다.

확인한 내용은 다음과 같다.

```text
1. CLI에서 v 입력 시 Pluto+ 없는 UI demo 실행을 확인하였다.
2. SCAN rail 주파수 목록 표시를 확인하였다.
3. SCAN 상태에서 marker 이동을 확인하였다.
4. PRECISION 상태에서 marker lock을 확인하였다.
5. 기존 precision 부채꼴 dashboard가 유지되는 것을 확인하였다.
6. OpenCV 창에서 q 또는 ESC 입력 시 CLI로 복귀하는 것을 확인하였다.
```

## 5. Scan-fixed / Same-window Validation

Scan-fixed mode에서는 scan을 통해 후보 주파수를 찾고, 후보가 검증되면 fixed precision dashboard로 진입하는 흐름을 확인하였다.

확인한 내용은 다음과 같다.

```text
1. sf 모드에서 scan → fixed 2.450GHz precision handoff 흐름을 확인하였다.
2. precision 단계에서 signal/AoA lost가 누적되면 AUTO-RETURN으로 scan 복귀하는 구조를 확인하였다.
3. scan과 precision의 OpenCV window name을 통일하였다.
4. handoff 시 scan renderer를 닫지 않도록 처리하였다.
5. auto-return 시 fixed precision renderer를 닫지 않도록 guard를 추가하였다.
6. scan 창의 위치와 크기를 조정하여 화면 잘림 문제를 완화하였다.
```

## 6. Key Result Summary

최종 시스템은 다음 정보를 실시간으로 제공할 수 있도록 구성되었다.

```text
- 탐지 여부
- 후보 주파수
- CNN Drone probability
- Top5 vote 상태
- Recheck 결과
- AoA angle
- Locked sector
- Coherence
- Raw signal strength profile
- Experimental range class
- SCAN / TRACK_AOA / HOLD runtime state
```

## 7. Known Limitations

현재 시스템의 한계는 다음과 같다.

```text
1. Range class는 정확한 거리 추정이 아니라 experimental coarse range indication이다.
2. Gain, center frequency, 안테나 배치가 바뀌면 range profile 재검증이 필요하다.
3. 장시간 운용 시 Pluto+ SDR thermal drift 및 false positive 가능성이 존재한다.
4. Same-window 구조는 사용자가 보기에는 하나의 창처럼 보이지만, 내부적으로는 완전한 단일 while-loop 상태머신은 아니다.
5. 다양한 드론 모델, 주파수, 거리, 각도 조건에 대한 데이터 일반화가 추가로 필요하다.
```

## 8. Final Interpretation

본 프로젝트는 저비용 SDR 기반으로 2.4GHz RF activity를 탐지하고, CNN 기반 후보 검증과 2채널 위상차 기반 AoA 추정을 결합하여 방향 정보를 제공하는 RF sensing prototype이다.

최종 검증을 통해 단순 RF energy detector나 단순 CNN 분류기를 넘어서, scan, verification, precision tracking, sector estimation, experimental range indication, runtime dashboard까지 연결된 통합 pipeline을 구현했음을 확인하였다.
