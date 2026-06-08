# 2026-06-08 RF 드론 탐지 시스템 SCAN + PRECISION OpenCV 통합 UI 구현 보고서

## 1. 작업 배경

기존 RF 드론 탐지 시스템은 CLI 기반 scan/runtime pipeline과 OpenCV 기반 AoA sector dashboard가 분리되어 있었다. CLI pipeline은 RawNoiseGate 기반 후보 주파수 탐색, CNN 기반 정밀 판정, AoA 및 sector 추정을 수행할 수 있었지만, 최종 시연 관점에서는 scan mode와 precision mode의 상태 전환이 직관적으로 보이지 않는 한계가 있었다.

특히 기존 OpenCV dashboard는 precision mode, 즉 이미 선택된 주파수에서 CNN/AoA/sector/range 결과를 보여주는 데 적합했다. 그러나 시스템이 실제로는 먼저 넓은 주파수 범위를 scan하고, 후보 주파수가 발견된 뒤에만 precision 분석으로 넘어간다는 구조가 화면에 드러나지 않았다.

따라서 이번 작업의 목적은 기존 precision dashboard의 부채꼴 AoA/sector/range 시각화는 유지하면서, 왼쪽에 얇은 SCAN rail을 추가하여 주파수 sweep과 candidate handoff 과정을 시각적으로 표현하는 것이다.

## 2. 구현 목표

이번 구현의 목표는 다음과 같다.

```text
1. 기존 부채꼴 precision dashboard 유지
2. 왼쪽에 SCAN rail 추가
3. SCAN 중에는 주파수 sweep marker 이동
4. 후보 주파수 발견 시 marker lock
5. SCAN mode 비활성화와 PRECISION mode 활성화를 색상으로 표시
6. CLI에서 s 입력 시 실제 OpenCV SCAN + PRECISION runtime 진입
7. CLI에서 v 입력 시 Pluto+ 없이 UI demo 실행
8. 기존 terminal scan loop는 t 명령으로 보존
```

## 3. UI 구조

최종 OpenCV 화면은 왼쪽 SCAN rail과 오른쪽 PRECISION dashboard로 구성하였다.

```text
┌───────────────┬──────────────────────────────────────────────┐
│ SCAN RAIL     │              PRECISION MODE                  │
│               │                                              │
│ 2.465 GHz     │   AoA / Sector / Range Fan Dashboard          │
│ 2.460 GHz     │                                              │
│ 2.455 GHz ◆   │   CNN Result                                 │
│ 2.450 GHz     │   AoA Angle                                  │
│ 2.445 GHz     │   Locked Sector                              │
│               │   Raw P99 / Coherence / Range Class          │
└───────────────┴──────────────────────────────────────────────┘
```

SCAN rail은 전체 화면의 일부만 차지하며, precision dashboard가 화면의 대부분을 유지하도록 설계하였다. 이는 기존에 구현된 부채꼴 AoA/sector/range 시각화가 최종 분석 결과의 핵심이기 때문이다.

## 4. SCAN rail 동작

SCAN rail은 `configs/scan.yaml`의 scan 범위를 읽어 주파수 목록을 자동 생성한다.

```yaml
scan:
  start_freq: 2435000000
  stop_freq: 2465000000
  step_freq: 5000000
```

이를 통해 다음과 같은 주파수 리스트가 생성된다.

```text
2.435 GHz
2.440 GHz
2.445 GHz
2.450 GHz
2.455 GHz
2.460 GHz
2.465 GHz
```

SCAN 상태에서는 현재 sweep 중인 주파수 위치에 marker가 이동한다. 이때 SCAN rail은 초록색 테두리로 표시되어 현재 시스템이 scan mode에 있음을 나타낸다.

```text
SCAN mode
- border color : green
- marker       : current frequency 위치에서 이동
- status       : SWEEPING
```

후보 주파수가 발견되면 marker는 해당 candidate frequency 위치에 고정된다. 이후 SCAN rail은 회색으로 비활성화되고, 오른쪽 PRECISION dashboard가 노란색으로 활성화된다.

```text
PRECISION mode
- SCAN rail border : gray
- marker           : locked frequency 위치에 고정
- precision border : yellow
- status           : HANDOFF
```

이를 통해 시스템이 scan mode에서 precision mode로 넘어갔다는 점을 직관적으로 확인할 수 있다.

## 5. CLI 구조 변경

기존 CLI의 `s` 명령은 terminal 기반 scan/runtime pipeline을 실행하는 역할이었다. 이번 작업에서는 최종 시연을 위해 `s`를 실제 OpenCV SCAN + PRECISION runtime으로 변경하고, 기존 terminal loop는 `t` 명령으로 보존하였다.

현재 CLI 명령 구조는 다음과 같다.

```text
[c] status        : calibration / pipeline 현재 상태창
[n] noise         : gain-wise noise calibration
[p] phase         : gain-wise phase/gain calibration
[s] start         : 실제 Pluto+ OpenCV SCAN + PRECISION runtime 구동
[v] view/demo     : Pluto 없이 OpenCV UI demo 구동
[t] terminal-loop : 기존 terminal scan/runtime pipeline 구동
[d] dataset       : CNN dataset capture
[r] rf4           : RF4 single block inference
[q] quit/shutdown : receiver close 후 종료
```

이 구조를 통해 실제 장비가 있는 상황과 없는 상황을 분리하였다.

```text
s : 실제 Pluto+ 기반 최종 runtime
v : 장비 없이 UI 확인용 demo
t : 기존 terminal 로그 기반 디버깅
```

## 6. 추가 및 수정 파일

이번 작업에서 추가한 주요 파일은 다음과 같다.

```text
src/viewer/scan_rail.py
scripts/experimental/test_scan_precision_rail_demo.py
src/runtime/opencv_scan_precision_runtime.py
```

수정한 주요 파일은 다음과 같다.

```text
src/runtime/cli.py
scripts/experimental/live_aoa_sector_dashboard.py
```

`src/viewer/scan_rail.py`는 왼쪽 SCAN rail 렌더링을 담당한다. 이 파일은 주파수 목록, current frequency, locked frequency, mode 상태를 받아 OpenCV canvas 왼쪽 영역에 scan 상태를 그린다.

`scripts/experimental/test_scan_precision_rail_demo.py`는 Pluto+ 없이 SCAN rail과 PRECISION dashboard 전환을 확인하기 위한 UI demo script이다.

`src/runtime/opencv_scan_precision_runtime.py`는 실제 Pluto+ 기반 scan/precision runtime과 OpenCV dashboard를 연결하기 위한 runtime 파일이다.

## 7. 현재 검증 결과

현재 Pluto+가 없는 환경에서 다음 항목을 검증하였다.

```text
1. CLI에서 v 입력 시 OpenCV UI demo 실행
2. SCAN rail 주파수 목록 자동 표시
3. SCAN 상태에서 marker 이동
4. PRECISION 상태에서 marker 고정
5. SCAN rail 초록색 활성화 표시
6. PRECISION 진입 후 SCAN rail 회색 비활성화 표시
7. 오른쪽 precision dashboard 유지
8. OpenCV 창에서 q 또는 ESC 입력 시 CLI 복귀
```

또한 SCAN rail의 하단 `CUR` 표시는 SCAN 중에는 현재 sweep 주파수를 따라 움직이고, PRECISION 진입 후에는 locked frequency에서 멈추도록 수정하였다. 이를 통해 사용자가 현재 시스템이 어느 주파수를 scan 중인지, 또는 어느 주파수에 lock 되었는지를 더 명확하게 확인할 수 있게 되었다.

## 8. 시스템 구조적 의의

이번 작업은 단순한 UI 수정이 아니라, 시스템의 실제 처리 구조를 화면에 반영했다는 점에서 의미가 있다.

기존 precision dashboard는 후보 주파수가 이미 선택된 이후의 분석 결과를 보여주는 데 초점이 있었다. 반면 이번 통합 UI는 다음 흐름을 하나의 OpenCV 화면 안에서 보여준다.

```text
주파수 sweep
→ 후보 주파수 탐색
→ candidate frequency lock
→ precision mode 진입
→ CNN 판정
→ AoA / Sector / Range 표시
```

즉, 최종 시연에서 사용자는 시스템이 단순히 한 주파수만 보는 것이 아니라, scan front-end를 통해 후보 주파수를 찾고 해당 주파수에서만 정밀 분석을 수행한다는 점을 확인할 수 있다.

또한 모든 주파수에서 CNN/AoA를 수행하는 구조가 아니라, RawNoiseGate를 통과한 후보 주파수에 대해서만 precision 분석을 수행한다는 pipeline 구조도 시각적으로 설명할 수 있게 되었다.

## 9. 남은 작업

현재 UI demo와 CLI 연결은 검증되었지만, 실제 Pluto+ 기반 `s` runtime은 장비 연결 후 추가 검증이 필요하다.

다음 작업은 다음과 같다.

```text
1. Pluto+ 연결 후 s 모드 실행 확인
2. configs/scan.yaml 범위대로 실제 주파수 sweep 되는지 확인
3. 후보 신호 발생 시 SCAN rail marker가 locked frequency에 고정되는지 확인
4. PrecisionAnalyzer 결과가 오른쪽 dashboard에 정상 표시되는지 확인
5. CNN / AoA / Sector / Range 결과가 기존 dashboard와 동일하게 표시되는지 확인
6. q 또는 ESC로 OpenCV 창 종료 후 CLI로 정상 복귀되는지 확인
7. 실제 시연용 scan 범위와 gain 조건 최종 결정
```

## 10. 결론

2026-06-08 작업을 통해 RF 드론 탐지 시스템의 최종 시연용 UI 구조가 크게 개선되었다. 기존 부채꼴 precision dashboard를 유지하면서 왼쪽에 SCAN rail을 추가하여 scan mode와 precision mode의 상태 전환을 직관적으로 표현할 수 있게 되었다.

또한 CLI 구조를 `s`, `v`, `t`로 분리함으로써 실제 장비 기반 runtime, 장비 없는 UI demo, 기존 terminal debug loop를 명확히 구분하였다.

현재까지는 Pluto+ 없는 환경에서 UI demo와 CLI 연결이 정상 동작함을 확인하였다. 다음 단계에서는 실제 Pluto+를 연결한 상태에서 `s` 모드를 실행하여 RawNoiseGate scan, candidate lock, PrecisionAnalyzer, CNN/AoA/sector/range dashboard 연결을 최종 검증할 예정이다.
