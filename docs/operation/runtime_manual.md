# Runtime Operation Manual

이 문서는 SDR 기반 RF 드론 탐지 시스템의 실행 방법, CLI 구조, calibration 절차, 현장 운용 순서를 정리한 문서이다.

## 1. Runtime Entry Point

```bash
PYTHONPATH=. python -m src.runtime.cli
```

## 2. CLI Modes

| Key | Mode | Purpose |
|---|---|---|
| c | status | calibration / pipeline 상태 확인 |
| n | noise calibration | gain-wise noise profile 생성 |
| p | phase/gain calibration | RX0/RX1 phase offset 및 gain profile 생성 |
| s | clean scan | 후보 주파수 탐색 |
| sf | scan-fixed handoff | scan 후 precision tracking 진입 |
| f | fixed precision | fixed 2.450GHz AoA/Sector dashboard 실행 |
| v | UI demo | Pluto+ 없이 OpenCV UI demo 실행 |
| t | terminal loop | terminal 기반 legacy runtime |
| d | dataset capture | CNN 학습용 데이터 수집 |
| r | RF4 inference | single block inference |
| q | quit | 종료 |

## 3. Recommended Field Procedure

```text
1. Pluto+ SDR 냉각 및 연결 확인
2. c로 calibration 상태 확인
3. 환경이 바뀌었으면 n 수행
4. 안테나/케이블/포트/gain 조건이 바뀌었으면 p 수행
5. s로 clean scan observe 확인
6. sf로 scan-to-precision runtime 실행
7. q 또는 ESC로 종료
```

## 4. Calibration Notes

Noise calibration과 phase/gain calibration은 다음 조건이 바뀌면 다시 수행하는 것이 좋다.

```text
- 실험 장소
- SDR 냉각 조건
- 감쇠기 삽입 여부
- 안테나 위치
- 케이블 / 포트 변경
- gain 변경
```

## 5. Runtime Stability Notes

장시간 운용 시 Pluto+ SDR의 thermal drift, spur, RF front-end 상태 변화로 false positive가 증가할 수 있다.

따라서 시연 전에는 다음을 확인한다.

```text
- SDR 냉각 유지
- calibration 상태 확인
- warm-up block discard
- clean scan observe
- auto-return 정책 유지
```

