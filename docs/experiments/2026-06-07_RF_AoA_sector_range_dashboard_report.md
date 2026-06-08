# 2026-06-07 RF 드론 탐지 AoA Sector Range Dashboard 및 CSV Replay 구현 보고서

## 1. 개요

본 보고서는 Pluto+ SDR 기반 RF 드론 탐지 캡스톤 프로젝트에서 2026-06-07에 수행한 AoA sector dashboard 개선 및 sector별 거리 구간 분류 기능 구현 내용을 정리한 것이다.

기존 시스템은 RawNoiseGate, CNN binary classifier, temporal voting, AoA/coherence 계산, fixed-bin sector consensus를 이용해 드론 RF activity의 존재 여부와 방향 sector를 판단하는 구조였다. 오늘 작업의 목적은 기존 판단 정책을 변경하지 않고, 이미 계산된 sector 및 raw feature 결과를 이용하여 **sector별 coarse range class**를 dashboard에 표시하는 것이었다.

본 기능은 정확한 거리값을 추정하는 기능이 아니라, 실험적으로 수집된 sector profile CSV를 기반으로 다음 두 구간을 구분하는 보조 기능이다.

```text
WITHIN_9M        : 약 9m 이내
RANGE_9_TO_15M   : 약 9m 초과 ~ 15m 이내 구간
SECTOR_ONLY      : sector는 신뢰되지만 거리 구간은 불안정하므로 sector 전체 표시
```

따라서 본 기능은 최종 거리 추정기가 아니라, **AoA sector 기반 방향 표시 + sector-specific feature 기반 coarse range classification**으로 정의한다.

---

## 2. 기존 구조와 문제점

### 2.1 기존 AoA sector dashboard

기존 OpenCV dashboard는 다음 정보를 표시했다.

```text
Status
Locked Sector
Instant Sector
Angle Median
Angle Spread
Median Coherence
Raw P99
CNN Raw / Confidence / Vote
Raw Pass
Top-K
Valid AoA
Gain / Center Frequency
Phase Delta
Reason
```

또한 trusted sector가 발생하면 부채꼴 fan에서 해당 sector를 점등했다. 그러나 거리 profile을 수집했음에도 dashboard에서는 거리 구간을 시각적으로 구분하지 못했다.

### 2.2 단순 거리 추정의 한계

초기 분석에서는 `median_raw_p99` 하나만 이용하여 6m, 9m, 12m, 15m를 모두 구분하려고 했으나, sector별 분포가 다르게 나타났다. 특히 CENTER와 RIGHT 계열 sector에서는 거리 증가에 따라 p99가 단조 감소하지 않는 경우가 있었다.

따라서 다음과 같이 방향을 수정했다.

```text
정확한 거리값 추정 X
6m / 9m / 12m / 15m 세부 분류 X
sector별 feature 조합 기반 2구간 coarse classification O
```

---

## 3. 설계 원칙

오늘 작업에서 적용한 설계 원칙은 다음과 같다.

```text
1. 기존 CNN/AoA/sector 판단 정책을 변경하지 않는다.
2. dashboard는 판단 결과를 받아서 보여주는 역할만 한다.
3. 거리 구간 판단은 experimental 옵션으로 둔다.
4. feature를 과도하게 사용하지 않고 sector별 1~3개 feature만 사용한다.
5. 거리 구분이 애매하면 결과를 억지로 출력하지 않고 SECTOR_ONLY로 처리한다.
6. SDR 없이도 CSV replay로 실제 수집 데이터를 재생할 수 있게 한다.
```

이 원칙을 통해 UI 개선이 기존 탐지 정책에 영향을 주지 않도록 분리하였다.

---

## 4. 구현 내용

### 4.1 Sector range profile builder

새로 추가한 파일은 다음과 같다.

```text
scripts/experimental/build_sector_range_profile.py
```

역할은 다음과 같다.

```text
여러 sector_profile.csv 입력
→ 7-sector 결과를 5-sector로 mapping
→ condition-level median aggregation
→ sector별 candidate feature 계산
→ WITHIN_9M vs RANGE_9_TO_15M 분리 profile 생성
→ sector_range_profile.json 저장
```

row 단위 그대로 학습하면 같은 조건에서 반복 저장된 row 때문에 성능이 과대평가될 수 있다. 이를 줄이기 위해 다음 단위로 median aggregation을 수행했다.

```text
source_file
+ distance_m
+ true_angle_deg
+ phase_offset_live_delta_deg
+ locked 5-sector
```

출력 파일 예시는 다음과 같다.

```text
outputs/sector_range_profiles/gain35_cf2450000000_nearfar_profile.json
```

### 4.2 SectorRangeEstimator

새로 추가한 핵심 모듈은 다음과 같다.

```text
src/viewer/sector_range_estimator.py
```

역할은 다음과 같다.

```text
profile JSON 로드
→ runtime feature dict 정리
→ 필요한 ratio feature 자동 계산
→ sector별 score 계산
→ WITHIN_9M / RANGE_9_TO_15M / SECTOR_ONLY 반환
```

반환 결과에는 다음 정보가 포함된다.

```text
range_class
range_label_ko
display_mode
sector_fill
confidence
reliability
score
threshold
margin
features_used
reason
```

특히 `display_mode`는 dashboard 시각화에 사용된다.

```text
range_bin    : 9m 이내 또는 9~15m 구간 중 하나의 cell 점등
sector_only  : 거리 구분 불안정, sector 전체 점등
none         : sector 없음, 표시하지 않음
```

### 4.3 Sector별 feature 조합

현재 생성된 profile 기준 sector별 feature 조합은 다음과 같다.

| 5-sector | 사용 feature | 개수 | 설명 |
|---|---|---:|---|
| LEFT_OUTER | `median_raw_p99` | 1 | p99 기반 단일 feature |
| LEFT_INNER | `frame_power_p99 + ratio_framepower_to_rms2` | 2 | power 및 정규화 비율 조합 |
| CENTER | `frame_power_p99` | 1 | CENTER에서 가장 단순하게 분리된 feature |
| RIGHT_INNER | `median_raw_p99 + ratio_p99_to_mean` | 2 | 절대 p99와 평균 대비 p99 비율 |
| RIGHT_OUTER | `raw_abs_mean + raw_abs_p99 + ratio_framepower_to_rms2` | 3 | 평균, p99, frame power ratio 조합 |

이 구조는 전체 sector에 하나의 feature를 공통 적용하는 방식이 아니라, **sector별로 설명 가능한 최소 feature 조합을 선택**하는 방식이다.

### 4.4 Profile 생성 결과

5개 CSV를 이용하여 profile을 생성한 결과는 다음과 같았다.

| Sector | Enabled | Reliability | Balanced Accuracy | Conditions | Feature |
|---|---:|---|---:|---:|---|
| LEFT_OUTER | True | HIGH | 1.000 | 22 | `median_raw_p99` |
| LEFT_INNER | True | HIGH | 1.000 | 19 | `frame_power_p99 + ratio_framepower_to_rms2` |
| CENTER | True | MID | 0.849 | 27 | `frame_power_p99` |
| RIGHT_INNER | True | HIGH | 0.850 | 19 | `median_raw_p99 + ratio_p99_to_mean` |
| RIGHT_OUTER | True | HIGH | 0.850 | 22 | `raw_abs_mean + raw_abs_p99 + ratio_framepower_to_rms2` |

단, 위 수치는 동일 CSV 묶음에서 profile을 생성하고 확인한 결과이므로 일반화 성능으로 해석하면 안 된다. 추후 leave-one-file-out 또는 새 실험일 데이터 검증이 필요하다.

### 4.5 OpenCV dashboard range cell 시각화

기존에는 sector fan 위에 별도 overlay를 얹는 방식으로 range band를 표시하려고 했다. 그러나 꼭지점과 sector 경계가 맞지 않아 시각적으로 부자연스러웠다.

최종적으로 다음 방식으로 수정했다.

```text
기존 fan + 별도 overlay 방식 폐기
→ _draw_sector_fan_v2()에서 sector grid와 range cell을 같은 polygon 좌표계로 직접 그림
```

현재 표시 방식은 다음과 같다.

```text
WITHIN_9M
→ 해당 sector의 안쪽 cell 점등

RANGE_9_TO_15M
→ 해당 sector의 바깥쪽 cell 점등

SECTOR_ONLY
→ 해당 sector 전체 점등
```

또한 오른쪽 텍스트 패널에는 다음 정보가 추가되었다.

```text
GT Angle
GT Distance
Distance Bin
Range Conf
Range Score
Range Feature
```

CSV replay에서는 `GT Angle`, `GT Distance`가 CSV에 저장된 원본 라벨값이며, live SDR 실행에서는 현재 실행 상태의 라벨값 또는 기본값으로 표시된다.

### 4.6 CSV replay dashboard

새로 추가한 파일은 다음과 같다.

```text
scripts/experimental/replay_sector_profile_dashboard.py
```

역할은 다음과 같다.

```text
저장된 sector_profile.csv 읽기
→ 각 row의 locked_sector_name / distance_m / true_angle_deg / raw feature 재구성
→ SectorRangeEstimator 적용
→ live dashboard renderer로 재생
```

실행 예시는 다음과 같다.

```bash
PYTHONPATH=. python scripts/experimental/replay_sector_profile_dashboard.py   --fps 2   --only-trusted
```

특정 CSV를 지정할 수도 있다.

```bash
PYTHONPATH=. python scripts/experimental/replay_sector_profile_dashboard.py   --csv outputs/aoa_sector_profiles/20260607_175750_gain35_cf2450000000_sector_profile.csv   --fps 2   --only-trusted
```

이를 통해 실제 SDR 없이도 수집된 CSV에서 sector 및 range cell이 어떻게 표시되는지 검증할 수 있다.

---

## 5. 실행 명령어 정리

### 5.1 Range profile 생성

```bash
PYTHONPATH=. python scripts/experimental/build_sector_range_profile.py
```

직접 CSV를 지정하는 경우:

```bash
PYTHONPATH=. python scripts/experimental/build_sector_range_profile.py   outputs/aoa_sector_profiles/20260606_190242_gain35_cf2450000000_sector_profile.csv   outputs/aoa_sector_profiles/20260606_191956_gain35_cf2450000000_sector_profile.csv   outputs/aoa_sector_profiles/20260607_173001_gain35_cf2450000000_sector_profile.csv   outputs/aoa_sector_profiles/20260607_175750_gain35_cf2450000000_sector_profile.csv
```

### 5.2 CSV replay

```bash
PYTHONPATH=. python scripts/experimental/replay_sector_profile_dashboard.py   --fps 2   --only-trusted
```

### 5.3 SDR live dashboard

실제 SDR 실행 전 demo mode는 반드시 꺼야 한다.

```yaml
sector_dashboard:
  demo_cycle: false
```

실행 명령어:

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_dashboard.py   --source-type sdr   --gain 35   --center-freq 2450000000   --target-fps 5
```

현재 range profile은 gain 35, center frequency 2.45 GHz 조건으로 생성되었으므로 위 조건과 맞다.

---

## 6. 검증 결과

### 6.1 기능 검증

다음 항목이 정상 동작함을 확인했다.

```text
1. SectorRangeEstimator import 및 profile loading
2. sector별 range_class 계산
3. profile_path 누락 또는 feature 부족 시 SECTOR_ONLY 처리
4. dashboard 오른쪽 panel에 range 정보 표시
5. fan_v2에서 range cell이 sector 칸에 맞게 점등
6. CSV replay에서 원본 distance_m / true_angle_deg 라벨 표시
7. sim/dummy demo_cycle을 통한 5-sector / 2-range UI 순환 확인
```

### 6.2 Git 반영

오늘 작업 중 다음 커밋을 생성했다.

```text
Add sector range profile estimator and dashboard range cells
Add CSV replay and range-cell dashboard visualization
Add AoA sector experiment reports
```

---

## 7. 한계 및 주의사항

### 7.1 일반화 한계

현재 profile은 특정 조건에서 수집된 CSV에 기반한다.

```text
gain = 35
center_freq = 2.45 GHz
동일 장비
동일 안테나 배치
동일 또는 유사 실험 환경
수동 입력된 distance_m / true_angle_deg 라벨
```

따라서 다른 gain, 다른 주파수, 다른 안테나 배치, 다른 실험 장소에서는 성능이 달라질 수 있다.

### 7.2 거리값 추정이 아님

현재 기능은 정확한 거리값을 출력하지 않는다.

```text
정확한 거리 추정 X
6m / 9m / 12m / 15m 세부 분류 X
9m 이내 / 9~15m 구간 coarse classification O
```

따라서 보고서와 발표에서는 “거리 추정”보다는 **실험적 거리 구간 분류** 또는 **coarse range indicator**로 표현하는 것이 안전하다.

### 7.3 검증 부족

현재 balanced accuracy는 profile 생성에 사용한 CSV 기반 결과이므로 과대평가 가능성이 있다. 다음 검증이 필요하다.

```text
1. Leave-one-file-out 검증
2. Leave-one-angle-out 검증
3. 새 실험일 데이터 검증
4. gain 30 / 40 조건 profile 추가 생성
5. 실제 드론 조종기 자세 변화에 대한 안정성 확인
```

---

## 8. 결론

2026-06-07 작업을 통해 기존 AoA sector dashboard는 단순 방향 표시에서 한 단계 확장되어, sector별 raw feature profile을 이용한 실험적 거리 구간 표시 기능을 갖게 되었다.

최종 구조는 다음과 같다.

```text
Sector profile CSV
→ build_sector_range_profile.py
→ sector_range_profile.json
→ SectorRangeEstimator
→ live_aoa_sector_dashboard.py
→ fan_v2 range cell visualization
```

이 구조는 기존 CNN/AoA/sector 판단 정책을 변경하지 않고, 별도 profile 기반 estimator와 dashboard 표시 계층을 추가한 것이다. 따라서 향후 profile만 교체하거나 검증 방식을 강화해도 runtime 판단 구조에는 영향을 최소화할 수 있다.

현재 기능은 아직 일반화 검증 전이므로 experimental로 유지하는 것이 적절하다. 다만 CSV replay, sim/dummy demo, SDR live dashboard를 모두 지원하게 되었기 때문에, 앞으로 추가 실험 데이터를 수집하고 검증 결과를 누적하기에 적합한 기반이 마련되었다.
