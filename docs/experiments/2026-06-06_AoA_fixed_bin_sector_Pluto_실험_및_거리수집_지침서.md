# 2026-06-06 AoA Fixed-Bin Sector Stabilizer 플루토 실험 및 거리 수집 지침서

## 0. 문서 목적

이 문서는 RF 드론 탐지 캡스톤 프로젝트에서 새로 만든 AoA Fixed-Bin Sector Stabilizer를 실제 Pluto SDR 환경에서 검증하고, 이후 sector별 거리 데이터를 수집하기 위한 현장 지침서이다.

이번 실험의 핵심은 다음이다.

```text
20블럭 update
→ raw gate 통과 block 중 top-K 선택
→ top-K CNN raw 판정
→ CNN Drone 후보 block만 AoA 후보로 사용
→ fixed-bin sector consensus 판단
→ trusted이면 locked sector 갱신
→ 실패하면 previous locked sector hold
```

거리 추정은 바로 하지 않는다.
먼저 AoA sector가 안정적으로 잡히는지 확인한 뒤, trusted sector 상태에서만 sector별 거리-세기 데이터를 수집한다.

---

## 1. 오늘까지 완료된 내용

### 완료된 구현

```text
[완료] configs/aoa_sector.yaml 생성
[완료] scripts/experimental/live_aoa_sector_experiment.py 생성
[완료] fixed_bins_7sector preset 생성
[완료] fixed_bins_3way preset 생성
[완료] 20블럭 update 구조 적용
[완료] raw gate 통과 block 중 top-K 선택
[완료] top-K CNN raw 판정
[완료] CNN temporal voting은 update당 1번만 반영
[완료] CNN Drone 후보 block만 AoA 후보로 사용
[완료] AoA 후보 angle/coherence/raw feature 수집
[완료] fixed-bin sector vote
[완료] trusted / hold_no_valid_aoa / hold_no_consensus / uncertain / no_signal 상태 구분
[완료] s 키 sector profile CSV 저장 구조
[완료] sim/file/sdr source-type 지원
[완료] sim dry-run 테스트 성공
```

### 아직 남은 것

```text
[미완료] 실제 Pluto SDR 연결 후 RF 신호 검증
[미완료] 실제 드론 조종기 신호에서 sector 안정성 확인
[미완료] sector별 거리 데이터 수집
[미완료] range_collect 모드 자동화
[미완료] sector별 거리-세기 테이블 생성
[미완료] rough range estimation
```

---

## 2. 이번 실험에서 절대 지킬 원칙

### 2.1 기존 viewer는 건드리지 않는다

이번 기능은 기존 `scripts/live_rf_viewer_drone_aoa.py`를 직접 바꾸는 방식이 아니다.

실험은 아래 새 파일에서만 수행한다.

```text
scripts/experimental/live_aoa_sector_experiment.py
```

설정은 아래 새 YAML을 사용한다.

```text
configs/aoa_sector.yaml
```

### 2.2 YAML 우선, CLI는 임시 override

기본 실행값은 YAML에서 관리한다.

CLI 인자는 현장 실험에서 임시로 바꾸고 싶을 때만 쓴다.

예를 들어 `receiver.yaml`에 이미 `source_type: sdr`가 있으면 실제 플루토 실행에서는 `--source-type sdr`를 굳이 넣지 않는다.

추천 기본 실행:

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --gain 35
```

### 2.3 거리 데이터는 trusted일 때만 저장한다

거리 수집 시 가장 중요한 원칙은 다음이다.

```text
sector_status == trusted 일 때만 저장한다.
```

아래 상태에서는 저장하지 않는다.

```text
hold_no_valid_aoa
hold_no_consensus
uncertain
no_signal
RAW_GATE_BLOCKED
```

hold 상태는 이전 sector를 유지하고 있는 상태일 수 있으므로, 그때 저장하면 거리-세기 profile이 오염될 수 있다.

---

## 3. 현재 sector bin 구조

기본 preset은 `fixed_bins_7sector`이다.

```text
L60    : -60도 이상, -45도 미만
L45    : -45도 이상, -30도 미만
L30    : -30도 이상, -15도 미만
CENTER : -15도 이상, +15도 미만
R30    : +15도 이상, +30도 미만
R45    : +30도 이상, +45도 미만
R60    : +45도 이상, +60도 이하
```

우선 7-sector로 실험한다.

만약 실제 드론 조종기 신호에서 너무 많이 튀면, 나중에 `fixed_bins_3way`로 낮춘다.

```text
LEFT   : -60도 ~ -20도
CENTER : -20도 ~ +20도
RIGHT  : +20도 ~ +60도
```

---

## 4. 실험 전 준비

### 4.1 코드 문법 확인

```bash
cd ~/projects/rf-drone-detection-capstone

python -m py_compile scripts/experimental/live_aoa_sector_experiment.py
```

아무 출력 없이 끝나면 정상이다.

### 4.2 YAML 확인

```bash
python - <<'PY'
import yaml
from pathlib import Path

p = Path("configs/aoa_sector.yaml")
data = yaml.safe_load(p.read_text(encoding="utf-8"))

root = data["aoa_sector"]
preset_name = root["active_preset"]
preset = root["presets"][preset_name]

print("YAML OK")
print("active_preset:", preset_name)
print("blocks_per_update:", root["runtime"]["blocks_per_update"])
print("top_k:", root["runtime"]["top_k"])
print("bins:", [b["name"] for b in preset["bins"]])
PY
```

정상 출력 예시:

```text
YAML OK
active_preset: fixed_bins_7sector
blocks_per_update: 20
top_k: 5
bins: ['L60', 'L45', 'L30', 'CENTER', 'R30', 'R45', 'R60']
```

### 4.3 sim dry-run 확인

플루토 없이 코드 흐름만 확인할 때 사용한다.

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py \
  --source-type sim \
  --cnn-backend dummy \
  --cnn-dummy-class-name Drone \
  --cnn-dummy-confidence 1.0 \
  --disable-raw-gate \
  --gain 35
```

이 테스트는 실제 성능 검증이 아니다.
UI, top-K, sector lock, hold, s 키 저장 흐름이 정상인지 확인하는 용도이다.

---

## 5. 오늘 플루토 실험 1단계: 기본 연결 확인

### 5.1 Pluto 연결 후 기본 실행

실제 플루토 연결 후에는 우선 가장 간단한 명령어를 사용한다.

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --gain 35
```

이 명령은 기본적으로 YAML 설정을 따른다.

```text
receiver.yaml       → source_type, center_freq, sample_rate, block_size
aoa_sector.yaml     → blocks_per_update, top_k, sector preset
ml.yaml             → CNN model, threshold, temporal voting
aoa.yaml            → antenna spacing, phase/gain calibration, coherence
```

### 5.2 실행 후 화면에서 확인할 값

화면 오른쪽 panel에서 아래 값을 본다.

```text
raw_pass
topk
drone
CNN raw class/confidence
CNN vote confirmed
sector_status
instant sector
locked sector
validAoA
sector votes
angle median
angle spread
coherence median
raw_p99 median
reason
```

### 5.3 정상 동작 예시

```text
[TOPK] raw_pass=8 topk=5 drone=3
[CNN] Drone conf=0.91
[VOTE] 3/3 confirmed=True
[SECTOR] status=trusted
instant=R30 locked=R30
validAoA=3 votes=R30=2,R45=1
```

### 5.4 이상 동작 예시

```text
[SECTOR] status=hold_no_valid_aoa
validAoA=0
reason=no valid AoA candidates
```

의미:

```text
CNN 후보는 있을 수 있지만, AoA 후보가 coherence/angle/gate 조건을 통과하지 못함.
```

```text
[SECTOR] status=hold_no_consensus
votes=R30=1,L45=1,CENTER=1
reason=sector votes scattered
```

의미:

```text
AoA 후보는 있지만 sector가 한 방향으로 모이지 않음.
멀티패스, 신호 약함, 위상 불안정 가능성 있음.
```

---

## 6. 오늘 플루토 실험 2단계: sector 방향 검증

거리 수집 전에 반드시 sector 방향이 실제 방향과 맞는지 확인한다.

### 6.1 권장 실험 조건

```text
center_freq = 2.450 GHz
gain = 35 먼저 시작
blocks_per_update = 20
top_k = 5
preset = fixed_bins_7sector
```

기본 명령어:

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --gain 35
```

필요하면 gain 40도 테스트한다.

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --gain 40
```

### 6.2 확인할 각도

처음부터 모든 각도를 다 하지 말고 아래 순서로 확인한다.

```text
0도
+30도
-30도
+45도
-45도
+60도
-60도
```

### 6.3 기대 sector

```text
0도    → CENTER
+30도  → R30 또는 R45 경계 부근
-30도  → L30 또는 L45 경계 부근
+45도  → R45
-45도  → L45
+60도  → R60
-60도  → L60
```

주의:

```text
+30도, -30도처럼 bin 경계 근처는 R30/R45 또는 L30/L45 사이에서 흔들릴 수 있다.
이 경우 실제 문제일 수도 있지만, 경계값 특성일 수도 있다.
```

### 6.4 판정 기준

좋은 상태:

```text
sector_status = trusted
validAoA >= 2
dominant_sector_ratio >= 0.60
locked_sector가 실제 방향과 대체로 일치
median_coherence가 0.70 이상
angle_spread가 지나치게 크지 않음
```

나쁜 상태:

```text
trusted가 거의 안 뜸
validAoA가 계속 0
hold_no_consensus가 계속 발생
locked sector가 실제 방향과 반대로 나옴
+각도인데 L sector가 뜸
-각도인데 R sector가 뜸
```

---

## 7. 오늘 플루토 실험 기록표

실험하면서 아래처럼 기록한다.

```text
날짜:
장소:
신호원:
안테나:
center_freq:
sample_rate:
block_size:
blocks_per_update:
top_k:
sector preset:
gain:
```

### 방향 검증 기록표

| 실제 각도 | 거리 | gain | raw_pass | drone | validAoA | status | locked sector | votes | coh_med | raw_p99_med | 비고 |
| ----: | -: | ---: | -------: | ----: | -------: | ------ | ------------- | ----- | ------: | ----------: | -- |
|    0도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  +30도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  -30도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  +45도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  -45도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  +60도 | 3m |   35 |          |       |          |        |               |       |         |             |    |
|  -60도 | 3m |   35 |          |       |          |        |               |       |         |             |    |

---

## 8. sector 검증 후 거리 데이터 수집 원칙

거리 데이터 수집은 다음 조건을 만족한 뒤 시작한다.

```text
1. sector_status가 trusted로 안정적으로 뜬다.
2. locked sector가 실제 방향과 대체로 일치한다.
3. validAoA가 보통 2개 이상 나온다.
4. median coherence가 0.70 이상으로 유지된다.
5. raw_pass_count와 cnn_drone_count가 너무 낮지 않다.
```

거리 수집에서는 절대 모든 update를 저장하지 않는다.

저장 조건:

```text
sector_status == trusted
validAoA >= 2
cnn_drone_count >= 2
locked_sector_name != ""
median_coherence >= 0.70
```

저장하지 말아야 할 조건:

```text
hold_no_valid_aoa
hold_no_consensus
uncertain
no_signal
RAW_GATE_BLOCKED
```

---

## 9. sector별 거리 수집 방법

### 9.1 기본 아이디어

거리 추정 모델을 바로 만드는 것이 아니라, 먼저 아래 형태의 데이터를 모은다.

```text
sector + distance + raw strength profile
```

예:

```text
CENTER, 3m, median_raw_p99
CENTER, 6m, median_raw_p99
CENTER, 9m, median_raw_p99
R30, 3m, median_raw_p99
R30, 6m, median_raw_p99
R45, 3m, median_raw_p99
...
```

### 9.2 추천 거리 간격

처음에는 3m 단위로 한다.

```text
3m
6m
9m
12m
15m
18m
21m
```

가능하면 추가:

```text
24m
27m
30m
```

다만 30m 근처는 각도 안정 범위가 줄어들 수 있으므로, 처음에는 CENTER와 ±30도 부근부터 보는 것이 좋다.

### 9.3 추천 수집 순서

1차 수집:

```text
CENTER 3m
CENTER 6m
CENTER 9m
CENTER 12m
CENTER 15m
```

2차 수집:

```text
R30 3m
R30 6m
R30 9m
R30 12m

L30 3m
L30 6m
L30 9m
L30 12m
```

3차 수집:

```text
R45 3m
R45 6m
R45 9m

L45 3m
L45 6m
L45 9m
```

4차 수집:

```text
R60
L60
```

단, R60/L60은 실제 조종기 신호에서 튈 수 있으므로 후순위로 둔다.

---

## 10. 거리 수집 현장 절차

예시: CENTER 6m 수집

### 10.1 준비

```text
gain = 35
center_freq = 2.450 GHz
실제 각도 = 0도
실제 거리 = 6m
목표 sector = CENTER
```

실행:

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py \
  --gain 35 \
  --distance-m 6 \
  --memo "CENTER_6m_gain35"
```

### 10.2 화면 확인

아래 상태가 될 때까지 기다린다.

```text
sector_status = trusted
locked = CENTER
validAoA >= 2
coh_med >= 0.70
cnn drone count >= 2
```

### 10.3 저장

`s` 키를 누른다.

한 거리/sector당 처음에는 10~30개 정도 저장한다.

```text
CENTER 6m에서 trusted 상태일 때 s 키 10~30회 저장
```

단, 화면이 hold 상태로 바뀌면 저장을 멈춘다.

---

## 11. 저장 파일 확인

`s` 키를 누르면 아래 폴더에 CSV가 저장된다.

```text
outputs/aoa_sector_profiles/
```

확인 명령어:

```bash
ls -lh outputs/aoa_sector_profiles
```

최근 파일 확인:

```bash
ls -lt outputs/aoa_sector_profiles | head
```

CSV 앞부분 확인:

```bash
head -n 5 outputs/aoa_sector_profiles/*.csv
```

---

## 12. 거리 수집 기록표

| sector | 실제 각도 | 실제 거리 | gain | 저장 개수 | trusted 비율 | raw_p99 경향 | coherence 경향 | 비고 |
| ------ | ----: | ----: | ---: | ----: | ---------: | ---------- | ------------ | -- |
| CENTER |    0도 |    3m |   35 |       |            |            |              |    |
| CENTER |    0도 |    6m |   35 |       |            |            |              |    |
| CENTER |    0도 |    9m |   35 |       |            |            |              |    |
| R30    |  +30도 |    3m |   35 |       |            |            |              |    |
| R30    |  +30도 |    6m |   35 |       |            |            |              |    |
| L30    |  -30도 |    3m |   35 |       |            |            |              |    |
| L30    |  -30도 |    6m |   35 |       |            |            |              |    |

---

## 13. 거리 수집 후 만들 테이블

나중에 CSV를 모아서 아래 형태의 summary table을 만든다.

```text
gain | sector | distance_bin | count | median_raw_p99 | iqr_raw_p99 | median_coherence | trusted_ratio
35   | CENTER | 3m           | 30    | ...            | ...         | ...              | ...
35   | CENTER | 6m           | 30    | ...            | ...         | ...              | ...
35   | CENTER | 9m           | 30    | ...            | ...         | ...              | ...
35   | R30    | 3m           | 30    | ...            | ...         | ...              | ...
35   | R30    | 6m           | 30    | ...            | ...         | ...              | ...
```

처음 range estimation은 정확한 m 단위가 아니라 rough bin으로 시작한다.

```text
near
mid
far
```

또는:

```text
3m bin
6m bin
9m bin
12m bin
```

---

## 14. 나중에 range_collect 모드로 확장할 때의 구조

현재는 수동 `s` 저장 방식으로 충분하다.

나중에 자동화할 때는 아래 옵션을 추가할 수 있다.

```bash
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py \
  --mode range_collect \
  --gain 35 \
  --true-distance-m 9 \
  --true-angle-deg 30 \
  --distance-bin-m 3 \
  --auto-save-trusted
```

자동 저장 조건:

```text
sector_status == trusted
validAoA >= 2
cnn_drone_count >= 2
median_coherence >= 0.70
```

자동 저장 금지 조건:

```text
hold_no_valid_aoa
hold_no_consensus
uncertain
no_signal
```

---

## 15. 현장에서 문제가 생겼을 때 판단법

### 15.1 raw_pass가 거의 0일 때

가능성:

```text
신호가 약함
gain이 낮음
raw_noise_gate threshold가 너무 높음
주파수가 맞지 않음
안테나 방향이 안 맞음
```

대응:

```text
gain 35 → 40으로 증가
center_freq 확인
신호원 거리 줄이기
안테나 방향 확인
```

### 15.2 cnn_drone_count가 0일 때

가능성:

```text
raw 신호는 들어오지만 CNN이 Drone으로 못 봄
모델 threshold가 너무 높음
드론 조종기 신호 패턴이 약함
주파수 또는 gain 조건이 학습 데이터와 다름
```

대응:

```text
gain 변경
거리 줄이기
조종기 방향 변경
threshold는 바로 낮추지 말고 로그 먼저 확인
```

### 15.3 validAoA가 0일 때

가능성:

```text
AoA angle이 valid 범위 밖으로 튐
coherence가 낮음
phase/gain calibration 문제
멀티패스 영향
두 채널 신호 품질 차이
```

대응:

```text
안테나 간격/방향 확인
phase_gain calibration profile 확인
신호원 거리 줄이기
주변 반사체 줄이기
min_stft_coherence를 임시로 0.65 정도로 낮춰 비교 가능
```

### 15.4 hold_no_consensus가 많을 때

가능성:

```text
top-K 후보 각도들이 서로 다른 sector로 흩어짐
멀티패스 영향
경계 각도에서 sector가 갈라짐
신호가 순간적으로 튐
```

대응:

```text
fixed_bins_3way로 낮춰 비교
top_k를 3으로 낮춰 비교
coherence threshold를 높여 나쁜 후보 제거
실제 각도를 bin 중앙에 가깝게 배치
```

---

## 16. 오늘 실험 성공 기준

오늘 플루토 실험의 성공 기준은 거리 추정이 아니다.

오늘 성공 기준:

```text
1. Pluto SDR 연결 후 새 sector viewer가 실행된다.
2. raw_pass_count가 정상적으로 나온다.
3. top-K CNN 후보가 표시된다.
4. Drone 신호에서 cnn_drone_count가 나온다.
5. AoA 후보 validAoA가 2개 이상 나오는 구간이 있다.
6. trusted sector가 발생한다.
7. 실제 방향과 locked sector가 대체로 일치한다.
8. s 키로 CSV 저장이 된다.
```

오늘 실패해도 괜찮은 것:

```text
거리 추정 안 됨
R60/L60이 불안정함
30m에서 validAoA가 부족함
hold_no_consensus가 일부 발생함
```

오늘 반드시 피해야 할 것:

```text
trusted가 아닌 상태에서 거리 데이터 저장
기존 live viewer 수정
거리 추정 알고리즘을 먼저 구현
threshold를 근거 없이 계속 낮추기
```

---

## 17. 오늘 실험 후 정리할 내용

실험이 끝나면 아래 내용을 보고서에 남긴다.

```text
실험 날짜:
실험 장소:
사용 장비:
안테나 종류:
center_freq:
gain:
blocks_per_update:
top_k:
sector preset:
```

결과 요약:

```text
CENTER sector 안정 여부:
R30/L30 sector 안정 여부:
R45/L45 sector 안정 여부:
R60/L60 sector 안정 여부:
trusted 발생 비율:
주요 실패 상태:
가장 안정적 gain:
거리별 raw_p99 변화:
coherence 경향:
```

다음 작업:

```text
sector 안정화 튜닝
range_collect 모드 추가
sector별 거리-세기 table 생성
rough range estimation 구현
```

