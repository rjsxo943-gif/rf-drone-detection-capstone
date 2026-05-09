# Pluto+ SDR 기반 CNN 데이터 수집 사용설명서

## 0. 목적

이 문서는 Pluto+ SDR과 안테나를 연결한 뒤, 2.4GHz 대역 RF 신호를 스캔하고, 감지된 신호를 CNN 학습용 spectrogram 데이터로 저장하기 위한 절차를 설명한다.

최종 목표는 아래 구조의 학습 데이터를 만드는 것이다.

```text
data/processed/cnn_capture/
├── background/
├── wifi/
├── bluetooth/
└── drone_like/
```

각 샘플은 `.npz` 파일로 저장되며, 내부에는 CNN 입력용 `spectrogram`, `cnn_input`, `metadata_json`이 저장된다.

---

## 1. 시작 전 준비물

### 1.1 하드웨어

```text
Pluto+ SDR
2.4GHz 안테나 1개 또는 2개
USB 또는 Ethernet 연결
노트북/PC
테스트 신호원
```

테스트 신호원 예시:

```text
WiFi: 아이폰 핫스팟, 공유기, 학교 WiFi
Bluetooth: 무선 이어폰, 블루투스 스피커, 마우스
Drone-like: 실제 드론 조종기 또는 공개 IQ 데이터셋
Background: 의도적인 신호원을 켜지 않은 주변 환경
```

### 1.2 소프트웨어 위치

프로젝트 폴더로 이동한다.

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

---

## 2. Pluto+ 연결 확인

### 2.1 SDR 연결 상태 확인

먼저 Pluto+가 잡히는지 확인한다.

```bash
iio_info -s
```

정상이라면 `ip:192.168.2.1` 또는 비슷한 URI가 보여야 한다.

직접 IP로 확인한다.

```bash
sudo iio_info -u usb:
```

### 2.2 주의

`iio_info -s`에서 Avahi 관련 경고가 나와도, 아래 명령이 성공하면 사용할 수 있다.

```bash
iio_info -u ip:192.168.2.1
```

중요한 것은 Pluto+에 직접 접속이 되는지다.

---

## 3. `receiver.yaml` 설정

실제 SDR을 사용할 때는 `configs/receiver.yaml`을 SDR 모드로 바꾼다.

```bash
code configs/receiver.yaml
```

기본 예시는 아래와 같다.

```yaml
source_type: sdr

uri: ip:192.168.2.1

sample_rate: 5000000
center_freq: 2400000000
rf_bandwidth: 5000000
block_size: 16384
num_samples: 16384

num_channels: 2
rx_channels:
  - 0
  - 1

gain: 40.0
```

### 설정 의미

```text
source_type: sdr
→ 실제 Pluto+ SDR을 사용한다.

sample_rate: 5000000
→ 5 MSPS로 수신한다.

center_freq: 2400000000
→ 초기 중심 주파수. 실제 scan에서는 주파수를 바꿔가며 탐색한다.

rf_bandwidth: 5000000
→ 수신 대역폭 설정. sample_rate와 맞춰 5 MHz 기준으로 둔다.

block_size: 16384
→ 한 번 처리할 IQ block 크기.

num_channels: 2
→ RX0/RX1 두 채널 사용.

gain: 40.0
→ 수신 gain. 너무 크면 포화될 수 있고, 너무 작으면 신호가 약하게 잡힌다.
```

---

## 4. RX0/RX1 수신 shape 확인

캘리브레이션 전에 실제로 IQ block이 들어오는지 확인한다.

```bash
PYTHONPATH=. python - <<'PY'
from src.core import load_all_configs, get_block_size
from src.receiver import build_receiver

configs = load_all_configs("configs")
receiver_cfg = configs["receiver"]
block_size = get_block_size(configs)

rx = build_receiver(receiver_cfg)

try:
    iq = rx.read_block(block_size)
    print("iq shape:", iq.shape)
    print("iq dtype:", iq.dtype)
    print("first samples:", iq[..., :5])
finally:
    if hasattr(rx, "close"):
        rx.close()
PY
```

### 정상 기준

2채널 설정이면 보통 아래처럼 나와야 한다.

```text
iq shape: (2, 16384)
```

1채널만 쓰는 설정이면 아래처럼 나올 수 있다.

```text
iq shape: (1, 16384)
```

### 문제 상황: `RX mapping exceeds available channels`

채널 설정이 실제 SDR 설정과 안 맞는 경우다.

확인할 것:

```text
receiver.yaml의 num_channels
receiver.yaml의 rx_channels
Pluto+가 실제 2RX를 지원하도록 설정되어 있는지
```

---

## 5. Noise Calibration 실행

CNN capture 모드에서는 noise calibration 결과를 기준으로 신호 후보를 판단한다. 따라서 먼저 noise calibration을 실행해야 한다.

CLI 실행:

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

메뉴가 뜨면:

```text
select> n
```

### Noise calibration 할 때 상태

이때는 의도적인 신호원을 꺼두는 것이 좋다.

```text
아이폰 핫스팟 끄기
블루투스 테스트 기기 끄기
드론 조종기 끄기
가까운 테스트 송신원 치우기
```

완전한 무신호 환경은 아니어도 괜찮다. 다만 테스트하려는 강한 신호원을 켜둔 상태에서 noise calibration을 하면 threshold가 너무 높게 잡힐 수 있다.

### 성공 기준

캘리브레이션이 끝난 뒤 CLI 시작 시 calibration status에 noise 값이 보여야 한다.

예상 출력 형태:

```text
[Noise]
noise_floor : ...
threshold   : ...
method      : ...
source      : ...
```

---

## 6. Phase/Gain Calibration 실행

CNN 학습용 spectrogram만 저장할 때는 phase/gain calibration이 필수는 아니다. 하지만 나중에 AoA까지 같이 검증하려면 실행해두는 것이 좋다.

CLI에서:

```text
select> p
```

### Phase/Gain calibration 할 때 상태

신호원을 두 안테나의 정면 0도 방향에 둔다.

```text
RX0 안테나와 RX1 안테나를 나란히 배치
두 안테나의 중앙 정면에 신호원 배치
가능하면 일정 거리 유지
신호원이 계속 송신하도록 유지
```

예시:

```text
아이폰 핫스팟 또는 공유기
두 안테나 정면 중앙
너무 가까이 붙이지 않기
```

### 성공 기준

CLI 시작 시 아래 값들이 보이면 된다.

```text
[Phase/Gain]
gain_correction : ...
phase_offset    : ... rad
phase_offset    : ... deg
coherence_like  : ...
```

---

## 7. CNN 데이터 수집 시작

CLI에서 start를 선택한다.

```text
select> s
```

그러면 아래 입력을 받는다.

```text
label ex) wifi / bluetooth / drone_like / background >
max_saved [default=50] >
rx_index [default=0] >
save raw iq? [y/N] >
```

---

## 8. 첫 테스트: WiFi 3개만 저장

처음부터 많이 저장하지 말고, 반드시 3개만 저장해본다.

### 8.1 WiFi 신호원 준비

아이폰 핫스팟이나 공유기를 사용한다.

권장 방식:

```text
아이폰 핫스팟 ON
다른 기기를 핫스팟에 연결
연결된 기기에서 유튜브 재생 또는 파일 다운로드
안테나는 핫스팟을 켠 기기 근처에 둔다
```

핫스팟만 켜두는 것보다 실제 데이터 통신이 있어야 신호가 더 잘 보인다.

### 8.2 입력 예시

```text
label ex) wifi / bluetooth / drone_like / background > wifi
max_saved [default=50] > 3
rx_index [default=0] > 0
save raw iq? [y/N] > n
```

### 8.3 중간 정지

스캔 중 멈추고 싶으면:

```text
q
```

입력 후 Enter.

현재 block 읽기가 끝난 뒤 capture가 종료되고 CLI로 돌아온다.

---

## 9. 저장 결과 확인

수집이 끝나면 아래 명령어로 파일을 확인한다.

```bash
find data/processed/cnn_capture -maxdepth 5 -type f | head -20
```

정상이라면 아래처럼 나온다.

```text
data/processed/cnn_capture/wifi/20260507_123456/session.json
data/processed/cnn_capture/wifi/20260507_123456/metadata.jsonl
data/processed/cnn_capture/wifi/20260507_123456/sample_000001.npz
data/processed/cnn_capture/wifi/20260507_123456/sample_000002.npz
data/processed/cnn_capture/wifi/20260507_123456/sample_000003.npz
```

---

## 10. 저장된 샘플 검사

검사 스크립트를 실행한다.

```bash
PYTHONPATH=. python scripts/check_cnn_capture_sample.py
```

최신 sample을 자동으로 검사한다.

### PNG까지 저장하고 싶으면

```bash
PYTHONPATH=. python scripts/check_cnn_capture_sample.py --save-png
```

### 특정 파일을 검사하고 싶으면

```bash
PYTHONPATH=. python scripts/check_cnn_capture_sample.py data/processed/cnn_capture/wifi/세션폴더/sample_000001.npz
```

### 정상 기준

아래 형태가 나오면 좋다.

```text
[spectrogram]
shape      : (512, 125)
dtype      : float32
min        : 0
max        : 1
has_nan    : False
has_inf    : False

[cnn_input]
shape      : (512, 125, 1)
dtype      : float32
has_nan    : False
has_inf    : False
```

중요한 기준:

```text
spectrogram이 2D인지
cnn_input이 3D인지
마지막 차원이 1인지
NaN이 없는지
Inf가 없는지
값 범위가 0~1 근처인지
metadata label이 맞는지
```

---

## 11. 본격 수집 순서

첫 WiFi 테스트가 성공하면 라벨별로 나눠서 수집한다.

### 11.1 Background

```text
label: background
```

상태:

```text
의도적인 신호원 없음
핫스팟 끄기
블루투스 테스트 끄기
드론 조종기 끄기
```

입력 예시:

```text
label ex) wifi / bluetooth / drone_like / background > background
max_saved [default=50] > 50
rx_index [default=0] > 0
save raw iq? [y/N] > n
```

주의:

학교 WiFi나 주변 신호는 완전히 제거하기 어렵다. 그래도 실제 환경 배경 데이터로 의미가 있다.

### 11.2 WiFi

```text
label: wifi
```

상태:

```text
아이폰 핫스팟 ON 또는 공유기 근처
연결된 기기에서 트래픽 발생
유튜브 재생, 파일 다운로드 등
```

입력 예시:

```text
label ex) wifi / bluetooth / drone_like / background > wifi
max_saved [default=50] > 50
rx_index [default=0] > 0
save raw iq? [y/N] > n
```

### 11.3 Bluetooth

```text
label: bluetooth
```

상태:

```text
무선 이어폰 연결
블루투스 스피커 음악 재생
블루투스 마우스/키보드 사용
연결/해제 반복도 가능
```

입력 예시:

```text
label ex) wifi / bluetooth / drone_like / background > bluetooth
max_saved [default=50] > 50
rx_index [default=0] > 0
save raw iq? [y/N] > n
```

### 11.4 Drone-like

```text
label: drone_like
```

상태:

```text
실제 드론 조종기 ON
드론과 조종기 연결
가능하면 모터 동작 없이 통신 상태 유지
안전한 환경에서 진행
```

입력 예시:

```text
label ex) wifi / bluetooth / drone_like / background > drone_like
max_saved [default=50] > 50
rx_index [default=0] > 0
save raw iq? [y/N] > n
```

주의:

실제 드론을 사용할 때는 주변 안전과 전파 사용 환경을 확인해야 한다.

---

## 12. raw IQ 저장 여부

입력 중 아래 질문이 나온다.

```text
save raw iq? [y/N] >
```

### 기본 추천

처음에는 `n`.

```text
n
```

이유:

```text
저장 용량이 작다
CNN 입력 확인이 빠르다
팀원에게 넘기기 쉽다
```

### raw IQ를 저장해야 하는 경우

```text
나중에 STFT 파라미터를 바꿔서 다시 만들고 싶을 때
CNN 입력이 이상해서 원본 IQ를 확인해야 할 때
AoA나 phase 분석까지 같이 검증하고 싶을 때
```

그때는:

```text
y
```

단, 저장 용량이 빠르게 커진다.

---

## 13. 수집 후 Git 주의사항

수집된 데이터는 보통 Git에 올리지 않는다.

아래 폴더는 커밋하지 않는 것이 원칙이다.

```text
data/processed/cnn_capture/
outputs/
```

상태 확인:

```bash
git status --short
```

만약 데이터 파일이 많이 뜨면, 실수로 커밋하지 않는다.

커밋해야 하는 것은 코드와 문서다.

```text
src/runtime/cnn_capture_actions.py
src/runtime/cli.py
scripts/check_cnn_capture_sample.py
docs/planning/SDR_CAPTURE_MANUAL.md
```

---

## 14. 문제 해결

### 14.1 CLI가 실행되지 않음

확인:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_runtime_cli.py
```

그래도 안 되면 compile 확인:

```bash
PYTHONPATH=. python -m compileall src/runtime/cnn_capture_actions.py src/runtime/cli.py scripts/run_runtime_cli.py
```

### 14.2 noise calibration이 없다고 나옴

에러 예시:

```text
noise calibration result not found. Run noise calibration first.
```

해결:

```text
CLI 실행
select> n
```

noise calibration을 먼저 실행한다.

### 14.3 sample이 저장되지 않음

가능한 원인:

```text
threshold가 너무 높음
신호원이 너무 약함
안테나 위치가 멀다
gain이 너무 낮다
scan_blocks 또는 min_detection_ratio 조건이 빡세다
receiver.yaml이 sim/file/sdr 중 잘못 설정됨
```

먼저 할 것:

```text
1. 신호원 가까이 두기
2. WiFi 트래픽 강하게 발생시키기
3. max_saved=3으로 다시 테스트
4. receiver.yaml source_type 확인
5. noise calibration 다시 실행
```

### 14.4 spectrogram shape가 예상과 다름

예상:

```text
spectrogram: (512, 125)
cnn_input: (512, 125, 1)
```

다를 수 있는 이유:

```text
block_size가 바뀜
nperseg가 바뀜
hop_size가 바뀜
nfft가 바뀜
```

STFT 설정을 바꿨다면 shape가 달라지는 것은 정상일 수 있다. 다만 CNN 학습 코드와 입력 shape는 반드시 맞춰야 한다.

### 14.5 q를 눌러도 바로 안 멈춤

정상이다.

현재 구조는 `read_block()`이 끝난 뒤 stop을 확인한다.

즉:

```text
q 입력 후 Enter
현재 block 수신 완료
stop 감지
capture 종료
CLI 복귀
```

완전한 즉시 강제 종료가 필요하면 `Ctrl+C`를 사용할 수 있다.

---

## 15. 전체 운용 요약

실제 SDR 연결 후 전체 순서는 아래와 같다.

```text
1. Pluto+ 연결
2. iio_info -u ip:192.168.2.1 확인
3. configs/receiver.yaml을 sdr 모드로 설정
4. RX0/RX1 IQ shape 확인
5. CLI 실행
6. noise calibration 실행
7. phase/gain calibration 실행
8. start 선택
9. label 입력
10. max_saved=3으로 테스트 수집
11. check_cnn_capture_sample.py로 npz 확인
12. 문제 없으면 label별 50개 이상 수집
13. 수집 데이터는 Git에 올리지 않음
```

가장 먼저 실제로 할 테스트는 이거다.

```bash
PYTHONPATH=. python scripts/run_runtime_cli.py
```

```text
select> n
select> s
label> wifi
max_saved> 3
rx_index> 0
save raw iq? [y/N]> n
```

그리고 확인:

```bash
PYTHONPATH=. python scripts/check_cnn_capture_sample.py --save-png
```

이 과정이 성공하면, Pluto+ 기반 CNN 학습 데이터 수집 파이프라인은 1차로 정상 동작한다고 보면 된다.

cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

ping -c 3 192.168.2.1
iio_info -u ip:192.168.2.1


데이터 수집
PYTHONPATH=. python scripts/capture_wifi_compare_128.py \
  --label home_wifihot_ch6_on_2437_gain10_1m \
  --blocks 400 \
  --center-freq 2437000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

PYTHONPATH=. python scripts/capture_wifi_compare_128.py \
  --label home_wifi_ch6_off_2437_gain10 \
  --blocks 30 \
  --center-freq 2437000000 \
  --gain 20 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

PYTHONPATH=. python scripts/analyze_capture_folder.py \
  --folder data/processed/cnn_capture/home_wifihot_ch6_on_2437_gain10_1m/20260509_160250

PYTHONPATH=. python scripts/analyze_capture_folder.py \
  --folder data/processed/cnn_capture/home_wifi_ch6_off_2437_gain10/20260509_154302


PYTHONPATH=. python scripts/select_meaningful_capture_blocks.py \
  --folder data/processed/cnn_capture/home_wifihot_ch6_on_2437_gain10_1m/20260509_172543

1. capture_wifi_compare_128.py
   → IQ/STFT 블록 수집

2. analyze_capture_folder.py
   → 전체적으로 신호가 제대로 잡혔는지 확인

3. select_meaningful_capture_blocks.py
   → 의미 있는 블록만 selected_meaningful 폴더로 정리

4. 눈으로 selected_meaningful 확인
   → 진짜 이상한 블록만 수동 제거

find data/processed/cnn_capture/home_wifihot_ch6_on_2437_gain10_1m/WIFI \
  -type f -name "*Zone.Identifier*" -delete



BT
  PYTHONPATH=. python scripts/capture_wifi_compare_128.py \
  --label home_bt_audio_on_2450_gain10_1m \
  --blocks 400 \
  --center-freq 2450000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

  PYTHONPATH=. python scripts/capture_wifi_compare_128.py \
  --label home_bt_audio_on_2437_gain10_1m \
  --blocks 400 \
  --center-freq 2437000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

  PYTHONPATH=. python scripts/capture_wifi_compare_128.py \
  --label home_bt_audio_on_2460_gain10_1m \
  --blocks 400 \
  --center-freq 2460000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

데이터 수집용
PYTHONPATH=. python scripts/capture_bluetooth_selected_only_128hop32.py \
  --label home_bt_audio_on_2437_gain10_0.4m \
  --blocks 4000 \
  --center-freq 2437000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40


  PYTHONPATH=. python scripts/analyze_capture_folder.py \
  --folder data/processed/cnn_capture/home_bt_audio_on_2437_gain10_1m/20260509_185407


PYTHONPATH=. python scripts/select_bluetooth_meaningful_blocks.py \
  --folder data/processed/cnn_capture/home_bt_audio_on_2437_gain10_1m/20260509_185407

PYTHONPATH=. python scripts/select_bluetooth_meaningful_blocks.py \
--folder data/processed/cnn_capture/home_bt_audio_on_2450_gain10_1m/20260509_184541

PYTHONPATH=. python scripts/select_bluetooth_meaningful_blocks.py \
--folder data/processed/cnn_capture/home_bt_audio_on_2460_gain10_1m/20260509_185751


background

cd ~/projects/rf-drone-detection-capstone

PYTHONPATH=. python scripts/capture_background_selected_only_128hop32.py \
  --label home_bg_2437_gain10 \
  --blocks 1000 \
  --target-selected 500 \
  --center-freq 2437000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40

  PYTHONPATH=. python scripts/capture_background_selected_only_128hop32.py \
  --label home_bg_2450_gain10 \
  --blocks 1000 \
  --target-selected 500 \
  --center-freq 2450000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40


  PYTHONPATH=. python scripts/capture_background_selected_only_128hop32.py \
  --label home_bg_2460_gain10 \
  --blocks 1000 \
  --target-selected 500 \
  --center-freq 2460000000 \
  --gain 10 \
  --channel 0 \
  --vmin -40 \
  --vmax 40