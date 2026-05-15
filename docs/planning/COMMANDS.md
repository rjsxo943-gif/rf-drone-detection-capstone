# RF Drone Detection Capstone - Commands

설명 + 주석 + 명령어를 같이 적어두는 작업용 문서

사용 원칙
- outputs/runs/latest/ 는 매 실행마다 덮어쓴다.
- 의미 있는 결과만 baseline_날짜시간/ 으로 따로 저장한다.
- 지금 단계에서는 자동 누적 저장보다 latest 확인 + baseline 수동 백업이 더 중요하다.

--------------------------------------------------
0) 프로젝트 폴더로 이동
현재 프로젝트 루트로 들어간다.

명령:
cd ~/projects/rf-drone-detection-capstone

--------------------------------------------------
1) 가상환경 활성화
파이썬 패키지 충돌 방지를 위해 먼저 켠다.

명령:
source .venv/bin/activate

--------------------------------------------------
2) 최소 파이프라인 실행
현재 기준 파이프라인:
sim input -> preprocess -> framing -> FFT -> energy detect -> summary/plot 저장

명령:
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

PYTHONPATH=. python scripts/파일이름.py
--------------------------------------------------
3) 최신 결과 목록 보기
최신 실행 결과 파일들이 제대로 생겼는지 확인한다.

명령:
ls outputs/runs/latest

기대 파일 예시
- summary.json
- frame_energies.npy
- detections.npy
- fft_mag.npy
- energy_plot.png

--------------------------------------------------
4) 최신 summary 보기
숫자 결과를 빠르게 확인한다.

명령:
cat outputs/runs/latest/summary.json

주요 확인 값
- num_frames
- num_detections
- noise_floor
- threshold
- detection_ratio

--------------------------------------------------
5) 결과 시각화 이미지 열기
에너지 그래프와 threshold 선을 직접 본다.

명령:
explorer.exe "$(wslpath -w outputs/runs/latest/energy_plot.png)"

--------------------------------------------------
6) 최신 결과 폴더 통째로 열기
최신 결과 파일들을 탐색기에서 한 번에 본다.

명령:
explorer.exe "$(wslpath -w outputs/runs/latest)"

--------------------------------------------------
7) baseline 저장
지금 결과를 비교 기준점으로 남기고 싶을 때만 실행한다.

명령:
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=outputs/runs/baseline_$STAMP
mkdir -p "$BASE"
cp -r outputs/runs/latest/* "$BASE"/
cp configs/receiver.yaml "$BASE"/
cp configs/detect.yaml "$BASE"/
echo "saved baseline to $BASE"
ls "$BASE"

언제 저장하면 좋은가
- 파이프라인이 처음 안정적으로 돌았을 때
- threshold 설정이 마음에 들 때
- sim 설정을 바꾸기 직전
- raw/real SDR로 넘어가기 직전

--------------------------------------------------
8) 저장된 baseline 목록 보기
기준점으로 남겨둔 결과들을 확인한다.

명령:
ls outputs/runs

--------------------------------------------------
9) 현재 detector 설정값 확인
detector 쪽 핵심 파라미터를 본다.

명령:
grep -n "threshold_multiplier\|frame_size\|hop_size\|window" configs/detect.yaml

주요 의미
- threshold_multiplier: 탐지 민감도
- frame_size: 한 프레임 길이
- hop_size: 프레임 이동 간격
- window: FFT 전에 적용하는 윈도우 종류

--------------------------------------------------
10) 현재 sim 설정값 확인
synthetic 입력 난이도를 확인한다.

명령:
grep -n "noise_std\|burst_amplitude\|burst_period\|burst_length\|tone_freq_norm" configs/receiver.yaml

주요 의미
- noise_std: 잡음 세기
- burst_amplitude: burst 신호 크기
- burst_period: burst 반복 주기
- burst_length: burst 길이
- tone_freq_norm: 합성 톤 위치

--------------------------------------------------
11) threshold_multiplier 빠르게 변경
detector 민감도를 바로 바꾼다.

명령:
sed -i 's/threshold_multiplier:.*/threshold_multiplier: 5.0/' configs/detect.yaml

팁
- 값을 올리면 보통 덜 민감해진다.
- 값을 내리면 보통 더 민감해진다.

--------------------------------------------------
12) burst_amplitude 빠르게 변경
sim 신호를 더 약하게 또는 더 강하게 만든다.

명령:
sed -i 's/burst_amplitude:.*/burst_amplitude: 1.5/' configs/receiver.yaml

팁
- 너무 쉬운 sim이면 amplitude를 내려본다.
- detector 튜닝 효과가 안 보일 때 유용하다.

--------------------------------------------------
13) noise_std 빠르게 변경
sim 잡음을 더 크게 만들어 난이도를 올린다.

명령:
sed -i 's/noise_std:.*/noise_std: 0.25/' configs/receiver.yaml

팁
- 현실성은 보통 noise를 조금 올릴수록 좋아진다.
- 너무 올리면 detector가 아예 힘들어질 수 있다.

--------------------------------------------------
14) 실행 + summary 확인 한 번에
수정 후 숫자 결과만 빠르게 보고 싶을 때 쓴다.

명령:
python scripts/run_pipeline.py && cat outputs/runs/latest/summary.json

--------------------------------------------------
15) 실행 + plot 열기 한 번에
수정 후 바로 그래프를 보고 싶을 때 쓴다.

명령:
python scripts/run_pipeline.py && explorer.exe "$(wslpath -w outputs/runs/latest/energy_plot.png)"

--------------------------------------------------
16) latest 폴더 상태 확인
최신 결과 저장 상태만 다시 본다.

명령:
ls outputs/runs/latest

--------------------------------------------------
17) 이 문서 열기
명령어 모음 문서를 편집한다.

명령:
code docs/planning/COMMANDS.md


1. check_raw_iq_store.py 완
   ↓
2. check_stage1_branch.py 완
   ↓
3. check_stage1_coherence.py 완
   ↓
4. calibrate.py 완
   ↓
5. preprocess_dataset.py 완(데이터셋없음)
   ↓
6. train_model.py
   ↓
7. export_model.py
   ↓
8. run_pipeline.py
완

1. run_pipeline.py 먼저 정리
2. SimReceiver / FileReceiver / PlutoReceiver 흐름 연결
3. DC offset 제거
4. STFT spectrogram 생성 확인
5. coherence / phase offset / AoA 계산 확인
6. 그다음 CNN dataset 전처리
7. train.py
8. export.py

지금 = 임시 (테스트용)
나중 = 고정 threshold (실제 시스템용)

scan 만들었음. 사이클돌면서 주파수대역 
신호크기 임계값넘으면 

raw데이터 쪼개서 cnn입력만들기
PYTHONPATH=. python scripts/auto_mavic_pipeline.py \
  --max-wide-blocks 0 \
  --top-k 3 \
  --power-ratio 1.2 \
  --save-top-if-none \
  --out-dir data/processed/mavic_auto_full

git 브런치 분기
   가장 자주 쓰는 안전한 흐름

너는 그냥 이 패턴만 기억하면 돼.

cd ~/projects/rf-drone-detection-capstone

git switch main
git pull origin main

git switch -c experiment/stft-128-hop32

# 코드 수정
# 실행 테스트

git status
git add .
git commit -m "experiment: test STFT nperseg 128 hop 32"

망했으면:

git switch main
git branch -D experiment/stft-128-hop32

A. 변경사항 임시보관
git stash
git switch main

나중에 다시 꺼내기:

git stash pop
B. 변경사항 버리기

정말 필요 없으면:

git reset --hard
git clean -fd
git switch main


# sdr 연결시 순서 

   1. 노트북에 sdr 연결
   2. 터미널  iio_info -u ip:192.168.2.1
   3. 파일위치랑 가상환경on
      cd ~/projects/rf-drone-detection-capstone

      source .venv/bin/activate
      
   4. PYTHONPATH=. python scripts/pluto_start1.py
   5. PYTHONPATH=. python scripts/pluto+exp.py