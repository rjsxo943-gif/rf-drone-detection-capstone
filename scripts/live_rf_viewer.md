Full mode 실행:

PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 5 \
  --profile-blocks 20 \
  --model outputs/ml/path/to/best_model.pt \
  --cnn-backend torch \
  --cnn-device cpu \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json


모델 없이 통합 화면만 먼저 확인하려면:
  PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 5 \
  --profile-blocks 20 \
  --cnn-backend dummy \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json



  첫 현장 테스트는 fast mode
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode fast \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 10

fast mode는 RF burst 움직임을 실시간 확인하는 용도고, full mode는 연산량 때문에 3~5 FPS 권장이라 현장 패턴 확인은 fast를 먼저 쓰는 게 맞다.

그다음 profile mode로 gain feature 저장 확인
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode profile \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 10 \
  --profile-blocks 20

여기서 s 눌렀을 때:

outputs/viewer/gain_feature_profiles.csv
outputs/viewer/gain_feature_profiles_latest.json

이 두 개 생기면 성공.

CNN은 먼저 dummy로 화면만 확인
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode cnn \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 5 \
  --cnn-backend dummy

이게 되면 그다음에 .pt 모델 넣고 torch로 돌리면 됨.

마지막에 full mode
PYTHONPATH=. python scripts/live_rf_viewer.py \
  --mode full \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 30 \
  --target-fps 5 \
  --profile-blocks 20 \
  --cnn-backend dummy \
  --aoa-phase-calibration-json configs/calibration/current_phase_offset.json \
  --aoa-gain-phase-table configs/calibration/gain_phase_table_2450.json