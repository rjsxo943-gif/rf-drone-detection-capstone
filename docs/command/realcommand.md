cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate

ping -c 3 192.168.2.1
iio_info -u ip:192.168.2.1



PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --display-backend opencv \
  --view-mode dual \
  --db-preset strong \
  --gain 25 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "home_ui_test_gain25"

--db-preset {basic,weak,strong,custom}

PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --display-backend opencv \
  --view-mode dual \
  --db-preset strong \
  --gain 25 \
  --center-freq 2437000000 \
  --distance-m 1.0 \
  --blocks-per-update 1 \
  --max-updates 50000 \
  --overload-peak 50 \
  --overload-clip-ratio 0.99 \
  --memo "youtube_static_g25_1m_100blocks"



  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_linked_motor_off_g25_d1m_front \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 200 \
  --until-max-saved \
  --max-total-blocks 3000 \
  --skip-inference

PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_linked_motor_off_g25_d1p5m_left   \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 100 \
  --until-max-saved \
  --max-total-blocks 4000 \
  --skip-inference

1. drone_linked_motor_off_g25_d1m_front      100
2. drone_linked_motor_off_g25_d1m_left       100
3. drone_linked_motor_off_g25_d1m_right      100
1. drone_linked_motor_off_g25_d1m_front_run2      100장
2. drone_linked_motor_off_g25_d2m_front_run2      100장


4. drone_linked_motor_off_g25_d1p5m_front    100
drone_linked_motor_off_g25_d1p5m_front_retry 100
7. drone_linked_motor_off_g25_d2m_front      100

5. drone_linked_motor_off_g25_d1p5m_left     100

6. drone_linked_motor_off_g25_d1p5m_right    100



8. drone_linked_motor_off_g25_d2m_left       100
9. drone_linked_motor_off_g25_d2m_right      100


PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label  drone_hover_g25_d1p5m_front   \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --skip-inference

1. drone_linked_motor_on_g25_d1m_front        150장
2. drone_linked_motor_on_g25_d1p5m_front      150장
3. drone_linked_motor_on_g25_d2m_front        150장

4. drone_linked_motor_on_g25_d1p5m_left       100장
5. drone_linked_motor_on_g25_d1p5m_right      100장

6. drone_hover_g25_d1p5m_front                150장
7. drone_hover_g25_d2m_front                  150장
8. drone_moving_lr_g25_d1p5m_front            100장
9. drone_moving_lr_g25_d2m_front              100장

PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_motor_on_stick_taps_g25_d2m_front \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --start-discard-sec 5.0 \
  --skip-inference

drone_motor_on_stick_taps_g25_d1m_front       150
drone_motor_on_stick_taps_g25_d1p5m_front     150
drone_motor_on_stick_taps_g25_d2m_front       150






--------------------------------------------------------
non-drone
--------------------------------------------------------

PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_outdoor_g25_people_moving  \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-saved 300 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --start-discard-sec 5.0 \
  --skip-inference

 
 
  background_outdoor_g25_clean_front    300
  background_outdoor_g25_angle_changed    200
  background_outdoor_g25_people_moving    300




  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label wifi_hotspot_handheld_moving \
  --center-freq 2437000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 6000 \
  --start-discard-sec 5.0 \
  --skip-inference


1. wifi_hotspot_static_d1m          200장
2. wifi_hotspot_static_d1p5m        200장
3. wifi_hotspot_static_d2m          200장

4. wifi_hotspot_handheld_static     150장
5. wifi_hotspot_handheld_moving     150장


  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label bluetooth_g25_connected_d1m \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 250 \
  --until-max-saved \
  --max-total-blocks 7000 \
  --skip-inference


bluetooth_g25_connected_d1m              200
bluetooth_g25_music_playing_d1p5m        200
bluetooth_g25_call_active_d1p5m          200
bluetooth_g25_phone_moving_d2m           200



  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label mixed_wifi_bluetooth_g25 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 300 \
  --until-max-saved \
  --max-total-blocks 8000 \
  --skip-inference

  mixed_wifi_bluetooth_g25    300


별도
  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label controller_only_g25_d1m_side90 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 200 \
  --until-max-saved \
  --max-total-blocks 7000 \
  --skip-inference

  controller_only_g25_d1m_side90    200


--------------------------------------------
viewer
--------------------------------------------
모드 설명서
 PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py --help | grep -E "decision-mode|cnn-model|threshold|temporal|candidate|confirmed"


기존 viewer처럼 CNN 없이 실행
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 30 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "viewer_none_mode_test" \
  --decision-mode none


Raw CNN 모드
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 30 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "raw_cnn_mode_test" \
  --decision-mode raw \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50


Gain-aware 모드
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 30 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "gain_aware_mode_test" \
  --decision-mode gain-aware \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80


최종 추천 Hybrid 모드
PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 30 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "hybrid_gain_aware_temporal_test" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3

  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 5 \
  --center-freq 2437000000 \
  --distance-m 1.0 \
  --memo "wifi_ui_test_compact_g5" \
  --blocks-per-update 5 \
  --update-interval-sec 1.0


  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 25 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "hybrid_gain_aware_temporal_test" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 0.2

  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 25 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "hybrid_gain_aware_temporal_test_overload_peak20" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 0.2 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001



  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 25 \
  --center-freq 2450000000 \
  --distance-m 1 \
  --memo "" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 1.0 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001


  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer.py \
  --gain 10 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "drone_fixed_gain25_5m" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 1.0 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001


  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_aoa.py \
  --gain 25 \
  --center-freq 2450000000 \
  --distance-m 5.0 \
  --memo "drone_aoa_test_5m_gain25" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 0.2 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001 \
  --enable-aoa \
  --aoa-antenna-spacing-m 0.061 \
  --aoa-calibration-deg 0



MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_aoa.py \
  --gain 15 \
  --center-freq 2450000000 \
  --distance-m 6.0 \
  --memo "aoa_phase_cal_front_0deg" \
  --decision-mode hybrid \
  --cnn-model outputs/ml/rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt \
  --drone-threshold 0.50 \
  --drone-threshold-g25 0.35 \
  --drone-threshold-g30 0.80 \
  --temporal-window 5 \
  --candidate-vote-k 2 \
  --confirmed-vote-k 3 \
  --blocks-per-update 5 \
  --update-interval-sec 0.2 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001 \
  --enable-aoa \
  --aoa-antenna-spacing-m 0.061 \
  --aoa-calibration-deg 0 \
  --aoa-min-coherence 0.05 \
  --aoa-min-signal-ratio 5.0 \
  --aoa-gate-mode candidate \
  --aoa-auto-phase-calibration \
  --aoa-calibration-blocks 30





  MPLBACKEND=QtAgg PYTHONPATH=. python scripts/live_siggen_aoa_viewer.py \
  --gain 20 \
  --center-freq 2450000000 \
  --distance-m 1.0 \
  --memo "siggen_aoa_phase_cal_front_0deg_g15" \
  --decision-mode none \
  --blocks-per-update 5 \
  --update-interval-sec 0.2 \
  --overload-peak 30 \
  --overload-clip-ratio 0.001 \
  --enable-aoa \
  --aoa-antenna-spacing-m 0.061 \
  --aoa-auto-phase-calibration \
  --aoa-calibration-blocks 50 \
  --aoa-calibration-deg 0 \
  --aoa-min-coherence 0.05 \
  --aoa-smooth-window 5 \
  --aoa-smooth-min-valid 3


  PYTHONPATH=. python scripts/calibrate_phase_offset_outdoor.py \
  --uri ip:192.168.2.1 \
  --center-freq 2450000000 \
  --sample-rate 5000000 \
  --gain 20\
  --num-blocks 50 \
  --min-coherence 0.80 \
  --memo "outdoor_0deg_2m_gain25"




  [필수 1]
scripts/calibrate_phase_offset_outdoor.py
→ 단일 현장 기준 gain calibration을 robust 방식으로 변경

[필수 2]
scripts/build_gain_phase_table.py
→ 실외 gain별 delta table 제작용 신규 스크립트

[필수 3]
실시간 AoA 실행 스크립트
→ live_cnn_spectrogram_viewer.py 또는 AoA 계산하는 스크립트에 get_phase_offset_to_apply() 적용

[권장 4]
src/calibration/gain_phase_table.py
→ 공통 모듈로 유지

[권장 5]
configs/calibration/*.json
→ current ref offset, gain delta table, session state 분리 저장

[나중 6]
dashboard/viewer overlay
→ 현재 보정값과 quality 표시



PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_linked_g25_d3m_vertical_cf2450 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 4000 \
  --skip-inference


  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_linked_g25_d3m_vertical_cf2460 \
  --center-freq 2460000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 4000 \
  --skip-inference


  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_linked_g25_d3m_vertical_cf2465 \
  --center-freq 2465000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 2.0 \
  --valid-signal-ratio 5.0 \
  --max-saved 150 \
  --until-max-saved \
  --max-total-blocks 4000 \
  --skip-inference


PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_g25_d3m_vertical_cf2450 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-saved 100 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --start-discard-sec 0.0 \
  --skip-inference



  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_g25_d3m_vertical_cf2460 \
  --center-freq 2460000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-saved 100 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --start-discard-sec 0.0 \
  --skip-inference



  PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label background_g25_d3m_vertical_cf2465 \
  --center-freq 2465000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy no_signal_only \
  --max-background-signal-ratio 1.5 \
  --max-saved 100 \
  --until-max-saved \
  --max-total-blocks 5000 \
  --start-discard-sec 0.0 \
  --skip-inference

1. Blink 2450MHz

조종기 꺼둔 상태에서 명령어 먼저 실행 → 저장 대기 시작되면 조종기 ON.

PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_controller_blink_ref_g25_d1m_vertical_cf2450 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 1.3 \
  --valid-signal-ratio 2.5 \
  --max-saved 50 \
  --until-max-saved \
  --max-total-blocks 1500 \
  --start-discard-sec 0.0 \
  --skip-inference

3. Blink 2465MHz
PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_controller_blink_ref_g25_d1m_vertical_cf2465 \
  --center-freq 2465000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-signal-ratio 1.3 \
  --valid-signal-ratio 2.5 \
  --max-saved 50 \
  --until-max-saved \
  --max-total-blocks 1500 \
  --start-discard-sec 0.0 \
  --skip-inference

PYTHONPATH=. python scripts/capture_rf4_fixed_freq_dataset.py \
  --label drone_controller_blink_rawp99gt4p5_g25_d3m_vertical_cf2450 \
  --center-freq 2450000000 \
  --gain 25 \
  --rx-index 0 \
  --save-policy weak_or_valid \
  --min-raw-p99 4.5 \
  --max-saved 50 \
  --until-max-saved \
  --max-total-blocks 2000 \
  --start-discard-sec 0.0 \
  --skip-inference





  PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_yaml_drone_aoa.py \
  --decision-mode hybrid \
  --gain 40 \
  --center-freq 2450000000 \
  --blocks-per-update 20 \
  --update-interval-sec 1.0


  PYTHONPATH=. python scripts/live_rf_viewer_drone_aoa.py --mode full

  aoa_gate:
  enabled: true   #aoa게이트 전원
  require_voting_confirmed: false  #true하면 cnn게이팅들어감
  require_current_drone: false
  min_current_confidence: 0.00
  min_coherence: 0.90
  display_only_valid: true
  show_skip_reason: true


   PYTHONPATH=. python -m src.runtime.cli


  

평소/현장 기본 실행:
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --gain 35

YAML까지 gain=35로 맞춘 뒤:
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py

플루토 없는 테스트:
PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment.py --source-type sim --cnn-backend dummy --cnn-dummy-class-name Drone --disable-raw-gate



PYTHONPATH=. python scripts/experimental/live_aoa_sector_experiment_capture.py \
  --gain 35 \
  --distance-m 3 \
  --true-angle-deg 0 \
  --capture-trusted-n 30 \
  --memo "CENTER_3m_gain35"

aoa secter

PYTHONPATH=. python scripts/experimental/live_aoa_sector_dashboard.py \
  --source-type sdr \
  --gain 35 \
  --center-freq 2450000000 \
  --target-fps 5


  cd ~/projects/rf-drone-detection-capstone



PYTHONPATH=. python scripts/experimental/live_aoa_sector_dashboard.py \
  --source-type sim \
  --cnn-backend dummy \
  --cnn-dummy-class-name Drone \
  --cnn-dummy-confidence 1.0 \
  --disable-raw-gate \
  --target-fps 5

기존데이터로 리플레이
  PYTHONPATH=. python scripts/experimental/replay_sector_profile_dashboard.py \
  --fps 2 \
  --only-trusted

  