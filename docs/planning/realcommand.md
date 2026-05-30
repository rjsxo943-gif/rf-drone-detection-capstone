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