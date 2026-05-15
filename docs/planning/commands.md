predict_rf3_file.py
    PYTHONPATH=. python scripts/ml/predict_rf3_file.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --input "data/processed/cnn_capture/Wifi/home_wifihot_ch6_on_2437_gain10_1m/part3/selected_meaningful/0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy" \
  --true-label WiFi

  PYTHONPATH=. python scripts/ml/predict_rf3_file.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --input "data/processed/cnn_capture/Wifi/home_wifihot_ch6_on_2437_gain10_1m/part3/selected_meaningful/0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy" \
  --true-label WiFi \
  --threshold 0.70

  PYTHONPATH=. python scripts/ml/predict_rf3_file.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --input "data/processed/cnn_capture/Wifi/home_wifihot_ch6_on_2437_gain10_1m/part3/selected_meaningful/0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy" \
  --true-label WiFi \
  --save-json outputs/ml/rf3_cnn_baseline_v1/single_predict_0352.json \
  --save-png outputs/ml/rf3_cnn_baseline_v1/single_predict_0352.png

