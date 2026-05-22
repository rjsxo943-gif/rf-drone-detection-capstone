# 2026-05-21 RF4 Drone Detection Pipeline Progress Report

## 1. 작업 개요

오늘 작업에서는 기존 RF3 CNN 분류 파이프라인을 실제 드론 RF 데이터를 포함한 RF4 분류 구조로 확장하였다. 기존 클래스는 Background, WiFi, Bluetooth였고, 오늘 Drone-like 클래스를 새로 추가하였다.

현재 RF4 모델의 분류 클래스는 Background, WiFi, Bluetooth, Drone-like이다. 이번 작업의 핵심 목적은 실제 드론-조종기 RF 신호를 수집하고, 이를 CNN 학습 데이터셋에 통합한 뒤, 단일 파일 예측과 runtime 추론까지 연결 가능한 구조를 만드는 것이었다.

## 2. 드론 RF 데이터 수집

소형 2.4GHz 드론과 조종기를 이용하여 실제 Drone-like 데이터를 수집하였다. 수집 조건은 center frequency 2437 MHz, sample rate 5 MSPS, RF bandwidth 5 MHz, gain 20, distance 50 cm, block size 16384, STFT nperseg 128, noverlap 96, nfft 128, spectrogram shape 128 x 509이다.

드론은 안정적인 hover가 어려웠기 때문에, 안전성과 재현성을 위해 바닥 기반 조건으로 데이터를 수집하였다. 수집 상태는 connected_idle, motor_ground, control_move_ground 세 가지로 나누었다. 최종적으로 Drone-like 데이터 300장을 확보하였다.

## 3. RF4 데이터셋 구성

최종 통합 데이터셋 폴더는 data/processed/cnn_capture/Background, data/processed/cnn_capture/Wifi, data/processed/cnn_capture/Bluetooth, data/processed/cnn_capture/Drone-like로 구성하였다. 실제 폴더명은 Wifi이지만, manifest에서는 label을 WiFi로 정규화하였다.

초기에 Background 데이터가 150장만 반영되는 문제가 있었다. 원인은 Background 세션 2개의 파일명이 겹쳐 통합 폴더로 복사할 때 덮어쓰기 되었기 때문이다. 이후 세션명을 파일명에 포함하여 Background 통합 폴더를 다시 구성하였다.

최종 데이터 수는 Background 300, WiFi 300, Bluetooth 298, Drone-like 300이다. Balanced dataset은 최소 클래스인 Bluetooth 기준으로 클래스당 298개씩, 총 1192개로 구성하였다.

## 4. Manifest 및 Split 생성

RF4 데이터셋 관리를 위해 scripts/ml/build_rf4_manifest.py와 scripts/ml/split_rf4_manifest.py를 추가하였다. 생성된 주요 파일은 manifest_rf4_balanced_v1.csv와 rf4_random_v1/train.csv, val.csv, test.csv이다.

최종 split은 train 832개, validation 176개, test 184개이며, 각 클래스별로 train 208개, validation 44개, test 46개씩 균등하게 분배되었다.

## 5. RF4 CNN 학습 결과

RF4 학습 스크립트 scripts/ml/train_rf4_cnn_baseline.py를 추가하였다. 대표 모델은 outputs/ml/rf4_cnn_baseline_v2에 저장하였다. 모델은 RF3SmallCNN 구조를 재사용하였고, class 수를 4개로 확장하였다. 입력 shape는 1 x 128 x 509이며, epoch 20, batch size 32, learning rate 1e-3, Adam optimizer, CrossEntropyLoss 조건으로 CPU에서 학습하였다.

RF4 v2 최종 test accuracy는 99.46%, test loss는 0.0586, test sample 수는 184개이다. Class별 결과는 Background precision 0.9787, recall 1.0000, f1 0.9892, WiFi precision 1.0000, recall 1.0000, f1 1.0000, Bluetooth precision 1.0000, recall 0.9783, f1 0.9890, Drone-like precision 1.0000, recall 1.0000, f1 1.0000이다.

Confusion matrix 기준 오분류는 Bluetooth sample 1개가 Background로 분류된 경우뿐이었다. Drone-like class는 test set에서 46개 중 46개 모두 정확히 분류되었다.

## 6. 단일 파일 예측 스크립트 추가

RF4 모델을 이용해 spectrogram npy 파일 하나를 바로 예측할 수 있는 scripts/ml/predict_rf4_file.py를 작성하였다. Drone-like sample 테스트 결과 pred_class는 Drone-like, confidence는 0.9998, final_class는 Drone-like로 출력되었다.

## 7. RF4 Inference 모듈 추가

Runtime에서 재사용하기 위해 src/ml/rf4_inference.py를 작성하였다. 주요 클래스는 RF4Classifier와 RF4Result이다. RF4Classifier는 best_model.pt를 로드하고, checkpoint 내부 mean/std를 이용해 입력 spectrogram을 정규화한 뒤 RF4 추론을 수행한다.

반환 정보는 class_id, class_name, confidence, final_class, probabilities, applied_threshold, general_threshold, drone_threshold이다.

## 8. Threshold 정책 개선

처음에는 모든 클래스에 동일한 threshold를 적용하였다. 그러나 Bluetooth sample 중 confidence가 낮은 경우가 있어 Unknown으로 빠지는 경우가 많았다. 따라서 일반 클래스와 Drone-like의 threshold를 분리하였다.

현재 정책은 general_threshold 0.50, drone_threshold 0.70이다. Background, WiFi, Bluetooth는 confidence가 0.50 이상이면 해당 class로 인정하고, Drone-like는 confidence가 0.70 이상일 때만 Drone-like로 인정한다. 그 외는 Unknown으로 처리한다.

Smoke test 결과는 Background 50/50, WiFi 50/50, Bluetooth 42/50, Drone-like 50/50, 전체 192/200, accuracy 96.00%였다. Drone-like는 더 높은 threshold인 0.70을 적용했음에도 50개 모두 정확히 분류되었다.

## 9. Runtime CLI 연결

RF4 추론을 runtime CLI에 연결하였다. 새로 추가한 파일은 src/runtime/rf4_actions.py이며, src/runtime/cli.py에 r 메뉴를 추가하였다.

현재 runtime CLI 메뉴는 c calibration status, n noise calibration, p phase/gain calibration, s CNN dataset capture, r RF4 single block inference, q quit로 구성되어 있다. 메뉴 출력까지 정상 확인하였고, 실제 SDR에서 r 메뉴를 실행하여 RF4 결과를 보는 실험은 내일 진행할 예정이다.

## 10. 현재까지의 핵심 성과

오늘의 핵심 성과는 실제 드론 RF 데이터를 수집하여 Drone-like class를 구성한 것, 기존 RF3 데이터셋을 RF4 데이터셋으로 확장한 것, 클래스당 298개 기준의 balanced dataset을 구성한 것, RF4 CNN baseline v2에서 test accuracy 99.46%를 달성한 것이다.

또한 Drone-like class는 test set에서 precision, recall, f1-score 모두 1.0000을 기록하였다. 단일 파일 예측 스크립트, RF4Classifier inference module, threshold 정책, smoke test, runtime CLI 메뉴 연결까지 완료하였다.

## 11. 한계점

현재 결과는 매우 좋지만 random split 기반이므로 같은 수집 조건에서 얻어진 유사 sample이 train과 test에 함께 포함되었을 수 있다. Drone-like 데이터는 같은 날, 같은 장소, 같은 거리, 같은 gain, 같은 중심주파수에서 수집되었으므로 실제 일반화 성능은 아직 검증되지 않았다.

또한 사용한 드론은 소형 toy drone 한 종류이므로 다른 드론의 RF 특성에는 일반화되지 않을 수 있다. 저장된 학습 데이터는 raw IQ가 아니라 CNN-ready spectrogram이므로 추후 STFT 파라미터를 바꾸려면 raw IQ 저장 구조가 필요하다. Bluetooth와 WiFi 또는 Background 사이에는 여전히 일부 혼동이 발생한다.

## 12. 내일 할 일

내일은 runtime CLI에서 r 메뉴를 이용해 RF4 single block inference를 실제 실행한다. 아무 신호 없음, WiFi, Drone-like 조건별 runtime 결과를 확인한다. 이후 RF4 runtime 결과를 CSV 또는 JSON으로 저장하는 기능을 추가하고, scan precision 단계에 RF4Classifier를 연결한다.

추가로 다른 거리와 gain 조건에서 Drone-like 데이터를 더 수집하고, session-based split 또는 leave-session-out 평가를 설계한다. raw IQ 저장 옵션도 검토한다.

## 13. 결론

오늘 작업을 통해 RF 기반 드론 탐지 프로젝트는 데이터 수집, 학습, 추론, runtime 연결의 전체 흐름을 거의 완성하였다. 실제 드론-조종기 RF 데이터를 수집하여 Drone-like 클래스를 만들고, 이를 기존 Background, WiFi, Bluetooth 데이터와 통합하여 RF4 CNN 모델을 학습한 점이 가장 큰 성과이다.

RF4 CNN baseline v2는 test set에서 99.46%의 정확도를 기록하였고, Drone-like 클래스는 46개 test sample을 모두 정확히 분류하였다. 또한 단일 파일 예측, RF4 inference module, threshold 정책, smoke test, runtime CLI 메뉴 연결까지 완료하여 실제 runtime 적용을 위한 기반을 마련하였다.

다음 단계는 실제 SDR 수신 환경에서 RF4 inference를 실행하고, scan precision 단계와 연결하여 실시간 RF 드론 탐지 흐름을 완성하는 것이다.
