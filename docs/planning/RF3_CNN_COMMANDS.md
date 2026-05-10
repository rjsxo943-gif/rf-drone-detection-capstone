# RF 3-Class CNN 실행 명령어 정리

이 문서는 RF 드론 탐지 졸작 프로젝트에서 `Background / Bluetooth / WiFi` 3분류 CNN을 학습하고 평가할 때 사용하는 명령어를 순서대로 정리한 것이다.

현재 RF3 모델은 드론 탐지 모델이 아니라, 2.4GHz 대역의 비드론 RF 신호를 분류하는 baseline 모델이다.

---

## 0. 터미널 처음 켰을 때 항상 먼저 할 것

프로젝트 폴더로 이동하고 가상환경을 켠다.

```bash
cd ~/projects/rf-drone-detection-capstone
source .venv/bin/activate
```

현재 파이썬 위치 확인:

```bash
which python
```

정상이라면 대략 이렇게 나와야 한다.

```text
/home/rjsxo342/projects/rf-drone-detection-capstone/.venv/bin/python
```

---

## 1. 데이터 폴더 구조 확인

현재 RF3 데이터셋 루트는 다음 위치를 기준으로 한다.

```text
data/processed/cnn_capture
```

최상위 라벨 폴더는 다음 3개여야 한다.

```text
Background
Bluetooth
Wifi 또는 WIFI
```

구조 확인:

```bash
tree -L 4 data/processed/cnn_capture
```

라벨 폴더 위치 확인:

```bash
find data/processed/cnn_capture -maxdepth 2 -type d \( -iname "wifi" -o -iname "bluetooth" -o -iname "background" \)
```

---

## 2. 클래스별 `.npy` 개수 확인

전체 `.npy` 개수 확인:

```bash
find data/processed/cnn_capture -type f -name "*.npy" | wc -l
```

클래스별 `.npy` 개수 확인:

```bash
find data/processed/cnn_capture/Background -type f -name "*.npy" | wc -l
find data/processed/cnn_capture/Bluetooth -type f -name "*.npy" | wc -l
find data/processed/cnn_capture/Wifi -type f -name "*.npy" | wc -l
```

만약 WiFi 폴더 이름이 `WIFI`라면 마지막 명령어는 이렇게 쓴다.

```bash
find data/processed/cnn_capture/WIFI -type f -name "*.npy" | wc -l
```

---

## 3. Manifest 생성

Manifest는 학습에 사용할 `.npy` 파일 경로, 라벨, 세션 정보를 정리한 CSV 목록표다.

실행:

```bash
PYTHONPATH=. python scripts/ml/build_rf3_manifest.py \
  --root data/processed/cnn_capture
```

생성되는 파일:

```text
data/processed/cnn_capture/manifests/manifest_all.csv
data/processed/cnn_capture/manifests/manifest_rf3_balanced_v1.csv
```

정상 출력 예시:

```text
=== manifest_all ===
total: 2572

[label counts]
Background: 1500
Bluetooth: 554
WiFi: 518

=== manifest_rf3_balanced_v1 ===
total: 1500

[label counts]
Background: 500
Bluetooth: 500
WiFi: 500
```

---

## 4. Train / Val / Test split 생성

Balanced manifest 1500개를 랜덤으로 나누는 단계다.

기본 비율:

```text
train: 70%
val: 15%
test: 15%
```

실행:

```bash
PYTHONPATH=. python scripts/ml/split_rf3_manifest.py
```

생성되는 파일:

```text
data/processed/cnn_capture/splits/rf3_random_v1/train.csv
data/processed/cnn_capture/splits/rf3_random_v1/val.csv
data/processed/cnn_capture/splits/rf3_random_v1/test.csv
data/processed/cnn_capture/splits/rf3_random_v1/split_manifest.csv
```

정상 출력 예시:

```text
=== train ===
total: 1050
Background: 350
Bluetooth: 350
WiFi: 350

=== val ===
total: 225
Background: 75
Bluetooth: 75
WiFi: 75

=== test ===
total: 225
Background: 75
Bluetooth: 75
WiFi: 75
```

---

## 5. Spectrogram shape 확인

현재 `.npy`가 IQ 원본인지, 이미 spectrogram인지 확인하는 단계다.

실행:

```bash
python - <<'PY'
import csv
import numpy as np
from collections import Counter

manifest = "data/processed/cnn_capture/manifests/manifest_rf3_balanced_v1.csv"

shapes = Counter()
dtypes = Counter()

with open(manifest, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = np.load(row["filepath"])
        shapes[str(x.shape)] += 1
        dtypes[str(x.dtype)] += 1

print("shapes:")
for k, v in shapes.items():
    print(k, v)

print("\ndtypes:")
for k, v in dtypes.items():
    print(k, v)
PY
```

정상 출력:

```text
shapes:
(128, 509) 1500

dtypes:
float32 1500
```

이 결과가 나오면 STFT 변환 없이 바로 CNN 학습에 사용할 수 있다.

---

## 6. 코드 컴파일 확인

학습 전에 문법 오류가 없는지 확인한다.

```bash
PYTHONPATH=. python -m compileall src/ml scripts/ml
```

에러 없이 끝나면 정상이다.

---

## 7. RF3 CNN 학습: 대표 모델, lr=0.001

대표 baseline 모델 학습 명령어다.

```bash
PYTHONPATH=. python scripts/ml/train_rf3_cnn.py \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --out-dir outputs/ml/rf3_cnn_baseline_v1
```

결과 저장 위치:

```text
outputs/ml/rf3_cnn_baseline_v1/
```

주요 산출물:

```text
best_model.pt
summary.json
history.json
classification_report.txt
confusion_matrix.csv
confusion_matrix.png
```

현재 대표 모델 결과:

```text
test accuracy: 98.67%
test loss: 0.0880
오분류: 3개
```

---

## 8. RF3 CNN 학습: 보조 실험, lr=0.0003

학습률을 낮춘 비교 실험이다.

```bash
PYTHONPATH=. python scripts/ml/train_rf3_cnn.py \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0003 \
  --out-dir outputs/ml/rf3_cnn_baseline_lr3e4
```

결과 저장 위치:

```text
outputs/ml/rf3_cnn_baseline_lr3e4/
```

현재 보조 실험 결과:

```text
test accuracy: 97.33%
test loss: 0.2213
오분류: 6개
```

---

## 9. 오분류 파일 찾기: 대표 모델

대표 모델에서 어떤 파일이 틀렸는지 찾는 명령어다.

```bash
PYTHONPATH=. python scripts/ml/find_rf3_misclassified.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --csv data/processed/cnn_capture/splits/rf3_random_v1/test.csv \
  --out-dir outputs/ml/rf3_cnn_baseline_v1/error_analysis
```

생성되는 파일:

```text
outputs/ml/rf3_cnn_baseline_v1/error_analysis/predictions.csv
outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified.csv
outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified_png/
```

오분류 이미지 폴더 열기:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified_png
```

---

## 10. 오분류 파일 찾기: lr=0.0003 모델

```bash
PYTHONPATH=. python scripts/ml/find_rf3_misclassified.py \
  --model outputs/ml/rf3_cnn_baseline_lr3e4/best_model.pt \
  --csv data/processed/cnn_capture/splits/rf3_random_v1/test.csv \
  --out-dir outputs/ml/rf3_cnn_baseline_lr3e4/error_analysis
```

오분류 이미지 폴더 열기:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_lr3e4/error_analysis/misclassified_png
```

---

## 11. Ambiguous 후보 목록 보관

오분류 중 사람이 봐도 애매한 샘플은 원본에서 삭제하지 않고 review 목록으로만 보관한다.

review 폴더 만들기:

```bash
mkdir -p data/processed/cnn_capture/review
```

대표 모델 오분류 목록을 ambiguous 후보로 복사:

```bash
cp outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified.csv \
   data/processed/cnn_capture/review/ambiguous_candidates_rf3_v1.csv
```

오분류 PNG도 review 폴더에 복사:

```bash
mkdir -p data/processed/cnn_capture/review/ambiguous_png_rf3_v1

cp outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified_png/*.png \
   data/processed/cnn_capture/review/ambiguous_png_rf3_v1/
```

원칙:

```text
원본 .npy 이동 금지
원본 .npy 삭제 금지
CSV 목록과 PNG로만 후보 관리
```

---

## 12. 학습 곡선 그래프 만들기: 대표 모델

`history.json`을 이용해 loss / accuracy 그래프를 만든다.

```bash
PYTHONPATH=. python scripts/ml/plot_rf3_history.py \
  --history outputs/ml/rf3_cnn_baseline_v1/history.json \
  --out-dir outputs/ml/rf3_cnn_baseline_v1/plots \
  --title "RF3 CNN Baseline v1 (lr=0.001)"
```

결과:

```text
outputs/ml/rf3_cnn_baseline_v1/plots/loss_curve.png
outputs/ml/rf3_cnn_baseline_v1/plots/accuracy_curve.png
```

그래프 폴더 열기:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_v1/plots
```

---

## 13. 학습 곡선 그래프 만들기: lr=0.0003 모델

```bash
PYTHONPATH=. python scripts/ml/plot_rf3_history.py \
  --history outputs/ml/rf3_cnn_baseline_lr3e4/history.json \
  --out-dir outputs/ml/rf3_cnn_baseline_lr3e4/plots \
  --title "RF3 CNN Baseline (lr=0.0003)"
```

결과:

```text
outputs/ml/rf3_cnn_baseline_lr3e4/plots/loss_curve.png
outputs/ml/rf3_cnn_baseline_lr3e4/plots/accuracy_curve.png
```

그래프 폴더 열기:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_lr3e4/plots
```

---

## 14. 단일 `.npy` 파일 예측

학습된 모델로 spectrogram `.npy` 하나를 예측한다.

예시: 오분류된 WiFi 파일 예측

```bash
PYTHONPATH=. python scripts/ml/predict_rf3_file.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --input "data/processed/cnn_capture/Wifi/home_wifihot_ch6_on_2437_gain10_1m/part3/selected_meaningful/0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy" \
  --true-label WiFi
```

PNG와 JSON까지 저장:

```bash
PYTHONPATH=. python scripts/ml/predict_rf3_file.py \
  --model outputs/ml/rf3_cnn_baseline_v1/best_model.pt \
  --input "data/processed/cnn_capture/Wifi/home_wifihot_ch6_on_2437_gain10_1m/part3/selected_meaningful/0352__20260509_163543__home_wifihot_ch6_on_2437_gain10_1m.npy" \
  --true-label WiFi \
  --save-json outputs/ml/rf3_cnn_baseline_v1/single_predict_0352.json \
  --save-png outputs/ml/rf3_cnn_baseline_v1/single_predict_0352.png
```

---

## 15. 결과 폴더 열기

대표 모델 결과:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_v1
```

보조 실험 결과:

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_lr3e4
```

review 폴더:

```bash
explorer.exe data/processed/cnn_capture/review
```

---

## 16. 실험 결과 문서 위치 추천

사람이 읽는 실험 기록은 `docs/experiments`에 둔다.

```text
docs/experiments/rf3_cnn_baseline_v1.md
```

실제 모델, 그래프, 결과 파일은 `outputs/ml`에 둔다.

```text
outputs/ml/rf3_cnn_baseline_v1/
outputs/ml/rf3_cnn_baseline_lr3e4/
```

---

## 17. 현재 대표 모델 정리

```text
대표 모델: rf3_cnn_baseline_v1
learning rate: 0.001
test accuracy: 98.67%
test samples: 225개
오분류: 3개
오분류 유형:
- Background → Bluetooth 1개
- WiFi → Bluetooth 2개
```

대표 모델 파일:

```text
outputs/ml/rf3_cnn_baseline_v1/best_model.pt
```

---

## 18. 현재 보조 실험 정리

```text
보조 실험: rf3_cnn_baseline_lr3e4
learning rate: 0.0003
test accuracy: 97.33%
test samples: 225개
오분류: 6개
```

보조 실험 모델 파일:

```text
outputs/ml/rf3_cnn_baseline_lr3e4/best_model.pt
```

---

## 19. 자주 헷갈리는 것 정리

### Q1. `.npy`를 직접 열 수 없는데 어떻게 보나?

오분류 분석 스크립트가 PNG로 변환해준다.

```bash
explorer.exe outputs/ml/rf3_cnn_baseline_v1/error_analysis/misclassified_png
```

### Q2. Background 1500장을 직접 줄여야 하나?

아니다. 원본은 그대로 두고 `manifest_rf3_balanced_v1.csv`에서 500개만 사용한다.

### Q3. 파일명 규칙이 달라도 괜찮나?

괜찮다. 라벨은 파일명이 아니라 상위 폴더명으로 판단한다.

### Q4. 정확도 98.67%를 최종 현실 성능이라고 말해도 되나?

아니다. 현재는 random split 기반 내부 데이터셋 성능이다. 새로운 장소, 거리, 주파수 조건에 대한 일반화 성능은 추가 데이터와 session split 평가가 필요하다.

### Q5. 드론 데이터 없이 이 모델은 무슨 의미인가?

현재 모델은 드론 탐지기가 아니라 `Background / Bluetooth / WiFi`를 구분하는 비드론 RF baseline 분류기다. 나중에 드론 데이터가 확보되면 `Drone-like` 클래스를 추가해 4분류 모델로 확장할 수 있다.
