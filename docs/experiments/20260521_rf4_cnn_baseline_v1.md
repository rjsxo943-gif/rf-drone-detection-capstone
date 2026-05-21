# 2026-05-21 RF4 CNN Baseline v1 Training Result

## 1. Experiment Purpose

This experiment trains a 4-class CNN classifier for RF spectrogram classification.

Target classes:

- Background
- WiFi
- Bluetooth
- Drone-like

The main purpose was to verify whether the newly collected real drone RF data can be separated from WiFi, Bluetooth, and Background signals.

---

## 2. Dataset and Split

Dataset root:

- data/processed/cnn_capture

Split directory:

- data/processed/cnn_capture/splits/rf4_random_v1

Training samples:

| Split | Samples |
|---|---:|
| Train | 729 |
| Validation | 154 |
| Test | 161 |

Note: The current split is not perfectly balanced in the test set. Background has fewer samples than the other classes.

---

## 3. Model Settings

| Parameter | Value |
|---|---:|
| Model | RF3SmallCNN used as RF4 classifier |
| Input shape | 1 x 128 x 509 |
| Number of classes | 4 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Device | CPU |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

Class order:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | WiFi |
| 2 | Bluetooth |
| 3 | Drone-like |

---

## 4. Training Result Summary

Final test result:

| Metric | Value |
|---|---:|
| Test loss | 0.2509 |
| Test accuracy | 0.9627 |

Best validation accuracy reached 1.0000 during training.

---

## 5. Per-Class Test Result

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Background | 0.9583 | 1.0000 | 0.9787 | 23 |
| WiFi | 0.9000 | 0.9783 | 0.9375 | 46 |
| Bluetooth | 1.0000 | 0.8913 | 0.9425 | 46 |
| Drone-like | 1.0000 | 1.0000 | 1.0000 | 46 |

Overall accuracy:

- 96.27%

---

## 6. Confusion Matrix

Rows are true labels, columns are predicted labels.

| True \ Pred | Background | WiFi | Bluetooth | Drone-like |
|---|---:|---:|---:|---:|
| Background | 23 | 0 | 0 | 0 |
| WiFi | 1 | 45 | 0 | 0 |
| Bluetooth | 0 | 5 | 41 | 0 |
| Drone-like | 0 | 0 | 0 | 46 |

Main observation:

- Drone-like was not confused with WiFi, Bluetooth, or Background in this test split.
- Most errors occurred between Bluetooth and WiFi.
- WiFi was occasionally confused with Background.

---

## 7. Interpretation

This RF4 baseline result is successful as a first experiment.

The model achieved high overall test accuracy, and the Drone-like class achieved perfect precision, recall, and F1-score on the current test split.

However, the result should be interpreted carefully because the drone data was collected under limited conditions:

- Same day
- Same location
- Same distance
- Same center frequency
- Same SDR gain
- Same toy drone

Therefore, the model may have learned not only drone-specific RF patterns but also some environment-specific characteristics.

---

## 8. Output Files

Training output directory:

- outputs/ml/rf4_cnn_baseline_v1

Generated files:

- best_model.pt
- summary.json
- history.json
- classification_report.txt
- confusion_matrix.csv
- confusion_matrix.png

---

## 9. Known Issue

The classification report title still says RF 3-Class Classification Report.

This is a text-label issue only. The actual experiment used 4 classes.

TODO:

- Change report title from RF 3-Class to RF 4-Class in the reporting utility.

---

## 10. Next Steps

Recommended next steps:

1. Fix the report title from RF3 to RF4.
2. Plot training history.
3. Inspect confusion_matrix.png.
4. Collect additional Background samples if needed.
5. Collect drone data under different distance, location, and gain conditions.
6. Run another RF4 experiment to check generalization.

Suggested next experiment name:

- outputs/ml/rf4_cnn_baseline_v2
