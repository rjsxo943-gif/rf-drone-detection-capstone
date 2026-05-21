# 2026-05-21 RF 4-Class Dataset Summary

## 1. Purpose

This dataset was built for 2.4 GHz RF signal classification using CNN-based spectrogram input.

Target classes:

- Background
- WiFi
- Bluetooth
- Drone-like

The main goal is to extend the previous RF3 dataset into an RF4 dataset by adding real drone RF data.

---

## 2. Dataset Root

```text
data/processed/cnn_capture
```

---

## 3. Class Counts Before Balancing

| Class | Samples |
|---|---:|
| Background | 150 |
| WiFi | 300 |
| Bluetooth | 298 |
| Drone-like | 300 |

Total raw integrated samples:

```text
1048
```

---

## 4. Drone Data Collection Conditions

Drone data was collected using a small 2.4 GHz toy drone and PlutoSDR.

### SDR Settings

| Parameter | Value |
|---|---:|
| Center frequency | 2437 MHz |
| Sample rate | 5 MSPS |
| RF bandwidth | 5 MHz |
| Gain | 20 dB |
| Distance | 50 cm |
| Block size | 16384 samples |
| STFT nperseg | 128 |
| STFT noverlap | 96 |
| STFT nfft | 128 |
| Window | Hann |

### Drone States

| State | Description |
|---|---|
| connected_idle | Drone and controller ON, no stick operation |
| motor_ground | Drone on the ground, propellers rotating |
| control_move_ground | Drone on the ground, right stick operated |

The drone was not required to maintain stable hover.

Ground-based operation was used to improve safety and repeatability.

---

## 5. Drone Dataset Structure

```text
data/processed/cnn_capture/20260521_drone_cf2437_g20p0_d50/
├── connected_idle
├── motor_ground
└── control_move_ground
```

The collected drone samples were integrated into:

```text
data/processed/cnn_capture/Drone-like
```

---

## 6. Final Learning Folders

```text
data/processed/cnn_capture/
├── Background
├── Wifi
├── Bluetooth
└── Drone-like
```

Note: the folder name is `Wifi`, but the manifest label is normalized to `WiFi`.

---

## 7. Balanced Manifest

Balanced manifest path:

```text
data/processed/cnn_capture/manifests/manifest_rf4_balanced_v1.csv
```

Balanced sample count:

| Class | Samples |
|---|---:|
| Background | 150 |
| WiFi | 150 |
| Bluetooth | 150 |
| Drone-like | 150 |

Total:

```text
600 samples
```

---

## 8. Train / Validation / Test Split

Split directory:

```text
data/processed/cnn_capture/splits/rf4_random_v1
```

| Split | Total | Background | WiFi | Bluetooth | Drone-like |
|---|---:|---:|---:|---:|---:|
| Train | 420 | 105 | 105 | 105 | 105 |
| Validation | 88 | 22 | 22 | 22 | 22 |
| Test | 92 | 23 | 23 | 23 | 23 |

---

## 9. Notes

- The RF4 dataset is balanced to 150 samples per class because Background has the smallest number of samples.
- Drone-like data was collected from real drone-controller RF activity.
- Hovering was not used as a required condition because the drone was difficult to stabilize.
- Ground-based motor and control operation were selected for repeatability and safety.
- The saved `.npy` files are CNN-ready spectrogram data, not raw IQ data.
- Existing RF3 data was reused and extended with real drone data.

---

## 10. Next Step

Train a 4-class CNN model using:

```text
Background / WiFi / Bluetooth / Drone-like
```

Required training changes:

- Change number of classes from 3 to 4.
- Add `Drone-like` to class names.
- Use RF4 split files.
- Save results under a new experiment name:

```text
outputs/ml/rf4_cnn_baseline_v1
```
