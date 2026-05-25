from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

SRC_ROOT = Path("data/processed/cnn_capture")
DST_ROOT = Path("data/processed/cnn_capture_canonical01")

CLASS_DIRS = ["Background", "Wifi", "Bluetooth", "Drone-like"]

VMIN = -40.0
VMAX = 40.0
EPS = 1e-8


def to_canonical01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # 이미 0~1 형태면 그대로 사용
    if xmin >= -EPS and xmax <= 1.0 + EPS:
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    # dB scale이면 고정 범위 기준으로 0~1 변환
    y = np.clip(x, VMIN, VMAX)
    y = (y - VMIN) / (VMAX - VMIN)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def main() -> None:
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    for class_dir in CLASS_DIRS:
        src_dir = SRC_ROOT / class_dir
        dst_dir = DST_ROOT / class_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for src in sorted(src_dir.glob("*.npy")):
            x = np.load(src)
            y = to_canonical01(x)
            np.save(dst_dir / src.name, y)
            count += 1

        print(f"{class_dir}: {count}")

    print(f"[OK] saved: {DST_ROOT}")


if __name__ == "__main__":
    main()
