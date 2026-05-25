from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

SRC_ROOT = Path("data/processed/cnn_capture")
DST_ROOT = Path("data/processed/cnn_capture_norm01")

CLASS_DIRS = [
    "Background",
    "Wifi",
    "Bluetooth",
    "Drone-like",
]

EPS = 1e-8


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    if xmax - xmin < EPS:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


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
            y = normalize_01(x)
            np.save(dst_dir / src.name, y)
            count += 1

        print(f"{class_dir}: {count}")

    print(f"[OK] saved normalized dataset to: {DST_ROOT}")


if __name__ == "__main__":
    main()
