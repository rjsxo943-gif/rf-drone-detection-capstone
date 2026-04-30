from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import stft
from PIL import Image


def load_iq_npy(path: Path) -> np.ndarray:
    iq = np.load(path)

    if iq.ndim == 2:
        # dual-channel이면 ch0만 CNN 입력으로 사용
        iq = iq[0]

    if not np.iscomplexobj(iq):
        raise ValueError(f"{path} is not complex IQ data")

    return iq.astype(np.complex64)


def iq_to_spectrogram(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    _, _, zxx = stft(
        iq,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    spec = np.abs(zxx)
    spec = np.fft.fftshift(spec, axes=0)

    spec_db = 20 * np.log10(spec + 1e-8)

    lo, hi = np.percentile(spec_db, [5, 95])
    spec_db = np.clip(spec_db, lo, hi)

    spec_norm = (spec_db - lo) / (hi - lo + 1e-8)
    return spec_norm.astype(np.float32)


def resize_spectrogram(spec: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((spec * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    sample_rate: float,
    nperseg: int,
    noverlap: int,
    image_size: int,
    save_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.npy"))

    if not files:
        print(f"No .npy files found in {input_dir}")
        return

    count = 0

    for path in files:
        label = path.parent.name

        label_npy_dir = output_dir / "npy" / label
        label_png_dir = output_dir / "png" / label

        label_npy_dir.mkdir(parents=True, exist_ok=True)

        if save_png:
            label_png_dir.mkdir(parents=True, exist_ok=True)

        try:
            iq = load_iq_npy(path)

            spec = iq_to_spectrogram(
                iq=iq,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
            )

            spec_resized = resize_spectrogram(spec, image_size)

            out_name = path.stem + "_spec.npy"
            np.save(label_npy_dir / out_name, spec_resized)

            if save_png:
                png_name = path.stem + "_spec.png"
                Image.fromarray((spec_resized * 255).astype(np.uint8)).save(
                    label_png_dir / png_name
                )

            count += 1
            print(f"[OK] {path} -> label={label}, shape={spec_resized.shape}")

        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    print()
    print(f"Done. Processed {count} files.")
    print(f"Saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw IQ .npy files into STFT spectrogram dataset."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw_iq"),
        help="Input directory containing label subfolders with .npy IQ files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spectrogram"),
        help="Output directory for processed spectrogram dataset.",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=5_000_000,
        help="Sample rate in Hz.",
    )

    parser.add_argument(
        "--nperseg",
        type=int,
        default=512,
        help="STFT window size.",
    )

    parser.add_argument(
        "--noverlap",
        type=int,
        default=384,
        help="STFT overlap size. hop = nperseg - noverlap.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Output spectrogram image size. Default: 64x64.",
    )

    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save spectrogram as PNG images.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        image_size=args.image_size,
        save_png=args.save_png,
    )


if __name__ == "__main__":
    main()

""" PYTHONPATH=. python scripts/run_pipeline.py
=== Pipeline Result ===
num_samples: 16384
num_frames: 31
num_detections: 11
noise_floor: 1.633805274963379
threshold: 8.169026374816895
detection_ratio: 0.3548387096774194

saved to: /home/rjsxo342/projects/rf-drone-detection-capstone/outputs/runs/latest"""