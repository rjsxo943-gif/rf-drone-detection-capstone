from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.spectrogram import compute_stft_branch


DEFAULT_INPUT = Path("data/raw_iq/mavic/mavic_offset_m22p5MHz_5msps.npy")
DEFAULT_OUT_DIR = Path("data/processed/spectrograms/mavic_m22p5")

SAMPLE_RATE = 5_000_000
BLOCK_SIZE = 16_384

NPERSEG = 512
NOVERLAP = 384
NFFT = 512
WINDOW = "hann"


def load_iq(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.complex64)

    # 2채널 파일이면 RX0만 사용
    if x.ndim == 2:
        x = x[0]

    if x.ndim != 1:
        raise ValueError(f"지원하지 않는 IQ shape입니다: {x.shape}")

    return x


def save_spectrogram_png(spec: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.colorbar(label="normalized magnitude")
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export block-wise CNN spectrogram dataset from 5 MSPS IQ file."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input 5 MSPS IQ .npy file",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for spectrogram dataset",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="drone_like",
        help="Class label name to store in metadata",
    )

    parser.add_argument(
        "--max-blocks",
        type=int,
        default=100,
        help="Maximum number of blocks to export",
    )

    parser.add_argument(
        "--start-block",
        type=int,
        default=0,
        help="Start block index",
    )

    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG images for visual inspection",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    out_dir = args.out_dir
    npy_dir = out_dir / "npy"
    png_dir = out_dir / "png"
    meta_dir = out_dir / "meta"

    npy_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    x = load_iq(args.input)

    total_blocks = len(x) // BLOCK_SIZE
    start_block = args.start_block
    end_block = min(total_blocks, start_block + args.max_blocks)

    print("=== Export Mavic Spectrogram Dataset ===")
    print(f"input       : {args.input}")
    print(f"out_dir     : {out_dir}")
    print(f"label       : {args.label}")
    print(f"iq_shape    : {x.shape}")
    print(f"total_blocks: {total_blocks}")
    print(f"export range: {start_block} ~ {end_block - 1}")
    print(f"save_png    : {args.save_png}")
    print()

    exported = []

    for block_idx in range(start_block, end_block):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = x[start:end]

        if len(block) != BLOCK_SIZE:
            continue

        # DC offset 제거
        block = block - np.mean(block)

        branch = compute_stft_branch(
            iq_block=block,
            sample_rate=SAMPLE_RATE,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            nfft=NFFT,
            window=WINDOW,
        )

        spec = branch.cnn_spectrogram.astype(np.float32)

        base_name = f"{args.label}_block_{block_idx:06d}"
        npy_path = npy_dir / f"{base_name}.npy"
        meta_path = meta_dir / f"{base_name}.json"

        np.save(npy_path, spec)

        metadata = {
            "label": args.label,
            "source_iq": str(args.input),
            "block_index": int(block_idx),
            "sample_start": int(start),
            "sample_end": int(end),
            "sample_rate": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
            "nfft": NFFT,
            "window": WINDOW,
            "spectrogram_shape": list(spec.shape),
            "spectrogram_npy": str(npy_path),
        }

        if args.save_png:
            png_path = png_dir / f"{base_name}.png"
            save_spectrogram_png(
                spec=spec,
                save_path=png_path,
                title=f"{args.label} | block {block_idx}",
            )
            metadata["spectrogram_png"] = str(png_path)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        exported.append(metadata)

        print(f"[{block_idx:06d}] saved {npy_path}")

    summary = {
        "input": str(args.input),
        "out_dir": str(out_dir),
        "label": args.label,
        "total_blocks_in_file": int(total_blocks),
        "start_block": int(start_block),
        "end_block_exclusive": int(end_block),
        "num_exported": int(len(exported)),
        "sample_rate": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "nperseg": NPERSEG,
        "noverlap": NOVERLAP,
        "nfft": NFFT,
        "window": WINDOW,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=== Done ===")
    print(f"exported    : {len(exported)}")
    print(f"summary     : {summary_path}")
    print(f"npy_dir     : {npy_dir}")

    if args.save_png:
        print(f"png_dir     : {png_dir}")


if __name__ == "__main__":
    main()