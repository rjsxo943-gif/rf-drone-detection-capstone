from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core import get_block_size, load_all_configs
from src.features.spectrogram import compute_stft_branch
from src.ml import RF4Classifier
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver import build_receiver


def safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.ndim == 1:
        return iq.reshape(1, -1)
    if iq.ndim == 2:
        return iq
    raise ValueError(f"iq must be 1D or 2D, got shape={iq.shape}")


def read_block(receiver: Any, block_size: int) -> np.ndarray:
    if hasattr(receiver, "read_block"):
        return ensure_2d_iq(receiver.read_block(block_size))
    if hasattr(receiver, "read_samples"):
        return ensure_2d_iq(receiver.read_samples(block_size))
    raise AttributeError("receiver has neither read_block nor read_samples")


def close_receiver(receiver: Any) -> None:
    if hasattr(receiver, "close"):
        try:
            receiver.close()
        except Exception:
            pass


def set_center_freq(receiver: Any, center_freq: int) -> None:
    center_freq = int(center_freq)

    if hasattr(receiver, "center_freq"):
        try:
            receiver.center_freq = center_freq
        except Exception:
            pass

    if hasattr(receiver, "sdr"):
        sdr = getattr(receiver, "sdr")
        if sdr is not None and hasattr(sdr, "rx_lo"):
            sdr.rx_lo = center_freq
            return

    if hasattr(receiver, "_set_center_freq"):
        receiver._set_center_freq(center_freq)


def get_block_size_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        safe_get(configs, "receiver", "block_size"),
        safe_get(configs, "receiver", "num_samples"),
        safe_get(configs, "sdr", "block_size"),
        safe_get(configs, "sdr", "num_samples"),
    ]:
        if value is not None:
            return int(value)

    try:
        return int(get_block_size(configs))
    except Exception:
        return 16384


def get_sample_rate_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        safe_get(configs, "receiver", "sample_rate"),
        safe_get(configs, "sdr", "sample_rate"),
    ]:
        if value is not None:
            return int(value)
    return 5_000_000


def build_runtime_receiver(configs: dict[str, Any]) -> Any:
    receiver_cfg = configs.get("receiver", configs)
    try:
        return build_receiver(receiver_cfg)
    except TypeError:
        return build_receiver(configs)



def compute_canonical01_spectrogram(
    iq_1d: np.ndarray,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
    window: str = "hann",
    vmin: float = -40.0,
    vmax: float = 40.0,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(iq_1d).reshape(-1)

    if x.size < nperseg:
        raise ValueError(f"iq length {x.size} is shorter than nperseg {nperseg}")

    hop_size = nperseg - noverlap
    if hop_size <= 0:
        raise ValueError(f"invalid hop_size={hop_size}")

    if window == "hann":
        win = np.hanning(nperseg).astype(np.float32)
    elif window == "hamming":
        win = np.hamming(nperseg).astype(np.float32)
    else:
        win = np.ones(nperseg, dtype=np.float32)

    cols = []
    for start in range(0, x.size - nperseg + 1, hop_size):
        frame = x[start:start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        # FFT magnitude는 window 길이/합에 비례해서 커지므로,
        # 기존 dB spectrogram 스케일과 맞추기 위해 window coherent gain으로 정규화한다.
        mag = np.abs(spec) / (float(np.sum(win)) + eps)
        cols.append(mag.astype(np.float32))

    mag_spec = np.stack(cols, axis=1)

    db_spec = 20.0 * np.log10(mag_spec + eps)
    db_spec = np.clip(db_spec, vmin, vmax)

    norm_spec = (db_spec - vmin) / (vmax - vmin)
    return np.clip(norm_spec, 0.0, 1.0).astype(np.float32)

def spec_stats(spec: np.ndarray) -> dict[str, float]:
    x = np.asarray(spec, dtype=np.float32)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="live_wifi")
    parser.add_argument("--model", default="outputs/ml/rf4_cnn_baseline_v2/best_model.pt")
    parser.add_argument("--center-freq", type=int, default=2_437_000_000)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--warmup-reads", type=int, default=5)
    parser.add_argument("--out-root", default="outputs/debug/rf4_live")
    parser.add_argument("--general-threshold", type=float, default=0.50)
    parser.add_argument("--drone-threshold", type=float, default=0.70)
    args = parser.parse_args()

    configs = load_all_configs()
    block_size = get_block_size_from_configs(configs)
    sample_rate = get_sample_rate_from_configs(configs)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{session}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = RF4Classifier(
        checkpoint_path=args.model,
        general_threshold=args.general_threshold,
        drone_threshold=args.drone_threshold,
    )

    receiver = build_runtime_receiver(configs)

    rows = []

    try:
        set_center_freq(receiver, args.center_freq)

        for _ in range(args.warmup_reads):
            read_block(receiver, block_size)

        print("=== RF4 Live Debug Capture ===")
        print(f"label       : {args.label}")
        print(f"out_dir     : {out_dir}")
        print(f"center_freq : {args.center_freq}")
        print(f"sample_rate : {sample_rate}")
        print(f"block_size  : {block_size}")
        print(f"rx_index    : {args.rx_index}")
        print()

        for idx in range(args.num_blocks):
            iq = read_block(receiver, block_size)
            iq = remove_dc_offset(iq, axis=-1)
            iq = ensure_2d_iq(iq)

            if args.rx_index < 0 or args.rx_index >= iq.shape[0]:
                raise ValueError(f"rx_index out of range: {args.rx_index}, num_channels={iq.shape[0]}")

            iq_1d = iq[args.rx_index]

            spec = compute_canonical01_spectrogram(
                iq_1d,
                nperseg=128,
                noverlap=96,
                nfft=128,
                window="hann",
                vmin=-40.0,
                vmax=40.0,
            )
            stats = spec_stats(spec)
            result = classifier.predict_array(spec)

            # Strong-drone gate:
            # CNN raw가 Drone-like여도 p99/max가 충분히 강하지 않으면 Background로 낮춘다.
            is_strong_drone = (
                result.class_name == "Drone-like"
                and stats["p99"] >= 0.65
                and stats["max"] >= 0.80
            )

            if result.class_name == "Drone-like":
                gated_final_class = "Drone-like" if is_strong_drone else "Background"
            else:
                gated_final_class = result.final_class

            stats = spec_stats(spec)

            filename = (
                f"{idx:04d}__{args.label}"
                f"__raw_{result.class_name}"
                f"__final_{gated_final_class}"
                f"__conf_{result.confidence:.4f}.npy"
            )
            save_path = out_dir / filename
            np.save(save_path, spec)

            row = {
                "index": idx,
                "file": str(save_path),
                "raw_class": result.class_name,
                "final_class": gated_final_class,
                "confidence": result.confidence,
                **stats,
            }
            rows.append(row)

            print(
                f"[{idx:04d}] "
                f"raw={result.class_name:10s} "
                f"final={gated_final_class:10s} "
                f"conf={result.confidence:.4f} "
                f"median={stats['median']:.4f} "
                f"p99={stats['p99']:.4f} "
                f"max={stats['max']:.4f}"
            )

            time.sleep(0.02)

    finally:
        close_receiver(receiver)

    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print()
    print(f"[OK] saved debug spectrograms to: {out_dir}")
    print(f"[OK] saved summary: {csv_path}")


if __name__ == "__main__":
    main()
