from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core import load_all_configs
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver import build_receiver


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


def safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_runtime_receiver(configs: dict[str, Any]) -> Any:
    receiver_cfg = configs.get("receiver", configs)
    try:
        return build_receiver(receiver_cfg)
    except TypeError:
        return build_receiver(configs)


def frame_energies(x: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    x = np.asarray(x).reshape(-1)

    if x.size < frame_size:
        frames = x.reshape(1, -1)
    else:
        frames = np.stack(
            [x[i:i + frame_size] for i in range(0, x.size - frame_size + 1, hop_size)],
            axis=0,
        )

    if frames.shape[-1] == frame_size:
        frames = frames * np.hanning(frame_size).astype(np.float32)

    return np.mean(np.abs(frames) ** 2, axis=-1).astype(np.float32)


def fft_score_db(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).reshape(-1)
    mag = np.abs(np.fft.fftshift(np.fft.fft(x)))
    return float(20.0 * np.log10(np.max(mag) + eps))


def load_noise_threshold(path: str = "outputs/calibration/noise_latest.json") -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data.get("threshold", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2402000000)
    parser.add_argument("--stop", type=int, default=2482000000)
    parser.add_argument("--step", type=int, default=5000000)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--warmup-reads", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    args = parser.parse_args()

    configs = load_all_configs()
    receiver = build_runtime_receiver(configs)

    block_size = int(
        safe_get(configs, "sdr", "block_size", default=None)
        or safe_get(configs, "sdr", "num_samples", default=None)
        or safe_get(configs, "receiver", "block_size", default=None)
        or 16384
    )

    noise_threshold = load_noise_threshold()

    rows = []

    try:
        for freq in range(args.start, args.stop + 1, args.step):
            set_center_freq(receiver, freq)

            for _ in range(args.warmup_reads):
                read_block(receiver, block_size)

            ratios = []
            emaxs = []
            ep99s = []
            fft_scores = []

            for _ in range(args.num_blocks):
                iq = read_block(receiver, block_size)
                iq = remove_dc_offset(iq, axis=-1)
                iq = ensure_2d_iq(iq)

                if args.rx_index >= iq.shape[0]:
                    raise ValueError(f"rx_index={args.rx_index}, channels={iq.shape[0]}")

                x = iq[args.rx_index]
                e = frame_energies(x)

                ratio = float(np.mean(e >= noise_threshold)) if noise_threshold > 0 else 0.0
                ratios.append(ratio)
                emaxs.append(float(np.max(e)))
                ep99s.append(float(np.percentile(e, 99)))
                fft_scores.append(fft_score_db(x))

                time.sleep(args.sleep_sec)

            row = {
                "freq": freq,
                "mhz": freq / 1e6,
                "ratio_max": max(ratios),
                "ratio_mean": float(np.mean(ratios)),
                "energy_p99_mean": float(np.mean(ep99s)),
                "energy_max_highest": max(emaxs),
                "fft_db_max": max(fft_scores),
            }
            rows.append(row)

            print(
                f"{row['mhz']:8.1f} MHz | "
                f"ratio_max={row['ratio_max']:.3f} "
                f"ratio_mean={row['ratio_mean']:.3f} "
                f"e_p99={row['energy_p99_mean']:.4g} "
                f"e_max={row['energy_max_highest']:.4g} "
                f"fft_max={row['fft_db_max']:.2f} dB"
            )

    finally:
        close_receiver(receiver)

    print()
    print("=== Top by energy_max_highest ===")
    for row in sorted(rows, key=lambda r: r["energy_max_highest"], reverse=True)[:10]:
        print(
            f"{row['mhz']:8.1f} MHz | "
            f"ratio_max={row['ratio_max']:.3f} "
            f"ratio_mean={row['ratio_mean']:.3f} "
            f"e_p99={row['energy_p99_mean']:.4g} "
            f"e_max={row['energy_max_highest']:.4g} "
            f"fft_max={row['fft_db_max']:.2f} dB"
        )


if __name__ == "__main__":
    main()
