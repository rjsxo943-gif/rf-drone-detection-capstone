from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.receiver.pluto_receiver import PlutoReceiver
from src.preprocess.dc_blocker import remove_dc_offset
from src.features.spectrogram import compute_stft_branch


def burst_stats(spec: np.ndarray) -> dict:
    x = np.asarray(spec, dtype=np.float32)
    return {
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "burst_score": float(np.percentile(x, 99) - np.median(x)),
    }


def make_filename(
    index: int,
    state: str,
    center_freq_mhz: int,
    gain: float,
    distance_cm: int,
    attempt: int,
) -> str:
    gain_text = str(gain).replace(".", "p")

    return (
        f"{index:04d}__DRONE__{state}"
        f"__cf{center_freq_mhz}"
        f"__g{gain_text}"
        f"__d{distance_cm}"
        f"__attempt{attempt:04d}.npy"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--uri", default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, default=2437000000)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=5_000_000)
    parser.add_argument("--gain", type=float, default=20.0)
    parser.add_argument("--block-size", type=int, default=16_384)

    parser.add_argument("--state", required=True)
    parser.add_argument("--num-save", type=int, default=50)
    parser.add_argument("--distance-cm", type=int, default=50)
    parser.add_argument("--attempt", type=int, default=1)

    parser.add_argument("--min-burst-score", type=float, default=0.30)
    parser.add_argument("--sleep-sec", type=float, default=0.02)

    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=96)
    parser.add_argument("--nfft", type=int, default=128)
    parser.add_argument("--window", default="hann")

    args = parser.parse_args()

    center_freq_mhz = int(args.center_freq / 1e6)
    gain_text = str(args.gain).replace(".", "p")

    session_name = (
        f"drone_cf{center_freq_mhz}"
        f"_g{gain_text}"
        f"_d{args.distance_cm}"
    )

    out_dir = (
        Path("data/processed/cnn_capture")
        / session_name
        / args.state
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rx = PlutoReceiver(
        uri=args.uri,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        rf_bandwidth=args.rf_bandwidth,
        gain=args.gain,
        channels=[0],
        block_size=args.block_size,
    )

    saved = 0
    tried = 0

    print("=== Drone capture start ===")
    print(f"state       : {args.state}")
    print(f"center_freq : {center_freq_mhz} MHz")
    print(f"gain        : {args.gain}")
    print(f"distance    : {args.distance_cm} cm")
    print(f"target save : {args.num_save}")
    print(f"threshold   : {args.min_burst_score}")
    print(f"out_dir     : {out_dir}")

    try:
        while saved < args.num_save:
            tried += 1

            iq = rx.read_block(args.block_size)
            iq = remove_dc_offset(iq, axis=-1)

            if iq.ndim == 2:
                iq_1d = iq[0]
            else:
                iq_1d = iq

            stft_out = compute_stft_branch(
                iq_1d,
                sample_rate=args.sample_rate,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                nfft=args.nfft,
                window=args.window,
            )

            spec = stft_out.cnn_spectrogram.astype(np.float32)
            stats = burst_stats(spec)

            if stats["burst_score"] < args.min_burst_score:
                print(
                    f"[skip] tried={tried:04d} "
                    f"burst={stats['burst_score']:.2f} "
                    f"p99={stats['p99']:.2f} "
                    f"median={stats['median']:.2f}"
                )
                time.sleep(args.sleep_sec)
                continue

            filename = make_filename(
                index=saved,
                state=args.state,
                center_freq_mhz=center_freq_mhz,
                gain=args.gain,
                distance_cm=args.distance_cm,
                attempt=args.attempt,
            )

            save_path = out_dir / filename
            np.save(save_path, spec)

            print(
                f"[save] {filename} "
                f"burst={stats['burst_score']:.2f} "
                f"p99={stats['p99']:.2f} "
                f"median={stats['median']:.2f}"
            )

            saved += 1
            time.sleep(args.sleep_sec)

    except KeyboardInterrupt:
        print("\n=== Interrupted ===")

    finally:
        rx.close()

    print("=== Done ===")
    print(f"saved: {saved}")
    print(f"tried : {tried}")


if __name__ == "__main__":
    main()