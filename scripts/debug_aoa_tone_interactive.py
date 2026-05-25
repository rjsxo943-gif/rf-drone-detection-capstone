from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.core import load_yaml
from src.preprocess.dc_blocker import remove_dc_offset

from scripts.debug_aoa_tone_live import (
    build_runtime_receiver,
    close_receiver,
    read_block,
    set_center_freq,
    get_block_size,
    get_sample_rate,
    load_phase_gain,
    estimate_phase_from_tone_fft,
    wrap_phase_rad,
    phase_to_angle_deg,
    ensure_2d_iq,
)


def collect_phase_rows(
    receiver,
    *,
    block_size: int,
    sample_rate: int,
    signal_freq: int,
    center_freq: int,
    antenna_spacing: float,
    num_blocks: int,
    phase_offset: float,
    gain_correction: float,
    search_bw: float,
    invert: bool,
    verbose: bool = True,
):
    expected_offset_hz = float(signal_freq - center_freq)

    rows = []

    for i in range(num_blocks):
        iq = read_block(receiver, block_size)
        iq = remove_dc_offset(iq, axis=-1)
        iq = ensure_2d_iq(iq)

        if iq.shape[0] < 2:
            raise RuntimeError(f"2 channels required, got shape={iq.shape}")

        ref = iq[0].astype(np.complex64)
        target = iq[1].astype(np.complex64) * gain_correction

        est = estimate_phase_from_tone_fft(
            ref=ref,
            target=target,
            sample_rate=sample_rate,
            expected_offset_hz=expected_offset_hz,
            search_bw_hz=search_bw,
        )

        raw_phase = float(est["raw_phase_rad"])
        corrected_phase = float(wrap_phase_rad(raw_phase - phase_offset))

        angle_deg, _, clipped = phase_to_angle_deg(
            corrected_phase,
            carrier_freq_hz=float(signal_freq),
            antenna_spacing_m=antenna_spacing,
            invert=invert,
        )

        row = {
            "block": i,
            "peak_freq_hz": float(est["peak_freq_hz"]),
            "raw_phase_rad": raw_phase,
            "corrected_phase_rad": corrected_phase,
            "coherence": float(est["coherence"]),
            "angle_deg": angle_deg,
            "clipped": clipped,
        }
        rows.append(row)

        if verbose:
            print(
                f"[{i:04d}] "
                f"peak={row['peak_freq_hz']:+.0f} Hz "
                f"raw={row['raw_phase_rad']:+.4f} "
                f"corr={row['corrected_phase_rad']:+.4f} "
                f"coh={row['coherence']:.4f} "
                f"angle={row['angle_deg']:+.2f} "
                f"clip={row['clipped']}"
            )

    return rows


def summarize(rows) -> None:
    angles = np.array([r["angle_deg"] for r in rows], dtype=np.float32)
    cohs = np.array([r["coherence"] for r in rows], dtype=np.float32)
    raw = np.array([r["raw_phase_rad"] for r in rows], dtype=np.float32)

    print()
    print("=== Measure Summary ===")
    print(f"raw phase median : {np.median(raw):+.6f} rad")
    print(f"angle mean       : {np.mean(angles):+.2f} deg")
    print(f"angle median     : {np.median(angles):+.2f} deg")
    print(f"angle std        : {np.std(angles):.2f} deg")
    print(f"coherence mean   : {np.mean(cohs):.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-freq", type=int, default=2449000000)
    parser.add_argument("--signal-freq", type=int, default=2450000000)
    parser.add_argument("--antenna-spacing", type=float, default=0.060)
    parser.add_argument("--num-blocks", type=int, default=30)
    parser.add_argument("--search-bw", type=float, default=200000.0)
    parser.add_argument("--calib", default="outputs/calibration/phase_gain_latest.json")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    configs = load_yaml("configs/receiver.yaml")
    receiver = build_runtime_receiver(configs)

    block_size = get_block_size(configs)
    sample_rate = get_sample_rate(configs)

    phase_offset, gain_correction = load_phase_gain(Path(args.calib))

    print("=== AoA Tone Interactive Debug ===")
    print(f"signal_freq     : {args.signal_freq}")
    print(f"center_freq     : {args.center_freq}")
    print(f"sample_rate     : {sample_rate}")
    print(f"block_size      : {block_size}")
    print(f"phase_offset    : {phase_offset:+.6f} rad")
    print(f"gain_correction : {gain_correction:.6f}")
    print()
    print("commands:")
    print("  c     : current position을 0도로 보고 phase offset 보정")
    print("  Enter : 현재 phase offset으로 AoA 측정")
    print("  q     : 종료")
    print()

    try:
        set_center_freq(receiver, args.center_freq)

        for _ in range(5):
            read_block(receiver, block_size)

        while True:
            cmd = input("measure> ").strip().lower()

            if cmd in {"q", "quit", "exit"}:
                break

            if cmd in {"c", "cal", "calibrate", "0"}:
                print()
                print("=== Calibrate current position as 0 deg ===")
                rows = collect_phase_rows(
                    receiver,
                    block_size=block_size,
                    sample_rate=sample_rate,
                    signal_freq=args.signal_freq,
                    center_freq=args.center_freq,
                    antenna_spacing=args.antenna_spacing,
                    num_blocks=args.num_blocks,
                    phase_offset=0.0,
                    gain_correction=gain_correction,
                    search_bw=args.search_bw,
                    invert=args.invert,
                    verbose=False,
                )

                raw_phases = np.array([r["raw_phase_rad"] for r in rows], dtype=np.float32)
                cohs = np.array([r["coherence"] for r in rows], dtype=np.float32)

                phase_offset = float(np.median(raw_phases))

                print(f"[OK] new in-session phase_offset = {phase_offset:+.6f} rad")
                print(f"[OK] phase_offset_deg = {np.degrees(phase_offset):+.3f} deg")
                print(f"[OK] coherence mean = {np.mean(cohs):.4f}")
                print("이제 신호원 위치를 바꾼 뒤 Enter로 측정하면 됨.")
                print()
                continue

            rows = collect_phase_rows(
                receiver,
                block_size=block_size,
                sample_rate=sample_rate,
                signal_freq=args.signal_freq,
                center_freq=args.center_freq,
                antenna_spacing=args.antenna_spacing,
                num_blocks=args.num_blocks,
                phase_offset=phase_offset,
                gain_correction=gain_correction,
                search_bw=args.search_bw,
                invert=args.invert,
                verbose=True,
            )
            summarize(rows)

    finally:
        close_receiver(receiver)


if __name__ == "__main__":
    main()
