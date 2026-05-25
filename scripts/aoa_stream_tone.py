from __future__ import annotations

import argparse
import time
from collections import deque
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


def estimate_one_angle(
    receiver,
    *,
    block_size: int,
    sample_rate: int,
    signal_freq: int,
    center_freq: int,
    antenna_spacing: float,
    phase_offset: float,
    gain_correction: float,
    search_bw: float,
    invert: bool,
) -> tuple[float, float, float, float, bool]:
    expected_offset_hz = float(signal_freq - center_freq)

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

    return (
        angle_deg,
        float(est["coherence"]),
        float(est["peak_freq_hz"]),
        raw_phase,
        bool(clipped),
    )


def calibrate_zero_degree(
    receiver,
    *,
    block_size: int,
    sample_rate: int,
    signal_freq: int,
    center_freq: int,
    gain_correction: float,
    search_bw: float,
    num_blocks: int,
) -> float:
    expected_offset_hz = float(signal_freq - center_freq)

    raw_phases = []
    coherences = []

    print()
    print("=== 0-degree in-session calibration ===")
    print("신호발생기를 안테나 정면 0도에 둔 상태에서 보정 중...")

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

        raw_phases.append(float(est["raw_phase_rad"]))
        coherences.append(float(est["coherence"]))

        print(
            f"[cal {i+1:02d}/{num_blocks}] "
            f"raw={float(est['raw_phase_rad']):+.4f} rad "
            f"coh={float(est['coherence']):.4f}"
        )

    phase_offset = float(np.median(np.array(raw_phases, dtype=np.float32)))

    print()
    print("[OK] in-session phase offset updated")
    print(f"phase_offset     : {phase_offset:+.6f} rad")
    print(f"phase_offset_deg : {np.degrees(phase_offset):+.3f} deg")
    print(f"coherence mean   : {np.mean(coherences):.4f}")
    print()

    return phase_offset


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--center-freq", type=int, default=2449000000)
    parser.add_argument("--signal-freq", type=int, default=2450000000)
    parser.add_argument("--antenna-spacing", type=float, default=0.060)

    parser.add_argument("--search-bw", type=float, default=200000.0)
    parser.add_argument("--calib", default="outputs/calibration/phase_gain_latest.json")

    parser.add_argument("--cal-blocks", type=int, default=20)
    parser.add_argument("--avg-window", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.10)

    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--invert", action="store_true")

    args = parser.parse_args()

    configs = load_yaml("configs/receiver.yaml")
    receiver = build_runtime_receiver(configs)

    block_size = get_block_size(configs)
    sample_rate = get_sample_rate(configs)

    phase_offset, gain_correction = load_phase_gain(Path(args.calib))

    angle_window: deque[float] = deque(maxlen=max(1, args.avg_window))
    coh_window: deque[float] = deque(maxlen=max(1, args.avg_window))

    print("=== AoA Tone Stream ===")
    print(f"signal_freq     : {args.signal_freq} Hz")
    print(f"center_freq     : {args.center_freq} Hz")
    print(f"expected_offset : {args.signal_freq - args.center_freq:+d} Hz")
    print(f"sample_rate     : {sample_rate}")
    print(f"block_size      : {block_size}")
    print(f"antenna_spacing : {args.antenna_spacing:.4f} m")
    print(f"gain_correction : {gain_correction:.6f}")
    print(f"loaded phase    : {phase_offset:+.6f} rad")
    print(f"avg_window      : {args.avg_window}")
    print()

    try:
        set_center_freq(receiver, args.center_freq)

        for _ in range(5):
            read_block(receiver, block_size)

        if not args.no_calibrate:
            input("신호발생기를 0도 정면에 두고 Enter를 누르면 현재 세션 기준으로 보정합니다...")
            phase_offset = calibrate_zero_degree(
                receiver,
                block_size=block_size,
                sample_rate=sample_rate,
                signal_freq=args.signal_freq,
                center_freq=args.center_freq,
                gain_correction=gain_correction,
                search_bw=args.search_bw,
                num_blocks=args.cal_blocks,
            )

        print("=== Streaming angle output ===")
        print("Ctrl+C로 종료")
        print()

        idx = 0

        while True:
            angle, coherence, peak_freq, raw_phase, clipped = estimate_one_angle(
                receiver,
                block_size=block_size,
                sample_rate=sample_rate,
                signal_freq=args.signal_freq,
                center_freq=args.center_freq,
                antenna_spacing=args.antenna_spacing,
                phase_offset=phase_offset,
                gain_correction=gain_correction,
                search_bw=args.search_bw,
                invert=args.invert,
            )

            angle_window.append(angle)
            coh_window.append(coherence)

            angle_avg = float(np.mean(angle_window))
            coh_avg = float(np.mean(coh_window))

            print(
                f"[{idx:06d}] "
                f"angle={angle:+7.2f} deg "
                f"avg={angle_avg:+7.2f} deg "
                f"coh={coherence:.4f} "
                f"coh_avg={coh_avg:.4f} "
                f"peak={peak_freq:+.0f} Hz "
                f"raw={raw_phase:+.4f} "
                f"clip={clipped}"
            )

            idx += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print()
        print("[STOP] AoA streaming stopped by user.")

    finally:
        close_receiver(receiver)


if __name__ == "__main__":
    main()
