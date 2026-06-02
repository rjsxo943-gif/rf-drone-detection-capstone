from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.gain_noise_calibration import (  # noqa: E402
    calibrate_noise_by_gain,
    get_noise_threshold_for_gain,
    load_gain_noise_calibration,
)


def make_synthetic_noise_block(
    gain: float,
    *,
    num_channels: int = 2,
    block_size: int = 16_384,
) -> np.ndarray:
    """
    Pluto 없이 gain별 noise calibration 흐름을 테스트하기 위한 synthetic IQ block.

    gain이 커질수록 noise amplitude가 조금씩 커지는 것처럼 만든다.
    """
    rng_scale = 0.02 + (float(gain) - 20.0) * 0.002

    real = rng_scale * np.random.randn(num_channels, block_size)
    imag = rng_scale * np.random.randn(num_channels, block_size)

    return (real + 1j * imag).astype(np.complex64)


def collect_fn(gain: float, n_blocks: int) -> list[np.ndarray]:
    print(f"[sim collect] gain={gain:g}, n_blocks={n_blocks}")

    return [
        make_synthetic_noise_block(gain)
        for _ in range(n_blocks)
    ]


def main() -> None:
    output_path = PROJECT_ROOT / "outputs" / "calibration" / "noise_by_gain_sim_test.json"

    result_set = calibrate_noise_by_gain(
        gain_list=[20, 25, 30, 35, 40],
        collect_fn=collect_fn,
        num_blocks_per_gain=10,
        method="time_power",
        frame_size=1024,
        hop_size=512,
        threshold_multiplier=5.0,
        min_detection_ratio=0.05,
        sample_rate=5_000_000,
        center_freq=2_450_000_000,
    )

    save_path = result_set.save_json(output_path)

    print()
    print("=== Saved ===")
    print(save_path)

    loaded = load_gain_noise_calibration(save_path)

    print()
    print("=== Threshold lookup test ===")
    for gain in [20, 25, 30, 35, 40, 33]:
        threshold = get_noise_threshold_for_gain(loaded, gain)
        profile = loaded.get_profile(gain)

        print(
            f"gain={gain:g} | "
            f"threshold={threshold:.10g} | "
            f"safety={profile['safety']['status']} | "
            f"matched_gain={profile.get('matched_gain', gain)}"
        )


if __name__ == "__main__":
    main()