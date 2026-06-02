from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.gain_noise_runtime import load_gain_noise_runtime  # noqa: E402


def main() -> None:
    profile_path = (
        PROJECT_ROOT
        / "outputs"
        / "calibration"
        / "noise_by_gain_sim_test.json"
    )

    runtime = load_gain_noise_runtime(profile_path)

    block = (
        0.1 * np.random.randn(2, 16_384)
        + 1j * 0.1 * np.random.randn(2, 16_384)
    ).astype(np.complex64)

    print(runtime.summarize_block(block, gain=30))

    clipped = block.copy()
    clipped[:, :500] = 1.0 + 1.0j

    print(runtime.summarize_block(clipped, gain=30))


if __name__ == "__main__":
    main()