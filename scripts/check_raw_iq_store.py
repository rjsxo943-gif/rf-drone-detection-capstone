from __future__ import annotations

import numpy as np

from src.core.raw_iq_store import (
    create_raw_iq_session,
    load_raw_iq_block,
    save_raw_iq_block,
)


def make_test_dual_iq(
    sample_rate: float,
    block_size: int,
    tone_freq: float = 500_000,
    phase_offset_rad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2 * np.pi * tone_freq * t)
    rx1 = np.exp(1j * (2 * np.pi * tone_freq * t + phase_offset_rad))

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def main() -> None:
    sample_rate = 5_000_000
    center_freq = 2_437_000_000
    block_size = 16_384
    label = "test"

    rx0_iq, rx1_iq = make_test_dual_iq(
        sample_rate=sample_rate,
        block_size=block_size,
    )

    session_dir = create_raw_iq_session(
        root_dir="outputs/runs/latest/raw_iq_test",
        label=label,
        metadata={
            "device": "simulated",
            "purpose": "raw IQ save/load test",
            "block_size": block_size,
        },
    )

    saved_path = save_raw_iq_block(
        session_dir=session_dir,
        block_index=0,
        rx0_iq=rx0_iq,
        rx1_iq=rx1_iq,
        sample_rate=sample_rate,
        center_freq=center_freq,
        label=label,
        metadata={
            "note": "This is not real Pluto+ data.",
        },
    )

    loaded = load_raw_iq_block(saved_path)

    print("=== Raw IQ Store Check ===")
    print(f"saved_path: {saved_path}")
    print(f"label: {loaded['label']}")
    print(f"sample_rate: {loaded['sample_rate']}")
    print(f"center_freq: {loaded['center_freq']}")
    print(f"block_index: {loaded['block_index']}")
    print(f"rx0_iq shape: {loaded['rx0_iq'].shape}")
    print(f"rx1_iq shape: {loaded['rx1_iq'].shape}")
    print(f"rx0_iq dtype: {loaded['rx0_iq'].dtype}")
    print(f"rx1_iq dtype: {loaded['rx1_iq'].dtype}")
    print(f"metadata: {loaded['metadata']}")

    if loaded["rx0_iq"].shape == (block_size,) and loaded["rx1_iq"].shape == (block_size,):
        print("OK: raw IQ save/load success")
    else:
        print("WARNING: loaded IQ shape is wrong")


if __name__ == "__main__":
    main()  