# check_raw_iq_store.py
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
    """
    테스트용 2채널 complex IQ를 생성한다.

    RX1은 RX0보다 phase_offset_rad만큼 위상이 앞서도록 만든다.
    """
    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2.0 * np.pi * tone_freq * t)
    rx1 = np.exp(1j * (2.0 * np.pi * tone_freq * t + phase_offset_rad))

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def main() -> None:
    sample_rate = 5_000_000
    center_freq = 2_437_000_000
    block_size = 16_384
    label = "test"
    phase_offset_rad = 0.5

    rx0_iq, rx1_iq = make_test_dual_iq(
        sample_rate=sample_rate,
        block_size=block_size,
        phase_offset_rad=phase_offset_rad,
    )

    session_dir = create_raw_iq_session(
        root_dir="outputs/runs/latest/raw_iq_test",
        label=label,
        metadata={
            "device": "simulated",
            "purpose": "raw IQ save/load test",
            "block_size": block_size,
            "phase_offset_rad": phase_offset_rad,
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

    loaded_rx0 = loaded["rx0_iq"]
    loaded_rx1 = loaded["rx1_iq"]

    print("=== Raw IQ Store Check ===")
    print(f"saved_path   : {saved_path}")
    print(f"label        : {loaded['label']}")
    print(f"sample_rate  : {loaded['sample_rate']}")
    print(f"center_freq  : {loaded['center_freq']}")
    print(f"block_index  : {loaded['block_index']}")
    print(f"rx0_iq shape : {loaded_rx0.shape}")
    print(f"rx1_iq shape : {loaded_rx1.shape}")
    print(f"rx0_iq dtype : {loaded_rx0.dtype}")
    print(f"rx1_iq dtype : {loaded_rx1.dtype}")
    print(f"metadata     : {loaded['metadata']}")

    shape_ok = (
        loaded_rx0.shape == (block_size,)
        and loaded_rx1.shape == (block_size,)
    )

    dtype_ok = (
        loaded_rx0.dtype == np.complex64
        and loaded_rx1.dtype == np.complex64
    )

    value_ok = (
        np.allclose(rx0_iq, loaded_rx0)
        and np.allclose(rx1_iq, loaded_rx1)
    )

    meta_ok = (
        loaded["label"] == label
        and loaded["sample_rate"] == float(sample_rate)
        and loaded["center_freq"] == float(center_freq)
        and loaded["block_index"] == 0
    )

    print()
    print("=== Check Result ===")
    print(f"shape_ok : {shape_ok}")
    print(f"dtype_ok : {dtype_ok}")
    print(f"value_ok : {value_ok}")
    print(f"meta_ok  : {meta_ok}")

    if shape_ok and dtype_ok and value_ok and meta_ok:
        print("OK: raw IQ save/load success")
    else:
        raise RuntimeError("Raw IQ save/load check failed.")


if __name__ == "__main__":
    main()