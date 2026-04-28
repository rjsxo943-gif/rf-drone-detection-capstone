from __future__ import annotations

from pathlib import Path

import numpy as np


def save_stage1_artifacts(
    output_path: str | Path,
    block_index: int,
    sample_rate: float,
    center_freq: float,
    cnn_spectrogram: np.ndarray,
    rx0_complex_stft: np.ndarray | None = None,
    rx1_complex_stft: np.ndarray | None = None,
    rx0_phase: np.ndarray | None = None,
    rx1_phase: np.ndarray | None = None,
    rx0_log_magnitude: np.ndarray | None = None,
    rx1_log_magnitude: np.ndarray | None = None,
    save_complex_stft: bool = False,
    save_phase: bool = False,
    save_log_magnitude: bool = False,
) -> Path:
    """
    Stage 1 중간 산출물을 선택적으로 저장한다.

    기본 저장:
    - cnn_spectrogram

    옵션 저장:
    - complex STFT
    - phase
    - log magnitude

    raw IQ 저장은 raw_iq_store.py가 담당한다.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {
        "block_index": np.array(block_index, dtype=np.int64),
        "sample_rate": np.array(sample_rate, dtype=np.float64),
        "center_freq": np.array(center_freq, dtype=np.float64),
        "cnn_spectrogram": cnn_spectrogram.astype(np.float32),
    }

    if save_complex_stft:
        if rx0_complex_stft is not None:
            save_dict["rx0_complex_stft"] = rx0_complex_stft.astype(np.complex64)
        if rx1_complex_stft is not None:
            save_dict["rx1_complex_stft"] = rx1_complex_stft.astype(np.complex64)

    if save_phase:
        if rx0_phase is not None:
            save_dict["rx0_phase"] = rx0_phase.astype(np.float32)
        if rx1_phase is not None:
            save_dict["rx1_phase"] = rx1_phase.astype(np.float32)

    if save_log_magnitude:
        if rx0_log_magnitude is not None:
            save_dict["rx0_log_magnitude"] = rx0_log_magnitude.astype(np.float32)
        if rx1_log_magnitude is not None:
            save_dict["rx1_log_magnitude"] = rx1_log_magnitude.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)

    return output_path