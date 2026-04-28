from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import (
    ensure_parent_dir,
    ensure_suffix,
    get_sample_range,
    to_float32_array,
)


def save_stage1_artifacts(
    output_path: str | Path,
    block_index: int,
    sample_rate: float,
    center_freq: float,
    cnn_spectrogram: np.ndarray,
    block_size: int = 16_384,
    nperseg: int = 512,
    noverlap: int = 384,
    hop_length: int = 128,
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
    output_path = ensure_suffix(output_path, ".npz")
    ensure_parent_dir(output_path)

    block_index = int(block_index)
    block_size = int(block_size)

    sample_start, sample_end = get_sample_range(block_index, block_size)

    cnn_spectrogram = to_float32_array(cnn_spectrogram, "cnn_spectrogram")

    save_dict: dict[str, np.ndarray] = {
        "block_index": np.array(block_index, dtype=np.int64),
        "sample_start": np.array(sample_start, dtype=np.int64),
        "sample_end": np.array(sample_end, dtype=np.int64),
        "sample_rate": np.array(sample_rate, dtype=np.float64),
        "center_freq": np.array(center_freq, dtype=np.float64),
        "block_size": np.array(block_size, dtype=np.int64),
        "nperseg": np.array(nperseg, dtype=np.int64),
        "noverlap": np.array(noverlap, dtype=np.int64),
        "hop_length": np.array(hop_length, dtype=np.int64),
        "cnn_spectrogram": cnn_spectrogram,
    }

    if save_complex_stft:
        if rx0_complex_stft is not None:
            save_dict["rx0_complex_stft"] = np.asarray(rx0_complex_stft).astype(
                np.complex64,
                copy=False,
            )
        if rx1_complex_stft is not None:
            save_dict["rx1_complex_stft"] = np.asarray(rx1_complex_stft).astype(
                np.complex64,
                copy=False,
            )

    if save_phase:
        if rx0_phase is not None:
            save_dict["rx0_phase"] = to_float32_array(rx0_phase, "rx0_phase")
        if rx1_phase is not None:
            save_dict["rx1_phase"] = to_float32_array(rx1_phase, "rx1_phase")

    if save_log_magnitude:
        if rx0_log_magnitude is not None:
            save_dict["rx0_log_magnitude"] = to_float32_array(
                rx0_log_magnitude,
                "rx0_log_magnitude",
            )
        if rx1_log_magnitude is not None:
            save_dict["rx1_log_magnitude"] = to_float32_array(
                rx1_log_magnitude,
                "rx1_log_magnitude",
            )

    np.savez_compressed(output_path, **save_dict)

    return Path(output_path)