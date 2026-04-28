#src/ml/transforms.py
from __future__ import annotations

import numpy as np


def ensure_cnn_input_shape(
    spectrogram: np.ndarray,
    expected_shape: tuple[int, int, int] = (512, 125, 1),
) -> np.ndarray:
    """
    spectrogram을 CNN 입력 shape으로 맞춘다.

    현재 기준:
    - 입력 spectrogram: (512, 125)
    - CNN 입력: (512, 125, 1)

    Returns:
        np.ndarray, shape = (512, 125, 1)
    """
    spectrogram = np.asarray(spectrogram, dtype=np.float32)

    if spectrogram.size == 0:
        raise ValueError("spectrogram is empty.")

    if spectrogram.ndim == 2:
        spectrogram = spectrogram[..., np.newaxis]

    if spectrogram.ndim != 3:
        raise ValueError(
            f"Expected spectrogram shape (H, W) or (H, W, C), got {spectrogram.shape}"
        )

    if spectrogram.shape != expected_shape:
        raise ValueError(
            f"Unexpected CNN input shape: {spectrogram.shape}. "
            f"Expected {expected_shape}."
        )

    return spectrogram.astype(np.float32, copy=False)


def add_batch_dimension(x: np.ndarray) -> np.ndarray:
    """
    CNN 입력에 batch dimension을 추가한다.

    (512, 125, 1) -> (1, 512, 125, 1)
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D CNN input, got shape {x.shape}")

    return x[np.newaxis, ...]