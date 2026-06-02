from __future__ import annotations

import numpy as np

from src.preprocess import get_cnn_input_iq, normalize_iq, remove_dc_offset

EPS = 1e-12


def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"IQ block must be 1D or 2D, got shape={arr.shape}")


def _frame_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if x.size < frame_size:
        return x.reshape(1, -1)

    num_frames = 1 + (x.size - frame_size) // hop_size
    shape = (num_frames, frame_size)
    strides = (x.strides[0] * hop_size, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def compute_runtime_cnn_spectrogram(
    iq_block: np.ndarray,
    *,
    rx_index: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> np.ndarray:
    """
    live_cnn_spectrogram_viewer_yaml.py와 같은 CNN 입력 생성 경로.

    순서:
    - remove DC
    - rx_index 선택
    - peak normalize
    - Hann STFT
    - fftshift
    - log1p magnitude
    - minmax normalize
    - return shape: (freq_bins, time_frames)
    """
    iq_no_dc = remove_dc_offset(_ensure_2d_iq(iq_block))
    cnn_iq = get_cnn_input_iq(iq_no_dc, rx_index=int(rx_index))
    cnn_iq = normalize_iq(cnn_iq, method="peak")
    cnn_iq = np.asarray(cnn_iq).reshape(-1).astype(np.complex64)

    hop_size = int(nperseg) - int(noverlap)
    frames = _frame_1d(cnn_iq, frame_size=int(nperseg), hop_size=hop_size)

    window = np.hanning(int(nperseg)).astype(np.float32)
    windowed = frames * window.reshape(1, -1)

    stft = np.fft.fft(windowed, n=int(nfft), axis=1)
    stft = np.fft.fftshift(stft, axes=1)

    mag = np.abs(stft).astype(np.float32)
    spec = np.log1p(mag)

    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec = (spec - spec_min) / max(spec_max - spec_min, EPS)

    return spec.T.astype(np.float32)
