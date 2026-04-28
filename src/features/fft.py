from __future__ import annotations

import numpy as np

from src.features.window import get_window


def compute_fft_magnitude(
    frames: np.ndarray,
    window: str = "hann",
    fftshift: bool = True,
) -> np.ndarray:
    """
    frame 단위 IQ에 대해 FFT magnitude를 계산한다.

    입력:
    - frames shape = (num_frames, frame_size)
    - 또는 단일 frame shape = (frame_size,)

    출력:
    - magnitude shape = (num_frames, frame_size)

    주의:
    - 이 함수는 block 전체를 STFT spectrogram으로 바꾸는 함수가 아니다.
    - energy detector나 FFT 확인용으로 사용한다.
    """
    frames = np.asarray(frames)

    if frames.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    if not np.iscomplexobj(frames):
        raise TypeError(f"FFT input must be complex IQ, got dtype={frames.dtype}")

    if frames.ndim == 1:
        frames = frames[np.newaxis, :]

    if frames.ndim != 2:
        raise ValueError(
            f"frames must be 1D or 2D. Expected (N,) or (F, N), got {frames.shape}"
        )

    frame_size = frames.shape[1]
    win = get_window(window, frame_size)

    windowed = frames * win[np.newaxis, :]
    spectrum = np.fft.fft(windowed, axis=1)

    if fftshift:
        spectrum = np.fft.fftshift(spectrum, axes=1)

    magnitude = np.abs(spectrum)

    return magnitude.astype(np.float32)


def compute_fft_power(
    frames: np.ndarray,
    window: str = "hann",
    fftshift: bool = True,
    eps: float = 1e-12,
    log_scale: bool = False,
) -> np.ndarray:
    """
    frame 단위 IQ에 대해 FFT power를 계산한다.

    power = |FFT|^2

    Args:
        frames:
            shape = (num_frames, frame_size) 또는 (frame_size,)
        window:
            window 종류
        fftshift:
            중심 주파수를 가운데로 옮길지 여부
        eps:
            log 계산 시 0 방지용
        log_scale:
            True이면 10*log10(power + eps) 반환

    Returns:
        power 또는 log power
    """
    magnitude = compute_fft_magnitude(
        frames,
        window=window,
        fftshift=fftshift,
    )

    power = magnitude**2

    if log_scale:
        power = 10.0 * np.log10(power + eps)

    return power.astype(np.float32)


def compute_block_fft_magnitude(
    iq_block: np.ndarray,
    window: str = "hann",
    fftshift: bool = True,
) -> np.ndarray:
    """
    block 하나에 대해 FFT magnitude를 계산한다.

    입력:
    - 단일 채널: shape = (16384,)
    - 여러 채널: shape = (C, 16384)

    출력:
    - 단일 채널 입력이면 shape = (1, 16384)
    - 여러 채널 입력이면 shape = (C, 16384)

    주의:
    - 이 함수는 block 전체에 FFT 1번을 적용한다.
    - CNN용 spectrogram 생성은 spectrogram.py에서 STFT로 처리한다.
    """
    iq_block = np.asarray(iq_block)

    if iq_block.size == 0:
        raise ValueError("Input IQ block is empty.")

    if not np.iscomplexobj(iq_block):
        raise TypeError(f"IQ block must be complex, got dtype={iq_block.dtype}")

    if iq_block.ndim == 1:
        frames = iq_block[np.newaxis, :]

    elif iq_block.ndim == 2:
        frames = iq_block

    else:
        raise ValueError(
            f"iq_block must be 1D or 2D. Expected (N,) or (C, N), got {iq_block.shape}"
        )

    return compute_fft_magnitude(
        frames,
        window=window,
        fftshift=fftshift,
    )


def compute_block_fft_power(
    iq_block: np.ndarray,
    window: str = "hann",
    fftshift: bool = True,
    eps: float = 1e-12,
    log_scale: bool = False,
) -> np.ndarray:
    """
    block 하나에 대해 FFT power를 계산한다.

    입력:
    - 단일 채널: shape = (16384,)
    - 여러 채널: shape = (C, 16384)

    출력:
    - 단일 채널 입력이면 shape = (1, 16384)
    - 여러 채널 입력이면 shape = (C, 16384)
    """
    iq_block = np.asarray(iq_block)

    if iq_block.size == 0:
        raise ValueError("Input IQ block is empty.")

    if not np.iscomplexobj(iq_block):
        raise TypeError(f"IQ block must be complex, got dtype={iq_block.dtype}")

    if iq_block.ndim == 1:
        frames = iq_block[np.newaxis, :]

    elif iq_block.ndim == 2:
        frames = iq_block

    else:
        raise ValueError(
            f"iq_block must be 1D or 2D. Expected (N,) or (C, N), got {iq_block.shape}"
        )

    return compute_fft_power(
        frames,
        window=window,
        fftshift=fftshift,
        eps=eps,
        log_scale=log_scale,
    )