from __future__ import annotations

import numpy as np


def get_window(window: str, size: int) -> np.ndarray:
    """
    FFT/STFT 계산에 사용할 window를 생성한다.

    Args:
        window:
            사용할 window 이름.
            지원 목록:
            - "hann"
            - "hamming"
            - "rect"
            - "rectangle"
            - "none"

        size:
            window 길이

    Returns:
        np.ndarray
        shape = (size,)
        dtype = float32
    """
    if size <= 0:
        raise ValueError(f"Window size must be positive, got {size}")

    window = window.lower().strip()

    if window == "hann":
        return np.hanning(size).astype(np.float32)

    if window == "hamming":
        return np.hamming(size).astype(np.float32)

    if window in {"rect", "rectangle", "none"}:
        return np.ones(size, dtype=np.float32)

    raise ValueError(
        f"Unsupported window type: {window}. "
        "Expected one of ['hann', 'hamming', 'rect', 'rectangle', 'none']."
    )


def apply_window(iq: np.ndarray, window: str = "hann", axis: int = -1) -> np.ndarray:
    """
    IQ 데이터에 window를 적용한다.

    입력 예:
    - 1D frame: shape = (1024,)
    - 여러 frame: shape = (31, 1024)
    - 여러 채널: shape = (2, 16384)

    Args:
        iq:
            complex IQ ndarray
        window:
            window 이름
        axis:
            window를 적용할 축.
            기본값 -1은 sample 축이다.

    Returns:
        window가 적용된 complex64 ndarray
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    size = iq.shape[axis]
    win = get_window(window, size)

    # broadcasting을 위해 window shape 조정
    shape = [1] * iq.ndim
    shape[axis] = size
    win = win.reshape(shape)

    return (iq * win).astype(np.complex64, copy=False)