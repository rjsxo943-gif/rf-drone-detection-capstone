from __future__ import annotations

import numpy as np


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ 데이터를 (num_channels, num_samples) 형태로 맞춘다.

    입력 예:
    - (N,)      -> (1, N)
    - (C, N)   -> 그대로 사용

    Returns:
        shape = (num_channels, num_samples)
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    if iq.ndim == 1:
        iq = iq[np.newaxis, :]

    if iq.ndim != 2:
        raise ValueError(
            f"IQ array must be 1D or 2D. Expected (N,) or (C, N), got {iq.shape}"
        )

    return iq.astype(np.complex64, copy=False)


def split_into_blocks(
    iq: np.ndarray,
    block_size: int = 16_384,
    drop_last: bool = True,
) -> np.ndarray:
    """
    IQ 데이터를 block 단위로 자른다.

    현재 프로젝트 기준:
    - 1 block = 16,384 samples
    - 입력 shape = (num_channels, num_samples)
    - 출력 shape = (num_blocks, num_channels, block_size)

    Args:
        iq:
            complex IQ array.
            shape: (N,) 또는 (C, N)
        block_size:
            block 하나의 sample 수.
        drop_last:
            True이면 block_size보다 짧은 마지막 조각은 버린다.
            False이면 마지막 조각을 zero padding해서 포함한다.

    Returns:
        blocks:
            shape = (num_blocks, num_channels, block_size)
    """
    iq = ensure_2d_iq(iq)
    block_size = int(block_size)

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    num_channels, num_samples = iq.shape

    if num_samples < block_size:
        if drop_last:
            return np.empty((0, num_channels, block_size), dtype=np.complex64)

        padded = np.zeros((num_channels, block_size), dtype=np.complex64)
        padded[:, :num_samples] = iq
        return padded[np.newaxis, :, :]

    num_full_blocks = num_samples // block_size
    remainder = num_samples % block_size

    blocks = []

    for block_index in range(num_full_blocks):
        start = block_index * block_size
        end = start + block_size
        blocks.append(iq[:, start:end])

    if not drop_last and remainder > 0:
        start = num_full_blocks * block_size
        last = iq[:, start:]

        padded = np.zeros((num_channels, block_size), dtype=np.complex64)
        padded[:, : last.shape[1]] = last
        blocks.append(padded)

    if not blocks:
        return np.empty((0, num_channels, block_size), dtype=np.complex64)

    return np.stack(blocks, axis=0).astype(np.complex64, copy=False)


def get_num_blocks(
    num_samples: int,
    block_size: int = 16_384,
    drop_last: bool = True,
) -> int:
    """
    전체 sample 수에서 몇 개의 block이 나오는지 계산한다.
    """
    num_samples = int(num_samples)
    block_size = int(block_size)

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if drop_last:
        return num_samples // block_size

    return int(np.ceil(num_samples / block_size))


def frame_signal(iq: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    1D IQ 신호를 작은 frame 단위로 자른다.

    주의:
    - 이 함수는 전체 파이프라인 block 분할용이 아니다.
    - energy detector 내부에서 block 안을 더 작은 window로 나눌 때 사용한다.

    예:
    - block_size = 16384
    - frame_size = 1024
    - hop_size = 512
    - 출력 frame 개수 = floor((16384 - 1024) / 512) + 1 = 31

    Args:
        iq:
            1D complex IQ array.
        frame_size:
            frame 하나의 sample 수.
        hop_size:
            frame 이동 간격.

    Returns:
        frames:
            shape = (num_frames, frame_size)
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    if iq.ndim != 1:
        raise ValueError(
            f"frame_signal expects 1D IQ array. Got shape {iq.shape}. "
            "For block splitting, use split_into_blocks()."
        )

    frame_size = int(frame_size)
    hop_size = int(hop_size)

    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}")

    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    if len(iq) < frame_size:
        return np.empty((0, frame_size), dtype=np.complex64)

    frames = []

    for start in range(0, len(iq) - frame_size + 1, hop_size):
        frames.append(iq[start : start + frame_size])

    return np.asarray(frames, dtype=np.complex64)