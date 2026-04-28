from __future__ import annotations

from typing import Sequence

import numpy as np


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ 데이터를 (num_channels, num_samples) 형태로 맞춘다.

    입력 예:
    - (N,)    -> (1, N)
    - (C, N) -> 그대로 사용

    Returns:
        np.ndarray
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


def select_channel(iq: np.ndarray, channel: int = 0) -> np.ndarray:
    """
    여러 채널 중 하나의 채널만 선택한다.

    현재 프로젝트에서 주로 CNN branch에 사용한다.

    예:
    - 입력 shape: (2, 16384)
    - channel=0 -> RX0만 선택
    - 출력 shape: (16384,)

    Args:
        iq:
            complex IQ array, shape = (C, N) 또는 (N,)
        channel:
            선택할 채널 index

    Returns:
        선택된 1D complex IQ array, shape = (N,)
    """
    iq = ensure_2d_iq(iq)

    if channel < 0 or channel >= iq.shape[0]:
        raise IndexError(
            f"Channel index out of range: {channel}. "
            f"Available channels: 0 ~ {iq.shape[0] - 1}"
        )

    return iq[channel].astype(np.complex64, copy=False)


def select_channels(iq: np.ndarray, channels: Sequence[int]) -> np.ndarray:
    """
    여러 채널 중 원하는 채널들만 선택한다.

    예:
    - 입력 shape: (2, 16384)
    - channels=[0, 1]
    - 출력 shape: (2, 16384)

    Args:
        iq:
            complex IQ array, shape = (C, N)
        channels:
            선택할 채널 index 목록

    Returns:
        선택된 channel IQ array, shape = (len(channels), N)
    """
    iq = ensure_2d_iq(iq)

    if len(channels) == 0:
        raise ValueError("channels must not be empty.")

    for ch in channels:
        if ch < 0 or ch >= iq.shape[0]:
            raise IndexError(
                f"Channel index out of range: {ch}. "
                f"Available channels: 0 ~ {iq.shape[0] - 1}"
            )

    return iq[list(channels)].astype(np.complex64, copy=False)


def combine_channels(iq: np.ndarray, method: str = "mean_power") -> np.ndarray:
    """
    여러 채널을 CNN 입력용 단일 1D IQ 또는 power 신호로 합친다.

    주의:
    - AoA branch에서는 이 함수를 쓰면 안 된다.
    - AoA는 RX0/RX1 위상차가 필요하므로 채널을 합치면 정보가 사라진다.
    - 이 함수는 CNN branch나 에너지 확인용으로만 사용한다.

    Args:
        iq:
            complex IQ array, shape = (C, N)
        method:
            - "rx0"          : RX0만 사용
            - "mean_complex" : 복소 IQ를 채널 평균
            - "mean_power"   : 채널별 power 평균 후 sqrt로 amplitude 형태 생성
            - "sum_power"    : 채널별 power 합산 후 sqrt로 amplitude 형태 생성

    Returns:
        1D complex64 array, shape = (N,)
    """
    iq = ensure_2d_iq(iq)
    method = method.lower().strip()

    if method == "rx0":
        return iq[0].astype(np.complex64, copy=False)

    if method == "mean_complex":
        return np.mean(iq, axis=0).astype(np.complex64)

    if method == "mean_power":
        power = np.mean(np.abs(iq) ** 2, axis=0)
        amplitude = np.sqrt(power)

        # 위상 정보는 RX0의 위상을 사용한다.
        # CNN용 spectrogram power를 만들 목적이면 위상 영향은 크지 않다.
        phase = np.angle(iq[0])
        combined = amplitude * np.exp(1j * phase)

        return combined.astype(np.complex64)

    if method == "sum_power":
        power = np.sum(np.abs(iq) ** 2, axis=0)
        amplitude = np.sqrt(power)

        phase = np.angle(iq[0])
        combined = amplitude * np.exp(1j * phase)

        return combined.astype(np.complex64)

    raise ValueError(
        f"Unsupported combine method: {method}. "
        "Expected one of ['rx0', 'mean_complex', 'mean_power', 'sum_power']."
    )


def get_cnn_input_iq(
    iq: np.ndarray,
    mode: str = "rx0",
    channel: int = 0,
) -> np.ndarray:
    """
    CNN spectrogram 생성 전에 사용할 1D IQ를 선택한다.

    현재 추천:
    - 기본값 mode='rx0'
    - 즉, RX0만 사용해서 spectrogram 생성

    Args:
        iq:
            complex IQ array, shape = (C, N)
        mode:
            - "rx0"          : RX0 사용
            - "single"       : channel 인자로 지정한 채널 사용
            - "mean_complex" : 복소 평균
            - "mean_power"   : power 평균
            - "sum_power"    : power 합산
        channel:
            mode='single'일 때 사용할 채널 번호

    Returns:
        1D complex64 IQ array, shape = (N,)
    """
    mode = mode.lower().strip()

    if mode == "single":
        return select_channel(iq, channel=channel)

    if mode == "rx0":
        return select_channel(iq, channel=0)

    return combine_channels(iq, method=mode)