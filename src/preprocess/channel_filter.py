from __future__ import annotations

import numpy as np


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ 데이터를 (num_channels, num_samples) 형태로 맞춘다.

    입력 예:
    - (N,)      -> (1, N)
    - (C, N)   -> 그대로 사용

    주의:
    - 이 함수는 데이터 값을 보정하지 않는다.
    - shape만 확인하고 필요하면 2D로 바꾼다.
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


def select_rx(iq: np.ndarray, rx_index: int = 0) -> np.ndarray:
    """
    RX0 또는 RX1 중 하나를 선택한다.

    입력:
    - iq shape = (num_channels, num_samples)
    - Pluto+ 2채널이면 보통 (2, 16384)

    출력:
    - 선택된 1D IQ block
    - shape = (num_samples,)

    주의:
    - 이 함수는 데이터 값을 바꾸지 않는다.
    - DC offset 제거, 정규화, gain matching, phase offset 보정은 여기서 하지 않는다.
    """
    iq = ensure_2d_iq(iq)

    if rx_index < 0 or rx_index >= iq.shape[0]:
        raise IndexError(
            f"rx_index out of range: {rx_index}. "
            f"Available RX index: 0 ~ {iq.shape[0] - 1}"
        )

    return iq[rx_index].astype(np.complex64, copy=False)


def get_cnn_input_iq(iq: np.ndarray, rx_index: int = 0) -> np.ndarray:
    """
    CNN spectrogram 생성에 사용할 IQ 채널을 선택한다.

    현재 기본 정책:
    - CNN branch는 RX0 사용
    - AoA branch는 RX0/RX1 둘 다 사용

    이 함수는 값 보정이 아니라 선택만 수행한다.
    """
    return select_rx(iq, rx_index=rx_index)