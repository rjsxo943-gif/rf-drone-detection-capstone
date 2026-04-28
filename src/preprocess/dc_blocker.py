from __future__ import annotations

import numpy as np


def remove_dc_offset(iq: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    IQ 데이터에서 DC offset을 제거한다.

    현재 프로젝트 기준:
    - 입력은 보통 block 단위 IQ
    - shape 예시:
      - 단일 채널 block: (1, 16384)
      - 2채널 block: (2, 16384)
    - 채널별로 평균값을 빼서 DC offset을 제거한다.

    Args:
        iq:
            complex IQ ndarray
        axis:
            평균을 계산할 축.
            기본값 -1은 sample 축을 의미한다.

    Returns:
        DC offset이 제거된 complex64 ndarray
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    dc = np.mean(iq, axis=axis, keepdims=True)
    corrected = iq - dc

    return corrected.astype(np.complex64, copy=False)


def estimate_dc_offset(iq: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    IQ 데이터의 DC offset 값을 추정한다.

    Args:
        iq:
            complex IQ ndarray
        axis:
            평균을 계산할 축.

    Returns:
        채널별 DC offset.
        예:
        - 입력 shape (2, 16384)
        - 출력 shape (2, 1)
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    return np.mean(iq, axis=axis, keepdims=True).astype(np.complex64)


class DCBlocker:
    """
    DC offset 제거기.

    함수형으로는 remove_dc_offset()을 쓰면 되고,
    파이프라인에서 객체 형태가 필요하면 이 클래스를 사용한다.
    """

    def __init__(self, axis: int = -1) -> None:
        self.axis = axis
        self.last_dc_offset: np.ndarray | None = None

    def transform(self, iq: np.ndarray) -> np.ndarray:
        """
        IQ 데이터에서 DC offset을 제거한다.
        """
        self.last_dc_offset = estimate_dc_offset(iq, axis=self.axis)
        corrected = np.asarray(iq) - self.last_dc_offset

        return corrected.astype(np.complex64, copy=False)

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        return self.transform(iq)