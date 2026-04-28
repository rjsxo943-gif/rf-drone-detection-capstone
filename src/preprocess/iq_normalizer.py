from __future__ import annotations

import numpy as np


def normalize_iq(
    iq: np.ndarray,
    eps: float = 1e-12,
    axis: int = -1,
    method: str = "peak",
) -> np.ndarray:
    """
    IQ 신호의 진폭 크기를 정규화한다.

    현재 프로젝트 기준:
    - 입력 shape 예시:
      - 단일 채널: (1, 16384)
      - 2채널: (2, 16384)
    - 기본적으로 sample 축(axis=-1)을 기준으로 채널별 정규화한다.
    주의:
        - 이 함수는 gain matching이 아니다.
        - CNN/STFT 입력 안정화를 위한 amplitude normalization이다.
        - AoA branch에서는 보통 이 함수를 쓰지 않고 phaseoffset.py를 사용한다.
    

    Args:
        iq:
            complex IQ ndarray
        eps:
            0으로 나누는 것을 방지하기 위한 작은 값
        axis:
            정규화 기준을 계산할 축.
            기본값 -1은 sample 축이다.
        method:
            "peak": 최대 진폭 기준 정규화
            "rms" : RMS 기준 정규화

    Returns:
        정규화된 complex64 IQ ndarray
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    method = method.lower().strip()

    if method == "peak":
        scale = np.max(np.abs(iq), axis=axis, keepdims=True)

    elif method == "rms":
        scale = np.sqrt(np.mean(np.abs(iq) ** 2, axis=axis, keepdims=True))

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    scale = np.maximum(scale, eps)

    normalized = iq / scale

    return normalized.astype(np.complex64, copy=False)


def estimate_iq_scale(
    iq: np.ndarray,
    eps: float = 1e-12,
    axis: int = -1,
    method: str = "peak",
) -> np.ndarray:
    """
    IQ 정규화에 사용되는 scale 값을 계산한다.

    예:
    - 입력 shape: (2, 16384)
    - 출력 shape: (2, 1)
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ array is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ data must be complex, got dtype={iq.dtype}")

    method = method.lower().strip()

    if method == "peak":
        scale = np.max(np.abs(iq), axis=axis, keepdims=True)

    elif method == "rms":
        scale = np.sqrt(np.mean(np.abs(iq) ** 2, axis=axis, keepdims=True))

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return np.maximum(scale, eps).astype(np.float32)


class IQNormalizer:
    """
    IQ 정규화 객체.

    함수형으로는 normalize_iq()를 쓰면 되고,
    pipeline에서 객체 형태가 필요하면 이 클래스를 사용한다.
    """

    def __init__(
        self,
        eps: float = 1e-12,
        axis: int = -1,
        method: str = "peak",
    ) -> None:
        self.eps = eps
        self.axis = axis
        self.method = method
        self.last_scale: np.ndarray | None = None

    def transform(self, iq: np.ndarray) -> np.ndarray:
        self.last_scale = estimate_iq_scale(
            iq,
            eps=self.eps,
            axis=self.axis,
            method=self.method,
        )

        normalized = np.asarray(iq) / self.last_scale

        return normalized.astype(np.complex64, copy=False)

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        return self.transform(iq)