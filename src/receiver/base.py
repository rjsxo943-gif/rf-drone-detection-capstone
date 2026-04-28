from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseReceiver(ABC):
    """
    모든 Receiver의 공통 인터페이스.

    현재 프로젝트 기준:
    - 처리 단위는 block
    - 1 block = 16,384 samples
    - Receiver는 IQ 데이터를 complex ndarray로 반환한다.
    - 반환 shape은 항상 (num_channels, num_samples)를 권장한다.

    예:
    - 단일 채널: (1, 16384)
    - Pluto+ RX0/RX1 2채널: (2, 16384)
    """

    def __init__(self, sample_rate: int, center_freq: int, num_channels: int = 1) -> None:
        self.sample_rate = int(sample_rate)
        self.center_freq = int(center_freq)
        self.num_channels = int(num_channels)

    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        num_samples 만큼 IQ sample을 읽는다.

        반환 형식:
        - dtype: complex64 또는 complex128
        - shape: (num_channels, num_samples)

        예:
        - 단일 채널: (1, num_samples)
        - 2채널: (2, num_samples)
        """
        raise NotImplementedError

    def read_block(self, block_size: int = 16384) -> np.ndarray:
        """
        block_size 만큼 IQ sample을 읽는다.

        현재 프로젝트의 기본 처리 단위는 16,384 samples/block이다.
        """
        return self.read_samples(block_size)

    def validate_samples(self, samples: np.ndarray, expected_samples: int | None = None) -> np.ndarray:
        """
        Receiver가 반환한 samples의 형식을 점검한다.

        기대 shape:
        - (num_channels, num_samples)
        """
        samples = np.asarray(samples)

        if samples.ndim == 1:
            # 단일 채널이 1차원으로 들어온 경우 (N,) → (1, N)으로 변환
            samples = samples[np.newaxis, :]

        if samples.ndim != 2:
            raise ValueError(
                f"Receiver output must be 2D array with shape "
                f"(num_channels, num_samples), got shape {samples.shape}"
            )

        if samples.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channel(s), "
                f"but got {samples.shape[0]} channel(s). Shape: {samples.shape}"
            )

        if expected_samples is not None and samples.shape[1] != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, "
                f"but got {samples.shape[1]} samples. Shape: {samples.shape}"
            )

        if not np.iscomplexobj(samples):
            raise TypeError(
                f"Receiver output must be complex IQ samples, got dtype {samples.dtype}"
            )

        return samples.astype(np.complex64, copy=False)

    def close(self) -> None:
        """
        Receiver 자원 해제.

        SDR 연결을 닫거나, 파일 핸들을 닫을 때 하위 클래스에서 override한다.
        """
        return None

    def __enter__(self) -> "BaseReceiver":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()