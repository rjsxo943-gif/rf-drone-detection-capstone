from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GainMatchEstimate:
    """
    RX0/RX1 채널 간 gain mismatch 추정 결과.
    """

    ref_rms: float
    target_rms: float
    gain_correction: float
    num_samples: int


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ 데이터를 (num_channels, num_samples) 형태로 맞춘다.
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


def estimate_gain_mismatch(
    iq: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
    eps: float = 1e-12,
) -> GainMatchEstimate:
    """
    RX0/RX1 사이의 amplitude gain 차이를 추정한다.

    기준:
    - ref_channel의 RMS 크기를 기준으로 둔다.
    - target_channel에 곱할 보정 계수를 계산한다.

    계산식:
        gain_correction = rms(ref) / rms(target)

    즉:
        RX1이 RX0보다 작게 들어오면 gain_correction > 1
        RX1이 RX0보다 크게 들어오면 gain_correction < 1
    """
    iq = ensure_2d_iq(iq)

    num_channels, num_samples = iq.shape

    if num_channels < 2:
        raise ValueError(
            f"At least 2 channels are required to estimate gain mismatch, got {iq.shape}"
        )

    if ref_channel < 0 or ref_channel >= num_channels:
        raise IndexError(f"ref_channel out of range: {ref_channel}")

    if target_channel < 0 or target_channel >= num_channels:
        raise IndexError(f"target_channel out of range: {target_channel}")

    if ref_channel == target_channel:
        raise ValueError("ref_channel and target_channel must be different.")

    ref = iq[ref_channel]
    target = iq[target_channel]

    ref_rms = float(np.sqrt(np.mean(np.abs(ref) ** 2)))
    target_rms = float(np.sqrt(np.mean(np.abs(target) ** 2)))

    gain_correction = float(ref_rms / (target_rms + eps))

    return GainMatchEstimate(
        ref_rms=ref_rms,
        target_rms=target_rms,
        gain_correction=gain_correction,
        num_samples=num_samples,
    )


def apply_gain_correction(
    iq: np.ndarray,
    gain_correction: float,
    target_channel: int = 1,
) -> np.ndarray:
    """
    target_channel에 gain 보정 계수를 적용한다.
    """
    iq = ensure_2d_iq(iq).copy()

    num_channels, _ = iq.shape

    if target_channel < 0 or target_channel >= num_channels:
        raise IndexError(f"target_channel out of range: {target_channel}")

    iq[target_channel] = iq[target_channel] * float(gain_correction)

    return iq.astype(np.complex64, copy=False)


def estimate_and_apply_gain_correction(
    iq: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> tuple[np.ndarray, GainMatchEstimate]:
    """
    현재 IQ block에서 gain mismatch를 추정한 뒤 바로 보정한다.

    주의:
    - 실험/확인용으로는 편하다.
    - 실제 시스템에서는 calibration 단계에서 구한 고정 gain_correction을
      계속 적용하는 방식이 더 안정적이다.
    """
    estimate = estimate_gain_mismatch(
        iq,
        ref_channel=ref_channel,
        target_channel=target_channel,
    )

    corrected = apply_gain_correction(
        iq,
        gain_correction=estimate.gain_correction,
        target_channel=target_channel,
    )

    return corrected, estimate


class GainMatcher:
    """
    RX0/RX1 gain mismatch 보정기.

    사용 방식:
    1. calibration 신호로 gain_correction 추정
    2. 이후 모든 block에 같은 gain_correction 적용
    """

    def __init__(
        self,
        gain_correction: float = 1.0,
        ref_channel: int = 0,
        target_channel: int = 1,
    ) -> None:
        self.gain_correction = float(gain_correction)
        self.ref_channel = int(ref_channel)
        self.target_channel = int(target_channel)

    def fit(self, iq: np.ndarray) -> GainMatchEstimate:
        """
        calibration용 IQ에서 gain mismatch를 추정하고 내부 값으로 저장한다.
        """
        estimate = estimate_gain_mismatch(
            iq,
            ref_channel=self.ref_channel,
            target_channel=self.target_channel,
        )

        self.gain_correction = estimate.gain_correction

        return estimate

    def transform(self, iq: np.ndarray) -> np.ndarray:
        """
        저장된 gain_correction을 이용해서 IQ를 보정한다.
        """
        return apply_gain_correction(
            iq,
            gain_correction=self.gain_correction,
            target_channel=self.target_channel,
        )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        return self.transform(iq)