from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseOffsetEstimate:
    """
    RX0/RX1 채널 간 고정 위상 오프셋 추정 결과.
    """

    phase_offset_rad: float
    phase_offset_deg: float
    coherence_like: float
    num_samples: int


def wrap_phase_rad(phase: float | np.ndarray) -> float | np.ndarray:
    """
    위상을 -pi ~ +pi 범위로 정리한다.
    """
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ 데이터를 (num_channels, num_samples) 형태로 맞춘다.

    예:
    - (16384,)   -> (1, 16384)
    - (2, 16384) -> 그대로 사용
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


def estimate_phase_offset(
    iq: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
    eps: float = 1e-12,
) -> PhaseOffsetEstimate:
    """
    RX0/RX1 사이의 고정 위상 오프셋을 추정한다.

    기준:
    - ref_channel = 0이면 RX0 기준
    - target_channel = 1이면 RX1의 위상이 RX0보다 얼마나 밀렸는지 계산

    계산식:
        offset = angle(mean(RX_target * conj(RX_ref)))

    즉:
        phase_offset_rad = RX1 위상 - RX0 위상

    Args:
        iq:
            complex IQ block 또는 calibration용 IQ.
            shape = (2, N) 권장.
        ref_channel:
            기준 채널. 기본 RX0.
        target_channel:
            보정할 채널. 기본 RX1.
        eps:
            0 나눗셈 방지용.

    Returns:
        PhaseOffsetEstimate
    """
    iq = ensure_2d_iq(iq)

    num_channels, num_samples = iq.shape

    if num_channels < 2:
        raise ValueError(
            f"At least 2 channels are required to estimate phase offset, got {iq.shape}"
        )

    if ref_channel < 0 or ref_channel >= num_channels:
        raise IndexError(f"ref_channel out of range: {ref_channel}")

    if target_channel < 0 or target_channel >= num_channels:
        raise IndexError(f"target_channel out of range: {target_channel}")

    if ref_channel == target_channel:
        raise ValueError("ref_channel and target_channel must be different.")

    ref = iq[ref_channel]
    target = iq[target_channel]

    cross = target * np.conj(ref)
    mean_cross = np.mean(cross)

    phase_offset_rad = float(wrap_phase_rad(np.angle(mean_cross)))
    phase_offset_deg = float(np.rad2deg(phase_offset_rad))

    numerator = np.abs(np.mean(cross))
    denominator = np.sqrt(
        np.mean(np.abs(ref) ** 2) * np.mean(np.abs(target) ** 2)
    ) + eps

    coherence_like = float(numerator / denominator)

    return PhaseOffsetEstimate(
        phase_offset_rad=phase_offset_rad,
        phase_offset_deg=phase_offset_deg,
        coherence_like=coherence_like,
        num_samples=num_samples,
    )


def remove_phase_offset(
    iq: np.ndarray,
    phase_offset_rad: float,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> np.ndarray:
    """
    target_channel의 고정 위상 오프셋을 제거한다.

    만약 phase_offset_rad가 RX1 - RX0라면,
    RX1에 exp(-j * phase_offset_rad)를 곱해서 RX0 기준으로 맞춘다.

    Args:
        iq:
            complex IQ array.
            shape = (2, N)
        phase_offset_rad:
            제거할 위상 오프셋 [rad]
        ref_channel:
            기준 채널. 기본 RX0.
        target_channel:
            보정할 채널. 기본 RX1.

    Returns:
        phase offset이 제거된 IQ.
        shape은 입력과 동일.
    """
    iq = ensure_2d_iq(iq).copy()

    num_channels, _ = iq.shape

    if ref_channel < 0 or ref_channel >= num_channels:
        raise IndexError(f"ref_channel out of range: {ref_channel}")

    if target_channel < 0 or target_channel >= num_channels:
        raise IndexError(f"target_channel out of range: {target_channel}")

    if ref_channel == target_channel:
        raise ValueError("ref_channel and target_channel must be different.")

    correction = np.exp(-1j * float(phase_offset_rad)).astype(np.complex64)

    iq[target_channel] = iq[target_channel] * correction

    return iq.astype(np.complex64, copy=False)


def estimate_and_remove_phase_offset(
    iq: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> tuple[np.ndarray, PhaseOffsetEstimate]:
    """
    현재 IQ block에서 phase offset을 추정한 뒤 바로 제거한다.

    주의:
    - 이 함수는 실험/확인용으로는 편하다.
    - 실제 AoA에서는 calibration 단계에서 구한 고정 phase_offset_rad를
      계속 적용하는 방식이 더 안정적이다.
    """
    estimate = estimate_phase_offset(
        iq,
        ref_channel=ref_channel,
        target_channel=target_channel,
    )

    corrected = remove_phase_offset(
        iq,
        phase_offset_rad=estimate.phase_offset_rad,
        ref_channel=ref_channel,
        target_channel=target_channel,
    )

    return corrected, estimate


class PhaseOffsetCorrector:
    """
    RX0/RX1 phase offset 보정기.

    사용 방식:
    1. calibration 신호로 phase_offset_rad를 추정
    2. 이후 모든 block에 같은 phase_offset_rad를 적용
    """

    def __init__(
        self,
        phase_offset_rad: float = 0.0,
        ref_channel: int = 0,
        target_channel: int = 1,
    ) -> None:
        self.phase_offset_rad = float(phase_offset_rad)
        self.ref_channel = int(ref_channel)
        self.target_channel = int(target_channel)

    def fit(self, iq: np.ndarray) -> PhaseOffsetEstimate:
        """
        calibration용 IQ에서 phase offset을 추정하고 내부 값으로 저장한다.
        """
        estimate = estimate_phase_offset(
            iq,
            ref_channel=self.ref_channel,
            target_channel=self.target_channel,
        )

        self.phase_offset_rad = estimate.phase_offset_rad

        return estimate

    def transform(self, iq: np.ndarray) -> np.ndarray:
        """
        저장된 phase_offset_rad를 이용해서 IQ를 보정한다.
        """
        return remove_phase_offset(
            iq,
            phase_offset_rad=self.phase_offset_rad,
            ref_channel=self.ref_channel,
            target_channel=self.target_channel,
        )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        return self.transform(iq)