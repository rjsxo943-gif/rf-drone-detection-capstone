from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseDiffResult:
    """
    block 하나에 대한 위상차 계산 결과.
    """

    phase_diff_rad: float
    phase_diff_deg: float
    coherence_like: float
    num_samples: int


def wrap_phase_rad(phase: float | np.ndarray) -> float | np.ndarray:
    """
    위상을 -pi ~ +pi 범위로 정리한다.
    """
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def estimate_phase_diff(
    iq_block: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
    eps: float = 1e-12,
) -> PhaseDiffResult:
    """
    RX0/RX1 block의 평균 위상차를 계산한다.

    현재 프로젝트 기준:
    - 입력 shape = (num_channels, block_size)
    - Pluto+ 2채널이면 shape = (2, 16384)
    - ref_channel=0이면 RX0 기준
    - target_channel=1이면 RX1 위상에서 RX0 위상을 뺀 값 계산

    계산식:
        phase_diff = angle(mean(RX_target * conj(RX_ref)))

    Args:
        iq_block:
            complex IQ block.
            shape = (2, 16384) 권장
        ref_channel:
            기준 채널. 기본 RX0.
        target_channel:
            비교 채널. 기본 RX1.
        eps:
            0 나눗셈 방지용 작은 값.

    Returns:
        PhaseDiffResult
    """
    iq_block = _ensure_2d_iq(iq_block)

    num_channels, num_samples = iq_block.shape

    if ref_channel < 0 or ref_channel >= num_channels:
        raise IndexError(
            f"ref_channel out of range: {ref_channel}. "
            f"Available channels: 0 ~ {num_channels - 1}"
        )

    if target_channel < 0 or target_channel >= num_channels:
        raise IndexError(
            f"target_channel out of range: {target_channel}. "
            f"Available channels: 0 ~ {num_channels - 1}"
        )

    if ref_channel == target_channel:
        raise ValueError("ref_channel and target_channel must be different.")

    ref = iq_block[ref_channel]
    target = iq_block[target_channel]

    # RX_target * conj(RX_ref)
    # angle 값은 target phase - ref phase 의미
    cross = target * np.conj(ref)

    mean_cross = np.mean(cross)

    phase_diff_rad = float(np.angle(mean_cross))
    phase_diff_rad = float(wrap_phase_rad(phase_diff_rad))
    phase_diff_deg = float(np.rad2deg(phase_diff_rad))

    # coherence와 비슷한 신뢰도 지표.
    # 1에 가까울수록 두 채널 위상 관계가 안정적이라는 의미.
    numerator = np.abs(np.mean(cross))
    denominator = np.sqrt(
        np.mean(np.abs(ref) ** 2) * np.mean(np.abs(target) ** 2)
    ) + eps

    coherence_like = float(numerator / denominator)

    return PhaseDiffResult(
        phase_diff_rad=phase_diff_rad,
        phase_diff_deg=phase_diff_deg,
        coherence_like=coherence_like,
        num_samples=num_samples,
    )


def estimate_phase_diff_rad(
    iq_block: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> float:
    """
    위상차 radian 값만 간단히 반환한다.
    """
    result = estimate_phase_diff(
        iq_block,
        ref_channel=ref_channel,
        target_channel=target_channel,
    )
    return result.phase_diff_rad


def estimate_phase_diff_deg(
    iq_block: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> float:
    """
    위상차 degree 값만 간단히 반환한다.
    """
    result = estimate_phase_diff(
        iq_block,
        ref_channel=ref_channel,
        target_channel=target_channel,
    )
    return result.phase_diff_deg


def compute_instant_phase_diff(
    iq_block: np.ndarray,
    ref_channel: int = 0,
    target_channel: int = 1,
) -> np.ndarray:
    """
    sample별 순간 위상차를 계산한다.

    출력:
        shape = (block_size,)

    주의:
    - AoA 최종 계산에는 보통 estimate_phase_diff()의 평균 위상차를 사용한다.
    - 이 함수는 디버깅이나 위상 흔들림 확인용이다.
    """
    iq_block = _ensure_2d_iq(iq_block)

    ref = iq_block[ref_channel]
    target = iq_block[target_channel]

    phase_diff = np.angle(target * np.conj(ref))
    phase_diff = wrap_phase_rad(phase_diff)

    return phase_diff.astype(np.float32)


def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    """
    IQ block을 (num_channels, num_samples) 형태로 맞춘다.
    """
    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("Input IQ block is empty.")

    if not np.iscomplexobj(iq):
        raise TypeError(f"IQ block must be complex, got dtype={iq.dtype}")

    if iq.ndim == 1:
        iq = iq[np.newaxis, :]

    if iq.ndim != 2:
        raise ValueError(
            f"IQ block must be 1D or 2D. "
            f"Expected (N,) or (C, N), got shape {iq.shape}"
        )

    if iq.shape[0] < 2:
        raise ValueError(
            f"At least 2 channels are required for phase difference, "
            f"got shape {iq.shape}"
        )

    return iq.astype(np.complex64, copy=False)