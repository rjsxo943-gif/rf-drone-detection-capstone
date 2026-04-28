from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoherenceResult:
    coherence: float
    passed: bool
    threshold: float
    used_bins: int


def compute_stft_coherence(
    z0: np.ndarray,
    z1: np.ndarray,
    energy_percentile: float = 75.0,
    eps: float = 1e-12,
) -> tuple[float, int]:
    """
    RX0/RX1 complex STFT로부터 coherence 값을 계산한다.

    Parameters
    ----------
    z0, z1:
        complex STFT matrices.
        shape = (freq_bins, time_frames)

    energy_percentile:
        에너지가 높은 bin만 coherence 계산에 사용하기 위한 percentile.
        기본 75는 상위 25% 에너지 영역만 사용한다는 뜻이다.

    Returns
    -------
    coherence:
        0~1 사이 값.
        1에 가까울수록 두 채널이 같은 신호를 안정적으로 보고 있다는 뜻.

    used_bins:
        coherence 계산에 사용된 STFT bin 개수.
    """

    z0 = np.asarray(z0, dtype=np.complex64)
    z1 = np.asarray(z1, dtype=np.complex64)

    if z0.shape != z1.shape:
        raise ValueError(f"z0 and z1 must have same shape. got {z0.shape}, {z1.shape}")

    if z0.ndim != 2:
        raise ValueError(f"z0 and z1 must be 2-D STFT matrices. got ndim={z0.ndim}")

    power = 0.5 * (np.abs(z0) ** 2 + np.abs(z1) ** 2)

    threshold = np.percentile(power, energy_percentile)
    mask = power >= threshold

    if np.count_nonzero(mask) < 10:
        mask = np.ones_like(power, dtype=bool)

    x = z0[mask]
    y = z1[mask]

    cross = np.mean(x * np.conj(y))
    p0 = np.mean(np.abs(x) ** 2)
    p1 = np.mean(np.abs(y) ** 2)

    coherence = (np.abs(cross) ** 2) / (p0 * p1 + eps)

    coherence = float(np.clip(coherence, 0.0, 1.0))
    used_bins = int(x.size)

    return coherence, used_bins


def coherence_gate(
    z0: np.ndarray,
    z1: np.ndarray,
    threshold: float = 0.6,
    energy_percentile: float = 75.0,
) -> CoherenceResult:
    """
    coherence를 계산하고 threshold gate를 통과했는지 반환한다.
    """

    coherence, used_bins = compute_stft_coherence(
        z0=z0,
        z1=z1,
        energy_percentile=energy_percentile,
    )

    passed = coherence >= threshold

    return CoherenceResult(
        coherence=coherence,
        passed=passed,
        threshold=threshold,
        used_bins=used_bins,
    )