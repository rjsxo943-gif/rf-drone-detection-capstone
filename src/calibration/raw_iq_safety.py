from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from src.preprocess import ensure_2d_iq


@dataclass(frozen=True)
class RawIQSafetyResult:
    """
    Raw IQ block의 안전 상태를 요약한 결과.

    목적:
    - ADC clipping / saturation 여부 확인
    - DC offset 과다 여부 확인
    - 신호가 너무 약하거나 너무 강한지 확인
    """

    status: str
    is_safe: bool
    max_abs: float
    rms: float
    mean_i: float
    mean_q: float
    dc_abs: float
    saturation_ratio: float
    near_saturation_ratio: float
    num_samples: int
    num_channels: int
    full_scale: float
    saturation_threshold: float
    near_saturation_threshold: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_project_iq(block: np.ndarray) -> np.ndarray:
    """
    프로젝트 표준 IQ shape으로 정리한다.

    표준:
    - shape = (num_channels, num_samples)

    예외:
    - (num_samples, num_channels)로 들어오면 transpose
    """
    iq = ensure_2d_iq(np.asarray(block))

    if iq.ndim != 2:
        raise ValueError(f"IQ block must be 2D, got shape={iq.shape}")

    if iq.shape[0] in (1, 2):
        return iq.astype(np.complex64, copy=False)

    if iq.shape[1] in (1, 2):
        return iq.T.astype(np.complex64, copy=False)

    raise ValueError(
        f"Unsupported IQ shape. Expected (C, N) or (N, C), got {iq.shape}"
    )


def check_raw_iq_safety(
    block: np.ndarray,
    *,
    full_scale: float = 1.0,
    saturation_ratio_warn: float = 0.001,
    saturation_ratio_clip: float = 0.01,
    near_saturation_level: float = 0.85,
    saturation_level: float = 0.98,
    min_rms: float | None = None,
    max_dc_abs: float | None = None,
) -> RawIQSafetyResult:
    """
    Raw IQ block의 safety 상태를 검사한다.

    Parameters
    ----------
    block:
        IQ block. shape은 (C, N) 또는 (N, C) 허용.

    full_scale:
        정규화된 IQ 기준이면 1.0.
        ADC integer raw 기준이면 예: 2048.0.

    saturation_level:
        full_scale의 몇 배 이상이면 saturation 후보로 볼지.
        기본 0.98.

    near_saturation_level:
        full_scale의 몇 배 이상이면 near-saturation으로 볼지.
        기본 0.85.

    saturation_ratio_warn:
        saturation sample 비율이 이 값 이상이면 WARNING.

    saturation_ratio_clip:
        saturation sample 비율이 이 값 이상이면 CLIPPED.

    min_rms:
        너무 약한 신호를 보고 싶을 때 사용.
        None이면 검사하지 않음.

    max_dc_abs:
        DC offset 크기 제한.
        None이면 검사하지 않음.

    Returns
    -------
    RawIQSafetyResult
    """
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")

    iq = _to_project_iq(block)

    abs_iq = np.abs(iq).astype(np.float64)
    max_abs = float(np.max(abs_iq))
    rms = float(np.sqrt(np.mean(abs_iq**2)))

    mean_complex = np.mean(iq)
    mean_i = float(np.real(mean_complex))
    mean_q = float(np.imag(mean_complex))
    dc_abs = float(np.abs(mean_complex))

    saturation_threshold = float(full_scale * saturation_level)
    near_saturation_threshold = float(full_scale * near_saturation_level)

    saturation_ratio = float(np.mean(abs_iq >= saturation_threshold))
    near_saturation_ratio = float(np.mean(abs_iq >= near_saturation_threshold))

    num_channels = int(iq.shape[0])
    num_samples = int(iq.shape[1])

    status = "SAFE"
    notes: list[str] = []

    if saturation_ratio >= saturation_ratio_clip:
        status = "CLIPPED"
        notes.append("saturation_ratio exceeds clipped threshold")
    elif saturation_ratio >= saturation_ratio_warn:
        status = "WARNING"
        notes.append("saturation_ratio exceeds warning threshold")
    elif near_saturation_ratio > 0:
        status = "WARNING"
        notes.append("near saturation samples detected")

    if min_rms is not None and rms < float(min_rms):
        if status == "SAFE":
            status = "WEAK"
        notes.append("rms is below min_rms")

    if max_dc_abs is not None and dc_abs > float(max_dc_abs):
        if status == "SAFE":
            status = "WARNING"
        notes.append("dc_abs exceeds max_dc_abs")

    is_safe = status == "SAFE"

    if not notes:
        notes.append("raw IQ block is within safety limits")

    return RawIQSafetyResult(
        status=status,
        is_safe=is_safe,
        max_abs=max_abs,
        rms=rms,
        mean_i=mean_i,
        mean_q=mean_q,
        dc_abs=dc_abs,
        saturation_ratio=saturation_ratio,
        near_saturation_ratio=near_saturation_ratio,
        num_samples=num_samples,
        num_channels=num_channels,
        full_scale=float(full_scale),
        saturation_threshold=saturation_threshold,
        near_saturation_threshold=near_saturation_threshold,
        note="; ".join(notes),
    )


def is_raw_iq_safe(
    block: np.ndarray,
    **kwargs: Any,
) -> bool:
    """
    간단한 boolean safety check helper.
    """
    return check_raw_iq_safety(block, **kwargs).is_safe


def summarize_raw_iq_safety(result: RawIQSafetyResult) -> str:
    """
    CLI / live viewer overlay용 한 줄 요약 문자열.
    """
    return (
        f"status={result.status} | "
        f"max_abs={result.max_abs:.4g} | "
        f"rms={result.rms:.4g} | "
        f"dc_abs={result.dc_abs:.4g} | "
        f"sat={result.saturation_ratio * 100:.3f}% | "
        f"near_sat={result.near_saturation_ratio * 100:.3f}%"
    )