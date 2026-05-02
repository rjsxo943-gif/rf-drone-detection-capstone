from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import json
import numpy as np

from src.core import now_iso
from src.preprocess import (
    ensure_2d_iq,
    remove_dc_offset,
    estimate_dc_offset,
    estimate_gain_mismatch,
    apply_gain_correction,
    estimate_phase_offset,
)


EPS = 1e-12


@dataclass
class PhaseGainCalibrationResult:
    mode: str
    created_at: str

    num_blocks: int
    block_size: int
    num_channels: int

    sample_rate: float | None
    center_freq: float | None

    ref_channel: int
    target_channel: int

    gain_ref_rms_mean: float
    gain_ref_rms_std: float
    gain_target_rms_mean: float
    gain_target_rms_std: float

    gain_correction_mean: float
    gain_correction_std: float
    gain_correction_min: float
    gain_correction_max: float

    phase_offset_rad_mean: float
    phase_offset_rad_std: float
    phase_offset_deg_mean: float
    phase_offset_deg_std: float

    phase_offset_rad_min: float
    phase_offset_rad_max: float
    phase_offset_deg_min: float
    phase_offset_deg_max: float

    coherence_like_mean: float
    coherence_like_std: float
    coherence_like_min: float
    coherence_like_max: float

    dc_offset_ch0_real: float | None
    dc_offset_ch0_imag: float | None
    dc_offset_ch1_real: float | None
    dc_offset_ch1_imag: float | None

    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return path


def _to_project_iq(block: np.ndarray) -> np.ndarray:
    """
    프로젝트 표준 IQ shape으로 정리한다.

    표준:
    - shape = (num_channels, num_samples)
    - phase/gain calibration은 최소 2채널 필요

    예외적으로 (num_samples, num_channels) 형태가 들어오면 transpose한다.
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


def _estimate_dc_per_block(iq: np.ndarray) -> np.ndarray:
    """
    기존 src.preprocess.estimate_dc_offset() 사용.
    최종 shape은 항상 (C,)로 맞춘다.
    """
    dc = estimate_dc_offset(iq, axis=-1)
    dc = np.asarray(dc, dtype=np.complex64).squeeze()

    if dc.ndim == 0:
        dc = dc.reshape(1)

    if dc.ndim != 1:
        dc = np.mean(iq, axis=-1).astype(np.complex64)

    return dc


def _circular_mean_rad(phases: np.ndarray) -> float:
    phases = np.asarray(phases, dtype=np.float64)
    return float(np.angle(np.mean(np.exp(1j * phases))))


def _circular_std_rad(phases: np.ndarray) -> float:
    phases = np.asarray(phases, dtype=np.float64)

    r = np.abs(np.mean(np.exp(1j * phases)))
    r = np.clip(r, EPS, 1.0)

    return float(np.sqrt(-2.0 * np.log(r)))


def calibrate_phase_gain_from_blocks(
    blocks: list[np.ndarray],
    *,
    ref_channel: int = 0,
    target_channel: int = 1,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> PhaseGainCalibrationResult:
    """
    위상 캘리브레이션 + 게인 매칭 본체.

    이 파일은 새 알고리즘을 만드는 곳이 아니라,
    기존 src.preprocess 부품들을 묶어서 calibration result를 만드는 wrapper다.

    사용 기존 부품:
    - remove_dc_offset()
    - estimate_gain_mismatch()
    - apply_gain_correction()
    - estimate_phase_offset()

    처리 순서:
    1. IQ shape을 프로젝트 표준 (C, N)으로 정리
    2. DC offset 제거
    3. 기존 gain_matcher로 gain_correction 추정
    4. target channel에 gain correction 적용
    5. 기존 phaseoffset으로 phase_offset 추정
    6. 여러 block의 평균/표준편차/최솟값/최댓값 저장

    주의:
    - normalize_iq는 사용하지 않는다.
    - 신호원은 두 안테나 정면 0도 방향에 두는 것이 기준이다.
    """
    if not blocks:
        raise ValueError("No IQ blocks were provided for phase/gain calibration")

    ref_rms_values: list[float] = []
    target_rms_values: list[float] = []
    gain_corrections: list[float] = []
    phase_offsets_rad: list[float] = []
    phase_offsets_deg: list[float] = []
    coherence_values: list[float] = []
    dc_offsets: list[np.ndarray] = []

    block_size: int | None = None
    num_channels: int | None = None

    for block_index, block in enumerate(blocks):
        iq = _to_project_iq(block)

        if iq.shape[0] < 2:
            raise ValueError(
                f"Phase/gain calibration requires 2 channels, got shape={iq.shape}"
            )

        if block_size is None:
            block_size = iq.shape[1]
            num_channels = iq.shape[0]
        else:
            if iq.shape[1] != block_size:
                raise ValueError(
                    f"Inconsistent block size at index {block_index}: "
                    f"expected {block_size}, got {iq.shape[1]}"
                )

            if iq.shape[0] != num_channels:
                raise ValueError(
                    f"Inconsistent channel count at index {block_index}: "
                    f"expected {num_channels}, got {iq.shape[0]}"
                )

        dc = _estimate_dc_per_block(iq)
        dc_offsets.append(dc)

        iq_dc_removed = remove_dc_offset(iq, axis=-1)

        gain_estimate = estimate_gain_mismatch(
            iq_dc_removed,
            ref_channel=ref_channel,
            target_channel=target_channel,
        )

        iq_gain_corrected = apply_gain_correction(
            iq_dc_removed,
            gain_correction=gain_estimate.gain_correction,
            target_channel=target_channel,
        )

        phase_estimate = estimate_phase_offset(
            iq_gain_corrected,
            ref_channel=ref_channel,
            target_channel=target_channel,
        )

        ref_rms_values.append(float(gain_estimate.ref_rms))
        target_rms_values.append(float(gain_estimate.target_rms))
        gain_corrections.append(float(gain_estimate.gain_correction))

        phase_offsets_rad.append(float(phase_estimate.phase_offset_rad))
        phase_offsets_deg.append(float(phase_estimate.phase_offset_deg))
        coherence_values.append(float(phase_estimate.coherence_like))

    ref_rms_arr = np.asarray(ref_rms_values, dtype=np.float64)
    target_rms_arr = np.asarray(target_rms_values, dtype=np.float64)
    gain_arr = np.asarray(gain_corrections, dtype=np.float64)
    phase_rad_arr = np.asarray(phase_offsets_rad, dtype=np.float64)
    coherence_arr = np.asarray(coherence_values, dtype=np.float64)

    phase_rad_mean = _circular_mean_rad(phase_rad_arr)
    phase_rad_std = _circular_std_rad(phase_rad_arr)

    phase_rad_min = float(np.min(phase_rad_arr))
    phase_rad_max = float(np.max(phase_rad_arr))

    max_channels = max(len(x) for x in dc_offsets)

    dc_matrix = np.full(
        shape=(len(dc_offsets), max_channels),
        fill_value=np.nan + 1j * np.nan,
        dtype=np.complex64,
    )

    for i, dc in enumerate(dc_offsets):
        dc_matrix[i, : len(dc)] = dc

    dc_mean = np.nanmean(dc_matrix, axis=0)

    ch0 = dc_mean[0] if len(dc_mean) >= 1 else None
    ch1 = dc_mean[1] if len(dc_mean) >= 2 else None

    return PhaseGainCalibrationResult(
        mode="PHASE_GAIN_CALIBRATION",
        created_at=now_iso(),
        num_blocks=len(blocks),
        block_size=int(block_size or 0),
        num_channels=int(num_channels or 0),
        sample_rate=sample_rate,
        center_freq=center_freq,
        ref_channel=ref_channel,
        target_channel=target_channel,
        gain_ref_rms_mean=float(np.mean(ref_rms_arr)),
        gain_ref_rms_std=float(np.std(ref_rms_arr)),
        gain_target_rms_mean=float(np.mean(target_rms_arr)),
        gain_target_rms_std=float(np.std(target_rms_arr)),
        gain_correction_mean=float(np.mean(gain_arr)),
        gain_correction_std=float(np.std(gain_arr)),
        gain_correction_min=float(np.min(gain_arr)),
        gain_correction_max=float(np.max(gain_arr)),
        phase_offset_rad_mean=float(phase_rad_mean),
        phase_offset_rad_std=float(phase_rad_std),
        phase_offset_deg_mean=float(np.rad2deg(phase_rad_mean)),
        phase_offset_deg_std=float(np.rad2deg(phase_rad_std)),
        phase_offset_rad_min=phase_rad_min,
        phase_offset_rad_max=phase_rad_max,
        phase_offset_deg_min=float(np.rad2deg(phase_rad_min)),
        phase_offset_deg_max=float(np.rad2deg(phase_rad_max)),
        coherence_like_mean=float(np.mean(coherence_arr)),
        coherence_like_std=float(np.std(coherence_arr)),
        coherence_like_min=float(np.min(coherence_arr)),
        coherence_like_max=float(np.max(coherence_arr)),
        dc_offset_ch0_real=float(np.real(ch0)) if ch0 is not None else None,
        dc_offset_ch0_imag=float(np.imag(ch0)) if ch0 is not None else None,
        dc_offset_ch1_real=float(np.real(ch1)) if ch1 is not None else None,
        dc_offset_ch1_imag=float(np.imag(ch1)) if ch1 is not None else None,
        note=(
            "Phase/gain calibration result. "
            "This module reuses src.preprocess.estimate_gain_mismatch, "
            "src.preprocess.apply_gain_correction, and "
            "src.preprocess.estimate_phase_offset. "
            "Assumes a known source at broadside 0 deg. "
            "Apply correction later as target_channel *= gain_correction "
            "and then target_channel *= exp(-j * phase_offset). "
            "Do not use IQ normalization for phase/gain calibration."
        ),
    )