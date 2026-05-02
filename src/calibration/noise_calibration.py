from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import json
import numpy as np

from src.core import now_iso
from src.detect import EnergyDetector
from src.preprocess import ensure_2d_iq, remove_dc_offset, estimate_dc_offset


@dataclass
class NoiseCalibrationResult:
    mode: str
    created_at: str

    num_blocks: int
    block_size: int
    num_channels: int

    sample_rate: float | None
    center_freq: float | None

    detector_method: str
    frame_size: int
    hop_size: int
    threshold_multiplier: float

    noise_floor: float
    noise_mean: float
    noise_std: float
    noise_min: float
    noise_max: float
    threshold: float

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

    현재 receiver 출력 기준:
    - shape = (num_channels, num_samples)
    - 예: (2, 16384)

    혹시 (num_samples, num_channels)로 들어오면 transpose한다.
    """
    iq = ensure_2d_iq(np.asarray(block))

    if iq.ndim != 2:
        raise ValueError(f"IQ block must be 2D, got shape={iq.shape}")

    # 정상: (1, N), (2, N)
    if iq.shape[0] in (1, 2):
        return iq.astype(np.complex64, copy=False)

    # 예외 대응: (N, 1), (N, 2)
    if iq.shape[1] in (1, 2):
        return iq.T.astype(np.complex64, copy=False)

    raise ValueError(
        f"Unsupported IQ shape. Expected (C, N) or (N, C), got {iq.shape}"
    )


def _estimate_dc_per_block(iq: np.ndarray) -> np.ndarray:
    """
    기존 estimate_dc_offset()을 사용하되,
    최종 shape은 항상 (C,)로 맞춘다.
    """
    dc = estimate_dc_offset(iq, axis=-1)
    dc = np.asarray(dc, dtype=np.complex64).squeeze()

    if dc.ndim == 0:
        dc = dc.reshape(1)

    if dc.ndim != 1:
        dc = np.mean(iq, axis=-1).astype(np.complex64)

    return dc


def _build_energy_detector(
    *,
    method: str,
    frame_size: int,
    hop_size: int,
    threshold_multiplier: float,
    calibration_blocks: int,
    min_detection_ratio: float,
) -> EnergyDetector:
    """
    EnergyDetector 생성부.

    현재 energy_detector.py의 정확한 __init__ 전체 시그니처를 아직 직접 본 건 아니므로,
    가장 가능성 높은 인자를 먼저 넣고, 실패하면 최소 인자로 fallback한다.
    """
    kwargs = {
        "method": method,
        "frame_size": frame_size,
        "hop_size": hop_size,
        "threshold_multiplier": threshold_multiplier,
        "calibration_blocks": calibration_blocks,
        "min_detection_ratio": min_detection_ratio,
    }

    try:
        return EnergyDetector(**kwargs)
    except TypeError:
        pass

    try:
        return EnergyDetector(
            method=method,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_multiplier=threshold_multiplier,
        )
    except TypeError:
        pass

    try:
        return EnergyDetector(
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_multiplier=threshold_multiplier,
        )
    except TypeError:
        pass

    return EnergyDetector()


def _get_detector_value(detector: EnergyDetector, names: list[str]) -> float | None:
    """
    EnergyDetector 내부 속성명이 noise_floor인지 threshold인지 아직 확정하지 않고도
    최대한 값을 가져오기 위한 보조 함수.
    """
    for name in names:
        if hasattr(detector, name):
            value = getattr(detector, name)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue

    return None


def calibrate_noise_from_blocks(
    blocks: list[np.ndarray],
    *,
    method: str = "time_power",
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold_multiplier: float = 5.0,
    calibration_blocks: int | None = None,
    min_detection_ratio: float = 0.05,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> NoiseCalibrationResult:
    """
    노이즈 캘리브레이션 본체.

    전제:
    - 신호원이 없는 상태에서 수집한 IQ block들을 넣어야 한다.
    - 여기서는 CNN용 normalize_iq를 절대 사용하지 않는다.
    - 실제 탐지와 같은 기준을 쓰기 위해 EnergyDetector.compute_frame_energies()를 재사용한다.

    처리:
    1. block shape을 프로젝트 표준 (C, N)으로 정리
    2. DC offset 추정 및 제거
    3. EnergyDetector.compute_frame_energies()로 frame energy 계산
    4. 전체 frame energy median을 noise_floor로 사용
    5. threshold = noise_floor * threshold_multiplier
    """
    if not blocks:
        raise ValueError("No IQ blocks were provided for noise calibration")

    if calibration_blocks is None:
        calibration_blocks = len(blocks)

    detector = _build_energy_detector(
        method=method,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold_multiplier=threshold_multiplier,
        calibration_blocks=calibration_blocks,
        min_detection_ratio=min_detection_ratio,
    )

    all_frame_energies: list[np.ndarray] = []
    dc_offsets: list[np.ndarray] = []

    block_size: int | None = None
    num_channels: int | None = None

    for block_index, block in enumerate(blocks):
        iq = _to_project_iq(block)

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

        frame_energies = detector.compute_frame_energies(iq_dc_removed)
        frame_energies = np.asarray(frame_energies, dtype=np.float64).reshape(-1)

        if frame_energies.size == 0:
            raise ValueError(f"Empty frame energies at block index {block_index}")

        all_frame_energies.append(frame_energies)

    merged = np.concatenate(all_frame_energies).astype(np.float64)

    noise_floor = float(np.median(merged))
    noise_mean = float(np.mean(merged))
    noise_std = float(np.std(merged))
    noise_min = float(np.min(merged))
    noise_max = float(np.max(merged))
    threshold = float(noise_floor * threshold_multiplier)

    # 가능하면 EnergyDetector의 fit도 호출해서 detector 내부 상태와 동일하게 맞춰둔다.
    try:
        detector.fit(merged)
    except Exception:
        pass

    detector_noise_floor = _get_detector_value(
        detector,
        ["noise_floor", "noise_floor_", "calibrated_noise_floor"],
    )
    detector_threshold = _get_detector_value(
        detector,
        ["threshold", "threshold_", "energy_threshold"],
    )

    if detector_noise_floor is not None:
        noise_floor = detector_noise_floor
        threshold = float(noise_floor * threshold_multiplier)

    if detector_threshold is not None:
        threshold = detector_threshold

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

    return NoiseCalibrationResult(
        mode="NOISE_CALIBRATION",
        created_at=now_iso(),
        num_blocks=len(blocks),
        block_size=int(block_size or 0),
        num_channels=int(num_channels or 0),
        sample_rate=sample_rate,
        center_freq=center_freq,
        detector_method=method,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold_multiplier=threshold_multiplier,
        noise_floor=float(noise_floor),
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_min=noise_min,
        noise_max=noise_max,
        threshold=float(threshold),
        dc_offset_ch0_real=float(np.real(ch0)) if ch0 is not None else None,
        dc_offset_ch0_imag=float(np.imag(ch0)) if ch0 is not None else None,
        dc_offset_ch1_real=float(np.real(ch1)) if ch1 is not None else None,
        dc_offset_ch1_imag=float(np.imag(ch1)) if ch1 is not None else None,
        note=(
            "Noise calibration result. "
            "Frame energies are computed using src.detect.EnergyDetector. "
            "DC offset removal uses src.preprocess.remove_dc_offset. "
            "Do not use IQ normalization for noise calibration."
        ),
    )