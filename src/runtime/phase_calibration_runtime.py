from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.calibration import (
    get_phase_offset_to_apply,
    load_gain_phase_table,
    wrap_phase_rad,
)


@dataclass
class PhaseCalibrationState:
    enabled: bool
    current_ref_phase_offset_rad: float
    current_ref_phase_offset_deg: float
    reference_gain: int | None
    current_gain: float | None
    phase_offset_to_apply_rad: float
    phase_offset_to_apply_deg: float
    uncertainty_rad: float
    uncertainty_deg: float
    source: str
    quality: str


def load_current_phase_offset(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"current phase offset file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_phase_offset_to_apply(
    current_phase_path: str | Path = "configs/calibration/current_phase_offset.json",
    gain_table_path: str | Path | None = None,
    current_gain: float | None = None,
) -> PhaseCalibrationState:
    """
    실시간 AoA에 적용할 phase_offset 계산.

    case 1:
      gain_table_path 없음
      → current_phase_offset.json의 phase_offset_rad 그대로 사용

    case 2:
      gain_table_path 있음 + current_gain 있음
      → current_ref_phase_offset + gain_delta_table[current_gain] 사용
    """
    current = load_current_phase_offset(current_phase_path)

    current_ref_rad = float(current["phase_offset_rad"])
    current_ref_deg = float(np.rad2deg(current_ref_rad))
    quality = str(current.get("quality", "UNKNOWN"))

    if gain_table_path is None:
        phase_to_apply = float(wrap_phase_rad(current_ref_rad))
        uncertainty = float(current.get("phase_std_rad", 0.0))

        return PhaseCalibrationState(
            enabled=True,
            current_ref_phase_offset_rad=current_ref_rad,
            current_ref_phase_offset_deg=current_ref_deg,
            reference_gain=current.get("gain"),
            current_gain=current_gain,
            phase_offset_to_apply_rad=phase_to_apply,
            phase_offset_to_apply_deg=float(np.rad2deg(phase_to_apply)),
            uncertainty_rad=uncertainty,
            uncertainty_deg=float(np.rad2deg(uncertainty)),
            source="current_phase_offset_only",
            quality=quality,
        )

    if current_gain is None:
        raise ValueError("current_gain is required when gain_table_path is provided.")

    table_data = load_gain_phase_table(gain_table_path)
    gain_table = table_data["gain_table"]
    reference_gain = int(table_data["reference_gain"])

    phase_to_apply, uncertainty = get_phase_offset_to_apply(
        current_ref_phase_offset=current_ref_rad,
        table=gain_table,
        current_gain=current_gain,
    )

    return PhaseCalibrationState(
        enabled=True,
        current_ref_phase_offset_rad=current_ref_rad,
        current_ref_phase_offset_deg=current_ref_deg,
        reference_gain=reference_gain,
        current_gain=float(current_gain),
        phase_offset_to_apply_rad=float(phase_to_apply),
        phase_offset_to_apply_deg=float(np.rad2deg(phase_to_apply)),
        uncertainty_rad=float(uncertainty),
        uncertainty_deg=float(np.rad2deg(uncertainty)),
        source="current_ref_plus_gain_delta_table",
        quality=quality,
    )


def apply_phase_offset_to_iq(
    iq: np.ndarray,
    phase_offset_rad: float,
    target_channel: int = 1,
) -> np.ndarray:
    """
    RX1에 exp(-j * phase_offset)를 곱해서 RX0 기준으로 phase 보정.

    iq shape:
      (2, N) 권장
    """
    iq = np.asarray(iq)

    if iq.ndim != 2:
        raise ValueError(f"iq must be 2-D array, got shape={iq.shape}")

    if target_channel < 0 or target_channel >= iq.shape[0]:
        raise IndexError(
            f"target_channel={target_channel} out of range for iq shape={iq.shape}"
        )

    corrected = iq.astype(np.complex64, copy=True)
    correction = np.exp(-1j * float(phase_offset_rad)).astype(np.complex64)
    corrected[target_channel] = corrected[target_channel] * correction

    return corrected


def print_phase_calibration_state(state: PhaseCalibrationState) -> None:
    print("=== Phase Calibration Runtime ===")
    print(f"enabled        : {state.enabled}")
    print(f"source         : {state.source}")
    print(f"quality        : {state.quality}")
    print(f"reference_gain : {state.reference_gain}")
    print(f"current_gain   : {state.current_gain}")
    print(f"ref_offset     : {state.current_ref_phase_offset_deg:+.3f} deg")
    print(f"apply_offset   : {state.phase_offset_to_apply_deg:+.3f} deg")
    print(f"uncertainty    : {state.uncertainty_deg:.3f} deg")
