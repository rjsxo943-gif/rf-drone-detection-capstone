from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.calibration.gain_noise_calibration import (
    GainNoiseCalibrationSet,
    get_noise_profile_for_gain,
    get_noise_threshold_for_gain,
    load_gain_noise_calibration,
)
from src.calibration.phase_gain_by_gain_calibration import (
    GainPhaseGainCalibrationSet,
    get_phase_gain_correction_for_gain,
    get_phase_gain_profile_for_gain,
    load_phase_gain_by_gain_calibration,
)
from src.calibration.raw_iq_safety import (
    RawIQSafetyResult,
    check_raw_iq_safety,
)


@dataclass(frozen=True)
class RuntimeNoiseCalibrationResult:
    gain: float
    threshold: float
    matched_gain: float
    matched_by: str
    profile_safety_status: str
    raw_safety_status: str
    raw_is_safe: bool
    raw_max_abs: float
    raw_rms: float
    raw_dc_abs: float
    raw_saturation_ratio: float
    raw_near_saturation_ratio: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePhaseGainResult:
    gain: float
    matched_gain: float
    matched_by: str
    gain_correction: float
    phase_offset_rad: float
    phase_offset_deg: float
    quality: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CalibrationRuntime:
    """
    CLI / OpenCV viewer / GUI viewer가 공통으로 사용하는 calibration adapter.

    담당:
    - gain별 noise profile load / threshold lookup
    - raw IQ safety check
    - gain별 phase/gain profile load / correction lookup
    """

    def __init__(
        self,
        *,
        noise_profile: GainNoiseCalibrationSet | None = None,
        phase_gain_profile: GainPhaseGainCalibrationSet | None = None,
        allow_nearest: bool = True,
        full_scale: float = 1.0,
    ) -> None:
        self.noise_profile = noise_profile
        self.phase_gain_profile = phase_gain_profile
        self.allow_nearest = bool(allow_nearest)
        self.full_scale = float(full_scale)

    @classmethod
    def from_files(
        cls,
        *,
        noise_profile_path: str | Path | None = None,
        phase_gain_profile_path: str | Path | None = None,
        allow_nearest: bool = True,
        full_scale: float = 1.0,
    ) -> "CalibrationRuntime":
        noise_profile = None
        if noise_profile_path:
            path = Path(noise_profile_path)
            if path.exists():
                noise_profile = load_gain_noise_calibration(path)

        phase_gain_profile = None
        if phase_gain_profile_path:
            path = Path(phase_gain_profile_path)
            if path.exists():
                phase_gain_profile = load_phase_gain_by_gain_calibration(path)

        return cls(
            noise_profile=noise_profile,
            phase_gain_profile=phase_gain_profile,
            allow_nearest=allow_nearest,
            full_scale=full_scale,
        )

    @property
    def has_noise_profile(self) -> bool:
        return self.noise_profile is not None

    @property
    def has_phase_gain_profile(self) -> bool:
        return self.phase_gain_profile is not None

    def check_noise(
        self,
        iq: np.ndarray,
        *,
        gain: int | float,
    ) -> RuntimeNoiseCalibrationResult:
        raw_safety = check_raw_iq_safety(
            iq,
            full_scale=self.full_scale,
        )

        threshold = float("nan")
        matched_gain = float(gain)
        matched_by = "none"
        profile_safety_status = "NO_PROFILE"

        if self.noise_profile is not None:
            profile = get_noise_profile_for_gain(
                self.noise_profile,
                gain,
                allow_nearest=self.allow_nearest,
            )
            threshold = get_noise_threshold_for_gain(
                self.noise_profile,
                gain,
                allow_nearest=self.allow_nearest,
            )
            matched_gain = float(profile.get("matched_gain", profile.get("gain", gain)))
            matched_by = str(profile.get("matched_by", "exact"))
            profile_safety_status = str(
                profile.get("safety", {}).get("status", "UNKNOWN")
            )

        return RuntimeNoiseCalibrationResult(
            gain=float(gain),
            threshold=threshold,
            matched_gain=matched_gain,
            matched_by=matched_by,
            profile_safety_status=profile_safety_status,
            raw_safety_status=raw_safety.status,
            raw_is_safe=raw_safety.is_safe,
            raw_max_abs=raw_safety.max_abs,
            raw_rms=raw_safety.rms,
            raw_dc_abs=raw_safety.dc_abs,
            raw_saturation_ratio=raw_safety.saturation_ratio,
            raw_near_saturation_ratio=raw_safety.near_saturation_ratio,
            note=raw_safety.note,
        )

    def get_phase_gain(
        self,
        *,
        gain: int | float,
    ) -> RuntimePhaseGainResult | None:
        if self.phase_gain_profile is None:
            return None

        profile = get_phase_gain_profile_for_gain(
            self.phase_gain_profile,
            gain,
            allow_nearest=self.allow_nearest,
        )
        gain_correction, phase_offset_rad = get_phase_gain_correction_for_gain(
            self.phase_gain_profile,
            gain,
            allow_nearest=self.allow_nearest,
        )

        matched_gain = float(profile.get("matched_gain", profile.get("gain", gain)))
        matched_by = str(profile.get("matched_by", "exact"))

        return RuntimePhaseGainResult(
            gain=float(gain),
            matched_gain=matched_gain,
            matched_by=matched_by,
            gain_correction=float(gain_correction),
            phase_offset_rad=float(phase_offset_rad),
            phase_offset_deg=float(np.rad2deg(phase_offset_rad)),
            quality=str(profile.get("quality", "UNKNOWN")),
            source="phase_gain_by_gain_profile",
        )

    def apply_phase_gain(
        self,
        iq: np.ndarray,
        *,
        gain: int | float,
        target_channel: int = 1,
    ) -> np.ndarray:
        pg = self.get_phase_gain(gain=gain)
        if pg is None:
            return np.asarray(iq).astype(np.complex64, copy=False)

        corrected = np.array(iq, dtype=np.complex64, copy=True)
        corrected[target_channel] *= np.complex64(pg.gain_correction)
        corrected[target_channel] *= np.exp(-1j * pg.phase_offset_rad).astype(np.complex64)
        return corrected

    def summarize(
        self,
        iq: np.ndarray,
        *,
        gain: int | float,
    ) -> str:
        noise = self.check_noise(iq, gain=gain)
        pg = self.get_phase_gain(gain=gain)

        phase_text = "phase_gain=NO_PROFILE"
        if pg is not None:
            phase_text = (
                f"phase_gain={pg.matched_gain:g}/{pg.matched_by} "
                f"corr={pg.gain_correction:.4g} "
                f"phase={pg.phase_offset_deg:.2f}deg"
            )

        return (
            f"noise={noise.matched_gain:g}/{noise.matched_by} "
            f"thr={noise.threshold:.4g} "
            f"raw={noise.raw_safety_status} "
            f"sat={noise.raw_saturation_ratio * 100:.3f}% | "
            f"{phase_text}"
        )


def load_calibration_runtime(
    *,
    noise_profile_path: str | Path | None = None,
    phase_gain_profile_path: str | Path | None = None,
    allow_nearest: bool = True,
    full_scale: float = 1.0,
) -> CalibrationRuntime:
    return CalibrationRuntime.from_files(
        noise_profile_path=noise_profile_path,
        phase_gain_profile_path=phase_gain_profile_path,
        allow_nearest=allow_nearest,
        full_scale=full_scale,
    )