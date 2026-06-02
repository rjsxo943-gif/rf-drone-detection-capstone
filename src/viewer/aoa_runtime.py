from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.aoa import coherence_gate, estimate_phase_diff, phase_diff_to_angle
from src.features.spectrogram import compute_stft_branch
from src.calibration.phase_gain_by_gain_calibration import (
    GainPhaseGainCalibrationSet,
    get_phase_gain_correction_for_gain,
    get_phase_gain_profile_for_gain,
    load_phase_gain_by_gain_calibration,
)


PHASE_OFFSET_RAD_KEYS = (
    "phase_offset_to_apply_rad",
    "phase_offset_rad",
    "offset_rad",
    "correction_rad",
    "rx1_phase_offset_rad",
)
PHASE_OFFSET_DEG_KEYS = (
    "phase_offset_to_apply_deg",
    "phase_offset_deg",
    "offset_deg",
    "correction_deg",
    "rx1_phase_offset_deg",
)
GAIN_KEYS = ("gain", "gain_db", "rx_gain", "hardware_gain")


@dataclass
class AoARuntime:
    """Runtime wrapper for phase-calibrated two-channel AoA estimation.

    The class keeps the live viewer thin. It loads optional phase calibration
    files, updates gain-dependent phase correction, computes coherence, and
    returns a plain dict that can be rendered as overlay text or logged.
    """

    carrier_freq: float
    sample_rate: float
    antenna_spacing_m: float = 0.0625
    speed_of_light: float = 300_000_000.0
    phase_calibration_json: str | Path | None = None
    gain_phase_table_json: str | Path | None = None
    phase_gain_profile_json: str | Path | None = None
    manual_phase_offset_deg: float = 0.0
    ref_channel: int = 0
    target_channel: int = 1
    coherence_threshold: float = 0.6
    energy_percentile: float = 75.0
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 512
    window: str = "hann"
    compute_coherence: bool = True
    clip_angle_input: bool = True
    gain: float = 0.0
    phase_offset_to_apply_rad: float = field(init=False)
    phase_offset_source: str = field(init=False, default="manual")
    gain_phase_entries: list[dict[str, float]] = field(init=False, default_factory=list)
    phase_gain_profiles: GainPhaseGainCalibrationSet | None = field(init=False, default=None)
    gain_correction_to_apply: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.phase_offset_to_apply_rad = float(np.deg2rad(self.manual_phase_offset_deg))
        self.phase_offset_source = "manual"

        calibration_offset = self._load_phase_offset_file(self.phase_calibration_json)
        if calibration_offset is not None:
            self.phase_offset_to_apply_rad = calibration_offset
            self.phase_offset_source = str(self.phase_calibration_json)

        self.gain_phase_entries = self._load_gain_phase_table(self.gain_phase_table_json)
        self.phase_gain_profiles = self._load_phase_gain_profile(
            self.phase_gain_profile_json
        )
        self.update_gain(self.gain)

    def update_gain(self, gain: float) -> None:
        """Update RX1 gain/phase compensation when receiver gain changes."""

        self.gain = float(gain)

        # New path: gain-wise phase/gain profile JSON.
        if self.phase_gain_profiles is not None:
            profile = get_phase_gain_profile_for_gain(
                self.phase_gain_profiles,
                self.gain,
                allow_nearest=True,
            )
            gain_correction, phase_offset_rad = get_phase_gain_correction_for_gain(
                self.phase_gain_profiles,
                self.gain,
                allow_nearest=True,
            )

            self.gain_correction_to_apply = float(gain_correction)
            self.phase_offset_to_apply_rad = float(phase_offset_rad)

            matched_gain = float(profile.get("matched_gain", profile.get("gain", self.gain)))
            matched_by = str(profile.get("matched_by", "exact"))
            quality = str(profile.get("quality", "UNKNOWN"))

            self.phase_offset_source = (
                f"{self.phase_gain_profile_json} "
                f"gain={matched_gain:.1f} {matched_by} quality={quality}"
            )
            return

        # Legacy fallback: gain phase table.
        self.gain_correction_to_apply = 1.0
        if not self.gain_phase_entries:
            return

        nearest = min(
            self.gain_phase_entries,
            key=lambda item: abs(float(item["gain"]) - self.gain),
        )
        self.phase_offset_to_apply_rad = float(nearest["phase_offset_rad"])
        self.phase_offset_source = (
            f"{self.gain_phase_table_json} nearest_gain={nearest['gain']:.1f}"
        )

    def process(self, iq: np.ndarray) -> dict[str, Any]:
        """Compute AoA metrics from a two-channel IQ block."""

        iq_2d = self._ensure_2d_iq(iq)
        raw_phase = estimate_phase_diff(
            iq_2d,
            ref_channel=self.ref_channel,
            target_channel=self.target_channel,
        )

        corrected_iq = self._apply_target_phase_correction(iq_2d)
        corrected_phase = estimate_phase_diff(
            corrected_iq,
            ref_channel=self.ref_channel,
            target_channel=self.target_channel,
        )

        angle = phase_diff_to_angle(
            phase_diff_rad=corrected_phase.phase_diff_rad,
            carrier_freq=float(self.carrier_freq),
            antenna_spacing_m=float(self.antenna_spacing_m),
            speed_of_light=float(self.speed_of_light),
            phase_offset_rad=0.0,
            clip_input=bool(self.clip_angle_input),
        )

        result: dict[str, Any] = {
            "aoa_angle_deg": float(angle.angle_deg),
            "aoa_angle_rad": float(angle.angle_rad),
            "aoa_valid": bool(angle.valid),
            "aoa_arcsin_input": float(angle.arcsin_input),
            "phase_diff_raw_rad": float(raw_phase.phase_diff_rad),
            "phase_diff_raw_deg": float(raw_phase.phase_diff_deg),
            "phase_diff_corrected_rad": float(corrected_phase.phase_diff_rad),
            "phase_diff_corrected_deg": float(corrected_phase.phase_diff_deg),
            "phase_offset_to_apply_rad": float(self.phase_offset_to_apply_rad),
            "phase_offset_to_apply_deg": float(np.rad2deg(self.phase_offset_to_apply_rad)),
            "gain_correction_to_apply": float(self.gain_correction_to_apply),
            "phase_offset_source": self.phase_offset_source,
            "coherence_like": float(corrected_phase.coherence_like),
            "num_samples": int(corrected_phase.num_samples),
            "gain": float(self.gain),
        }

        if self.compute_coherence:
            try:
                z0, z1 = self._compute_dual_stft(corrected_iq)
                coh = coherence_gate(
                    z0=z0,
                    z1=z1,
                    threshold=float(self.coherence_threshold),
                    energy_percentile=float(self.energy_percentile),
                )
                result.update(
                    {
                        "stft_coherence": float(coh.coherence),
                        "stft_coherence_passed": bool(coh.passed),
                        "stft_coherence_threshold": float(coh.threshold),
                        "stft_coherence_used_bins": int(coh.used_bins),
                    }
                )
            except Exception as exc:  # Keep live viewer running in field tests.
                result.update(
                    {
                        "stft_coherence": float("nan"),
                        "stft_coherence_passed": False,
                        "stft_coherence_error": str(exc),
                    }
                )

        return result

    def status_text(self) -> str:
        return (
            f"AOA gain_corr={self.gain_correction_to_apply:.4g} "
            f"offset={np.rad2deg(self.phase_offset_to_apply_rad):.2f} deg "
            f"src={self.phase_offset_source}"
        )

    def _apply_target_phase_correction(self, iq_2d: np.ndarray) -> np.ndarray:
        corrected = np.array(iq_2d, dtype=np.complex64, copy=True)
        correction = np.exp(-1j * float(self.phase_offset_to_apply_rad)).astype(np.complex64)
        corrected[self.target_channel] *= np.complex64(self.gain_correction_to_apply)
        corrected[self.target_channel] *= correction
        return corrected

    def _compute_dual_stft(self, iq_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rx0 = iq_2d[self.ref_channel]
        rx1 = iq_2d[self.target_channel]
        stft0 = compute_stft_branch(
            iq_block=rx0,
            sample_rate=float(self.sample_rate),
            nperseg=int(self.nperseg),
            noverlap=int(self.noverlap),
            nfft=int(self.nfft),
            window=self.window,
        )
        stft1 = compute_stft_branch(
            iq_block=rx1,
            sample_rate=float(self.sample_rate),
            nperseg=int(self.nperseg),
            noverlap=int(self.noverlap),
            nfft=int(self.nfft),
            window=self.window,
        )
        return stft0.complex_stft, stft1.complex_stft

    @staticmethod
    def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
        arr = np.asarray(iq)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"iq must be 1-D or 2-D complex array. got shape={arr.shape}")
        if not np.iscomplexobj(arr):
            raise TypeError(f"iq must be complex. got dtype={arr.dtype}")
        if arr.shape[0] < 2:
            raise ValueError(f"AoA mode requires at least 2 channels. got shape={arr.shape}")
        return arr.astype(np.complex64, copy=False)

    @staticmethod
    def _load_phase_gain_profile(
        path: str | Path | None,
    ) -> GainPhaseGainCalibrationSet | None:
        if path is None:
            return None

        path_obj = Path(path)
        if not path_obj.exists():
            return None

        return load_phase_gain_by_gain_calibration(path_obj)

    @classmethod
    def _load_phase_offset_file(cls, path: str | Path | None) -> float | None:
        if path is None:
            return None
        path_obj = Path(path)
        if not path_obj.exists():
            return None
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._extract_phase_offset_rad(data)

    @classmethod
    def _load_gain_phase_table(cls, path: str | Path | None) -> list[dict[str, float]]:
        if path is None:
            return []
        path_obj = Path(path)
        if not path_obj.exists():
            return []
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rows = cls._extract_table_rows(data)
        entries: list[dict[str, float]] = []
        for row in rows:
            gain = cls._extract_gain(row)
            offset = cls._extract_phase_offset_rad(row)
            if gain is None or offset is None:
                continue
            entries.append({"gain": float(gain), "phase_offset_rad": float(offset)})
        return sorted(entries, key=lambda item: item["gain"])

    @staticmethod
    def _extract_table_rows(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("gain_phase_table", "entries", "rows", "profiles", "table"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            # Also support {"25": {"phase_offset_deg": ...}, "30": ...}
            rows: list[dict[str, Any]] = []
            for key, value in data.items():
                try:
                    gain = float(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    row = {"gain": gain, **value}
                else:
                    row = {"gain": gain, "phase_offset_deg": value}
                rows.append(row)
            return rows
        return []

    @staticmethod
    def _extract_gain(data: Any) -> float | None:
        if isinstance(data, dict):
            for key in GAIN_KEYS:
                if key in data:
                    try:
                        return float(data[key])
                    except (TypeError, ValueError):
                        return None
        return None

    @staticmethod
    def _extract_phase_offset_rad(data: Any) -> float | None:
        if isinstance(data, (int, float)):
            return float(np.deg2rad(float(data)))
        if not isinstance(data, dict):
            return None

        for key in PHASE_OFFSET_RAD_KEYS:
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    return None
        for key in PHASE_OFFSET_DEG_KEYS:
            if key in data:
                try:
                    return float(np.deg2rad(float(data[key])))
                except (TypeError, ValueError):
                    return None
        return None
