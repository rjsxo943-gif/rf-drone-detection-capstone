from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from src.preprocess import (
    apply_gain_correction,
    remove_phase_offset,
)


DEFAULT_CALIBRATION_DIR = Path("outputs") / "calibration"
DEFAULT_NOISE_PATH = DEFAULT_CALIBRATION_DIR / "noise_latest.json"
DEFAULT_PHASE_GAIN_PATH = DEFAULT_CALIBRATION_DIR / "phase_gain_latest.json"


@dataclass(frozen=True)
class NoiseCalibrationParams:
    """
    noise_latest.json에서 실제 파이프라인에 필요한 값만 뽑은 객체.
    """

    noise_floor: float
    threshold: float
    threshold_multiplier: float
    detector_method: str
    frame_size: int
    hop_size: int
    sample_rate: float | None
    center_freq: float | None
    source_path: str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_path: str = "",
    ) -> "NoiseCalibrationParams":
        return cls(
            noise_floor=float(data["noise_floor"]),
            threshold=float(data["threshold"]),
            threshold_multiplier=float(data.get("threshold_multiplier", 5.0)),
            detector_method=str(data.get("detector_method", "time_power")),
            frame_size=int(data.get("frame_size", 1024)),
            hop_size=int(data.get("hop_size", 512)),
            sample_rate=_optional_float(data.get("sample_rate")),
            center_freq=_optional_float(data.get("center_freq")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_floor": self.noise_floor,
            "threshold": self.threshold,
            "threshold_multiplier": self.threshold_multiplier,
            "detector_method": self.detector_method,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "center_freq": self.center_freq,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class PhaseGainCalibrationParams:
    """
    phase_gain_latest.json에서 실제 파이프라인에 필요한 값만 뽑은 객체.
    """

    gain_correction: float
    phase_offset_rad: float
    phase_offset_deg: float
    coherence_like: float
    ref_channel: int
    target_channel: int
    sample_rate: float | None
    center_freq: float | None
    source_path: str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_path: str = "",
    ) -> "PhaseGainCalibrationParams":
        return cls(
            gain_correction=float(data["gain_correction_mean"]),
            phase_offset_rad=float(data["phase_offset_rad_mean"]),
            phase_offset_deg=float(data.get("phase_offset_deg_mean", 0.0)),
            coherence_like=float(data.get("coherence_like_mean", 0.0)),
            ref_channel=int(data.get("ref_channel", 0)),
            target_channel=int(data.get("target_channel", 1)),
            sample_rate=_optional_float(data.get("sample_rate")),
            center_freq=_optional_float(data.get("center_freq")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain_correction": self.gain_correction,
            "phase_offset_rad": self.phase_offset_rad,
            "phase_offset_deg": self.phase_offset_deg,
            "coherence_like": self.coherence_like,
            "ref_channel": self.ref_channel,
            "target_channel": self.target_channel,
            "sample_rate": self.sample_rate,
            "center_freq": self.center_freq,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class CalibrationParams:
    """
    noise + phase/gain calibration을 한 번에 들고 다니는 객체.
    """

    noise: NoiseCalibrationParams | None = None
    phase_gain: PhaseGainCalibrationParams | None = None

    @property
    def has_noise(self) -> bool:
        return self.noise is not None

    @property
    def has_phase_gain(self) -> bool:
        return self.phase_gain is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise": self.noise.to_dict() if self.noise is not None else None,
            "phase_gain": (
                self.phase_gain.to_dict() if self.phase_gain is not None else None
            ),
        }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_noise_calibration(
    path: str | Path = DEFAULT_NOISE_PATH,
) -> NoiseCalibrationParams:
    """
    noise_latest.json을 읽어서 NoiseCalibrationParams로 변환한다.
    """
    path = Path(path)

    data = _load_json(path)

    return NoiseCalibrationParams.from_dict(
        data,
        source_path=str(path),
    )


def load_phase_gain_calibration(
    path: str | Path = DEFAULT_PHASE_GAIN_PATH,
) -> PhaseGainCalibrationParams:
    """
    phase_gain_latest.json을 읽어서 PhaseGainCalibrationParams로 변환한다.
    """
    path = Path(path)

    data = _load_json(path)

    return PhaseGainCalibrationParams.from_dict(
        data,
        source_path=str(path),
    )


def load_calibration_params(
    *,
    noise_path: str | Path = DEFAULT_NOISE_PATH,
    phase_gain_path: str | Path = DEFAULT_PHASE_GAIN_PATH,
    require_noise: bool = False,
    require_phase_gain: bool = False,
) -> CalibrationParams:
    """
    noise / phase_gain calibration을 한 번에 로드한다.

    require_noise=False이면 파일이 없어도 None으로 둔다.
    require_phase_gain=False이면 파일이 없어도 None으로 둔다.

    상태머신에서는 보통:
    - SCAN 시작 전 require_noise=True
    - BAND_HOLD/AoA 시작 전 require_phase_gain=True
    이런 식으로 쓸 수 있다.
    """
    noise: NoiseCalibrationParams | None = None
    phase_gain: PhaseGainCalibrationParams | None = None

    noise_path = Path(noise_path)
    phase_gain_path = Path(phase_gain_path)

    if noise_path.exists():
        noise = load_noise_calibration(noise_path)
    elif require_noise:
        raise FileNotFoundError(f"Noise calibration file not found: {noise_path}")

    if phase_gain_path.exists():
        phase_gain = load_phase_gain_calibration(phase_gain_path)
    elif require_phase_gain:
        raise FileNotFoundError(
            f"Phase/gain calibration file not found: {phase_gain_path}"
        )

    return CalibrationParams(
        noise=noise,
        phase_gain=phase_gain,
    )


def apply_phase_gain_calibration(
    iq: np.ndarray,
    params: PhaseGainCalibrationParams,
) -> np.ndarray:
    """
    IQ block에 phase/gain calibration 결과를 적용한다.

    적용 순서:
    1. target channel gain correction
    2. target channel phase offset removal

    주의:
    - DC offset 제거는 이 함수 밖에서 먼저 수행하는 것을 권장한다.
    - 즉, pipeline에서는 보통:
        iq = remove_dc_offset(iq, axis=-1)
        iq = apply_phase_gain_calibration(iq, params)
    """
    corrected = apply_gain_correction(
        iq,
        gain_correction=params.gain_correction,
        target_channel=params.target_channel,
    )

    corrected = remove_phase_offset(
        corrected,
        phase_offset_rad=params.phase_offset_rad,
        ref_channel=params.ref_channel,
        target_channel=params.target_channel,
    )

    return corrected.astype(np.complex64, copy=False)


def apply_phase_gain_if_available(
    iq: np.ndarray,
    calibration: CalibrationParams,
) -> np.ndarray:
    """
    phase/gain calibration이 있으면 적용하고,
    없으면 원본 IQ를 그대로 반환한다.

    초기 개발 중에는 calibration 파일이 없을 수도 있으므로
    SCAN 연결 테스트에서 유용하다.
    """
    if calibration.phase_gain is None:
        return iq

    return apply_phase_gain_calibration(iq, calibration.phase_gain)


def get_energy_threshold(
    calibration: CalibrationParams,
    *,
    fallback_threshold: float | None = None,
) -> float:
    """
    noise calibration에서 energy threshold를 가져온다.

    파일이 없으면 fallback_threshold를 사용한다.
    fallback_threshold도 없으면 에러를 낸다.
    """
    if calibration.noise is not None:
        return calibration.noise.threshold

    if fallback_threshold is not None:
        return float(fallback_threshold)

    raise ValueError("Noise calibration is not loaded and fallback_threshold is None.")