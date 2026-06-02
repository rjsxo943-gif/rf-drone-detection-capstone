from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable
from src.calibration.raw_iq_safety import check_raw_iq_safety

import json
import numpy as np

from src.calibration.noise_calibration import (
    NoiseCalibrationResult,
    calibrate_noise_from_blocks,
)
from src.core import now_iso


CollectNoiseBlocksFn = Callable[[float, int], list[np.ndarray]]


@dataclass
class GainNoiseCalibrationResult:
    """
    단일 gain에서 측정한 noise calibration 결과.

    기존 NoiseCalibrationResult를 감싸서 gain 정보를 함께 보관한다.
    기존 calibrate_noise_from_blocks() 호출 방식은 그대로 유지한다.
    """

    gain: float
    result: NoiseCalibrationResult

    def to_dict(self) -> dict[str, Any]:
        data = self.result.to_dict()
        data["gain"] = float(self.gain)
        return data


@dataclass
class GainNoiseCalibrationSet:
    """
    여러 gain에서 측정한 noise calibration 결과 묶음.

    integrated CLI / live viewer / runtime pipeline에서 공통으로 로드해서 사용한다.
    """

    mode: str
    created_at: str
    gain_list: list[float]
    num_gains: int
    num_blocks_per_gain: int
    sample_rate: float | None
    center_freq: float | None
    detector_method: str
    frame_size: int
    hop_size: int
    threshold_multiplier: float
    min_detection_ratio: float
    profiles: dict[str, dict[str, Any]]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def get_profile(self, gain: int | float) -> dict[str, Any]:
        return get_noise_profile_for_gain(self, gain)

    def get_threshold(self, gain: int | float) -> float:
        profile = self.get_profile(gain)
        return float(profile["threshold"])


def gain_to_key(gain: int | float) -> str:
    """JSON key용 gain 문자열 생성."""
    gain_value = float(gain)
    if gain_value.is_integer():
        return str(int(gain_value))
    return str(gain_value)


def _profile_key_to_float(key: str) -> float:
    try:
        return float(key)
    except ValueError as exc:
        raise ValueError(f"Invalid gain profile key: {key}") from exc


def _validate_gain_list(gain_list: list[int | float]) -> list[float]:
    if not gain_list:
        raise ValueError("gain_list must not be empty")

    gains = [float(gain) for gain in gain_list]
    return sorted(dict.fromkeys(gains))

def summarize_blocks_safety(
    blocks: list[np.ndarray],
    *,
    full_scale: float = 1.0,
    saturation_ratio_warn: float = 0.001,
    saturation_ratio_clip: float = 0.01,
    near_saturation_level: float = 0.85,
    saturation_level: float = 0.98,
) -> dict[str, Any]:
    """
    여러 IQ block의 Raw IQ safety 결과를 요약한다.

    gain별 noise calibration 중 수집된 blocks가 포화 상태였는지 확인하기 위한 용도.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")

    results = [
        check_raw_iq_safety(
            block,
            full_scale=full_scale,
            saturation_ratio_warn=saturation_ratio_warn,
            saturation_ratio_clip=saturation_ratio_clip,
            near_saturation_level=near_saturation_level,
            saturation_level=saturation_level,
        )
        for block in blocks
    ]

    status_rank = {
        "SAFE": 0,
        "WEAK": 1,
        "WARNING": 2,
        "CLIPPED": 3,
    }

    worst = max(
        results,
        key=lambda item: status_rank.get(item.status, 99),
    )

    max_abs_values = np.asarray([x.max_abs for x in results], dtype=np.float64)
    rms_values = np.asarray([x.rms for x in results], dtype=np.float64)
    dc_abs_values = np.asarray([x.dc_abs for x in results], dtype=np.float64)
    saturation_values = np.asarray(
        [x.saturation_ratio for x in results],
        dtype=np.float64,
    )
    near_saturation_values = np.asarray(
        [x.near_saturation_ratio for x in results],
        dtype=np.float64,
    )

    return {
        "status": worst.status,
        "is_safe": bool(worst.status == "SAFE"),
        "num_blocks": int(len(results)),
        "max_abs_mean": float(np.mean(max_abs_values)),
        "max_abs_max": float(np.max(max_abs_values)),
        "rms_mean": float(np.mean(rms_values)),
        "rms_max": float(np.max(rms_values)),
        "dc_abs_mean": float(np.mean(dc_abs_values)),
        "dc_abs_max": float(np.max(dc_abs_values)),
        "saturation_ratio_mean": float(np.mean(saturation_values)),
        "saturation_ratio_max": float(np.max(saturation_values)),
        "near_saturation_ratio_mean": float(np.mean(near_saturation_values)),
        "near_saturation_ratio_max": float(np.max(near_saturation_values)),
        "full_scale": float(full_scale),
        "saturation_level": float(saturation_level),
        "near_saturation_level": float(near_saturation_level),
        "note": worst.note,
    }


def calibrate_noise_by_gain(
    gain_list: list[int | float],
    collect_fn: CollectNoiseBlocksFn,
    *,
    num_blocks_per_gain: int = 50,
    method: str = "time_power",
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold_multiplier: float = 5.0,
    min_detection_ratio: float = 0.05,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> GainNoiseCalibrationSet:
    """
    gain별 noise calibration을 수행한다.

    collect_fn:
        collect_fn(gain, num_blocks) -> list[np.ndarray]

    설계 의도:
    - 기존 calibrate_noise_from_blocks()는 수정하지 않는다.
    - integrated CLI와 live viewer는 collect_fn만 다르게 주입해서 같은 모듈을 쓴다.
    - gain별 결과는 하나의 JSON profile set으로 저장한다.
    """
    gains = _validate_gain_list(gain_list)
    profiles: dict[str, dict[str, Any]] = {}

    print("=" * 60)
    print("GAIN-WISE NOISE CALIBRATION 시작")
    print("=" * 60)
    print(f"gain_list            : {gains}")
    print(f"num_blocks_per_gain  : {num_blocks_per_gain}")
    print(f"method               : {method}")
    print(f"frame_size           : {frame_size}")
    print(f"hop_size             : {hop_size}")
    print(f"threshold_multiplier : {threshold_multiplier}")
    print("=" * 60)

    for gain in gains:
        print(f"\n[gain={gain:g}] noise block 수집 중...")
        blocks = collect_fn(gain, int(num_blocks_per_gain))

        result = calibrate_noise_from_blocks(
            blocks,
            method=method,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_multiplier=threshold_multiplier,
            calibration_blocks=int(num_blocks_per_gain),
            min_detection_ratio=min_detection_ratio,
            sample_rate=sample_rate,
            center_freq=center_freq,
        )

        safety = summarize_blocks_safety(
            blocks,
            full_scale=2048.0,
        )

        profile = GainNoiseCalibrationResult(
            gain=gain,
            result=result,
        )

        profile_dict = profile.to_dict()
        profile_dict["safety"] = safety

        profiles[gain_to_key(gain)] = profile_dict

        print(
            f"  gain={gain:g} | "
            f"noise_floor={result.noise_floor:.10g} | "
            f"threshold={result.threshold:.10g} | "
            f"noise_std={result.noise_std:.10g} | "
            f"safety={safety['status']} | "
            f"sat_max={safety['saturation_ratio_max'] * 100:.3f}%"
        )

    return GainNoiseCalibrationSet(
        mode="GAIN_NOISE_CALIBRATION",
        created_at=now_iso(),
        gain_list=gains,
        num_gains=len(gains),
        num_blocks_per_gain=int(num_blocks_per_gain),
        sample_rate=sample_rate,
        center_freq=center_freq,
        detector_method=method,
        frame_size=int(frame_size),
        hop_size=int(hop_size),
        threshold_multiplier=float(threshold_multiplier),
        min_detection_ratio=float(min_detection_ratio),
        profiles=profiles,
        note=(
            "Gain-wise noise calibration profile set. "
            "Each gain profile is generated by existing calibrate_noise_from_blocks(). "
            "Use get_noise_profile_for_gain() or get_noise_threshold_for_gain() "
            "from integrated CLI / live viewer / runtime pipeline."
        ),
    )


def load_gain_noise_calibration(path: str | Path) -> GainNoiseCalibrationSet:
    """gain별 noise calibration JSON을 로드한다."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("mode") != "GAIN_NOISE_CALIBRATION":
        raise ValueError(
            "Invalid gain noise calibration file: "
            f"mode={data.get('mode')!r}"
        )

    if "profiles" not in data or not isinstance(data["profiles"], dict):
        raise ValueError("Invalid gain noise calibration file: missing profiles")

    return GainNoiseCalibrationSet(
        mode=str(data["mode"]),
        created_at=str(data.get("created_at", "")),
        gain_list=[float(x) for x in data.get("gain_list", [])],
        num_gains=int(data.get("num_gains", len(data["profiles"]))),
        num_blocks_per_gain=int(data.get("num_blocks_per_gain", 0)),
        sample_rate=_optional_float(data.get("sample_rate")),
        center_freq=_optional_float(data.get("center_freq")),
        detector_method=str(data.get("detector_method", "time_power")),
        frame_size=int(data.get("frame_size", 1024)),
        hop_size=int(data.get("hop_size", 512)),
        threshold_multiplier=float(data.get("threshold_multiplier", 5.0)),
        min_detection_ratio=float(data.get("min_detection_ratio", 0.05)),
        profiles=data["profiles"],
        note=str(data.get("note", "")),
    )


def get_noise_profile_for_gain(
    profile_set: GainNoiseCalibrationSet | dict[str, Any],
    gain: int | float,
    *,
    allow_nearest: bool = True,
) -> dict[str, Any]:
    """
    현재 gain에 해당하는 noise profile을 반환한다.

    allow_nearest=True이면 정확히 같은 gain이 없을 때 가장 가까운 gain profile을 쓴다.
    live viewer에서 slider/input gain이 테이블 gain과 조금 다를 수 있으므로 기본값은 True.
    """
    profiles = profile_set.profiles if isinstance(profile_set, GainNoiseCalibrationSet) else profile_set["profiles"]

    key = gain_to_key(gain)
    if key in profiles:
        return dict(profiles[key])

    if not allow_nearest:
        available = ", ".join(sorted(profiles.keys(), key=_profile_key_to_float))
        raise KeyError(f"gain={gain} profile not found. Available gains: {available}")

    target = float(gain)
    nearest_key = min(
        profiles.keys(),
        key=lambda item: abs(_profile_key_to_float(item) - target),
    )
    profile = dict(profiles[nearest_key])
    profile["requested_gain"] = float(gain)
    profile["matched_gain"] = _profile_key_to_float(nearest_key)
    profile["matched_by"] = "nearest"
    return profile


def get_noise_threshold_for_gain(
    profile_set: GainNoiseCalibrationSet | dict[str, Any],
    gain: int | float,
    *,
    allow_nearest: bool = True,
) -> float:
    """현재 gain에 맞는 energy threshold를 반환한다."""
    profile = get_noise_profile_for_gain(
        profile_set,
        gain,
        allow_nearest=allow_nearest,
    )
    return float(profile["threshold"])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
