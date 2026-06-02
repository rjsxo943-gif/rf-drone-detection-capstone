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
from src.calibration.raw_iq_safety import (
    RawIQSafetyResult,
    check_raw_iq_safety,
    summarize_raw_iq_safety,
)


@dataclass(frozen=True)
class GainNoiseRuntimeResult:
    """
    현재 gain과 현재 IQ block 기준 runtime 판단 결과.

    CLI / live viewer / integrated runtime에서 공통 사용.
    """

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


class GainNoiseRuntime:
    """
    gain별 noise profile JSON을 runtime에서 쉽게 쓰기 위한 adapter.

    목적:
    - integrated CLI에서 현재 gain threshold 조회
    - live viewer에서 gain 변경 시 threshold 자동 반영
    - 현재 Raw IQ block safety check 수행
    """

    def __init__(
        self,
        profile_set: GainNoiseCalibrationSet,
        *,
        allow_nearest: bool = True,
        full_scale: float = 1.0,
    ) -> None:
        self.profile_set = profile_set
        self.allow_nearest = bool(allow_nearest)
        self.full_scale = float(full_scale)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        allow_nearest: bool = True,
        full_scale: float = 1.0,
    ) -> "GainNoiseRuntime":
        profile_set = load_gain_noise_calibration(path)
        return cls(
            profile_set,
            allow_nearest=allow_nearest,
            full_scale=full_scale,
        )

    def get_profile(self, gain: int | float) -> dict[str, Any]:
        return get_noise_profile_for_gain(
            self.profile_set,
            gain,
            allow_nearest=self.allow_nearest,
        )

    def get_threshold(self, gain: int | float) -> float:
        return get_noise_threshold_for_gain(
            self.profile_set,
            gain,
            allow_nearest=self.allow_nearest,
        )

    def check_block(
        self,
        block: np.ndarray,
        gain: int | float,
    ) -> GainNoiseRuntimeResult:
        """
        현재 gain에서 사용할 threshold와 Raw IQ safety를 같이 반환한다.
        """
        profile = self.get_profile(gain)
        threshold = float(profile["threshold"])

        matched_gain = float(profile.get("matched_gain", profile.get("gain", gain)))
        matched_by = str(profile.get("matched_by", "exact"))

        profile_safety = profile.get("safety", {})
        profile_safety_status = str(profile_safety.get("status", "UNKNOWN"))

        raw_safety = check_raw_iq_safety(
            block,
            full_scale=self.full_scale,
        )

        return GainNoiseRuntimeResult(
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

    def summarize_block(
        self,
        block: np.ndarray,
        gain: int | float,
    ) -> str:
        """
        CLI 출력 / live viewer overlay용 한 줄 요약.
        """
        result = self.check_block(block, gain)

        return (
            f"gain={result.gain:g} "
            f"(profile={result.matched_gain:g}, {result.matched_by}) | "
            f"threshold={result.threshold:.4g} | "
            f"profile_safety={result.profile_safety_status} | "
            f"raw_safety={result.raw_safety_status} | "
            f"max_abs={result.raw_max_abs:.4g} | "
            f"rms={result.raw_rms:.4g} | "
            f"sat={result.raw_saturation_ratio * 100:.3f}%"
        )


def load_gain_noise_runtime(
    path: str | Path,
    *,
    allow_nearest: bool = True,
    full_scale: float = 1.0,
) -> GainNoiseRuntime:
    """
    외부에서 간단히 호출하기 위한 helper.
    """
    return GainNoiseRuntime.from_json(
        path,
        allow_nearest=allow_nearest,
        full_scale=full_scale,
    )