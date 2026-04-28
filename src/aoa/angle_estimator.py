from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AngleEstimateResult:
    """
    block 하나에 대한 AoA 각도 계산 결과.
    """

    angle_deg: float
    angle_rad: float
    phase_diff_rad: float
    phase_diff_deg: float
    arcsin_input: float
    valid: bool


def phase_diff_to_angle(
    phase_diff_rad: float,
    carrier_freq: float = 2_400_000_000,
    antenna_spacing_m: float = 0.0625,
    speed_of_light: float = 300_000_000,
    phase_offset_rad: float = 0.0,
    clip_input: bool = True,
) -> AngleEstimateResult:
    """
    RX0/RX1 위상차를 AoA 각도로 변환한다.

    현재 프로젝트 기준:
    - carrier_freq = 2.4 GHz
    - wavelength = 0.125 m
    - antenna_spacing = 0.0625 m = lambda / 2
    - 입력 phase_diff_rad는 RX1 위상 - RX0 위상 기준

    공식:
        theta = arcsin((phase_diff * wavelength) / (2*pi*d))

    Args:
        phase_diff_rad:
            RX1 - RX0 위상차 [rad]
        carrier_freq:
            중심 주파수 [Hz]
        antenna_spacing_m:
            안테나 간격 [m]
        speed_of_light:
            전파 속도 [m/s]
        phase_offset_rad:
            RX0/RX1 하드웨어 고정 위상 오차 보정값 [rad]
        clip_input:
            arcsin 입력값을 -1~1로 제한할지 여부

    Returns:
        AngleEstimateResult
    """
    if carrier_freq <= 0:
        raise ValueError(f"carrier_freq must be positive, got {carrier_freq}")

    if antenna_spacing_m <= 0:
        raise ValueError(
            f"antenna_spacing_m must be positive, got {antenna_spacing_m}"
        )

    if speed_of_light <= 0:
        raise ValueError(f"speed_of_light must be positive, got {speed_of_light}")

    wavelength_m = speed_of_light / carrier_freq

    # 하드웨어 고정 위상 오차 보정
    corrected_phase = wrap_phase_rad(phase_diff_rad - phase_offset_rad)

    arcsin_input = (corrected_phase * wavelength_m) / (
        2.0 * np.pi * antenna_spacing_m
    )

    valid = bool(-1.0 <= arcsin_input <= 1.0)

    if clip_input:
        arcsin_input_used = float(np.clip(arcsin_input, -1.0, 1.0))
    else:
        if not valid:
            return AngleEstimateResult(
                angle_deg=float("nan"),
                angle_rad=float("nan"),
                phase_diff_rad=float(corrected_phase),
                phase_diff_deg=float(np.rad2deg(corrected_phase)),
                arcsin_input=float(arcsin_input),
                valid=False,
            )

        arcsin_input_used = float(arcsin_input)

    angle_rad = float(np.arcsin(arcsin_input_used))
    angle_deg = float(np.rad2deg(angle_rad))

    return AngleEstimateResult(
        angle_deg=angle_deg,
        angle_rad=angle_rad,
        phase_diff_rad=float(corrected_phase),
        phase_diff_deg=float(np.rad2deg(corrected_phase)),
        arcsin_input=float(arcsin_input),
        valid=valid,
    )


def estimate_angle_from_phase_result(
    phase_result: Any,
    carrier_freq: float = 2_400_000_000,
    antenna_spacing_m: float = 0.0625,
    speed_of_light: float = 300_000_000,
    phase_offset_rad: float = 0.0,
    clip_input: bool = True,
) -> AngleEstimateResult:
    """
    PhaseDiffResult 객체를 받아 AoA를 계산한다.

    phase_diff.py의 estimate_phase_diff() 결과와 연결할 때 사용한다.
    """
    return phase_diff_to_angle(
        phase_diff_rad=phase_result.phase_diff_rad,
        carrier_freq=carrier_freq,
        antenna_spacing_m=antenna_spacing_m,
        speed_of_light=speed_of_light,
        phase_offset_rad=phase_offset_rad,
        clip_input=clip_input,
    )


def wrap_phase_rad(phase: float | np.ndarray) -> float | np.ndarray:
    """
    위상을 -pi ~ +pi 범위로 정리한다.
    """
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def angle_to_phase_diff(
    angle_deg: float,
    carrier_freq: float = 2_400_000_000,
    antenna_spacing_m: float = 0.0625,
    speed_of_light: float = 300_000_000,
) -> float:
    """
    테스트용 역변환 함수.

    특정 AoA 각도에서 기대되는 위상차를 계산한다.

    공식:
        phase_diff = 2π d sin(theta) / wavelength
    """
    if carrier_freq <= 0:
        raise ValueError(f"carrier_freq must be positive, got {carrier_freq}")

    if antenna_spacing_m <= 0:
        raise ValueError(
            f"antenna_spacing_m must be positive, got {antenna_spacing_m}"
        )

    if speed_of_light <= 0:
        raise ValueError(f"speed_of_light must be positive, got {speed_of_light}")

    wavelength_m = speed_of_light / carrier_freq
    angle_rad = np.deg2rad(angle_deg)

    phase_diff_rad = (
        2.0 * np.pi * antenna_spacing_m * np.sin(angle_rad)
    ) / wavelength_m

    return float(wrap_phase_rad(phase_diff_rad))