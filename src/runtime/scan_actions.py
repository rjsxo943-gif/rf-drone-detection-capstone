from __future__ import annotations

from pathlib import Path

from src.calibration import load_calibration_params
from src.runtime.calibration_actions import (
    DEFAULT_NOISE_OUTPUT,
    DEFAULT_PHASE_GAIN_OUTPUT,
)
from src.runtime.scan_loop import run_continuous_scan_loop


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_loaded_calibration_status(
    *,
    require_noise: bool,
    require_phase_gain: bool,
) -> None:
    calibration = load_calibration_params(
        noise_path=DEFAULT_NOISE_OUTPUT,
        phase_gain_path=DEFAULT_PHASE_GAIN_OUTPUT,
        require_noise=require_noise,
        require_phase_gain=require_phase_gain,
    )

    print()
    print("=== Scan Calibration Check ===")

    if calibration.noise is not None:
        print("[Noise] loaded")
        print(f"threshold   : {calibration.noise.threshold:.10g}")
        print(f"noise_floor : {calibration.noise.noise_floor:.10g}")
        print(f"source      : {calibration.noise.source_path}")
    else:
        print("[Noise] not loaded")

    if calibration.phase_gain is not None:
        print()
        print("[Phase/Gain] loaded")
        print(f"gain_correction : {calibration.phase_gain.gain_correction:.10g}")
        print(f"phase_offset    : {calibration.phase_gain.phase_offset_rad:.10g} rad")
        print(f"phase_offset    : {calibration.phase_gain.phase_offset_deg:.6f} deg")
        print(f"coherence_like  : {calibration.phase_gain.coherence_like:.10g}")
        print(f"source          : {calibration.phase_gain.source_path}")
    else:
        print()
        print("[Phase/Gain] not loaded")
        print("[WARN] phase/gain calibration이 없으면 AoA 보정은 아직 신뢰하기 어렵다.")

    print()


def run_scan_action(
    *,
    require_noise: bool = True,
    require_phase_gain: bool = False,
    stop_key: str = "q",
    cycle_delay_sec: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    runtime CLI에서 continuous scan loop를 실행한다.

    더 이상 scripts/run_scan.py를 반복 subprocess로 실행하지 않는다.
    대신 src.runtime.scan_loop.run_continuous_scan_loop()를 직접 호출한다.
    """
    _print_loaded_calibration_status(
        require_noise=require_noise,
        require_phase_gain=require_phase_gain,
    )

    return run_continuous_scan_loop(
        stop_key=stop_key,
        cycle_delay_sec=cycle_delay_sec,
        verbose=verbose,
    )