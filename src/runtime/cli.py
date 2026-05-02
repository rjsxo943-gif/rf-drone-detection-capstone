from __future__ import annotations

from pathlib import Path

from src.calibration import load_calibration_params
from src.runtime.calibration_actions import (
    DEFAULT_NOISE_OUTPUT,
    DEFAULT_PHASE_GAIN_OUTPUT,
    run_noise_calibration_action,
    run_phase_gain_calibration_action,
)


def _format_exists(path: Path) -> str:
    return "있음" if path.exists() else "없음"


def print_calibration_status() -> None:
    noise_path = Path(DEFAULT_NOISE_OUTPUT)
    phase_gain_path = Path(DEFAULT_PHASE_GAIN_OUTPUT)

    print()
    print("=== Calibration Status ===")
    print(f"noise calibration      : {_format_exists(noise_path)}")
    print(f"phase/gain calibration : {_format_exists(phase_gain_path)}")

    try:
        calib = load_calibration_params(
            noise_path=noise_path,
            phase_gain_path=phase_gain_path,
            require_noise=False,
            require_phase_gain=False,
        )

        if calib.noise is not None:
            print()
            print("[Noise]")
            print(f"noise_floor : {calib.noise.noise_floor:.10g}")
            print(f"threshold   : {calib.noise.threshold:.10g}")
            print(f"method      : {calib.noise.detector_method}")
            print(f"source      : {calib.noise.source_path}")

        if calib.phase_gain is not None:
            print()
            print("[Phase/Gain]")
            print(f"gain_correction : {calib.phase_gain.gain_correction:.10g}")
            print(f"phase_offset    : {calib.phase_gain.phase_offset_rad:.10g} rad")
            print(f"phase_offset    : {calib.phase_gain.phase_offset_deg:.6f} deg")
            print(f"coherence_like  : {calib.phase_gain.coherence_like:.10g}")
            print(f"source          : {calib.phase_gain.source_path}")

    except Exception as e:
        print(f"[WARN] calibration status load failed: {e}")

    print()


def print_menu() -> None:
    print()
    print("=== RF Drone Detection Runtime CLI ===")
    print("[c] calibration status")
    print("[n] noise calibration")
    print("[p] phase/gain calibration")
    print("[s] scan start")
    print("[q] quit")


def run_cli() -> None:
    print_calibration_status()

    while True:
        print_menu()
        cmd = input("select> ").strip().lower()

        if cmd == "c":
            print_calibration_status()

        elif cmd == "n":
            print()
            print("=== Run Noise Calibration ===")
            print("주의: 의도적인 신호원을 끄고 진행하는 것이 좋다.")
            run_noise_calibration_action()

        elif cmd == "p":
            print()
            print("=== Run Phase/Gain Calibration ===")
            print("주의: 신호원을 두 안테나의 정면 0도 방향에 두고 진행하는 것이 기준이다.")
            run_phase_gain_calibration_action()

        elif cmd == "s":
            print()
            print("=== Scan Start ===")
            print("아직 runtime CLI에서는 scan을 직접 실행하지 않는다.")
            print("현재는 아래 명령어로 실행하면 된다:")
            print()
            print("PYTHONPATH=. python scripts/run_scan.py")
            print()
            print("다음 단계에서 SCAN/BAND_HOLD 상태머신에 연결하면 된다.")

        elif cmd == "q":
            print("exit runtime cli")
            break

        else:
            print(f"unknown command: {cmd}")