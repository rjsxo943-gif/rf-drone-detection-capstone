#cli.py
from __future__ import annotations

from pathlib import Path

from src.runtime.cnn_capture_actions import run_cnn_capture_action
from src.runtime.scan_actions import run_scan_action
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
    print("[n] noise calibration")
    print("[p] phase/gain calibration")
    print("[s] start")
    print("[q] quit")


def run_cli() -> None:
    print_calibration_status()

    while True:
        print_menu()

        try:
            cmd = input("select> ").strip().lower()
        except KeyboardInterrupt:
            print()
            print("exit runtime cli")
            break

        if cmd == "q":
            print("exit runtime cli")
            break

        elif cmd == "c":
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
            print("=== Start CNN Dataset Capture ===")
            print("신호원을 켜둔 상태에서 label을 직접 입력하면,")
            print("스캔으로 잡힌 후보 신호를 CNN 학습용 spectrogram으로 저장한다.")
            print()

            label = input("label ex) wifi / bluetooth / drone_like / background > ").strip()

            max_saved_text = input("max_saved [default=50] > ").strip()
            max_saved = int(max_saved_text) if max_saved_text else 50

            rx_index_text = input("rx_index [default=0] > ").strip()
            rx_index = int(rx_index_text) if rx_index_text else 0

            save_raw_text = input("save raw iq? [y/N] > ").strip().lower()
            save_raw_iq = save_raw_text in ("y", "yes")

            print()
            print("=== Capture Config ===")
            print(f"label       : {label}")
            print(f"max_saved   : {max_saved}")
            print(f"rx_index    : {rx_index}")
            print(f"save_raw_iq : {save_raw_iq}")
            print()

            try:
                return_code = run_cnn_capture_action(
                    label=label,
                    max_saved=max_saved,
                    rx_index=rx_index,
                    save_raw_iq=save_raw_iq,
                    require_noise=True,
                    require_phase_gain=False,
                    stop_key="q",
                    cycle_delay_sec=0.0,
                    verbose=True,
                )

                if return_code != 0:
                    print(f"[WARN] capture finished with non-zero return code: {return_code}")

            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                print("먼저 [n] noise calibration을 실행해야 한다.")

            except Exception as e:
                print(f"[ERROR] cnn capture action failed: {e}")

        elif cmd == "":
            continue

        else:
            print(f"unknown command: {cmd}")