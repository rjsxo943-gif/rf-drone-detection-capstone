#cli.py
from __future__ import annotations

from pathlib import Path
import re

from src.runtime.cnn_capture_actions import run_cnn_capture_action
from src.runtime.scan_actions import run_scan_action
from src.runtime.rf4_actions import run_rf4_single_block_action
from src.calibration import (
    load_gain_noise_calibration,
    load_phase_gain_by_gain_calibration,
)
from src.runtime.scan_loop import run_continuous_scan_loop
from src.runtime.calibration_actions import (
    DEFAULT_GAIN_LIST,
    DEFAULT_GAIN_NOISE_OUTPUT,
    DEFAULT_GAIN_PHASE_GAIN_OUTPUT,
    run_gain_wise_noise_calibration_action,
    run_gain_wise_phase_gain_calibration_action,
)


def _format_exists(path: Path) -> str:
    return "있음" if path.exists() else "없음"


def _parse_gain_list_text(text: str, default: tuple[float, ...] = DEFAULT_GAIN_LIST) -> list[float]:
    value = text.strip()
    if not value:
        return [float(x) for x in default]

    # accepts: 20,25,30 or 20 25 30
    tokens = [x for x in re.split(r"[,\s]+", value) if x]
    gains = sorted(dict.fromkeys(float(x) for x in tokens))
    if not gains:
        raise ValueError("gain list must not be empty")
    return gains


def _ask_gain_list() -> list[float]:
    default_text = ",".join(f"{x:g}" for x in DEFAULT_GAIN_LIST)
    text = input(f"gain list [default={default_text}] > ").strip()
    return _parse_gain_list_text(text)


def _ask_int(prompt: str, default: int) -> int:
    text = input(f"{prompt} [default={default}] > ").strip()
    return int(text) if text else int(default)


def print_calibration_status() -> None:
    noise_path = Path(DEFAULT_GAIN_NOISE_OUTPUT)
    phase_gain_path = Path(DEFAULT_GAIN_PHASE_GAIN_OUTPUT)

    print()
    print("=== Gain-wise Calibration Status ===")
    print(f"noise_by_gain profile      : {_format_exists(noise_path)}")
    print(f"phase_gain_by_gain profile : {_format_exists(phase_gain_path)}")

    if noise_path.exists():
        try:
            noise_set = load_gain_noise_calibration(noise_path)
            print()
            print("[Gain-wise Noise]")
            print(f"source     : {noise_path}")
            print(f"gain_list  : {noise_set.gain_list}")
            print(f"blocks/gain: {noise_set.num_blocks_per_gain}")
            for gain_key, profile in sorted(noise_set.profiles.items(), key=lambda item: float(item[0])):
                safety = profile.get("safety", {})
                print(
                    f"  gain={float(gain_key):g} "
                    f"thr={float(profile.get('threshold', float('nan'))):.10g} "
                    f"safety={safety.get('status', 'UNKNOWN')} "
                    f"sat_max={float(safety.get('saturation_ratio_max', 0.0)) * 100:.3f}%"
                )
        except Exception as e:
            print(f"[WARN] gain-wise noise status load failed: {e}")

    if phase_gain_path.exists():
        try:
            pg_set = load_phase_gain_by_gain_calibration(phase_gain_path)
            print()
            print("[Gain-wise Phase/Gain]")
            print(f"source     : {phase_gain_path}")
            print(f"gain_list  : {pg_set.gain_list}")
            print(f"blocks/gain: {pg_set.num_blocks_per_gain}")
            for gain_key, profile in sorted(pg_set.profiles.items(), key=lambda item: float(item[0])):
                print(
                    f"  gain={float(gain_key):g} "
                    f"corr={float(profile.get('gain_correction_mean', float('nan'))):.6g} "
                    f"phase={float(profile.get('phase_offset_deg_mean', float('nan'))):.3f}deg "
                    f"coh={float(profile.get('coherence_like_mean', float('nan'))):.6g}"
                )
        except Exception as e:
            print(f"[WARN] gain-wise phase/gain status load failed: {e}")

    print()

def print_menu() -> None:
    print()
    print("=== RF Drone Detection Runtime CLI ===")
    print("[c] gain-wise calibration status")
    print("[n] gain-wise noise calibration")
    print("[p] gain-wise phase/gain calibration")
    print("[s] start")
    print("[r] RF4 single block inference")
    print("[a] AoA sector scan")
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
            print("=== Run Gain-wise Noise Calibration ===")
            print("주의: 드론/조종기/와이파이 신호원을 최대한 끄고 배경 noise만 받는 상태에서 진행한다.")
            gain_list = _ask_gain_list()
            num_blocks = _ask_int("num_blocks_per_gain", 50)
            run_gain_wise_noise_calibration_action(
                gain_list=gain_list,
                num_blocks_per_gain=num_blocks,
            )

        elif cmd == "p":
            print()
            print("=== Run Gain-wise Phase/Gain Calibration ===")
            print("주의: 기준 신호원을 두 안테나 정면 0도 방향에 두고 진행한다.")
            gain_list = _ask_gain_list()
            num_blocks = _ask_int("num_blocks_per_gain", 50)
            run_gain_wise_phase_gain_calibration_action(
                gain_list=gain_list,
                num_blocks_per_gain=num_blocks,
            )

        elif cmd == "a":
            print()
            print("=== AoA Sector Scan Start ===")
            print("PrecisionAnalyzer 기반 scan loop를 실행한다.")
            print("angle_deg와 8-sector 방향을 함께 출력한다.")
            print("scan cycle 사이에 q 입력 후 Enter를 누르면 중단한다.")
            print()

            try:
                return_code = run_continuous_scan_loop(
                    stop_key="q",
                    cycle_delay_sec=0.0,
                    verbose=True,
                )

                if return_code != 0:
                    print(f"[WARN] AoA sector scan finished with non-zero return code: {return_code}")

            except Exception as e:
                print(f"[ERROR] AoA sector scan failed: {e}")

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
                print("먼저 [n] gain-wise noise calibration을 실행해야 한다.")

            except Exception as e:
                print(f"[ERROR] cnn capture action failed: {e}")


        elif cmd == "r":
            print()
            print("=== RF4 Single Block Inference ===")

            model_path = input(
                "model path [default=outputs/ml/rf4_cnn_live2450_v2/best_model.pt] > "
            ).strip()
            if not model_path:
                model_path = "outputs/ml/rf4_cnn_live2450_v2/best_model.pt"

            center_freq_text = input("center_freq Hz [default=2437000000] > ").strip()
            center_freq = int(center_freq_text) if center_freq_text else 2_437_000_000

            rx_index_text = input("rx_index [default=0] > ").strip()
            rx_index = int(rx_index_text) if rx_index_text else 0

            general_threshold_text = input("general_threshold [default=0.50] > ").strip()
            general_threshold = (
                float(general_threshold_text) if general_threshold_text else 0.50
            )

            drone_threshold_text = input("drone_threshold [default=0.70] > ").strip()
            drone_threshold = (
                float(drone_threshold_text) if drone_threshold_text else 0.70
            )

            num_blocks_text = input("num_blocks [default=10] > ").strip()
            num_blocks = int(num_blocks_text) if num_blocks_text else 10

            min_drone_votes_text = input("min_drone_votes [default=3] > ").strip()
            min_drone_votes = (
                int(min_drone_votes_text) if min_drone_votes_text else 3
            )

            try:
                return_code = run_rf4_single_block_action(
                    model_path=model_path,
                    center_freq=center_freq,
                    rx_index=rx_index,
                    general_threshold=general_threshold,
                    drone_threshold=drone_threshold,
                    num_blocks=num_blocks,
                    min_drone_votes=min_drone_votes,
                )

                if return_code != 0:
                    print(f"[WARN] RF4 inference finished with non-zero return code: {return_code}")

            except FileNotFoundError as e:
                print(f"[ERROR] {e}")

            except Exception as e:
                print(f"[ERROR] RF4 inference failed: {e}")

        elif cmd == "":
            continue

        else:
            print(f"unknown command: {cmd}")
if __name__ == "__main__":
    run_cli()
