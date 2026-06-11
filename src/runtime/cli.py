#cli.py
from __future__ import annotations

from pathlib import Path
from src.runtime.fixed2450_precision_runtime import run_fixed2450_precision_runtime
import re
import runpy
import sys

from src.runtime.cnn_capture_actions import run_cnn_capture_action
from src.runtime.rf4_actions import run_rf4_single_block_action
from src.calibration import (
    load_gain_noise_calibration,
    load_phase_gain_by_gain_calibration,
)
from src.runtime.scan_loop import run_continuous_scan_loop
from src.runtime.opencv_scan_precision_runtime import run_opencv_scan_precision_runtime
from src.runtime.scan_activity_cnn_runtime import run_scan_activity_cnn_runtime
from src.runtime.calibration_actions import (
    DEFAULT_GAIN_LIST,
    DEFAULT_GAIN_NOISE_OUTPUT,
    DEFAULT_GAIN_PHASE_GAIN_OUTPUT,
    run_gain_wise_noise_calibration_action,
    run_gain_wise_phase_gain_calibration_action,
)


EXIT_COMMANDS = {"q", "quit", "exit", "shutdown"}
STATUS_COMMANDS = {"c", "status"}
NOISE_COMMANDS = {"n", "noise"}
PHASE_COMMANDS = {"p", "phase", "phase_gain"}
PIPELINE_COMMANDS = {"s", "start", "run", "pipeline", "a", "aoa"}
SCAN_HANDOFF_COMMANDS = {"sf", "scan_fixed", "scan_precision", "scan_handoff"}
FIXED_AOA_COMMANDS = {"f", "fixed", "fixed2450", "fixed_2450", "aoa_fixed"}
TERMINAL_PIPELINE_COMMANDS = {"t", "terminal", "terminal_loop", "scan_loop"}
DEMO_COMMANDS = {"v", "view", "viewer", "demo", "ui_demo"}
CAPTURE_COMMANDS = {"d", "dataset", "capture"}
RF4_COMMANDS = {"r", "rf4"}


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
    """
    현재 runtime 시작 전 상태창 역할.

    확인 항목:
    - gain-wise noise calibration 존재 여부
    - gain-wise phase/gain calibration 존재 여부
    - gain별 threshold / raw safety profile
    - gain별 phase offset / coherence profile
    """
    noise_path = Path(DEFAULT_GAIN_NOISE_OUTPUT)
    phase_gain_path = Path(DEFAULT_GAIN_PHASE_GAIN_OUTPUT)

    print()
    print("=== Runtime Calibration / Pipeline Status ===")
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
    else:
        print("[WARN] noise calibration이 아직 없습니다. 먼저 [n]을 권장합니다.")

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
    else:
        print("[WARN] phase/gain calibration이 아직 없습니다. AoA 신뢰도 확보 전에는 [p]를 권장합니다.")

    print()


def print_menu() -> None:
    print()
    print("=== RF Drone Detection Runtime CLI ===")
    print("[c] status        : calibration / pipeline 현재 상태창")
    print("[n] noise         : gain-wise noise calibration")
    print("[p] phase         : gain-wise phase/gain calibration")
    print("[s] start         : clean SCAN(raw gate) + 후보 freq CNN Top5 vote 확인용, precision 진입 안 함")
    print("[sf] scan_fixed   : clean SCAN + CNN Top5 vote 통과 시 fixed 2.450 precision 진입")
    print("[f] fixed-2450    : SCAN 없이 2.450GHz 고정 AoA/거리 dashboard 구동")
    print("[v] view/demo     : Pluto 없이 OpenCV UI demo 구동")
    print("[t] terminal-loop : 기존 terminal scan/runtime pipeline 구동")
    print("[d] dataset       : CNN dataset capture")
    print("[r] rf4           : RF4 single block inference")
    print("[q] quit/shutdown : receiver close 후 종료")


def _run_pipeline_start_action() -> None:
    print()
    print("=== Clean SCAN Activity + CNN Verify Only Start ===")
    print("흐름: CLI [s] → raw gate sweep → 후보 freq CNN Top5 vote 확인, precision 진입 안 함")
    print("중단: OpenCV 창에서 q 또는 ESC")
    print()
    print_calibration_status()

    try:
        return_code = run_scan_activity_cnn_runtime(
            handoff_to_precision=False,
            config_dir="configs",
            stop_key="q",
            verbose=True,
        )

        if return_code != 0:
            print(f"[WARN] clean scan observe runtime finished with non-zero return code: {return_code}")

    except SystemExit as e:
        code = e.code
        if code not in (None, 0):
            print(f"[WARN] OpenCV UI finished with non-zero return code: {code}")

    except Exception as e:
        print(f"[ERROR] clean scan observe runtime failed: {e}")


def _run_fixed_2450_dashboard_action() -> None:
    print()
    print("=== Fixed 2.450GHz AoA/Distance Dashboard Start ===")
    print("흐름: CLI [f] → fixed2450_precision_runtime → 2.450GHz 고정 관측")
    print("SCAN sweep / candidate selection / scan policy를 사용하지 않습니다.")
    print("중단: OpenCV 창에서 q 또는 ESC")
    print()
    print_calibration_status()

    try:
        run_fixed2450_precision_runtime(
            config_dir="configs",
            center_freq_hz=2.450e9,
        )

    except SystemExit as e:
        code = e.code
        if code not in (None, 0):
            print(f"[WARN] fixed 2.450 precision runtime finished with non-zero return code: {code}")

    except Exception as e:
        print(f"[ERROR] fixed 2.450 precision runtime failed: {e}")



def _run_scan_handoff_action() -> None:
    print()
    print("=== Clean SCAN Activity + CNN Verify → Fixed Precision Start ===")
    print("흐름: CLI [sf] → raw gate sweep → 후보 freq CNN Top5 vote → 통과 시 fixed 2.450 precision")
    print("중단: OpenCV 창에서 q 또는 ESC")
    print()
    print_calibration_status()

    try:
        return_code = run_scan_activity_cnn_runtime(
            handoff_to_precision=True,
            config_dir="configs",
            stop_key="q",
            verbose=True,
        )
        if return_code not in (None, 0):
            print(f"[WARN] clean scan handoff runtime finished with non-zero return code: {return_code}")

    except SystemExit as e:
        code = e.code
        if code not in (None, 0):
            print(f"[WARN] clean scan handoff runtime finished with non-zero return code: {code}")

    except Exception as e:
        print(f"[ERROR] clean scan handoff runtime failed: {e}")


def _run_opencv_demo_action() -> None:
    print()
    print("=== OpenCV SCAN + PRECISION UI Demo ===")
    print("Pluto+ 없이 UI 동작만 확인합니다.")
    print("중단: OpenCV 창에서 q 또는 ESC")
    print()

    try:
        runpy.run_path(
            "scripts/experimental/test_scan_precision_rail_demo.py",
            run_name="__main__",
        )

    except SystemExit as e:
        code = e.code
        if code not in (None, 0):
            print(f"[WARN] OpenCV demo finished with non-zero return code: {code}")

    except Exception as e:
        print(f"[ERROR] OpenCV demo failed: {e}")


def _run_terminal_pipeline_action() -> None:
    print()
    print("=== Terminal Integrated Scan/Runtime Pipeline Start ===")
    print("흐름: Scan → Candidate → Precision CNN → Coherence/AoA/Sector → Logging")
    print("중단: scan cycle 사이에 q 입력 후 Enter, 또는 Ctrl+C")
    print()
    print_calibration_status()

    try:
        return_code = run_continuous_scan_loop(
            stop_key="q",
            cycle_delay_sec=0.0,
            verbose=True,
        )

        if return_code != 0:
            print(f"[WARN] runtime pipeline finished with non-zero return code: {return_code}")

    except Exception as e:
        print(f"[ERROR] runtime pipeline failed: {e}")


def _run_cnn_dataset_capture_action() -> None:
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


def _run_rf4_single_block_inference_action() -> None:
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


def run_cli() -> None:
    print_calibration_status()

    while True:
        print_menu()

        try:
            cmd = input("select> ").strip().lower()
        except KeyboardInterrupt:
            print()
            print("shutdown runtime cli")
            break

        if cmd in EXIT_COMMANDS:
            print("shutdown runtime cli")
            break

        elif cmd in STATUS_COMMANDS:
            print_calibration_status()

        elif cmd in NOISE_COMMANDS:
            print()
            print("=== Run Gain-wise Noise Calibration ===")
            print("주의: 드론/조종기/와이파이 신호원을 최대한 끄고 배경 noise만 받는 상태에서 진행한다.")
            gain_list = _ask_gain_list()
            num_blocks = _ask_int("num_blocks_per_gain", 50)
            run_gain_wise_noise_calibration_action(
                gain_list=gain_list,
                num_blocks_per_gain=num_blocks,
            )
            print_calibration_status()

        elif cmd in PHASE_COMMANDS:
            print()
            print("=== Run Gain-wise Phase/Gain Calibration ===")
            print("주의: 기준 신호원을 두 안테나 정면 0도 방향에 두고 진행한다.")
            gain_list = _ask_gain_list()
            num_blocks = _ask_int("num_blocks_per_gain", 50)
            run_gain_wise_phase_gain_calibration_action(
                gain_list=gain_list,
                num_blocks_per_gain=num_blocks,
            )
            print_calibration_status()

        elif cmd in DEMO_COMMANDS:
            _run_opencv_demo_action()

        elif cmd in TERMINAL_PIPELINE_COMMANDS:
            _run_terminal_pipeline_action()

        elif cmd in SCAN_HANDOFF_COMMANDS:
            _run_scan_handoff_action()

        elif cmd in FIXED_AOA_COMMANDS:
            _run_fixed_2450_dashboard_action()

        elif cmd in PIPELINE_COMMANDS:
            _run_pipeline_start_action()

        elif cmd in CAPTURE_COMMANDS:
            _run_cnn_dataset_capture_action()

        elif cmd in RF4_COMMANDS:
            _run_rf4_single_block_inference_action()

        elif cmd == "":
            continue

        else:
            print(f"unknown command: {cmd}")


if __name__ == "__main__":
    run_cli()
