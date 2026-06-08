from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.core import load_all_configs
from src.features.spectrogram import compute_stft_branch
from src.ml.runtime_decision import load_runtime_decision_config, select_drone_threshold
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver.pluto_receiver import PlutoReceiver
from src.runtime.calibration_runtime import load_calibration_runtime
from src.runtime.raw_noise_gate import RawNoiseGate
from src.viewer import (
    AoARuntime,
    CNNRuntime,
    GainProfileRuntime,
    OpenCVRenderer,
    ViewerState,
    append_viewer_csv,
    compute_raw_features,
)


SUPPORTED_MODES = ("fast", "profile", "cnn", "aoa", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV-based live RF viewer for fast/profile/cnn/aoa/full experiment modes."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="fast")
    parser.add_argument("--uri", default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, default=2_450_000_000)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=None)
    parser.add_argument("--gain", type=float, default=30.0)
    parser.add_argument("--gain-step", type=float, default=1.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--max-gain", type=float, default=73.0)
    parser.add_argument("--block-size", type=int, default=16_384)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument(
        "--disable-dc-offset-removal",
        action="store_true",
        help="Disable per-block per-channel DC offset removal before feature/CNN/AoA processing.",
    )
    parser.add_argument("--num-channels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to the displayed spectrogram.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Optional fixed OpenCV display width in pixels. 0 keeps scale-based sizing.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="Optional fixed OpenCV display height in pixels. 0 keeps scale-based sizing.",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=("right", "image"),
        default="right",
        help="Overlay layout. 'right' puts text in a side panel; 'image' draws text on spectrogram.",
    )
    parser.add_argument(
        "--overlay-panel-width",
        type=int,
        default=520,
        help="Width of right-side status panel in pixels.",
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Disable automatic landscape orientation for spectrogram display.",
    )
    parser.add_argument(
        "--debug-shape",
        action="store_true",
        help="Print raw and display spectrogram shapes.",
    )
    parser.add_argument("--distance-m", type=float, default=0.0)
    parser.add_argument("--memo", default="")
    parser.add_argument(
        "--blocks-per-update",
        type=int,
        default=5,
        help="Read N RF blocks per viewer update and select one representative block for CNN/AoA.",
    )
    parser.add_argument(
        "--select-policy",
        choices=("raw_gate_pass_score_max", "raw_gate_score_max", "raw_p99", "frame_power_p99", "last"),
        default="raw_gate_pass_score_max",
        help="Representative block selection policy within each update set.",
    )
    parser.add_argument("--profile-blocks", type=int, default=20)
    parser.add_argument(
        "--profile-csv",
        default="outputs/viewer/gain_feature_profiles.csv",
    )
    parser.add_argument(
        "--profile-json",
        default="outputs/viewer/gain_feature_profiles_latest.json",
    )
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=96)
    parser.add_argument("--nfft", type=int, default=128)
    parser.add_argument("--window", default="hann")
    parser.add_argument(
        "--display-freq-bins",
        type=int,
        default=128,
        help="Expected frequency-axis bins for display. Default: 128, so display shape is (128, time_bins).",
    )
    parser.add_argument("--model", default=None, help="CNN model checkpoint path for cnn mode.")
    parser.add_argument("--cnn-backend", choices=("binary", "binary_flat", "rf4_binary", "drone_binary", "torch", "keras", "dummy"), default="torch")
    parser.add_argument("--cnn-device", default="cpu")
    parser.add_argument(
        "--class-names",
        default="Background,WiFi,Bluetooth,Drone-like",
        help="Comma-separated CNN class names in checkpoint order.",
    )
    parser.add_argument(
        "--cnn-positive-class-names",
        default="Drone-like,Drone,drone",
        help="Comma-separated class names treated as positive Drone-like candidates.",
    )
    parser.add_argument("--cnn-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--cnn-smooth-window", type=int, default=5)
    parser.add_argument("--cnn-confirm-votes", type=int, default=3)
    parser.add_argument("--cnn-dummy-class-name", default="Background")
    parser.add_argument("--cnn-dummy-confidence", type=float, default=0.0)
    parser.add_argument("--aoa-phase-calibration-json", default=None)
    parser.add_argument("--aoa-gain-phase-table", default=None)
    parser.add_argument("--aoa-phase-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--noise-profile",
        default="outputs/calibration/noise_by_gain_latest.json",
        help="Gain-wise noise calibration JSON path.",
    )
    parser.add_argument(
        "--phase-gain-profile",
        default="outputs/calibration/phase_gain_by_gain_latest.json",
        help="Gain-wise phase/gain calibration JSON path.",
    )
    parser.add_argument(
        "--disable-calibration-runtime",
        action="store_true",
        help="Disable gain-wise calibration runtime lookup.",
    )
    parser.add_argument("--aoa-ref-channel", type=int, default=0)
    parser.add_argument("--aoa-target-channel", type=int, default=1)
    parser.add_argument("--aoa-antenna-spacing-m", type=float, default=0.0625)
    parser.add_argument("--aoa-speed-of-light", type=float, default=300_000_000.0)
    parser.add_argument("--aoa-coherence-threshold", type=float, default=0.6)
    parser.add_argument("--aoa-energy-percentile", type=float, default=75.0)
    parser.add_argument(
        "--aoa-disable-stft-coherence",
        action="store_true",
        help="Disable STFT coherence calculation for faster AoA rendering.",
    )
    parser.add_argument(
        "--aoa-no-clip-angle-input",
        action="store_true",
        help="Return NaN angle when arcsin input is outside [-1, 1].",
    )
    parser.add_argument(
        "--log-csv",
        default="outputs/viewer/live_rf_viewer_log.csv",
        help="CSV path for full-mode live logging.",
    )
    parser.add_argument(
        "--log-every-n",
        type=int,
        default=1,
        help="Append one full-mode log row every N newly-read blocks.",
    )
    parser.add_argument(
        "--cli-log-every-n",
        type=int,
        default=5,
        help="Print one compact terminal status line every N update sets.",
    )
    parser.add_argument(
        "--disable-cli-log",
        action="store_true",
        help="Disable periodic terminal status logs.",
    )
    parser.add_argument(
        "--disable-log",
        action="store_true",
        help="Disable full-mode live CSV logging.",
    )
    parser.add_argument(
        "--overload-threshold",
        type=float,
        default=None,
        help="Optional abs(IQ) threshold for overload flag. Leave unset to disable.",
    )
    args = parser.parse_args()
    if args.target_fps is None:
        args.target_fps = 5.0 if args.mode in ("cnn", "full") else 10.0
    args.log_every_n = max(1, int(args.log_every_n))
    args.blocks_per_update = max(1, int(args.blocks_per_update))
    args.cli_log_every_n = max(1, int(args.cli_log_every_n))
    return args


def build_receiver(args: argparse.Namespace) -> PlutoReceiver:
    channels = [0, 1] if args.num_channels == 2 else [0]
    return PlutoReceiver(
        uri=args.uri,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        num_channels=args.num_channels,
        channels=channels,
        gain=args.gain,
        block_size=args.block_size,
        rf_bandwidth=args.rf_bandwidth,
    )


def _parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _flag_given(*flags: str) -> bool:
    argv = sys.argv[1:]
    for flag in flags:
        if flag in argv:
            return True
        if any(str(item).startswith(flag + "=") for item in argv):
            return True
    return False


def _nested_get(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _csv(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return ",".join(str(v) for v in values)


def _viewer_backend(value: Any) -> str:
    backend = str(value or "dummy").lower().strip()
    if backend in {"binary_flat", "rf4_binary", "drone_binary"}:
        return "binary"
    return backend


def _set_yaml_default(args: argparse.Namespace, attr: str, flags: tuple[str, ...], value: Any) -> None:
    if value is None:
        return
    if _flag_given(*flags):
        return
    setattr(args, attr, value)


def apply_yaml_defaults(args: argparse.Namespace) -> argparse.Namespace:
    configs = load_all_configs(args.config_dir)

    receiver_cfg = configs.get("receiver", {}) or {}
    sdr_cfg = receiver_cfg.get("sdr", {}) or {}
    ml_cfg = configs.get("ml", {}) or {}
    aoa_cfg = configs.get("aoa", {}) or {}
    ui_cfg = configs.get("ui", {}) or {}
    live_rf_viewer_cfg = ui_cfg.get("live_rf_viewer", {}) or {}
    aoa_gate_cfg = live_rf_viewer_cfg.get("aoa_gate", {}) or {}

    stft_cfg = ml_cfg.get("stft", {}) or {}
    spectrogram_cfg = ml_cfg.get("spectrogram", {}) or {}
    cnn_input_cfg = ml_cfg.get("cnn_input", {}) or {}
    inference_cfg = ml_cfg.get("inference", {}) or {}
    temporal_cfg = inference_cfg.get("temporal_voting", {}) or {}
    raw_safety_cfg = ml_cfg.get("raw_safety", {}) or {}
    overload_cfg = raw_safety_cfg.get("overload", {}) or {}

    decision_cfg = load_runtime_decision_config(ml_cfg)

    # Receiver / SDR
    _set_yaml_default(args, "uri", ("--uri",), sdr_cfg.get("uri"))
    _set_yaml_default(args, "center_freq", ("--center-freq",), sdr_cfg.get("center_freq", receiver_cfg.get("center_freq")))
    _set_yaml_default(args, "sample_rate", ("--sample-rate",), sdr_cfg.get("sample_rate", receiver_cfg.get("sample_rate")))
    _set_yaml_default(args, "rf_bandwidth", ("--rf-bandwidth",), sdr_cfg.get("rf_bandwidth", receiver_cfg.get("rf_bandwidth")))
    _set_yaml_default(args, "gain", ("--gain",), sdr_cfg.get("gain", receiver_cfg.get("gain")))
    _set_yaml_default(args, "block_size", ("--block-size",), sdr_cfg.get("block_size", receiver_cfg.get("block_size", ml_cfg.get("block_size"))))
    _set_yaml_default(args, "num_channels", ("--num-channels",), receiver_cfg.get("num_channels", len(sdr_cfg.get("channels", [0, 1]))))
    _set_yaml_default(args, "rx_index", ("--rx-index",), cnn_input_cfg.get("rx_index", 0))

    # STFT / spectrogram
    _set_yaml_default(args, "nperseg", ("--nperseg",), stft_cfg.get("nperseg"))
    _set_yaml_default(args, "noverlap", ("--noverlap",), stft_cfg.get("noverlap"))
    _set_yaml_default(args, "nfft", ("--nfft",), stft_cfg.get("nfft"))
    _set_yaml_default(args, "window", ("--window",), stft_cfg.get("window"))
    _set_yaml_default(
        args,
        "display_freq_bins",
        ("--display-freq-bins",),
        stft_cfg.get("expected_freq_bins", spectrogram_cfg.get("image_height")),
    )

    # CNN
    _set_yaml_default(args, "model", ("--model",), inference_cfg.get("model_path"))
    if not _flag_given("--cnn-backend"):
        args.cnn_backend = _viewer_backend(inference_cfg.get("backend", args.cnn_backend))
    else:
        args.cnn_backend = _viewer_backend(args.cnn_backend)

    _set_yaml_default(args, "cnn_device", ("--cnn-device",), inference_cfg.get("device"))
    _set_yaml_default(args, "class_names", ("--class-names",), _csv(ml_cfg.get("class_names", ["NotDrone", "Drone"])))
    _set_yaml_default(
        args,
        "cnn_positive_class_names",
        ("--cnn-positive-class-names",),
        str(ml_cfg.get("positive_class", temporal_cfg.get("positive_class", "Drone"))),
    )
    _set_yaml_default(
        args,
        "cnn_confidence_threshold",
        ("--cnn-confidence-threshold",),
        select_drone_threshold(decision_cfg, args.gain),
    )
    _set_yaml_default(args, "cnn_smooth_window", ("--cnn-smooth-window",), temporal_cfg.get("window_size"))
    _set_yaml_default(args, "cnn_confirm_votes", ("--cnn-confirm-votes",), temporal_cfg.get("confirmed_vote_k"))
    _set_yaml_default(args, "cnn_dummy_class_name", ("--cnn-dummy-class-name",), ml_cfg.get("negative_class", "NotDrone"))

    # AoA
    _set_yaml_default(args, "aoa_ref_channel", ("--aoa-ref-channel",), aoa_cfg.get("ref_channel"))
    _set_yaml_default(args, "aoa_target_channel", ("--aoa-target-channel",), aoa_cfg.get("target_channel"))
    _set_yaml_default(args, "aoa_antenna_spacing_m", ("--aoa-antenna-spacing-m",), aoa_cfg.get("antenna_spacing_m"))
    _set_yaml_default(args, "aoa_speed_of_light", ("--aoa-speed-of-light",), aoa_cfg.get("speed_of_light"))
    _set_yaml_default(args, "aoa_coherence_threshold", ("--aoa-coherence-threshold",), _nested_get(aoa_cfg, ["coherence", "threshold"]))
    if not _flag_given("--aoa-phase-offset-deg"):
        phase_offset_rad = aoa_cfg.get("phase_offset_rad")
        if phase_offset_rad is not None:
            args.aoa_phase_offset_deg = math.degrees(float(phase_offset_rad))

    # Calibration / raw safety
    _set_yaml_default(args, "overload_threshold", ("--overload-threshold",), overload_cfg.get("raw_peak_overload"))

    # Live RF viewer / OpenCV update policy
    # YAML is the default source; CLI arguments are temporary experiment overrides.
    _set_yaml_default(
        args,
        "blocks_per_update",
        ("--blocks-per-update",),
        live_rf_viewer_cfg.get("blocks_per_update"),
    )
    _set_yaml_default(
        args,
        "select_policy",
        ("--select-policy",),
        live_rf_viewer_cfg.get("select_policy"),
    )
    _set_yaml_default(
        args,
        "cli_log_every_n",
        ("--cli-log-every-n",),
        live_rf_viewer_cfg.get("cli_log_every_n"),
    )

    if not _flag_given("--disable-cli-log"):
        disable_cli_log = live_rf_viewer_cfg.get("disable_cli_log")
        if disable_cli_log is not None:
            args.disable_cli_log = bool(disable_cli_log)

    # AoA display gate.
    # detection_confirmed와 AoA 표시 조건을 분리한다.
    # confirmed=True가 유지 중이어도 현재 selected block이 NotDrone이면 AoA를 막는다.
    args.aoa_gate_enabled = bool(aoa_gate_cfg.get("enabled", True))
    args.aoa_require_voting_confirmed = bool(
        aoa_gate_cfg.get("require_voting_confirmed", True)
    )
    args.aoa_require_current_drone = bool(
        aoa_gate_cfg.get("require_current_drone", True)
    )
    args.aoa_min_current_confidence = float(
        aoa_gate_cfg.get("min_current_confidence", 0.90)
    )
    args.aoa_min_display_coherence = float(
        aoa_gate_cfg.get("min_coherence", 0.90)
    )
    args.aoa_display_only_valid = bool(
        aoa_gate_cfg.get("display_only_valid", True)
    )
    args.aoa_show_skip_reason = bool(
        aoa_gate_cfg.get("show_skip_reason", True)
    )

    # Normalize and validate values after YAML defaults are applied.
    args.blocks_per_update = max(1, int(args.blocks_per_update))
    args.cli_log_every_n = max(1, int(args.cli_log_every_n))

    allowed_select_policies = {
        "raw_gate_pass_score_max",
        "raw_gate_score_max",
        "raw_p99",
        "frame_power_p99",
        "last",
    }
    if args.select_policy not in allowed_select_policies:
        raise ValueError(
            f"Invalid live_rf_viewer.select_policy={args.select_policy!r}. "
            f"Allowed: {sorted(allowed_select_policies)}"
        )

    args._decision_cfg = decision_cfg
    args._ml_cfg = ml_cfg

    print("=== live_rf_viewer YAML config loaded ===")
    print(f"config_dir      : {args.config_dir}")
    print(f"receiver gain   : {args.gain}")
    print(f"center_freq     : {args.center_freq}")
    print(f"sample_rate     : {args.sample_rate}")
    print(f"block_size      : {args.block_size}")
    print(f"blocks_update   : {args.blocks_per_update}")
    print(f"select_policy   : {args.select_policy}")
    print(f"cli_log_every_n  : {args.cli_log_every_n}")
    print(f"disable_cli_log  : {args.disable_cli_log}")
    print(f"rx_index        : {args.rx_index}")
    print(f"stft            : nperseg={args.nperseg}, noverlap={args.noverlap}, nfft={args.nfft}")
    print(f"cnn_backend     : {args.cnn_backend}")
    print(f"cnn_model       : {args.model}")
    print(f"class_names     : {args.class_names}")
    print(f"positive_class  : {args.cnn_positive_class_names}")
    print(f"cnn_threshold   : {args.cnn_confidence_threshold}")
    print(f"cnn_voting      : window={args.cnn_smooth_window}, confirmed={args.cnn_confirm_votes}")
    print(
        "aoa_gate       : "
        f"enabled={args.aoa_gate_enabled} "
        f"cur_drone={args.aoa_require_current_drone} "
        f"min_conf={args.aoa_min_current_confidence} "
        f"min_coh={args.aoa_min_display_coherence}"
    )
    print(f"raw_overload_th : {args.overload_threshold}")
    print("=========================================")

    return args


def build_cnn_runtime(args: argparse.Namespace) -> CNNRuntime:
    return CNNRuntime(
        model_path=args.model,
        backend=_viewer_backend(args.cnn_backend),
        device=args.cnn_device,
        class_names=_parse_csv_values(args.class_names),
        positive_class_names=_parse_csv_values(args.cnn_positive_class_names),
        rx_index=int(args.rx_index),
        sample_rate=float(args.sample_rate),
        nperseg=int(args.nperseg),
        noverlap=int(args.noverlap),
        nfft=int(args.nfft),
        window=args.window,
        confidence_threshold=float(args.cnn_confidence_threshold),
        smooth_window=int(args.cnn_smooth_window),
        confirm_votes=int(args.cnn_confirm_votes),
        dummy_class_name=args.cnn_dummy_class_name,
        dummy_confidence=float(args.cnn_dummy_confidence),
    )




def _positive_class_set(args: argparse.Namespace) -> set[str]:
    return {
        item.strip().lower()
        for item in str(args.cnn_positive_class_names).split(",")
        if item.strip()
    }


def _is_current_cnn_drone(
    cnn_result: dict[str, Any] | None,
    args: argparse.Namespace,
) -> bool:
    if not cnn_result:
        return False

    raw_class = str(cnn_result.get("cnn_raw_class_name", "")).strip().lower()
    return raw_class in _positive_class_set(args)


def _current_cnn_confidence(cnn_result: dict[str, Any] | None) -> float:
    if not cnn_result:
        return 0.0

    try:
        return float(cnn_result.get("cnn_raw_confidence", 0.0))
    except Exception:
        return 0.0


def check_aoa_pre_gate(
    cnn_result: dict[str, Any] | None,
    raw_gate_blocks_aoa: bool,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """
    AoA 계산 전 gate.

    핵심:
    - cnn_confirmed=True는 최근 voting 상태다.
    - 현재 selected block의 raw CNN 결과가 NotDrone이면 AoA를 열지 않는다.
    - 그래서 NotDrone block이 이전 Drone voting 상태를 상속해서 AoA를 여는 문제를 막는다.
    """
    if not getattr(args, "aoa_gate_enabled", True):
        return True, "gate_disabled"

    if raw_gate_blocks_aoa:
        return False, "raw_noise_gate_failed"

    if getattr(args, "aoa_require_voting_confirmed", True):
        if not (cnn_result is not None and bool(cnn_result.get("cnn_confirmed", False))):
            return False, "cnn_not_confirmed"

    if getattr(args, "aoa_require_current_drone", True):
        if not _is_current_cnn_drone(cnn_result, args):
            return False, "current_not_drone"

    conf = _current_cnn_confidence(cnn_result)
    min_conf = float(getattr(args, "aoa_min_current_confidence", 0.90))
    if conf < min_conf:
        return False, f"low_current_conf:{conf:.3f}<{min_conf:.3f}"

    return True, "pass"


def apply_aoa_post_gate(
    aoa_result: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """
    AoA 계산 후 coherence gate.

    AoA는 계산해야 coherence를 알 수 있으므로,
    계산 후 coherence가 낮으면 valid=False 처리하고 화면 표시를 막는다.
    """
    if not aoa_result:
        return aoa_result

    coh_value = aoa_result.get(
        "stft_coherence",
        aoa_result.get("coherence_like", None),
    )
    if coh_value is None:
        return aoa_result

    try:
        coh = float(coh_value)
    except Exception:
        return aoa_result

    min_coh = float(getattr(args, "aoa_min_display_coherence", 0.90))
    if coh < min_coh:
        aoa_result["aoa_gated"] = True
        aoa_result["aoa_valid"] = False
        aoa_result["aoa_skipped_reason"] = f"low_coherence:{coh:.3f}<{min_coh:.3f}"

        if bool(getattr(args, "aoa_display_only_valid", True)):
            aoa_result["aoa_angle_deg"] = float("nan")

    return aoa_result

def build_aoa_runtime(args: argparse.Namespace) -> AoARuntime:
    if args.num_channels < 2:
        raise ValueError("AoA mode requires --num-channels 2")

    return AoARuntime(
        carrier_freq=float(args.center_freq),
        sample_rate=float(args.sample_rate),
        antenna_spacing_m=float(args.aoa_antenna_spacing_m),
        speed_of_light=float(args.aoa_speed_of_light),
        phase_calibration_json=args.aoa_phase_calibration_json,
        gain_phase_table_json=args.aoa_gain_phase_table,
        phase_gain_profile_json=args.phase_gain_profile,
        manual_phase_offset_deg=float(args.aoa_phase_offset_deg),
        ref_channel=int(args.aoa_ref_channel),
        target_channel=int(args.aoa_target_channel),
        coherence_threshold=float(args.aoa_coherence_threshold),
        energy_percentile=float(args.aoa_energy_percentile),
        nperseg=int(args.nperseg),
        noverlap=int(args.noverlap),
        nfft=int(args.nfft),
        window=args.window,
        compute_coherence=not bool(args.aoa_disable_stft_coherence),
        clip_angle_input=not bool(args.aoa_no_clip_angle_input),
        gain=float(args.gain),
    )


def apply_receiver_gain(receiver: PlutoReceiver, gain: float) -> None:
    receiver.gain = float(gain)
    for ch in receiver.channels:
        receiver._set_channel_gain(ch)  # PlutoReceiver exposes no public setter yet.


def select_iq_channel(iq: np.ndarray, rx_index: int) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        return arr.astype(np.complex64, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected IQ shape (channels, samples), got {arr.shape}")
    if rx_index < 0 or rx_index >= arr.shape[0]:
        raise IndexError(f"rx_index={rx_index} out of range for IQ shape {arr.shape}")
    return arr[rx_index].astype(np.complex64, copy=False)


def ensure_freq_time_spectrogram(
    spec: np.ndarray,
    expected_freq_bins: int = 128,
) -> np.ndarray:
    """
    Force spectrogram array layout to (frequency_bins, time_bins).

    Project convention:
        frequency axis = 128 bins
        time axis      = 509 bins or similar

    Therefore the display image should have:
        shape[0] = frequency axis
        shape[1] = time axis
    """
    arr = np.asarray(spec)

    if arr.ndim != 2:
        return arr

    expected_freq_bins = int(expected_freq_bins)

    # Already correct: (freq, time)
    if arr.shape[0] == expected_freq_bins:
        return arr

    # Wrong orientation: (time, freq)
    if arr.shape[1] == expected_freq_bins:
        return arr.T

    # Unknown shape. Do not guess.
    return arr


def compute_view_spectrogram(iq_1d: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    stft = compute_stft_branch(
        iq_block=iq_1d,
        sample_rate=float(args.sample_rate),
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        window=args.window,
    )
    spec = ensure_freq_time_spectrogram(
        stft.cnn_spectrogram,
        expected_freq_bins=args.display_freq_bins,
    )
    if getattr(args, "debug_shape", False):
        print("VIEW SPEC SHAPE:", "raw=", stft.cnn_spectrogram.shape, "display=", spec.shape)
    return spec


def _safe_float_for_selection(value: Any, default: float = float("-inf")) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def select_representative_block_index(
    raw_items: list[dict[str, Any]],
    raw_gate_results: list[Any],
    policy: str,
) -> int:
    """
    Select one representative block from a fast OpenCV update set.

    Main policy:
    - raw_gate_pass_score_max:
      1) If one or more blocks pass raw noise gate, choose the passed block
         with the largest raw gate score_max.
      2) If no block passes, choose the largest score_max anyway.
         The selected block will still fail raw gate later, so CNN/AoA stay blocked.

    This makes CNN voting operate per update-set instead of per raw block.
    """
    if not raw_items:
        raise ValueError("raw_items is empty")

    n = len(raw_items)
    policy = str(policy or "raw_gate_pass_score_max").strip()

    if policy == "last":
        return n - 1

    if policy == "raw_p99":
        return max(
            range(n),
            key=lambda i: _safe_float_for_selection(raw_items[i].get("raw_abs_p99")),
        )

    if policy == "frame_power_p99":
        return max(
            range(n),
            key=lambda i: _safe_float_for_selection(raw_items[i].get("frame_power_p99")),
        )

    if policy == "raw_gate_score_max":
        return max(
            range(n),
            key=lambda i: _safe_float_for_selection(raw_gate_results[i].score_max),
        )

    if policy == "raw_gate_pass_score_max":
        passed_indices = [
            i
            for i, result in enumerate(raw_gate_results)
            if bool((not result.enabled) or result.passed)
        ]
        if passed_indices:
            return max(
                passed_indices,
                key=lambda i: _safe_float_for_selection(raw_gate_results[i].score_max),
            )
        return max(
            range(n),
            key=lambda i: _safe_float_for_selection(raw_gate_results[i].score_max),
        )

    raise ValueError(f"Unknown select_policy: {policy}")


def format_calibration_status(
    calibration_noise_result: Any | None = None,
    calibration_phase_result: Any | None = None,
) -> str | None:
    """
    OpenCV overlay에 표시할 calibration 상태 한 줄을 만든다.
    """
    if calibration_noise_result is None and calibration_phase_result is None:
        return None

    parts: list[str] = []

    if calibration_noise_result is not None:
        parts.append(
            "CAL noise="
            f"{calibration_noise_result.matched_gain:g}/"
            f"{calibration_noise_result.matched_by} "
            f"thr={calibration_noise_result.threshold:.4g} "
            f"raw={calibration_noise_result.raw_safety_status} "
            f"sat={calibration_noise_result.raw_saturation_ratio * 100:.3f}%"
        )

    if calibration_phase_result is not None:
        parts.append(
            "phase_gain="
            f"{calibration_phase_result.matched_gain:g}/"
            f"{calibration_phase_result.matched_by} "
            f"corr={calibration_phase_result.gain_correction:.4g} "
            f"phase={calibration_phase_result.phase_offset_deg:.2f}deg "
            f"quality={calibration_phase_result.quality}"
        )

    return " | ".join(parts)


def _fmt_profile_value(value: Any, digits: int = 3) -> str:
    try:
        if value is None or value == "":
            return "n/a"
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def format_profile_summary_lines(
    profile_summaries: list[dict[str, Any]] | None,
    max_rows: int = 8,
) -> list[str]:
    """
    OpenCV 오른쪽 패널에 표시할 gain별 profile 대표 피쳐값을 만든다.

    대표값 기준:
    - raw_abs_p99_median      : gain별 대표 수신 세기
    - raw_rms_median          : 전체 에너지 대표값
    - frame_power_p99_median  : spectrogram/CNN 입력 쪽 세기 대표값
    - raw_abs_max_max         : 포화/이상치 확인용
    """
    if not profile_summaries:
        return []

    rows = list(profile_summaries)[-int(max_rows):]

    lines: list[str] = []
    lines.append("PROFILE SUMMARY median")
    lines.append("gain | dist | raw_p99 | rms | frame_p99 | max")

    for item in rows:
        gain = _fmt_profile_value(item.get("gain"), digits=4)
        dist = _fmt_profile_value(item.get("distance_m"), digits=3)
        raw_p99 = _fmt_profile_value(item.get("raw_abs_p99_median"), digits=4)
        rms = _fmt_profile_value(item.get("raw_rms_median"), digits=4)
        frame_p99 = _fmt_profile_value(item.get("frame_power_p99_median"), digits=4)
        raw_max = _fmt_profile_value(item.get("raw_abs_max_max"), digits=4)

        lines.append(
            f"G{gain} D{dist} p99={raw_p99} rms={rms} fp99={frame_p99} max={raw_max}"
        )

    latest = rows[-1]
    memo = str(latest.get("memo", "")).strip()
    if memo:
        lines.append(f"latest memo={memo}")

    return lines

def build_overlay(
    state: ViewerState,
    raw: dict[str, Any],
    profile_status: str | None = None,
    profile_summaries: list[dict[str, Any]] | None = None,
    aoa_result: dict[str, Any] | None = None,
    cnn_result: dict[str, Any] | None = None,
    log_status: str | None = None,
    calibration_status: str | None = None,
    raw_gate_status: str | None = None,
) -> list[str]:
    """
    Compact OpenCV side overlay.

    OpenCV right panel is not a full log view.
    Keep only field-critical information so AOA never falls below the visible area.
    Detailed values remain in CLI/CSV.
    """

    def short(text: Any, max_len: int = 58) -> str:
        value = str(text)
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."

    live_state = "PAUSED" if state.paused else "LIVE"

    lines: list[str] = [
        f"[RF] idx={state.update_index} {live_state} {state.mode}",
        f"cf={state.center_freq / 1e6:.3f}M sr={state.sample_rate / 1e6:.3f}M g={state.gain:.1f}",
        f"[RAW] sel={raw.get('selected_block_index', 0)}/{raw.get('blocks_per_update', 1)} p99={_fmt_cli_value(raw.get('raw_abs_p99'), 4)} rms={_fmt_cli_value(raw.get('raw_rms'), 4)}",
        f"max={_fmt_cli_value(raw.get('raw_abs_max'), 4)} fp99={_fmt_cli_value(raw.get('frame_power_p99'), 4)} ov={bool(raw.get('overloaded', False))}",
    ]

    if bool(raw.get("overloaded", False)):
        lines.append("[WARN] OVERLOAD -> TRY_LOWER_GAIN")

    # AOA must stay near the top.
    if aoa_result:
        skip_reason = aoa_result.get("aoa_skipped_reason", "")
        if skip_reason:
            lines.append(f"[AOA] skip={short(skip_reason, 48)}")
        else:
            angle = aoa_result.get("aoa_angle_deg", float("nan"))
            coh = aoa_result.get(
                "stft_coherence",
                aoa_result.get("coherence_like", float("nan")),
            )
            valid = bool(aoa_result.get("aoa_valid", False))
            phase = aoa_result.get("phase_diff_corrected_deg", float("nan"))
            lines.append(
                f"[AOA] { _fmt_cli_value(angle, 3)}deg "
                f"valid={valid} coh={_fmt_cli_value(coh, 3)}"
            )
            lines.append(f"phase={_fmt_cli_value(phase, 3)}deg")
    else:
        lines.append("[AOA] n/a")

    if cnn_result:
        lines.append(
            f"[CNN] {cnn_result.get('cnn_raw_class_name', 'Unknown')} "
            f"conf={_fmt_cli_value(cnn_result.get('cnn_raw_confidence'), 3)}"
        )
        lines.append(
            f"[VOTE] {cnn_result.get('cnn_smoothed_class_name', 'Unknown')} "
            f"{int(cnn_result.get('cnn_positive_votes', 0))}/"
            f"{int(cnn_result.get('cnn_confirm_votes', 0))} "
            f"ok={bool(cnn_result.get('cnn_confirmed', False))}"
        )

    if raw_gate_status:
        lines.append(f"[GATE] {short(raw_gate_status, 60)}")

    if calibration_status:
        lines.append(f"[CAL] {short(calibration_status, 60)}")

    if state.distance_m > 0:
        lines.append(f"[META] d={state.distance_m:.2f}m")
    if state.memo:
        lines.append(f"memo={short(state.memo, 50)}")

    # Do not dump profile summaries on OpenCV; they push AOA/CNN out of view.
    # Keep only a compact profile status line.
    if profile_status:
        lines.append(f"[PROFILE] {short(profile_status, 54)}")

    lines.append("q quit | p pause | [/] gain | s profile")

    # Hard cap for OpenCV readability.
    # CLI/CSV still preserve full details.
    max_lines = 13
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["... see CLI/CSV for details"]

    return lines



def build_live_log_row(
    state: ViewerState,
    args: argparse.Namespace,
    raw: dict[str, Any],
    cnn_result: dict[str, Any] | None = None,
    aoa_result: dict[str, Any] | None = None,
    profile_status: str | None = None,
    raw_gate_result: Any | None = None,
    calibration_noise_result: Any | None = None,
    calibration_phase_result: Any | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": state.mode,
        "update_index": int(state.update_index),
        "paused": bool(state.paused),
        "center_freq": int(state.center_freq),
        "sample_rate": int(state.sample_rate),
        "gain": float(state.gain),
        "distance_m": float(state.distance_m),
        "memo": str(state.memo),
        "rx_index": int(args.rx_index),
        "target_fps": float(state.target_fps),
        "profile_status": profile_status or "",
    }
    row.update(raw)
    if cnn_result:
        row.update(cnn_result)

    if aoa_result:
        row.update(aoa_result)

    if raw_gate_result is not None:
        row.update(
            {
                "raw_gate_enabled": raw_gate_result.enabled,
                "raw_gate_passed": raw_gate_result.passed,
                "raw_gate_label": raw_gate_result.label,
                "raw_gate_gain": raw_gate_result.gain,
                "raw_gate_matched_gain": raw_gate_result.matched_gain,
                "raw_gate_matched_by": raw_gate_result.matched_by,
                "raw_gate_detector_method": raw_gate_result.detector_method,
                "raw_gate_frame_size": raw_gate_result.frame_size,
                "raw_gate_hop_size": raw_gate_result.hop_size,
                "raw_gate_noise_floor": raw_gate_result.noise_floor,
                "raw_gate_threshold_multiplier": raw_gate_result.threshold_multiplier,
                "raw_gate_threshold": raw_gate_result.threshold,
                "raw_gate_detection_ratio": raw_gate_result.detection_ratio,
                "raw_gate_min_detection_ratio": raw_gate_result.min_detection_ratio,
                "raw_gate_score_max": raw_gate_result.score_max,
                "raw_gate_score_median": raw_gate_result.score_median,
                "raw_gate_reason": raw_gate_result.reason,
            }
        )

    if calibration_noise_result is not None:
        row.update(
            {
                "calib_noise_threshold": calibration_noise_result.threshold,
                "calib_noise_matched_gain": calibration_noise_result.matched_gain,
                "calib_noise_matched_by": calibration_noise_result.matched_by,
                "calib_profile_safety_status": calibration_noise_result.profile_safety_status,
                "raw_safety_status": calibration_noise_result.raw_safety_status,
                "raw_safety_is_safe": calibration_noise_result.raw_is_safe,
                "raw_safety_max_abs": calibration_noise_result.raw_max_abs,
                "raw_safety_rms": calibration_noise_result.raw_rms,
                "raw_safety_dc_abs": calibration_noise_result.raw_dc_abs,
                "raw_safety_saturation_ratio": calibration_noise_result.raw_saturation_ratio,
                "raw_safety_near_saturation_ratio": calibration_noise_result.raw_near_saturation_ratio,
            }
        )

    if calibration_phase_result is not None:
        row.update(
            {
                "calib_phase_gain_matched_gain": calibration_phase_result.matched_gain,
                "calib_phase_gain_matched_by": calibration_phase_result.matched_by,
                "calib_gain_correction": calibration_phase_result.gain_correction,
                "calib_phase_offset_rad": calibration_phase_result.phase_offset_rad,
                "calib_phase_offset_deg": calibration_phase_result.phase_offset_deg,
                "calib_phase_gain_quality": calibration_phase_result.quality,
            }
        )

    return row


def should_log_live_row(
    args: argparse.Namespace,
    state: ViewerState,
    read_new_block: bool,
) -> bool:
    if args.disable_log:
        return False
    if args.mode != "full":
        return False
    if not read_new_block:
        return False
    return int(state.update_index) % int(args.log_every_n) == 0


def should_print_cli_log(
    args: argparse.Namespace,
    state: ViewerState,
    read_new_block: bool,
) -> bool:
    if args.disable_cli_log:
        return False
    if not read_new_block:
        return False
    return int(state.update_index) % int(args.cli_log_every_n) == 0


def _fmt_cli_value(value: Any, digits: int = 3) -> str:
    try:
        if value is None or value == "":
            return "n/a"
        v = float(value)
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}g}"
    except Exception:
        return str(value)


def format_cli_status_line(
    state: ViewerState,
    raw: dict[str, Any],
    cnn_result: dict[str, Any] | None = None,
    aoa_result: dict[str, Any] | None = None,
    raw_gate_result: Any | None = None,
    calibration_noise_result: Any | None = None,
) -> str:
    """
    Grouped CLI status output.

    OpenCV overlay is compact.
    CLI keeps detailed grouped values for debugging.
    """
    lines: list[str] = []

    lines.append(
        "[RF] "
        f"idx={state.update_index} "
        f"mode={state.mode} "
        f"gain={state.gain:.1f} "
        f"cf={state.center_freq / 1e6:.3f}MHz "
        f"sr={state.sample_rate / 1e6:.3f}MS/s"
    )

    lines.append(
        "[RAW] "
        f"sel={raw.get('selected_block_index', 0)}/{raw.get('blocks_per_update', 1)} "
        f"policy={raw.get('select_policy', 'n/a')} "
        f"raw_p99={_fmt_cli_value(raw.get('raw_abs_p99'), 4)} "
        f"rms={_fmt_cli_value(raw.get('raw_rms'), 4)} "
        f"max={_fmt_cli_value(raw.get('raw_abs_max'), 4)} "
        f"fp99={_fmt_cli_value(raw.get('frame_power_p99'), 4)} "
        f"overload={bool(raw.get('overloaded', False))}"
    )

    if bool(raw.get("overloaded", False)):
        lines.append("[WARNING] status=OVERLOAD suggestion=TRY_LOWER_GAIN")

    if raw_gate_result is not None:
        lines.append(
            "[RAW_GATE] "
            f"{raw_gate_result.label} "
            f"pass={bool(raw_gate_result.passed)} "
            f"score={_fmt_cli_value(raw_gate_result.score_max, 4)} "
            f"noise={_fmt_cli_value(raw_gate_result.noise_floor, 4)} "
            f"thr={_fmt_cli_value(raw_gate_result.threshold, 4)} "
            f"x{_fmt_cli_value(raw_gate_result.threshold_multiplier, 3)} "
            f"ratio={_fmt_cli_value(raw_gate_result.detection_ratio, 3)}/"
            f"{_fmt_cli_value(raw_gate_result.min_detection_ratio, 3)} "
            f"match={raw_gate_result.matched_gain}({raw_gate_result.matched_by})"
        )

    if calibration_noise_result is not None:
        lines.append(
            "[CAL] "
            f"thr={_fmt_cli_value(calibration_noise_result.threshold, 4)} "
            f"raw={calibration_noise_result.raw_safety_status} "
            f"sat={_fmt_cli_value(calibration_noise_result.raw_saturation_ratio * 100, 4)}%"
        )

    if cnn_result:
        lines.append(
            "[CNN] "
            f"raw={cnn_result.get('cnn_raw_class_name', 'Unknown')} "
            f"conf={_fmt_cli_value(cnn_result.get('cnn_raw_confidence'), 3)}"
        )
        lines.append(
            "[CNN_VOTE] "
            f"smooth={cnn_result.get('cnn_smoothed_class_name', 'Unknown')} "
            f"avg={_fmt_cli_value(cnn_result.get('cnn_smoothed_confidence'), 3)} "
            f"votes={int(cnn_result.get('cnn_positive_votes', 0))}/"
            f"{int(cnn_result.get('cnn_confirm_votes', 0))} "
            f"confirmed={bool(cnn_result.get('cnn_confirmed', False))}"
        )

    if aoa_result:
        skip_reason = aoa_result.get("aoa_skipped_reason", "")
        if skip_reason:
            lines.append(f"[AOA] skip={skip_reason}")
        else:
            lines.append(
                "[AOA] "
                f"angle={_fmt_cli_value(aoa_result.get('aoa_angle_deg'), 3)}deg "
                f"valid={bool(aoa_result.get('aoa_valid', False))} "
                f"coh={_fmt_cli_value(aoa_result.get('stft_coherence', aoa_result.get('coherence_like')), 3)} "
                f"phase={_fmt_cli_value(aoa_result.get('phase_diff_corrected_deg'), 3)}deg"
            )

    return "\n".join(lines)



def handle_key(
    key: str | None,
    state: ViewerState,
    receiver: PlutoReceiver,
    profile: GainProfileRuntime | None,
    aoa_runtime: AoARuntime | None,
    cnn_runtime: CNNRuntime | None,
    args: argparse.Namespace,
) -> None:
    if key is None:
        return
    if key == "quit":
        state.running = False
        return
    if key == "pause":
        state.toggle_pause()
        return
    if key == "gain_down":
        state.step_gain(-args.gain_step, min_gain=args.min_gain, max_gain=args.max_gain)
        apply_receiver_gain(receiver, state.gain)
        if aoa_runtime is not None:
            aoa_runtime.update_gain(state.gain)
        if cnn_runtime is not None:
            cnn_runtime.reset_history()
        return
    if key == "gain_up":
        state.step_gain(args.gain_step, min_gain=args.min_gain, max_gain=args.max_gain)
        apply_receiver_gain(receiver, state.gain)
        if aoa_runtime is not None:
            aoa_runtime.update_gain(state.gain)
        if cnn_runtime is not None:
            cnn_runtime.reset_history()
        return
    if key == "save_profile" and profile is not None:
        profile.request_capture(
            gain=state.gain,
            distance_m=state.distance_m,
            memo=state.memo,
        )


def run() -> int:
    args = apply_yaml_defaults(parse_args())
    state = ViewerState(
        mode=args.mode,
        gain=args.gain,
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        distance_m=args.distance_m,
        memo=args.memo,
        target_fps=args.target_fps,
    )

    profile = None
    if args.mode in ("profile", "full"):
        profile = GainProfileRuntime(
            blocks=args.profile_blocks,
            csv_path=Path(args.profile_csv),
            json_path=Path(args.profile_json),
        )

    aoa_runtime = None
    if args.mode in ("aoa", "full"):
        aoa_runtime = build_aoa_runtime(args)

    cnn_runtime = None
    if args.mode in ("cnn", "full"):
        cnn_runtime = build_cnn_runtime(args)

    raw_gate = RawNoiseGate(
        detect_config_path=Path(args.config_dir) / "detect.yaml",
    )
    print("raw_noise_gate :", "enabled" if raw_gate.enabled else "disabled")
    if raw_gate.enabled:
        print(f"raw_gate_profile: {raw_gate.noise_profile_path}")
        print(
            "raw_gate_frame  : "
            f"method={raw_gate.detector_method}, "
            f"frame_size={raw_gate.frame_size}, hop_size={raw_gate.hop_size}"
        )

    calibration_runtime = None
    if not args.disable_calibration_runtime:
        calibration_runtime = load_calibration_runtime(
            noise_profile_path=args.noise_profile,
            phase_gain_profile_path=args.phase_gain_profile,
            allow_nearest=True,
            full_scale=2048.0,
        )

    renderer = OpenCVRenderer(
        window_name=f"RF Viewer Drone-AoA - {args.mode}",
        target_fps=args.target_fps,
        display_scale=args.display_scale,
        display_width=args.display_width,
        display_height=args.display_height,
        auto_orient=False,
        overlay_mode=args.overlay_mode,
        overlay_width=args.overlay_panel_width,
    )

    last_iq: np.ndarray | None = None
    last_selected_idx = 0
    last_blocks_per_update = 1
    recent_profile_summaries: list[dict[str, Any]] = []
    receiver = build_receiver(args)
    try:
        while state.running:
            read_new_block = False
            selected_idx = last_selected_idx
            blocks_this_update = last_blocks_per_update

            if not state.paused or last_iq is None:
                candidate_blocks: list[np.ndarray] = []
                candidate_raw_items: list[dict[str, Any]] = []
                candidate_raw_gate_results: list[Any] = []

                blocks_this_update = max(1, int(args.blocks_per_update))
                for _block_read_idx in range(blocks_this_update):
                    block_iq = receiver.read_block(args.block_size)
                    if not args.disable_dc_offset_removal:
                        block_iq = remove_dc_offset(block_iq, axis=-1)

                    block_raw = compute_raw_features(
                        block_iq,
                        overload_abs_threshold=args.overload_threshold,
                    )
                    block_raw_gate_result = raw_gate.evaluate(
                        block_iq,
                        gain=state.gain,
                    )

                    candidate_blocks.append(block_iq)
                    candidate_raw_items.append(block_raw)
                    candidate_raw_gate_results.append(block_raw_gate_result)

                selected_idx = select_representative_block_index(
                    raw_items=candidate_raw_items,
                    raw_gate_results=candidate_raw_gate_results,
                    policy=args.select_policy,
                )

                last_iq = candidate_blocks[selected_idx]
                raw = dict(candidate_raw_items[selected_idx])
                raw_gate_result = candidate_raw_gate_results[selected_idx]

                last_selected_idx = int(selected_idx)
                last_blocks_per_update = int(blocks_this_update)

                state.mark_update()
                read_new_block = True
            else:
                raw = compute_raw_features(
                    last_iq,
                    overload_abs_threshold=args.overload_threshold,
                )
                raw_gate_result = raw_gate.evaluate(last_iq, gain=state.gain)

            raw.update(
                {
                    "selected_block_index": int(selected_idx),
                    "blocks_per_update": int(blocks_this_update),
                    "select_policy": str(args.select_policy),
                }
            )

            calibration_noise_result = None
            calibration_phase_result = None
            calibration_status = None

            if calibration_runtime is not None:
                calibration_noise_result = calibration_runtime.check_noise(
                    last_iq,
                    gain=state.gain,
                )
                calibration_phase_result = calibration_runtime.get_phase_gain(
                    gain=state.gain,
                )
                calibration_status = format_calibration_status(
                    calibration_noise_result=calibration_noise_result,
                    calibration_phase_result=calibration_phase_result,
                )

            raw_gate_passed = bool(
                (not raw_gate_result.enabled) or raw_gate_result.passed
            )
            raw_gate_allows_cnn = bool(
                raw_gate_passed or not raw_gate.block_cnn_on_fail()
            )
            raw_gate_blocks_aoa = bool(
                raw_gate_result.enabled
                and (not raw_gate_result.passed)
                and raw_gate.block_aoa_on_fail()
            )
            raw_gate_status = raw_gate.status_text(raw_gate_result)

            cnn_result = None
            if cnn_runtime is not None and raw_gate_allows_cnn:
                # Gain-aware Drone threshold from configs/ml.yaml.
                if hasattr(args, "_decision_cfg"):
                    cnn_runtime.confidence_threshold = select_drone_threshold(
                        args._decision_cfg,
                        state.gain,
                    )
                image, cnn_result = cnn_runtime.process(last_iq)
            elif cnn_runtime is not None:
                if raw_gate.reset_cnn_history_on_fail():
                    cnn_runtime.reset_history()

                iq_1d = select_iq_channel(last_iq, args.rx_index)
                image = compute_view_spectrogram(iq_1d, args)

                cnn_result = {
                    "cnn_raw_class_name": "RAW_GATE_BLOCKED",
                    "cnn_raw_confidence": 0.0,
                    "cnn_smoothed_class_name": "RAW_GATE_BLOCKED",
                    "cnn_smoothed_confidence": 0.0,
                    "cnn_positive_votes": 0,
                    "cnn_confirm_votes": int(args.cnn_confirm_votes),
                    "cnn_confirmed": False,
                    "cnn_skipped": True,
                    "cnn_skipped_reason": "raw_noise_gate_failed",
                }
            else:
                iq_1d = select_iq_channel(last_iq, args.rx_index)
                image = compute_view_spectrogram(iq_1d, args)

            profile_status = None
            if profile is not None:
                summary = profile.update(
                    {
                        "update_index": state.update_index,
                        "center_freq": state.center_freq,
                        "sample_rate": state.sample_rate,
                        "rx_index": args.rx_index,
                        **raw,
                    }
                )
                profile_status = profile.status_text()
                if summary is not None:
                    recent_profile_summaries.append(summary)
                    del recent_profile_summaries[:-8]
                    profile_status = f"PROFILE saved {summary.get('captured_blocks')} blocks"

            # Drone-gated AoA:
            # detection confirmed와 AoA 표시 조건을 분리한다.
            # AoA는 다음 조건을 모두 만족할 때만 계산/표시한다.
            #   1) CNN temporal voting confirmed
            #   2) 현재 selected block의 raw CNN 결과도 Drone
            #   3) 현재 CNN confidence가 YAML 기준 이상
            #   4) raw gate가 AoA를 막지 않음
            # 계산 후 coherence가 낮으면 angle 표시를 무효화한다.
            aoa_result = None
            if aoa_runtime is not None:
                aoa_pre_ok, aoa_skip_reason = check_aoa_pre_gate(
                    cnn_result=cnn_result,
                    raw_gate_blocks_aoa=raw_gate_blocks_aoa,
                    args=args,
                )

                if aoa_pre_ok:
                    aoa_result = aoa_runtime.process(last_iq)
                    aoa_result = apply_aoa_post_gate(aoa_result, args)
                else:
                    aoa_result = {
                        "aoa_gated": True,
                        "aoa_skipped_reason": aoa_skip_reason,
                        "aoa_valid": False,
                    }

            log_status = None
            if args.mode == "full" and not args.disable_log:
                log_status = f"LOG every={args.log_every_n} -> {args.log_csv}"
            if should_log_live_row(args, state, read_new_block):
                append_viewer_csv(
                    args.log_csv,
                    build_live_log_row(
                        state=state,
                        args=args,
                        raw=raw,
                        cnn_result=cnn_result,
                        aoa_result=aoa_result,
                        profile_status=profile_status,
                        raw_gate_result=raw_gate_result,
                        calibration_noise_result=calibration_noise_result,
                        calibration_phase_result=calibration_phase_result,
                    ),
                )

            if should_print_cli_log(args, state, read_new_block):
                print(
                    format_cli_status_line(
                        state=state,
                        raw=raw,
                        cnn_result=cnn_result,
                        aoa_result=aoa_result,
                        raw_gate_result=raw_gate_result,
                        calibration_noise_result=calibration_noise_result,
                    ),
                    flush=True,
                )

            overlay = build_overlay(
                state=state,
                raw=raw,
                profile_status=profile_status,
                profile_summaries=recent_profile_summaries,
                aoa_result=aoa_result,
                cnn_result=cnn_result,
                log_status=log_status,
                calibration_status=calibration_status,
                raw_gate_status=raw_gate_status,
            )
            key = renderer.render(image, overlay)
            handle_key(key, state, receiver, profile, aoa_runtime, cnn_runtime, args)

    except KeyboardInterrupt:
        return 130
    finally:
        receiver.close()
        renderer.close()

    return 0
    


if __name__ == "__main__":
    sys.exit(run())
