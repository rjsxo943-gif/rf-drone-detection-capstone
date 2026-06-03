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

    args._decision_cfg = decision_cfg
    args._ml_cfg = ml_cfg

    print("=== live_rf_viewer YAML config loaded ===")
    print(f"config_dir      : {args.config_dir}")
    print(f"receiver gain   : {args.gain}")
    print(f"center_freq     : {args.center_freq}")
    print(f"sample_rate     : {args.sample_rate}")
    print(f"block_size      : {args.block_size}")
    print(f"rx_index        : {args.rx_index}")
    print(f"stft            : nperseg={args.nperseg}, noverlap={args.noverlap}, nfft={args.nfft}")
    print(f"cnn_backend     : {args.cnn_backend}")
    print(f"cnn_model       : {args.model}")
    print(f"class_names     : {args.class_names}")
    print(f"positive_class  : {args.cnn_positive_class_names}")
    print(f"cnn_threshold   : {args.cnn_confidence_threshold}")
    print(f"cnn_voting      : window={args.cnn_smooth_window}, confirmed={args.cnn_confirm_votes}")
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
) -> list[str]:
    lines = [
        f"mode={state.mode} idx={state.update_index} {'PAUSED' if state.paused else 'LIVE'}",
        f"cf={state.center_freq / 1e6:.3f} MHz sr={state.sample_rate / 1e6:.3f} MS/s gain={state.gain:.1f} dB",
        f"raw_p99={raw.get('raw_abs_p99', 0.0):.4g} rms={raw.get('raw_rms', 0.0):.4g} max={raw.get('raw_abs_max', 0.0):.4g}",
        f"frame_power_p99={raw.get('frame_power_p99', 0.0):.4g} overload={raw.get('overloaded', False)}",
        "keys: q quit | p pause | [/] gain | s profile",
    ]
    if state.distance_m > 0:
        lines.append(f"distance={state.distance_m:.2f} m")
    if state.memo:
        lines.append(f"memo={state.memo}")
    if profile_status:
        lines.append(profile_status)
    if profile_summaries:
        lines.extend(format_profile_summary_lines(profile_summaries))
    if log_status:
        lines.append(log_status)
    if calibration_status:
        lines.append(calibration_status)
    if cnn_result:
        raw_class = cnn_result.get("cnn_raw_class_name", "Unknown")
        raw_conf = float(cnn_result.get("cnn_raw_confidence", 0.0))
        smooth_class = cnn_result.get("cnn_smoothed_class_name", "Unknown")
        smooth_conf = float(cnn_result.get("cnn_smoothed_confidence", 0.0))
        votes = int(cnn_result.get("cnn_positive_votes", 0))
        needed = int(cnn_result.get("cnn_confirm_votes", 0))
        confirmed = bool(cnn_result.get("cnn_confirmed", False))
        lines.append(
            f"CNN raw={raw_class} conf={raw_conf:.3f} "
            f"smooth={smooth_class} avg={smooth_conf:.3f}"
        )
        lines.append(f"CNN candidate_votes={votes}/{needed} confirmed={confirmed}")
    if aoa_result:
        angle = aoa_result.get("aoa_angle_deg", float("nan"))
        phase = aoa_result.get("phase_diff_corrected_deg", float("nan"))
        coh = aoa_result.get("stft_coherence", aoa_result.get("coherence_like", float("nan")))
        passed = aoa_result.get("stft_coherence_passed", "n/a")
        offset = aoa_result.get("phase_offset_to_apply_deg", 0.0)
        lines.append(
            f"AOA angle={angle:.2f} deg phase={phase:.2f} deg "
            f"coh={coh:.3f} pass={passed}"
        )
        lines.append(f"AOA offset={offset:.2f} deg valid={aoa_result.get('aoa_valid', False)}")
    return lines



def build_live_log_row(
    state: ViewerState,
    args: argparse.Namespace,
    raw: dict[str, Any],
    cnn_result: dict[str, Any] | None = None,
    aoa_result: dict[str, Any] | None = None,
    profile_status: str | None = None,
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

    calibration_runtime = None
    if not args.disable_calibration_runtime:
        calibration_runtime = load_calibration_runtime(
            noise_profile_path=args.noise_profile,
            phase_gain_profile_path=args.phase_gain_profile,
            allow_nearest=True,
            full_scale=2048.0,
        )

    renderer = OpenCVRenderer(
        window_name=f"RF Viewer - {args.mode}",
        target_fps=args.target_fps,
        display_scale=args.display_scale,
        display_width=args.display_width,
        display_height=args.display_height,
        auto_orient=False,
        overlay_mode=args.overlay_mode,
        overlay_width=args.overlay_panel_width,
    )

    last_iq: np.ndarray | None = None
    recent_profile_summaries: list[dict[str, Any]] = []
    receiver = build_receiver(args)
    try:
        while state.running:
            read_new_block = False
            if not state.paused or last_iq is None:
                last_iq = receiver.read_block(args.block_size)
                if not args.disable_dc_offset_removal:
                    last_iq = remove_dc_offset(last_iq, axis=-1)
                state.mark_update()
                read_new_block = True

            raw = compute_raw_features(
                last_iq,
                overload_abs_threshold=args.overload_threshold,
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

            cnn_result = None
            if cnn_runtime is not None:
                # Gain-aware Drone threshold from configs/ml.yaml.
                if hasattr(args, "_decision_cfg"):
                    cnn_runtime.confidence_threshold = select_drone_threshold(
                        args._decision_cfg,
                        state.gain,
                    )
                image, cnn_result = cnn_runtime.process(last_iq)
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

            aoa_result = None
            if aoa_runtime is not None:
                aoa_result = aoa_runtime.process(last_iq)

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
                        calibration_noise_result=calibration_noise_result,
                        calibration_phase_result=calibration_phase_result,
                    ),
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
