#!/usr/bin/env python3
"""
Live CNN Spectrogram Viewer v1.

Purpose:
- Read IQ blocks from the configured receiver.
- Compute raw IQ features before normalization.
- Select one representative block per update.
- Convert only the selected block to the same CNN-style spectrogram image.
- Display/save the spectrogram and append feature logs to CSV.

Notes:
- This script does not run CNN inference.
- This script does not compute AoA.
- This script does not scan frequencies.
- YAML is loaded only once at startup. Restart the script to apply config changes.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from matplotlib.widgets import Button, TextBox
from scripts.ml.train_rf_binary_cnn import SmallRFBinaryCNN

from src.core import get_project_root, load_all_configs
from src.preprocess import get_cnn_input_iq, normalize_iq, remove_dc_offset
from src.receiver import build_receiver


EPS = 1e-12


@dataclass(frozen=True)
class RawFeatures:
    raw_rms: float
    raw_peak: float
    raw_abs_mean: float
    raw_abs_median: float
    raw_abs_p95: float
    raw_abs_p99: float
    clip_ratio: float
    frame_power_median: float
    frame_power_p95: float
    frame_power_p99: float
    noise_floor: float
    signal_ratio: float


@dataclass(frozen=True)
class CnnSpecFeatures:
    cnn_spec_mean: float
    cnn_spec_std: float
    cnn_spec_min: float
    cnn_spec_p50: float
    cnn_spec_p95: float
    cnn_spec_p99: float
    cnn_spec_max: float


@dataclass(frozen=True)
class ViewerThresholds:
    no_signal_ratio: float
    valid_signal_ratio: float
    overload_peak: float
    overload_clip_ratio: float


CSV_COLUMNS = [
    "timestamp",
    "session_id",
    "update_index",
    "center_freq",
    "sample_rate",
    "rf_bandwidth",
    "block_size",
    "rx_index",
    "gain",
    "distance_m",
    "memo",
    "blocks_per_update",
    "selected_block_index",
    "select_policy",
    "status",
    "suggestion",
    "raw_rms",
    "raw_peak",
    "raw_abs_mean",
    "raw_abs_median",
    "raw_abs_p95",
    "raw_abs_p99",
    "clip_ratio",
    "frame_power_median",
    "frame_power_p95",
    "frame_power_p99",
    "noise_floor",
    "signal_ratio",
    "cnn_spec_mean",
    "cnn_spec_std",
    "cnn_spec_min",
    "cnn_spec_p50",
    "cnn_spec_p95",
    "cnn_spec_p99",
    "cnn_spec_max",
    "decision_mode",
    "cnn_model_path",
    "cnn_prob_drone",
    "cnn_threshold",
    "cnn_raw_decision",
    "temporal_window",
    "candidate_vote_k",
    "confirmed_vote_k",
    "temporal_history",
    "candidate_status",
    "confirmed_status",
    "final_decision",
    "latency_sec",
    "processing_time_sec",
    "target_raw_abs_p99",
    "target_frame_power_p99",
    "target_raw_rms",
    "feature_match_status",
    "feature_match_max_error_pct",
    "feature_match_mean_error_pct",
    "feature_match_tolerance_pct",
    "aoa_enabled",
    "aoa_status",
    "aoa_deg",
    "aoa_phase_diff_rad",
    "aoa_phase_diff_raw_rad",
    "aoa_phase_offset_rad",
    "aoa_coherence",
    "aoa_antenna_spacing_m",
    "aoa_calibration_deg",
    "aoa_deg_smooth",
    "aoa_smooth_status",
    "aoa_smooth_count",
    "aoa_smooth_window",
    "aoa_smooth_min_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live viewer for CNN-style RF spectrogram input quality."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--center-freq", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--rf-bandwidth", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--distance-m", type=float, default=math.nan)
    parser.add_argument("--memo", default="")

    parser.add_argument("--update-interval-sec", type=float, default=1.0)
    parser.add_argument("--blocks-per-update", type=int, default=20)
    parser.add_argument("--select-policy", default="max_signal_ratio")
    parser.add_argument("--max-updates", type=int, default=None)

    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--no-signal-ratio", type=float, default=2.0)
    parser.add_argument("--valid-signal-ratio", type=float, default=5.0)
    parser.add_argument("--overload-peak", type=float, default=0.95)
    parser.add_argument("--overload-clip-ratio", type=float, default=0.001)

    parser.add_argument("--stft-nperseg", type=int, default=None)
    parser.add_argument("--stft-noverlap", type=int, default=None)
    parser.add_argument("--stft-nfft", type=int, default=None)

    parser.add_argument("--log-dir", default="outputs/live_viewer/logs")
    parser.add_argument("--latest-image-dir", default="outputs/live_viewer/latest")
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--no-display", action="store_true")

    # Decision / CNN inference modes
    parser.add_argument(
        "--decision-mode",
        default="none",
        choices=["none", "raw", "gain-aware", "temporal", "hybrid"],
        help=(
            "none: no CNN inference, raw: fixed threshold, "
            "gain-aware: gain-specific threshold, temporal: fixed threshold + vote, "
            "hybrid: gain-aware threshold + vote"
        ),
    )
    parser.add_argument(
        "--cnn-model",
        default="",
        help="Path to trained RF binary CNN model. Required unless decision-mode=none.",
    )
    parser.add_argument("--cnn-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--drone-threshold", type=float, default=0.50)
    parser.add_argument("--drone-threshold-g25", type=float, default=0.35)
    parser.add_argument("--drone-threshold-g30", type=float, default=0.80)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--candidate-vote-k", type=int, default=2)
    parser.add_argument("--confirmed-vote-k", type=int, default=3)
    parser.add_argument("--reset-temporal-on-no-signal", action="store_true")
    parser.add_argument("--show-candidate-as-drone", action="store_true")

    # AoA / dual-channel phase-difference estimate
    parser.add_argument("--enable-aoa", action="store_true")
    parser.add_argument(
        "--aoa-antenna-spacing-m",
        type=float,
        default=0.061,
        help="RX0-RX1 antenna spacing in meters. Default is about half wavelength at 2.45 GHz.",
    )
    parser.add_argument(
        "--aoa-calibration-deg",
        type=float,
        default=0.0,
        help="Angle offset added to estimated AoA after phase-to-angle conversion.",
    )
    parser.add_argument(
        "--aoa-phase-offset-rad",
        type=float,
        default=0.0,
        help="RX1-RX0 phase offset in radians. This is subtracted before AoA conversion.",
    )
    parser.add_argument(
        "--aoa-auto-phase-calibration",
        action="store_true",
        help="Estimate phase offset at startup. Put the source at boresight/front 0 degree.",
    )
    parser.add_argument(
        "--aoa-calibration-blocks",
        type=int,
        default=30,
        help="Number of blocks used for startup phase calibration.",
    )
    parser.add_argument(
        "--aoa-min-coherence",
        type=float,
        default=0.20,
        help="Minimum channel coherence for AoA to be considered usable.",
    )
    parser.add_argument(
        "--aoa-min-signal-ratio",
        type=float,
        default=5.0,
        help="Minimum signal_ratio required before AoA is computed.",
    )
    parser.add_argument(
        "--aoa-gate-mode",
        default="candidate",
        choices=["signal", "raw-drone", "candidate", "confirmed"],
        help=(
            "signal: valid RF signal only, "
            "raw-drone: CNN raw Drone only, "
            "candidate: Candidate or Confirmed Drone, "
            "confirmed: Confirmed Drone only"
        ),
    )
    parser.add_argument(
        "--aoa-smooth-window",
        type=int,
        default=5,
        help="Number of recent valid AoA estimates used for circular moving average.",
    )
    parser.add_argument(
        "--aoa-smooth-min-valid",
        type=int,
        default=3,
        help="Minimum valid AoA estimates required to output smoothed AoA.",
    )
    return parser.parse_args()


def now_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_project_root() / path


def nested_get(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_receiver_value(receiver_cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read common receiver values from either top-level or nested config blocks."""
    if key in receiver_cfg:
        return receiver_cfg[key]
    for section in ("sdr", "pluto", "pluto_plus", "file", "sim"):
        value = nested_get(receiver_cfg, [section, key], None)
        if value is not None:
            return value
    return default


def set_receiver_value(receiver_cfg: dict[str, Any], key: str, value: Any) -> None:
    """Set a receiver value in top-level and known nested blocks when present."""
    receiver_cfg[key] = value
    for section in ("sdr", "pluto", "pluto_plus", "file", "sim"):
        if isinstance(receiver_cfg.get(section), dict):
            receiver_cfg[section][key] = value


def load_viewer_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = load_all_configs(args.config_dir)
    receiver_cfg = dict(configs.get("receiver", {}))
    ml_cfg = dict(configs.get("ml", {}))

    if args.gain is not None:
        set_receiver_value(receiver_cfg, "gain", args.gain)
    if args.center_freq is not None:
        set_receiver_value(receiver_cfg, "center_freq", int(args.center_freq))
    if args.sample_rate is not None:
        set_receiver_value(receiver_cfg, "sample_rate", int(args.sample_rate))
    if args.rf_bandwidth is not None:
        set_receiver_value(receiver_cfg, "rf_bandwidth", int(args.rf_bandwidth))
    if args.block_size is not None:
        set_receiver_value(receiver_cfg, "block_size", args.block_size)

    return receiver_cfg, ml_cfg


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"IQ block must be 1D or 2D, got shape={arr.shape}")


def select_rx_1d(iq: np.ndarray, rx_index: int) -> np.ndarray:
    arr = ensure_2d_iq(iq)
    if rx_index < 0 or rx_index >= arr.shape[0]:
        raise IndexError(
            f"rx_index={rx_index} is out of range for IQ shape={arr.shape}"
        )
    return np.asarray(arr[rx_index], dtype=np.complex64)


def frame_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if x.size < frame_size:
        return x.reshape(1, -1)

    num_frames = 1 + (x.size - frame_size) // hop_size
    shape = (num_frames, frame_size)
    strides = (x.strides[0] * hop_size, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def compute_raw_features(
    iq_block: np.ndarray,
    rx_index: int,
    frame_size: int,
    hop_size: int,
    overload_peak: float,
) -> RawFeatures:
    raw = select_rx_1d(iq_block, rx_index)
    abs_raw = np.abs(raw).astype(np.float64)

    raw_rms = float(np.sqrt(np.mean(abs_raw ** 2)))
    raw_peak = float(np.max(abs_raw))
    raw_abs_mean = float(np.mean(abs_raw))
    raw_abs_median = float(np.median(abs_raw))
    raw_abs_p95 = float(np.percentile(abs_raw, 95))
    raw_abs_p99 = float(np.percentile(abs_raw, 99))
    clip_ratio = float(np.mean(abs_raw >= overload_peak))

    frames = frame_1d(raw, frame_size=frame_size, hop_size=hop_size)
    frame_power = np.mean(np.abs(frames).astype(np.float64) ** 2, axis=1)
    frame_power_median = float(np.median(frame_power))
    frame_power_p95 = float(np.percentile(frame_power, 95))
    frame_power_p99 = float(np.percentile(frame_power, 99))
    noise_floor = max(frame_power_median, EPS)
    signal_ratio = float(frame_power_p99 / noise_floor)

    return RawFeatures(
        raw_rms=raw_rms,
        raw_peak=raw_peak,
        raw_abs_mean=raw_abs_mean,
        raw_abs_median=raw_abs_median,
        raw_abs_p95=raw_abs_p95,
        raw_abs_p99=raw_abs_p99,
        clip_ratio=clip_ratio,
        frame_power_median=frame_power_median,
        frame_power_p95=frame_power_p95,
        frame_power_p99=frame_power_p99,
        noise_floor=noise_floor,
        signal_ratio=signal_ratio,
    )


def classify_signal_status(
    raw_features: RawFeatures,
    thresholds: ViewerThresholds,
) -> tuple[str, str]:
    if (
        raw_features.raw_peak >= thresholds.overload_peak
        or raw_features.clip_ratio >= thresholds.overload_clip_ratio
    ):
        return "OVERLOAD", "TRY_LOWER_GAIN"

    if raw_features.signal_ratio < thresholds.no_signal_ratio:
        return "NO_SIGNAL", "KEEP_GAIN"

    if raw_features.signal_ratio < thresholds.valid_signal_ratio:
        return "WEAK_SIGNAL", "TRY_HIGHER_GAIN_IF_REPEATED"

    return "VALID_SIGNAL", "KEEP_GAIN"


def select_representative_block(
    blocks: list[np.ndarray],
    raw_features_list: list[RawFeatures],
    select_policy: str,
) -> int:
    if not blocks:
        raise ValueError("No IQ blocks were collected")
    if select_policy != "max_signal_ratio":
        raise ValueError(f"Unsupported select_policy={select_policy!r}")
    scores = [features.signal_ratio for features in raw_features_list]
    return int(np.argmax(scores))


def infer_stft_params(
    ml_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    stft_cfg = ml_cfg.get("stft", {}) if isinstance(ml_cfg.get("stft", {}), dict) else {}

    nperseg = args.stft_nperseg or int(
        stft_cfg.get("nperseg", stft_cfg.get("frame_size", 128))
    )
    noverlap = args.stft_noverlap or int(
        stft_cfg.get("noverlap", nperseg - stft_cfg.get("hop_length", 32))
    )
    nfft = args.stft_nfft or int(stft_cfg.get("nfft", nperseg))

    if nperseg <= 0:
        raise ValueError("stft nperseg must be positive")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("stft noverlap must satisfy 0 <= noverlap < nperseg")
    if nfft < nperseg:
        raise ValueError("stft nfft must be >= nperseg")
    return nperseg, noverlap, nfft


def compute_cnn_input_spectrogram(
    iq_block: np.ndarray,
    rx_index: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> np.ndarray:
    """Create normalized log-magnitude STFT spectrogram for CNN input viewing."""
    iq_no_dc = remove_dc_offset(iq_block)
    cnn_iq = get_cnn_input_iq(iq_no_dc, rx_index=rx_index)
    cnn_iq = normalize_iq(cnn_iq, method="peak")
    cnn_iq = np.asarray(cnn_iq).reshape(-1).astype(np.complex64)

    hop = nperseg - noverlap
    frames = frame_1d(cnn_iq, frame_size=nperseg, hop_size=hop)
    window = np.hanning(nperseg).astype(np.float32)
    windowed = frames * window.reshape(1, -1)

    stft = np.fft.fft(windowed, n=nfft, axis=1)
    stft = np.fft.fftshift(stft, axes=1)
    mag = np.abs(stft).astype(np.float32)
    spec = np.log1p(mag)

    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec = (spec - spec_min) / max(spec_max - spec_min, EPS)

    # Image convention: frequency bins x time frames.
    return spec.T.astype(np.float32)


def compute_cnn_spec_features(spec: np.ndarray) -> CnnSpecFeatures:
    x = np.asarray(spec, dtype=np.float64)
    return CnnSpecFeatures(
        cnn_spec_mean=float(np.mean(x)),
        cnn_spec_std=float(np.std(x)),
        cnn_spec_min=float(np.min(x)),
        cnn_spec_p50=float(np.percentile(x, 50)),
        cnn_spec_p95=float(np.percentile(x, 95)),
        cnn_spec_p99=float(np.percentile(x, 99)),
        cnn_spec_max=float(np.max(x)),
    )


def empty_cnn_features() -> dict[str, float]:
    return {
        "cnn_spec_mean": math.nan,
        "cnn_spec_std": math.nan,
        "cnn_spec_min": math.nan,
        "cnn_spec_p50": math.nan,
        "cnn_spec_p95": math.nan,
        "cnn_spec_p99": math.nan,
        "cnn_spec_max": math.nan,
    }


def empty_aoa_features(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "aoa_enabled": bool(getattr(args, "enable_aoa", False)),
        "aoa_status": "DISABLED" if not getattr(args, "enable_aoa", False) else "UNAVAILABLE",
        "aoa_deg": math.nan,
        "aoa_phase_diff_rad": math.nan,
        "aoa_phase_diff_raw_rad": math.nan,
        "aoa_phase_offset_rad": float(getattr(args, "aoa_phase_offset_rad", 0.0)),
        "aoa_coherence": math.nan,
        "aoa_antenna_spacing_m": float(getattr(args, "aoa_antenna_spacing_m", math.nan)),
        "aoa_calibration_deg": float(getattr(args, "aoa_calibration_deg", 0.0)),
        "aoa_deg_smooth": math.nan,
        "aoa_smooth_status": "NO_SMOOTH",
        "aoa_smooth_count": 0,
        "aoa_smooth_window": int(getattr(args, "aoa_smooth_window", 5)),
        "aoa_smooth_min_valid": int(getattr(args, "aoa_smooth_min_valid", 3)),
    }


def wrap_to_pi(angle_rad: float) -> float:
    return float(np.angle(np.exp(1j * float(angle_rad))))


def estimate_dual_rx_phase_diff(iq_block: np.ndarray) -> tuple[float, float, str]:
    """
    Return raw RX1-RX0 phase difference and coherence.
    """
    arr = ensure_2d_iq(iq_block)
    if arr.shape[0] < 2:
        return math.nan, math.nan, "NEED_DUAL_RX"

    ch0 = np.asarray(arr[0], dtype=np.complex64).reshape(-1)
    ch1 = np.asarray(arr[1], dtype=np.complex64).reshape(-1)

    if ch0.size == 0 or ch1.size == 0:
        return math.nan, math.nan, "EMPTY_IQ"

    n = min(ch0.size, ch1.size)
    ch0 = ch0[:n]
    ch1 = ch1[:n]

    p0 = float(np.mean(np.abs(ch0).astype(np.float64) ** 2))
    p1 = float(np.mean(np.abs(ch1).astype(np.float64) ** 2))
    denom = math.sqrt(max(p0 * p1, EPS))

    cross = np.mean(ch1 * np.conj(ch0))
    phase_diff_raw = float(np.angle(cross))
    coherence = float(np.abs(cross) / denom)

    return phase_diff_raw, coherence, "OK"


def calibrate_aoa_phase_offset(
    receiver: Any,
    *,
    block_size: int,
    calibration_blocks: int,
    min_coherence: float,
) -> float:
    """
    Estimate RX1-RX0 phase offset using circular mean.
    Place known source at antenna front/0 degree before running.
    """
    phase_vectors: list[complex] = []
    used = 0

    print("[AOA CAL] Start phase calibration. Put source at front/0 degree.")

    for idx in range(max(1, int(calibration_blocks))):
        block = receiver.read_block(block_size)
        block = remove_dc_offset(block)

        phase_raw, coherence, status = estimate_dual_rx_phase_diff(block)

        if (
            status == "OK"
            and math.isfinite(phase_raw)
            and math.isfinite(coherence)
            and coherence >= float(min_coherence)
        ):
            phase_vectors.append(np.exp(1j * phase_raw))
            used += 1

        if idx % 5 == 0:
            print(
                f"[AOA CAL] block={idx:03d} status={status} "
                f"phase={phase_raw:.4f} coh={coherence:.3f} used={used}"
            )

    if not phase_vectors:
        print("[AOA CAL] Failed. No valid dual-channel phase samples.")
        return 0.0

    offset = float(np.angle(np.mean(np.asarray(phase_vectors, dtype=np.complex128))))
    print(f"[AOA CAL] Done. phase_offset_rad={offset:.6f} used={used}/{calibration_blocks}")
    return offset



def circular_mean_deg(values_deg: list[float]) -> float:
    if not values_deg:
        return math.nan

    radians = np.deg2rad(np.asarray(values_deg, dtype=np.float64))
    vector = np.mean(np.exp(1j * radians))

    if abs(vector) < EPS:
        return math.nan

    return float(np.rad2deg(np.angle(vector)))


def update_aoa_smoothing(
    aoa_features: dict[str, Any],
    aoa_history: list[float],
    *,
    window: int,
    min_valid: int,
) -> dict[str, Any]:
    window = max(1, int(window))
    min_valid = max(1, int(min_valid))

    aoa_status = str(aoa_features.get("aoa_status", ""))
    aoa_deg = float(aoa_features.get("aoa_deg", math.nan))

    if aoa_status == "OK" and math.isfinite(aoa_deg):
        aoa_history.append(aoa_deg)

    if len(aoa_history) > window:
        del aoa_history[:-window]

    count = len(aoa_history)

    aoa_features["aoa_smooth_count"] = count
    aoa_features["aoa_smooth_window"] = window
    aoa_features["aoa_smooth_min_valid"] = min_valid

    if count >= min_valid:
        smooth_deg = circular_mean_deg(aoa_history[-window:])
        aoa_features["aoa_deg_smooth"] = smooth_deg
        aoa_features["aoa_smooth_status"] = "OK" if math.isfinite(smooth_deg) else "BAD_MEAN"
    else:
        aoa_features["aoa_deg_smooth"] = math.nan
        aoa_features["aoa_smooth_status"] = "WAIT_VALID"

    return aoa_features



def should_compute_aoa(
    args: argparse.Namespace,
    status: str,
    raw_features: RawFeatures,
    cnn_raw_decision: str,
    final_decision: str,
) -> tuple[bool, str]:
    if not getattr(args, "enable_aoa", False):
        return False, "DISABLED"

    if status == "OVERLOAD":
        return False, "GATED_OVERLOAD"

    if status not in {"WEAK_SIGNAL", "VALID_SIGNAL"}:
        return False, "GATED_NO_SIGNAL"

    if raw_features.signal_ratio < float(args.aoa_min_signal_ratio):
        return False, "GATED_LOW_SR"

    mode = getattr(args, "aoa_gate_mode", "candidate")

    if mode == "signal":
        return True, "OK"

    if mode == "raw-drone":
        if cnn_raw_decision == "Drone":
            return True, "OK"
        return False, "GATED_RAW_NONDRONE"

    if mode == "candidate":
        if final_decision in {"Drone-like Candidate", "Confirmed Drone"}:
            return True, "OK"
        return False, "GATED_NOT_CANDIDATE"

    if mode == "confirmed":
        if final_decision == "Confirmed Drone":
            return True, "OK"
        return False, "GATED_NOT_CONFIRMED"

    return False, "GATED_UNKNOWN_MODE"



def compute_aoa_features(
    iq_block: np.ndarray,
    center_freq_hz: float,
    antenna_spacing_m: float,
    calibration_deg: float = 0.0,
    min_coherence: float = 0.20,
    phase_offset_rad: float = 0.0,
) -> dict[str, Any]:
    """
    Estimate rough AoA using phase difference between RX0 and RX1.

    Notes:
    - Requires coherent dual-channel IQ input with shape (2, N) or more.
    - Angle sign depends on antenna order and physical layout.
    - calibration_deg should be adjusted using a known 0-degree source.
    """
    result = {
        "aoa_enabled": True,
        "aoa_status": "UNAVAILABLE",
        "aoa_deg": math.nan,
        "aoa_phase_diff_rad": math.nan,
        "aoa_phase_diff_raw_rad": math.nan,
        "aoa_phase_offset_rad": float(phase_offset_rad),
        "aoa_coherence": math.nan,
        "aoa_antenna_spacing_m": float(antenna_spacing_m),
        "aoa_calibration_deg": float(calibration_deg),
    }

    arr = ensure_2d_iq(iq_block)
    if arr.shape[0] < 2:
        result["aoa_status"] = "NEED_DUAL_RX"
        return result

    if center_freq_hz is None or not math.isfinite(float(center_freq_hz)) or float(center_freq_hz) <= 0:
        result["aoa_status"] = "BAD_CENTER_FREQ"
        return result

    if antenna_spacing_m <= 0:
        result["aoa_status"] = "BAD_SPACING"
        return result

    ch0 = np.asarray(arr[0], dtype=np.complex64).reshape(-1)
    ch1 = np.asarray(arr[1], dtype=np.complex64).reshape(-1)

    if ch0.size == 0 or ch1.size == 0:
        result["aoa_status"] = "EMPTY_IQ"
        return result

    n = min(ch0.size, ch1.size)
    ch0 = ch0[:n]
    ch1 = ch1[:n]

    p0 = float(np.mean(np.abs(ch0).astype(np.float64) ** 2))
    p1 = float(np.mean(np.abs(ch1).astype(np.float64) ** 2))
    denom = math.sqrt(max(p0 * p1, EPS))

    cross = np.mean(ch1 * np.conj(ch0))
    phase_diff_raw = float(np.angle(cross))
    phase_diff = wrap_to_pi(phase_diff_raw - float(phase_offset_rad))
    coherence = float(np.abs(cross) / denom)

    result["aoa_phase_diff_raw_rad"] = phase_diff_raw
    result["aoa_phase_diff_rad"] = phase_diff
    result["aoa_phase_offset_rad"] = float(phase_offset_rad)
    result["aoa_coherence"] = coherence

    if coherence < min_coherence:
        result["aoa_status"] = "LOW_COHERENCE"
        return result

    c = 299_792_458.0
    wavelength = c / float(center_freq_hz)

    sin_theta = phase_diff * wavelength / (2.0 * math.pi * float(antenna_spacing_m))

    if abs(sin_theta) > 1.0:
        result["aoa_status"] = "AMBIGUOUS"
        sin_theta = max(-1.0, min(1.0, sin_theta))
    else:
        result["aoa_status"] = "OK"

    aoa_deg = math.degrees(math.asin(sin_theta)) + float(calibration_deg)
    result["aoa_deg"] = float(aoa_deg)
    return result



def prepare_matplotlib(no_display: bool):
    if no_display:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_latest_image(
    plt: Any,
    latest_image_dir: Path,
    spec: np.ndarray | None,
    title: str,
    side_text: str,
) -> Path | None:
    if spec is None:
        return None
    latest_image_dir.mkdir(parents=True, exist_ok=True)
    path = latest_image_dir / "live_cnn_spectrogram_latest.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(spec, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.text(
        1.02,
        0.5,
        side_text,
        transform=ax.transAxes,
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def update_display(
    plt: Any,
    image_handle: Any,
    text_handle: Any,
    spec: np.ndarray | None,
    title: str,
    side_text: str,
) -> tuple[Any, Any]:
    if spec is None:
        plt.gcf().suptitle(title)
        text_handle.set_text(side_text)
        plt.pause(0.001)
        return image_handle, text_handle

    if image_handle is None:
        image_handle = plt.imshow(spec, aspect="auto", origin="lower")
        plt.xlabel("Time frame")
        plt.ylabel("Frequency bin")
    else:
        image_handle.set_data(spec)
        image_handle.set_clim(float(np.min(spec)), float(np.max(spec)))

    plt.title(title)
    text_handle.set_text(side_text)
    plt.pause(0.001)
    return image_handle, text_handle



def resolve_cnn_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_binary_cnn_model(model_path: str, device: str) -> SmallRFBinaryCNN:
    if not model_path:
        raise ValueError("--cnn-model is required when decision-mode is not 'none'")

    model_file = to_project_path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"CNN model not found: {model_file}")

    model = SmallRFBinaryCNN(num_classes=2)
    ckpt = torch.load(model_file, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def infer_drone_probability(
    model: SmallRFBinaryCNN,
    spec: np.ndarray,
    device: str,
) -> float:
    arr = np.asarray(spec, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"CNN spectrogram must be 2D, got shape={arr.shape}")

    x = torch.from_numpy(arr[None, None, :, :]).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    return float(probs[0, 1].detach().cpu().item())


def select_drone_threshold(
    decision_mode: str,
    gain: Any,
    default_threshold: float,
    threshold_g25: float,
    threshold_g30: float,
) -> float:
    if decision_mode not in {"gain-aware", "hybrid"}:
        return default_threshold

    try:
        gain_value = float(gain)
    except (TypeError, ValueError):
        return default_threshold

    if abs(gain_value - 25.0) < 1e-6:
        return threshold_g25
    if abs(gain_value - 30.0) < 1e-6:
        return threshold_g30

    return default_threshold


def update_temporal_decision(
    history: deque[int],
    raw_decision: int,
    window: int,
    candidate_vote_k: int,
    confirmed_vote_k: int,
) -> tuple[list[int], bool, bool, str]:
    history.append(int(raw_decision))

    recent = list(history)[-window:]
    vote_count = sum(recent)

    candidate = vote_count >= candidate_vote_k
    confirmed = vote_count >= confirmed_vote_k

    if confirmed:
        final_decision = "Confirmed Drone"
    elif candidate:
        final_decision = "Drone-like Candidate"
    else:
        final_decision = "NonDrone"

    return recent, candidate, confirmed, final_decision



def append_csv_log(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})

def _parse_optional_float(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}g}"


def setup_gain_widgets(
    fig: Any,
    receiver: Any,
    initial_gain: float,
    initial_distance: float = math.nan,
    initial_memo: str = "",
) -> dict[str, Any]:
    main_ax = fig.gca()

    gain_state: dict[str, Any] = {
        "gain": float(initial_gain),
        "distance_m": float(initial_distance)
        if math.isfinite(float(initial_distance))
        else math.nan,
        "memo": str(initial_memo),
        "target_raw_abs_p99": math.nan,
        "target_frame_power_p99": math.nan,
        "target_raw_rms": math.nan,
        "tolerance_pct": 15.0,
        "message": "Gain/feature control ready",
        "paused": False,
        "latest_row": None,
    }

    fig.subplots_adjust(left=0.08, right=0.70, bottom=0.38, top=0.92)

    current_gain_text = fig.text(
        0.02,
        0.345,
        f"Current Gain: {gain_state['gain']:.1f} dB",
        fontsize=8,
    )

    message_text = fig.text(
        0.53,
        0.345,
        gain_state["message"],
        fontsize=8,
    )

    current_feature_text = fig.text(
        0.02,
        0.020,
        "Current features: waiting...",
        fontsize=7,
    )

    match_text = fig.text(
        0.53,
        0.020,
        "Target match: NO_TARGET",
        fontsize=7,
    )

    # row 1: gain / distance / memo
    gain_ax = fig.add_axes([0.10, 0.290, 0.11, 0.032])
    gain_button_ax = fig.add_axes([0.22, 0.290, 0.08, 0.032])
    distance_ax = fig.add_axes([0.39, 0.290, 0.10, 0.032])
    memo_ax = fig.add_axes([0.57, 0.290, 0.34, 0.032])

    gain_box = TextBox(gain_ax, "Gain", initial=f"{gain_state['gain']:.1f}")
    apply_button = Button(gain_button_ax, "Apply")
    distance_box = TextBox(
        distance_ax,
        "Dist",
        initial="" if math.isnan(gain_state["distance_m"]) else f"{gain_state['distance_m']:.2f}",
    )
    memo_box = TextBox(memo_ax, "Memo", initial=gain_state["memo"])

    # row 2: target features
    raw_p99_ax = fig.add_axes([0.13, 0.225, 0.12, 0.032])
    frame_p99_ax = fig.add_axes([0.36, 0.225, 0.12, 0.032])
    rms_ax = fig.add_axes([0.58, 0.225, 0.12, 0.032])
    tol_ax = fig.add_axes([0.80, 0.225, 0.07, 0.032])

    target_raw_abs_p99_box = TextBox(raw_p99_ax, "T.raw99", initial="")
    target_frame_power_p99_box = TextBox(frame_p99_ax, "T.frm99", initial="")
    target_raw_rms_box = TextBox(rms_ax, "T.rms", initial="")
    tolerance_box = TextBox(tol_ax, "Tol", initial="15")

    # row 3: target capture button
    capture_button_ax = fig.add_axes([0.13, 0.160, 0.20, 0.034])
    capture_button = Button(capture_button_ax, "Use Current")

    pause_button_ax = fig.add_axes([0.36, 0.160, 0.12, 0.034])
    pause_button = Button(pause_button_ax, "Pause")

    def apply_gain(_event: Any = None) -> None:
        try:
            new_gain = float(gain_box.text)

            if not hasattr(receiver, "set_gain"):
                raise AttributeError(
                    "receiver.set_gain()이 없습니다. "
                    "src/receiver/pluto_receiver.py에 set_gain()을 먼저 추가해야 합니다."
                )

            applied_gain = receiver.set_gain(new_gain, warmup_reads=1)
            gain_state["gain"] = float(applied_gain)
            gain_state["message"] = f"Gain updated: {applied_gain:.1f} dB"

        except Exception as exc:
            gain_state["message"] = f"Gain update failed: {exc}"

        current_gain_text.set_text(
            f"Current Gain: {gain_state['gain']:.1f} dB"
        )
        message_text.set_text(gain_state["message"])
        fig.canvas.draw_idle()

    def capture_target_from_current(_event: Any = None) -> None:
        row = gain_state.get("latest_row")
        if not row:
            gain_state["message"] = "No current feature row yet"
            message_text.set_text(gain_state["message"])
            fig.canvas.draw_idle()
            return

        target_raw_abs_p99_box.set_val(_fmt_float(row.get("raw_abs_p99"), 8))
        target_frame_power_p99_box.set_val(_fmt_float(row.get("frame_power_p99"), 8))
        target_raw_rms_box.set_val(_fmt_float(row.get("raw_rms"), 8))

        update_control_state_from_widgets(gain_state)
        gain_state["message"] = "Current features captured as target"
        message_text.set_text(gain_state["message"])
        fig.canvas.draw_idle()


    def toggle_pause(_event: Any = None) -> None:
        gain_state["paused"] = not bool(gain_state.get("paused", False))

        if gain_state["paused"]:
            gain_state["message"] = "PAUSED: acquisition/logging stopped"
            pause_button.label.set_text("Resume")
        else:
            gain_state["message"] = "RESUMED: acquisition/logging running"
            pause_button.label.set_text("Pause")

        message_text.set_text(gain_state["message"])
        fig.canvas.draw_idle()

    gain_box.on_submit(apply_gain)
    apply_button.on_clicked(apply_gain)
    pause_button.on_clicked(toggle_pause)
    capture_button.on_clicked(capture_target_from_current)

    gain_state["current_gain_text"] = current_gain_text
    gain_state["message_text"] = message_text
    gain_state["current_feature_text"] = current_feature_text
    gain_state["match_text"] = match_text

    gain_state["gain_box"] = gain_box
    gain_state["apply_button"] = apply_button
    gain_state["distance_box"] = distance_box
    gain_state["memo_box"] = memo_box

    gain_state["target_raw_abs_p99_box"] = target_raw_abs_p99_box
    gain_state["target_frame_power_p99_box"] = target_frame_power_p99_box
    gain_state["target_raw_rms_box"] = target_raw_rms_box
    gain_state["tolerance_box"] = tolerance_box
    gain_state["capture_button"] = capture_button
    gain_state["pause_button"] = pause_button

    fig.sca(main_ax)
    return gain_state


def update_control_state_from_widgets(gain_state: dict[str, Any]) -> None:
    gain_state["distance_m"] = _parse_optional_float(
        gain_state["distance_box"].text,
        default=math.nan,
    )
    gain_state["memo"] = str(gain_state["memo_box"].text)

    gain_state["target_raw_abs_p99"] = _parse_optional_float(
        gain_state["target_raw_abs_p99_box"].text,
        default=math.nan,
    )
    gain_state["target_frame_power_p99"] = _parse_optional_float(
        gain_state["target_frame_power_p99_box"].text,
        default=math.nan,
    )
    gain_state["target_raw_rms"] = _parse_optional_float(
        gain_state["target_raw_rms_box"].text,
        default=math.nan,
    )

    tolerance_pct = _parse_optional_float(
        gain_state["tolerance_box"].text,
        default=15.0,
    )
    if not math.isfinite(tolerance_pct) or tolerance_pct <= 0:
        tolerance_pct = 15.0

    gain_state["tolerance_pct"] = float(tolerance_pct)


def empty_feature_match_result() -> dict[str, Any]:
    return {
        "target_raw_abs_p99": math.nan,
        "target_frame_power_p99": math.nan,
        "target_raw_rms": math.nan,
        "feature_match_status": "NO_TARGET",
        "feature_match_max_error_pct": math.nan,
        "feature_match_mean_error_pct": math.nan,
        "feature_match_tolerance_pct": math.nan,
    }


def evaluate_feature_match(
    row: dict[str, Any],
    gain_state: dict[str, Any],
) -> dict[str, Any]:
    target_items = [
        ("raw_abs_p99", "target_raw_abs_p99"),
        ("frame_power_p99", "target_frame_power_p99"),
        ("raw_rms", "target_raw_rms"),
    ]

    errors_pct: list[float] = []

    for current_key, target_key in target_items:
        current_value = float(row.get(current_key, math.nan))
        target_value = float(gain_state.get(target_key, math.nan))

        if not math.isfinite(current_value) or not math.isfinite(target_value):
            continue

        denom = max(abs(target_value), EPS)
        error_pct = abs(current_value - target_value) / denom * 100.0
        errors_pct.append(float(error_pct))

    tolerance_pct = float(gain_state.get("tolerance_pct", 15.0))

    result = {
        "target_raw_abs_p99": gain_state.get("target_raw_abs_p99", math.nan),
        "target_frame_power_p99": gain_state.get("target_frame_power_p99", math.nan),
        "target_raw_rms": gain_state.get("target_raw_rms", math.nan),
        "feature_match_status": "NO_TARGET",
        "feature_match_max_error_pct": math.nan,
        "feature_match_mean_error_pct": math.nan,
        "feature_match_tolerance_pct": tolerance_pct,
    }

    if not errors_pct:
        return result

    max_error = float(max(errors_pct))
    mean_error = float(sum(errors_pct) / len(errors_pct))

    if row.get("status") == "OVERLOAD" or float(row.get("clip_ratio", 0.0)) > 0:
        status = "OVERLOAD"
    elif max_error <= tolerance_pct:
        status = "MATCH"
    else:
        status = "SEARCH"

    result.update(
        {
            "feature_match_status": status,
            "feature_match_max_error_pct": max_error,
            "feature_match_mean_error_pct": mean_error,
        }
    )
    return result


def update_feature_widget_text(
    gain_state: dict[str, Any],
    row: dict[str, Any],
    match_result: dict[str, Any],
) -> None:
    gain_state["latest_row"] = row

    current_text = (
        "Current features | "
        f"raw_p99={_fmt_float(row.get('raw_abs_p99'))}  "
        f"frame_p99={_fmt_float(row.get('frame_power_p99'))}  "
        f"rms={_fmt_float(row.get('raw_rms'))}  "
        f"SR={_fmt_float(row.get('signal_ratio'), 4)}  "
        f"clip={_fmt_float(row.get('clip_ratio'), 4)}  "
        f"AoA={_fmt_float(row.get('aoa_deg'), 4)}deg  "
        f"AoA_sm={_fmt_float(row.get('aoa_deg_smooth'), 4)}deg  "
        f"Coh={_fmt_float(row.get('aoa_coherence'), 3)}"
    )

    status = match_result.get("feature_match_status", "NO_TARGET")
    max_err = match_result.get("feature_match_max_error_pct", math.nan)
    mean_err = match_result.get("feature_match_mean_error_pct", math.nan)
    tol = match_result.get("feature_match_tolerance_pct", math.nan)

    match_text = (
        f"Target match: {status} | "
        f"max_err={_fmt_float(max_err, 4)}%  "
        f"mean_err={_fmt_float(mean_err, 4)}%  "
        f"tol={_fmt_float(tol, 4)}%"
    )

    gain_state["current_feature_text"].set_text(current_text)
    gain_state["match_text"].set_text(match_text)
    gain_state["message_text"].set_text(gain_state.get("message", ""))


def build_side_text(row: dict[str, Any]) -> str:
    lines = [
        f"center_freq: {row['center_freq']}",
        f"sample_rate: {row['sample_rate']}",
        f"rf_bandwidth: {row['rf_bandwidth']}",
        f"gain: {row['gain']}",
        f"rx_index: {row['rx_index']}",
        f"distance_m: {row['distance_m']}",
        f"memo: {row['memo']}",
        f"status: {row['status']}",
        f"suggestion: {row['suggestion']}",
        "",
        f"raw_rms: {row['raw_rms']:.6g}",
        f"raw_peak: {row['raw_peak']:.6g}",
        f"raw_p99: {row['raw_abs_p99']:.6g}",
        f"clip_ratio: {row['clip_ratio']:.6g}",
        f"noise_floor: {row['noise_floor']:.6g}",
        f"signal_ratio: {row['signal_ratio']:.3f}",
        f"frame_p99: {row['frame_power_p99']:.6g}",
        "",
        f"cnn_mean: {row['cnn_spec_mean']:.6g}",
        f"cnn_std: {row['cnn_spec_std']:.6g}",
        f"cnn_p95: {row['cnn_spec_p95']:.6g}",
        f"cnn_p99: {row['cnn_spec_p99']:.6g}",
        f"cnn_max: {row['cnn_spec_max']:.6g}",
        "",
        f"target_status: {row.get('feature_match_status', 'NA')}",
        f"target_raw_p99: {row.get('target_raw_abs_p99', math.nan):.6g}",
        f"target_frame_p99: {row.get('target_frame_power_p99', math.nan):.6g}",
        f"target_rms: {row.get('target_raw_rms', math.nan):.6g}",
        f"match_max_err_pct: {row.get('feature_match_max_error_pct', math.nan):.3f}",
        f"match_mean_err_pct: {row.get('feature_match_mean_error_pct', math.nan):.3f}",
        f"match_tol_pct: {row.get('feature_match_tolerance_pct', math.nan):.3f}",
        "",
        f"aoa_status: {row.get('aoa_status', 'NA')}",
        f"aoa_deg: {row.get('aoa_deg', math.nan):.2f}",
        f"aoa_smooth: {row.get('aoa_deg_smooth', math.nan):.2f}",
        f"aoa_sm_status: {row.get('aoa_smooth_status', 'NA')}",
        f"aoa_sm_count: {row.get('aoa_smooth_count', 0)}",
        f"aoa_phase_rad: {row.get('aoa_phase_diff_rad', math.nan):.3f}",
        f"aoa_raw_phase: {row.get('aoa_phase_diff_raw_rad', math.nan):.3f}",
        f"aoa_offset: {row.get('aoa_phase_offset_rad', math.nan):.3f}",
        f"aoa_coherence: {row.get('aoa_coherence', math.nan):.3f}",
        "",
        f"decision_mode: {row.get('decision_mode', 'none')}",
        f"cnn_prob_drone: {row.get('cnn_prob_drone', math.nan):.4f}",
        f"cnn_threshold: {row.get('cnn_threshold', math.nan):.4f}",
        f"raw_decision: {row.get('cnn_raw_decision', 'NA')}",
        f"history: {row.get('temporal_history', '')}",
        f"candidate: {row.get('candidate_status', False)}",
        f"confirmed: {row.get('confirmed_status', False)}",
        f"final: {row.get('final_decision', 'NA')}",
    ]
    return "\n".join(lines)


def print_update(row: dict[str, Any]) -> None:
    print(
        "[update={update_index:04d}] status={status:<12} "
        "signal_ratio={signal_ratio:.3f} raw_peak={raw_peak:.6g} "
        "clip_ratio={clip_ratio:.6g} suggestion={suggestion} "
        "mode={decision_mode} prob={cnn_prob_drone:.4f} "
        "th={cnn_threshold:.3f} raw={cnn_raw_decision} final={final_decision}".format(**row),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    receiver_cfg, ml_cfg = load_viewer_configs(args)

    session_id = now_session_id()
    log_dir = to_project_path(args.log_dir)
    latest_image_dir = to_project_path(args.latest_image_dir)
    csv_path = log_dir / f"{session_id}_live_cnn_viewer_log.csv"

    thresholds = ViewerThresholds(
        no_signal_ratio=args.no_signal_ratio,
        valid_signal_ratio=args.valid_signal_ratio,
        overload_peak=args.overload_peak,
        overload_clip_ratio=args.overload_clip_ratio,
    )

    block_size = int(infer_receiver_value(receiver_cfg, "block_size", args.block_size or 16384))
    center_freq = infer_receiver_value(receiver_cfg, "center_freq", args.center_freq)
    sample_rate = infer_receiver_value(receiver_cfg, "sample_rate", args.sample_rate)
    rf_bandwidth = infer_receiver_value(receiver_cfg, "rf_bandwidth", args.rf_bandwidth)
    gain = infer_receiver_value(receiver_cfg, "gain", args.gain)
    nperseg, noverlap, nfft = infer_stft_params(ml_cfg, args)

    cnn_device = resolve_cnn_device(args.cnn_device)
    cnn_model = None
    if args.decision_mode != "none":
        cnn_model = load_binary_cnn_model(args.cnn_model, cnn_device)

    temporal_history: deque[int] = deque(maxlen=args.temporal_window)

    print("=== Live CNN Spectrogram Viewer v1 ===")
    print(f"session_id         : {session_id}")
    print(f"csv_log            : {csv_path}")
    print(f"center_freq        : {center_freq}")
    print(f"sample_rate        : {sample_rate}")
    print(f"rf_bandwidth       : {rf_bandwidth}")
    print(f"gain               : {gain}")
    print(f"block_size         : {block_size}")
    print(f"rx_index           : {args.rx_index}")
    print(f"distance_m         : {args.distance_m}")
    print(f"memo               : {args.memo}")
    print(f"blocks_per_update  : {args.blocks_per_update}")
    print(f"stft               : nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}")
    print(f"decision_mode      : {args.decision_mode}")
    print(f"cnn_model          : {args.cnn_model or 'NA'}")
    print(f"cnn_device         : {cnn_device}")
    print(f"drone_threshold    : {args.drone_threshold}")
    print(f"threshold_g25/g30  : {args.drone_threshold_g25} / {args.drone_threshold_g30}")
    print(f"temporal_window    : {args.temporal_window}")
    print(f"candidate/confirmed: {args.candidate_vote_k} / {args.confirmed_vote_k}")
    print(f"enable_aoa         : {args.enable_aoa}")
    print(f"aoa_spacing_m      : {args.aoa_antenna_spacing_m}")
    print(f"aoa_calib_deg      : {args.aoa_calibration_deg}")
    print(f"aoa_phase_offset   : {args.aoa_phase_offset_rad}")
    print(f"aoa_auto_phase_cal : {args.aoa_auto_phase_calibration}")
    print("note               : YAML is loaded once at startup. Restart to apply changes.")
    print("Press Ctrl+C to stop.")

    plt = prepare_matplotlib(args.no_display)
    image_handle = None
    text_handle = None
    if not args.no_display:
        plt.ion()
        _, ax = plt.subplots(figsize=(12, 7))
        text_handle = ax.text(
            1.02,
            0.5,
            "starting...",
            transform=ax.transAxes,
            va="center",
            fontsize=7,
        )
    else:
        # Dummy text handle for no-display mode.
        class _TextHandle:
            def set_text(self, _text: str) -> None:
                return None

        text_handle = _TextHandle()

    receiver = build_receiver(receiver_cfg)

    if args.enable_aoa and args.aoa_auto_phase_calibration:
        args.aoa_phase_offset_rad = calibrate_aoa_phase_offset(
            receiver,
            block_size=int(block_size),
            calibration_blocks=int(args.aoa_calibration_blocks),
            min_coherence=float(args.aoa_min_coherence),
        )

    gain_state: dict[str, Any] | None = None
    if not args.no_display:
        gain_state = setup_gain_widgets(
            plt.gcf(),
            receiver,
            initial_gain=float(gain),
            initial_distance=args.distance_m,
            initial_memo=args.memo,
        )

    update_index = 0
    aoa_history: list[float] = []

    try:
        while args.max_updates is None or update_index < args.max_updates:
            loop_start = time.perf_counter()

            if gain_state is not None:
                update_control_state_from_widgets(gain_state)
                gain = gain_state["gain"]
                current_distance_m = gain_state["distance_m"]
                current_memo = gain_state["memo"]
            else:
                current_distance_m = args.distance_m
                current_memo = args.memo

            if gain_state is not None and gain_state.get("paused", False):
                plt.pause(0.05)
                time.sleep(0.05)
                continue

            blocks: list[np.ndarray] = []
            raw_features_list: list[RawFeatures] = []

            for _ in range(args.blocks_per_update):
                block = receiver.read_block(block_size)
                features = compute_raw_features(
                    block,
                    rx_index=args.rx_index,
                    frame_size=args.frame_size,
                    hop_size=args.hop_size,
                    overload_peak=args.overload_peak,
                )
                blocks.append(block)
                raw_features_list.append(features)

            selected_idx = select_representative_block(
                blocks,
                raw_features_list,
                select_policy=args.select_policy,
            )
            selected_block = blocks[selected_idx]
            raw_features = raw_features_list[selected_idx]
            status, suggestion = classify_signal_status(raw_features, thresholds)

            spec: np.ndarray | None = None
            cnn_features: dict[str, float]
            if status in {"WEAK_SIGNAL", "VALID_SIGNAL", "OVERLOAD"}:
                spec = compute_cnn_input_spectrogram(
                    selected_block,
                    rx_index=args.rx_index,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                )
                cnn_features = asdict(compute_cnn_spec_features(spec))
            else:
                cnn_features = empty_cnn_features()

            cnn_prob_drone = math.nan
            cnn_threshold = math.nan
            cnn_raw_decision = "NA"
            temporal_history_text = ""
            candidate_status = False
            confirmed_status = False
            final_decision = "NA"

            if args.decision_mode != "none":
                cnn_threshold = select_drone_threshold(
                    args.decision_mode,
                    gain,
                    args.drone_threshold,
                    args.drone_threshold_g25,
                    args.drone_threshold_g30,
                )

                if spec is not None and cnn_model is not None:
                    cnn_prob_drone = infer_drone_probability(cnn_model, spec, cnn_device)
                    raw_is_drone = int(cnn_prob_drone >= cnn_threshold)
                else:
                    raw_is_drone = 0

                cnn_raw_decision = "Drone" if raw_is_drone else "NonDrone"

                if args.reset_temporal_on_no_signal and status == "NO_SIGNAL":
                    temporal_history.clear()

                if args.decision_mode in {"temporal", "hybrid"}:
                    recent, candidate_status, confirmed_status, final_decision = update_temporal_decision(
                        temporal_history,
                        raw_is_drone,
                        args.temporal_window,
                        args.candidate_vote_k,
                        args.confirmed_vote_k,
                    )
                    temporal_history_text = "".join(str(x) for x in recent)

                    if args.show_candidate_as_drone and candidate_status:
                        final_decision = "Drone-like Candidate"
                else:
                    final_decision = cnn_raw_decision

            if args.enable_aoa:
                aoa_ok, aoa_gate_status = should_compute_aoa(
                    args,
                    status=status,
                    raw_features=raw_features,
                    cnn_raw_decision=cnn_raw_decision,
                    final_decision=final_decision,
                )

                if aoa_ok:
                    aoa_features = compute_aoa_features(
                        selected_block,
                        center_freq_hz=float(center_freq),
                        antenna_spacing_m=float(args.aoa_antenna_spacing_m),
                        calibration_deg=float(args.aoa_calibration_deg),
                        min_coherence=float(args.aoa_min_coherence),
                        phase_offset_rad=float(args.aoa_phase_offset_rad),
                    )
                else:
                    aoa_features = empty_aoa_features(args)
                    aoa_features["aoa_status"] = aoa_gate_status
            else:
                aoa_features = empty_aoa_features(args)

            if args.enable_aoa:
                aoa_features = update_aoa_smoothing(
                    aoa_features,
                    aoa_history,
                    window=int(args.aoa_smooth_window),
                    min_valid=int(args.aoa_smooth_min_valid),
                )

            processing_time_sec = time.perf_counter() - loop_start
            latency_sec = processing_time_sec

            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "session_id": session_id,
                "update_index": update_index,
                "center_freq": center_freq,
                "sample_rate": sample_rate,
                "rf_bandwidth": rf_bandwidth,
                "block_size": block_size,
                "rx_index": args.rx_index,
                "gain": gain,
                "distance_m": current_distance_m,
                "memo": current_memo,
                "blocks_per_update": args.blocks_per_update,
                "selected_block_index": selected_idx,
                "select_policy": args.select_policy,
                "status": status,
                "suggestion": suggestion,
                **asdict(raw_features),
                **cnn_features,
                "decision_mode": args.decision_mode,
                "cnn_model_path": args.cnn_model,
                "cnn_prob_drone": cnn_prob_drone,
                "cnn_threshold": cnn_threshold,
                "cnn_raw_decision": cnn_raw_decision,
                "temporal_window": args.temporal_window,
                "candidate_vote_k": args.candidate_vote_k,
                "confirmed_vote_k": args.confirmed_vote_k,
                "temporal_history": temporal_history_text,
                "candidate_status": candidate_status,
                "confirmed_status": confirmed_status,
                "final_decision": final_decision,
                "latency_sec": latency_sec,
                "processing_time_sec": processing_time_sec,
                **aoa_features,
            }

            if gain_state is not None:
                match_result = evaluate_feature_match(row, gain_state)
                row.update(match_result)
                update_feature_widget_text(gain_state, row, match_result)
            else:
                row.update(empty_feature_match_result())

            append_csv_log(csv_path, row)
            print_update(row)

            title = (
                f"Live CNN Spectrogram | {status} | {final_decision} | "
                f"gain={gain} | d={current_distance_m}m | update={update_index}"
            )
            side_text = build_side_text(row)

            if not args.no_display:
                image_handle, text_handle = update_display(
                    plt, image_handle, text_handle, spec, title, side_text
                )
            if args.save_latest:
                save_latest_image(plt, latest_image_dir, spec, title, side_text)

            update_index += 1
            elapsed = time.perf_counter() - loop_start
            sleep_sec = max(0.0, args.update_interval_sec - elapsed)
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        close = getattr(receiver, "close", None)
        if callable(close):
            close()
        if not args.no_display:
            plt.ioff()
        print(f"CSV log saved to: {csv_path}")
        if args.save_latest:
            print(f"Latest image dir: {latest_image_dir}")


if __name__ == "__main__":
    main()
