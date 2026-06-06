from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core import load_all_configs
from src.features.spectrogram import compute_stft_branch
from src.ml.runtime_decision import load_runtime_decision_config, select_drone_threshold
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver.pluto_receiver import PlutoReceiver
from src.receiver.factory import build_receiver as build_receiver_from_config
from src.runtime.raw_noise_gate import RawNoiseGate
from src.viewer import (
    AoARuntime,
    CNNRuntime,
    OpenCVRenderer,
    ViewerState,
    append_viewer_csv,
    compute_raw_features,
)


@dataclass
class SectorLockState:
    locked_sector_name: str | None = None
    locked_sector_label_deg: float | None = None
    status: str = "no_signal"
    hold_count: int = 0
    no_signal_count: int = 0
    reason: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimental AoA fixed-bin sector stabilizer viewer."
    )
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--source-type", choices=("sdr", "sim", "file"), default=None)
    p.add_argument("--file-path", default=None)
    p.add_argument("--disable-raw-gate", action="store_true")
    p.add_argument("--sector-config", default="configs/aoa_sector.yaml")
    p.add_argument("--uri", default=None)
    p.add_argument("--center-freq", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--rf-bandwidth", type=int, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--gain-step", type=float, default=1.0)
    p.add_argument("--min-gain", type=float, default=0.0)
    p.add_argument("--max-gain", type=float, default=73.0)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--num-channels", type=int, choices=(1, 2), default=None)
    p.add_argument("--rx-index", type=int, default=None)
    p.add_argument("--blocks-per-update", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--preset", default=None)
    p.add_argument("--mode", default="sector")
    p.add_argument("--model", default=None)
    p.add_argument("--cnn-backend", default=None)
    p.add_argument("--cnn-device", default=None)
    p.add_argument("--cnn-dummy-class-name", default=None)
    p.add_argument("--cnn-dummy-confidence", type=float, default=1.0)
    p.add_argument("--target-fps", type=float, default=5.0)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--overlay-panel-width", type=int, default=560)
    p.add_argument("--disable-dc-offset-removal", action="store_true")
    p.add_argument("--disable-cli-log", action="store_true")
    p.add_argument("--cli-log-every-n", type=int, default=None)
    p.add_argument("--distance-m", type=float, default=0.0)
    p.add_argument("--memo", default="")
    return p.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def csv_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def viewer_backend(value: Any) -> str:
    backend = str(value or "dummy").lower().strip()
    if backend in {"binary_flat", "rf4_binary", "drone_binary"}:
        return "binary"
    return backend


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    configs = load_all_configs(args.config_dir)
    sector_root = load_yaml(args.sector_config).get("aoa_sector", {}) or {}

    receiver_cfg = configs.get("receiver", {}) or {}
    sdr_cfg = receiver_cfg.get("sdr", {}) or {}
    ml_cfg = configs.get("ml", {}) or {}
    aoa_cfg = configs.get("aoa", {}) or {}
    ui_cfg = configs.get("ui", {}) or {}

    inference_cfg = ml_cfg.get("inference", {}) or {}
    stft_cfg = ml_cfg.get("stft", {}) or {}
    cnn_input_cfg = ml_cfg.get("cnn_input", {}) or {}
    temporal_cfg = inference_cfg.get("temporal_voting", {}) or {}
    live_cfg = (ui_cfg.get("live_rf_viewer", {}) or {})

    sector_runtime = sector_root.get("runtime", {}) or {}
    sector_quality = sector_root.get("quality_gate", {}) or {}
    sector_hold = sector_root.get("hold", {}) or {}
    sector_profile = sector_root.get("profile_save", {}) or {}

    args.source_type = str(args.source_type or receiver_cfg.get("source_type", "sdr")).strip().lower()
    args.file_path = args.file_path or receiver_cfg.get("file_path", "data/raw_iq/test.npy")

    args.uri = args.uri or sdr_cfg.get("uri", "ip:192.168.2.1")
    args.center_freq = int(args.center_freq or sdr_cfg.get("center_freq", receiver_cfg.get("center_freq", 2450000000)))
    args.sample_rate = int(args.sample_rate or sdr_cfg.get("sample_rate", receiver_cfg.get("sample_rate", 5000000)))
    args.rf_bandwidth = args.rf_bandwidth or sdr_cfg.get("rf_bandwidth", receiver_cfg.get("rf_bandwidth"))
    args.gain = float(args.gain if args.gain is not None else sdr_cfg.get("gain", 30.0))
    args.block_size = int(args.block_size or sdr_cfg.get("block_size", receiver_cfg.get("block_size", ml_cfg.get("block_size", 16384))))
    args.num_channels = int(args.num_channels or receiver_cfg.get("num_channels", 2))
    args.rx_index = int(args.rx_index if args.rx_index is not None else cnn_input_cfg.get("rx_index", 0))

    args.blocks_per_update = int(
        args.blocks_per_update
        or sector_runtime.get("blocks_per_update")
        or live_cfg.get("blocks_per_update", 20)
    )
    args.top_k = int(args.top_k or sector_runtime.get("top_k", 5))
    args.cli_log_every_n = int(args.cli_log_every_n or live_cfg.get("cli_log_every_n", 1))

    decision_cfg = load_runtime_decision_config(ml_cfg)

    args.model = args.model or inference_cfg.get("model_path")
    args.cnn_backend = viewer_backend(args.cnn_backend or inference_cfg.get("backend", "binary"))
    args.cnn_device = args.cnn_device or inference_cfg.get("device", "cpu")
    args.class_names = csv_values(ml_cfg.get("class_names", ["NotDrone", "Drone"]))
    args.positive_class_names = csv_values([ml_cfg.get("positive_class", temporal_cfg.get("positive_class", "Drone"))])
    args.negative_class = str(ml_cfg.get("negative_class", "NotDrone"))
    args.cnn_dummy_class_name = args.cnn_dummy_class_name or args.negative_class
    args.cnn_smooth_window = int(temporal_cfg.get("window_size", 5))
    args.cnn_confirm_votes = int(temporal_cfg.get("confirmed_vote_k", 3))
    args.cnn_threshold = float(select_drone_threshold(decision_cfg, args.gain))

    args.nperseg = int(stft_cfg.get("nperseg", 128))
    args.noverlap = int(stft_cfg.get("noverlap", 96))
    args.nfft = int(stft_cfg.get("nfft", 128))
    args.window = str(stft_cfg.get("window", "hann"))

    args.aoa_ref_channel = int(aoa_cfg.get("ref_channel", 0))
    args.aoa_target_channel = int(aoa_cfg.get("target_channel", 1))
    args.aoa_antenna_spacing_m = float(aoa_cfg.get("antenna_spacing_m", 0.06))
    args.aoa_speed_of_light = float(aoa_cfg.get("speed_of_light", 300000000.0))
    args.aoa_coherence_threshold = float((aoa_cfg.get("coherence", {}) or {}).get("threshold", 0.6))
    args.aoa_energy_percentile = float((aoa_cfg.get("coherence", {}) or {}).get("energy_percentile", 75.0))
    args.phase_gain_profile = str((aoa_cfg.get("calibration", {}) or {}).get("phase_gain_profile", "outputs/calibration/phase_gain_by_gain_latest.json"))

    args.sector_root = sector_root
    args.sector_quality = sector_quality
    args.sector_hold = sector_hold
    args.sector_profile = sector_profile
    args.sector_preset_name = args.preset or sector_root.get("active_preset", "fixed_bins_7sector")
    args.sector_preset = (sector_root.get("presets", {}) or {})[args.sector_preset_name]
    args.decision_cfg = decision_cfg

    print("=== AoA Sector Experiment Config ===")
    print(f"cf              : {args.center_freq}")
    print(f"sample_rate     : {args.sample_rate}")
    print(f"gain            : {args.gain}")
    print(f"source_type     : {args.source_type}")
    print(f"file_path       : {args.file_path}")
    print(f"block_size      : {args.block_size}")
    print(f"blocks/update   : {args.blocks_per_update}")
    print(f"top_k           : {args.top_k}")
    print(f"sector_preset   : {args.sector_preset_name}")
    print(f"cnn_backend     : {args.cnn_backend}")
    print(f"cnn_model       : {args.model}")
    print(f"cnn_threshold   : {args.cnn_threshold}")
    print("====================================")

    return args


def build_receiver(args: argparse.Namespace):
    """
    Build input receiver for the sector experiment.

    source_type:
    - sdr  : PlutoReceiver
    - sim  : SimReceiver
    - file : RawFileReceiver

    This keeps the experiment runnable without Pluto SDR.
    """
    configs = load_all_configs(args.config_dir)
    receiver_cfg = dict(configs.get("receiver", {}) or {})
    sdr_cfg = dict(receiver_cfg.get("sdr", {}) or {})

    receiver_cfg["source_type"] = str(args.source_type)
    receiver_cfg["sample_rate"] = int(args.sample_rate)
    receiver_cfg["center_freq"] = int(args.center_freq)
    receiver_cfg["block_size"] = int(args.block_size)
    receiver_cfg["num_samples"] = int(args.block_size)
    receiver_cfg["num_channels"] = int(args.num_channels)

    if args.source_type == "file":
        receiver_cfg["file_path"] = str(args.file_path)

    if args.source_type == "sdr":
        channels = [0, 1] if int(args.num_channels) == 2 else [0]
        sdr_cfg.update(
            {
                "uri": args.uri,
                "sample_rate": int(args.sample_rate),
                "center_freq": int(args.center_freq),
                "rf_bandwidth": args.rf_bandwidth,
                "gain": float(args.gain),
                "num_samples": int(args.block_size),
                "block_size": int(args.block_size),
                "channels": channels,
            }
        )
        receiver_cfg["sdr"] = sdr_cfg

    return build_receiver_from_config(receiver_cfg)


def build_cnn_runtime(args: argparse.Namespace) -> CNNRuntime:
    return CNNRuntime(
        model_path=args.model,
        backend=args.cnn_backend,
        device=args.cnn_device,
        class_names=args.class_names,
        positive_class_names=args.positive_class_names,
        rx_index=args.rx_index,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        window=args.window,
        confidence_threshold=args.cnn_threshold,
        smooth_window=args.cnn_smooth_window,
        confirm_votes=args.cnn_confirm_votes,
        dummy_class_name=args.cnn_dummy_class_name,
        dummy_confidence=float(args.cnn_dummy_confidence),
    )


def build_aoa_runtime(args: argparse.Namespace) -> AoARuntime:
    if args.num_channels < 2:
        raise ValueError("AoA sector experiment requires 2 channels.")
    return AoARuntime(
        carrier_freq=float(args.center_freq),
        sample_rate=float(args.sample_rate),
        antenna_spacing_m=float(args.aoa_antenna_spacing_m),
        speed_of_light=float(args.aoa_speed_of_light),
        phase_gain_profile_json=args.phase_gain_profile,
        ref_channel=int(args.aoa_ref_channel),
        target_channel=int(args.aoa_target_channel),
        coherence_threshold=float(args.aoa_coherence_threshold),
        energy_percentile=float(args.aoa_energy_percentile),
        nperseg=int(args.nperseg),
        noverlap=int(args.noverlap),
        nfft=int(args.nfft),
        window=args.window,
        compute_coherence=True,
        gain=float(args.gain),
    )


def apply_receiver_gain(receiver: Any, gain: float) -> None:
    """
    Apply hardware gain only when the receiver supports it.

    SimReceiver / RawFileReceiver do not have real RF gain control, so they only
    keep the viewer state gain value.
    """
    if hasattr(receiver, "gain"):
        try:
            receiver.gain = float(gain)
        except Exception:
            pass

    if hasattr(receiver, "_set_channel_gain") and hasattr(receiver, "channels"):
        for ch in receiver.channels:
            receiver._set_channel_gain(ch)


def select_iq_channel(iq: np.ndarray, rx_index: int) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        return arr.astype(np.complex64, copy=False)
    return arr[int(rx_index)].astype(np.complex64, copy=False)


def safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def select_topk_indices(
    raw_gate_results: list[Any],
    top_k: int,
) -> list[int]:
    passed = [
        i for i, r in enumerate(raw_gate_results)
        if bool((not r.enabled) or r.passed)
    ]

    source = passed if passed else list(range(len(raw_gate_results)))

    ranked = sorted(
        source,
        key=lambda i: safe_float(raw_gate_results[i].score_max),
        reverse=True,
    )
    return ranked[: max(1, int(top_k))]


def run_cnn_raw_no_history(
    cnn_runtime: CNNRuntime,
    iq: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    iq_1d = select_iq_channel(iq, cnn_runtime.rx_index)
    stft = compute_stft_branch(
        iq_block=iq_1d,
        sample_rate=float(cnn_runtime.sample_rate),
        nperseg=int(cnn_runtime.nperseg),
        noverlap=int(cnn_runtime.noverlap),
        nfft=int(cnn_runtime.nfft),
        window=cnn_runtime.window,
    )
    image = stft.cnn_spectrogram

    pred = cnn_runtime.classifier.predict(image)
    raw = pred.to_dict()

    raw_class = str(raw["class_name"])
    raw_conf = float(raw["confidence"])
    positive = {x.strip().lower() for x in cnn_runtime.positive_class_names}
    is_candidate = (
        raw_class.strip().lower() in positive
        and raw_conf >= float(cnn_runtime.confidence_threshold)
    )

    return image, {
        "cnn_raw_class_name": raw_class,
        "cnn_raw_class_index": int(raw["class_index"]),
        "cnn_raw_confidence": raw_conf,
        "cnn_probabilities": list(raw.get("probabilities", [])),
        "cnn_candidate": bool(is_candidate),
        "cnn_confidence_threshold": float(cnn_runtime.confidence_threshold),
    }


def update_cnn_history_once(
    cnn_runtime: CNNRuntime,
    raw_cnn: dict[str, Any],
) -> dict[str, Any]:
    cnn_runtime.history.append(
        {
            "class_name": raw_cnn["cnn_raw_class_name"],
            "confidence": float(raw_cnn["cnn_raw_confidence"]),
            "candidate": bool(raw_cnn["cnn_candidate"]),
        }
    )
    smooth = cnn_runtime._build_smoothing_result()

    result = dict(raw_cnn)
    result.update(
        {
            "cnn_confirmed": bool(smooth["confirmed"]),
            "cnn_smoothed_class_name": smooth["smoothed_class_name"],
            "cnn_smoothed_confidence": float(smooth["smoothed_confidence"]),
            "cnn_positive_votes": int(smooth["positive_votes"]),
            "cnn_history_size": int(len(cnn_runtime.history)),
            "cnn_smooth_window": int(cnn_runtime.smooth_window),
            "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
            "cnn_backend": str(cnn_runtime.backend),
            "cnn_model_path": str(cnn_runtime.model_path) if cnn_runtime.model_path else "",
            "cnn_rx_index": int(cnn_runtime.rx_index),
        }
    )
    return result


def pick_vote_cnn_result(cnn_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cnn_items:
        return None

    positive_items = [x for x in cnn_items if bool(x["cnn"]["cnn_candidate"])]
    if positive_items:
        best = max(
            positive_items,
            key=lambda x: float(x["cnn"]["cnn_raw_confidence"]),
        )
        return best["cnn"]

    best = max(
        cnn_items,
        key=lambda x: float(x["cnn"]["cnn_raw_confidence"]),
    )
    return best["cnn"]


def map_angle_to_bin(angle_deg: float, bins: list[dict[str, Any]]) -> dict[str, Any] | None:
    for i, b in enumerate(bins):
        mn = float(b["min_deg"])
        mx = float(b["max_deg"])
        is_last = i == len(bins) - 1

        if is_last:
            inside = mn <= angle_deg <= mx
        else:
            inside = mn <= angle_deg < mx

        if inside:
            return b

    return None


def build_aoa_candidates(
    topk_items: list[dict[str, Any]],
    aoa_runtime: AoARuntime,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    q = args.sector_quality
    min_conf = float(q.get("min_cnn_confidence", 0.80))
    min_coh = float(q.get("min_stft_coherence", 0.70))
    min_angle = float(q.get("valid_angle_min_deg", -60))
    max_angle = float(q.get("valid_angle_max_deg", 60))
    require_aoa_valid = bool(q.get("require_aoa_valid", True))
    require_coh_passed = bool(q.get("require_coherence_passed", False))

    bins = list(args.sector_preset.get("bins", []) or [])
    candidates: list[dict[str, Any]] = []

    for item in topk_items:
        cnn = item["cnn"]
        if not bool(cnn.get("cnn_candidate", False)):
            continue
        if float(cnn.get("cnn_raw_confidence", 0.0)) < min_conf:
            continue

        aoa = aoa_runtime.process(item["iq"])
        angle = safe_float(aoa.get("aoa_angle_deg"), default=float("nan"))

        stft_coh = aoa.get("stft_coherence", aoa.get("coherence_like", float("nan")))
        coh = safe_float(stft_coh, default=float("nan"))
        coh_passed = bool(aoa.get("stft_coherence_passed", True))

        valid = True
        reasons: list[str] = []

        if require_aoa_valid and not bool(aoa.get("aoa_valid", False)):
            valid = False
            reasons.append("aoa_invalid")

        if math.isnan(angle) or not (min_angle <= angle <= max_angle):
            valid = False
            reasons.append("angle_out_of_range")

        if not math.isnan(coh) and coh < min_coh:
            valid = False
            reasons.append(f"low_coherence:{coh:.3f}<{min_coh:.3f}")

        if require_coh_passed and not coh_passed:
            valid = False
            reasons.append("coherence_gate_failed")

        sector = map_angle_to_bin(angle, bins) if valid else None
        if sector is None and valid:
            valid = False
            reasons.append("no_matching_sector")

        cand = {
            "block_index": int(item["index"]),
            "valid": bool(valid),
            "reason": ",".join(reasons) if reasons else "valid",
            "angle_deg": angle,
            "stft_coherence": coh,
            "raw_abs_p99": item["raw"].get("raw_abs_p99"),
            "raw_abs_p95": item["raw"].get("raw_abs_p95"),
            "raw_rms": item["raw"].get("raw_rms"),
            "frame_power_p99": item["raw"].get("frame_power_p99"),
            "cnn_confidence": cnn.get("cnn_raw_confidence"),
            "cnn_class_name": cnn.get("cnn_raw_class_name"),
            "sector_name": sector.get("name") if sector else None,
            "sector_label_deg": sector.get("label_deg") if sector else None,
            "aoa": aoa,
        }
        candidates.append(cand)

    return candidates


def update_sector_lock(
    lock: SectorLockState,
    candidates: list[dict[str, Any]],
    raw_pass_count: int,
    cnn_drone_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    preset = args.sector_preset
    hold_cfg = args.sector_hold

    valid = [c for c in candidates if bool(c["valid"])]
    valid_count = len(valid)

    min_valid = int(preset.get("min_valid_aoa_candidates", 2))
    min_count = int(preset.get("dominant_sector_min_count", 2))
    min_ratio = float(preset.get("dominant_sector_min_ratio", 0.60))
    hold_max = int(hold_cfg.get("hold_max_updates", 8))
    no_signal_clear = int(hold_cfg.get("no_signal_clear_updates", 10))

    votes = Counter(str(c["sector_name"]) for c in valid if c.get("sector_name"))
    vote_text = ",".join(f"{k}={v}" for k, v in votes.items()) if votes else ""

    trusted = False
    instant_sector_name = None
    instant_sector_label = None
    dominant_count = 0
    dominant_ratio = 0.0

    if valid_count >= min_valid and votes:
        instant_sector_name, dominant_count = votes.most_common(1)[0]
        dominant_ratio = dominant_count / max(1, valid_count)

        if dominant_count >= min_count and dominant_ratio >= min_ratio:
            for c in valid:
                if c["sector_name"] == instant_sector_name:
                    instant_sector_label = float(c["sector_label_deg"])
                    break
            trusted = True

    if trusted:
        lock.locked_sector_name = instant_sector_name
        lock.locked_sector_label_deg = instant_sector_label
        lock.status = "trusted"
        lock.hold_count = 0
        lock.no_signal_count = 0
        lock.reason = "sector consensus passed"
    else:
        if raw_pass_count <= 0 or cnn_drone_count <= 0:
            lock.no_signal_count += 1
        else:
            lock.no_signal_count = 0

        lock.hold_count += 1

        if lock.no_signal_count >= no_signal_clear:
            lock.locked_sector_name = None
            lock.locked_sector_label_deg = None
            lock.status = "no_signal"
            lock.reason = "raw/cnn signal disappeared too long"
        elif valid_count == 0:
            lock.status = "hold_no_valid_aoa" if lock.hold_count <= hold_max else "uncertain"
            lock.reason = "no valid AoA candidates"
        else:
            lock.status = "hold_no_consensus" if lock.hold_count <= hold_max else "uncertain"
            lock.reason = "sector votes scattered"

    angles = [float(c["angle_deg"]) for c in valid if not math.isnan(float(c["angle_deg"]))]
    cohs = [float(c["stft_coherence"]) for c in valid if not math.isnan(float(c["stft_coherence"]))]
    p99s = [float(c["raw_abs_p99"]) for c in valid if c.get("raw_abs_p99") is not None]

    return {
        "sector_status": lock.status,
        "instant_sector_name": instant_sector_name or "",
        "instant_sector_label_deg": instant_sector_label if instant_sector_label is not None else "",
        "locked_sector_name": lock.locked_sector_name or "",
        "locked_sector_label_deg": lock.locked_sector_label_deg if lock.locked_sector_label_deg is not None else "",
        "hold_count": int(lock.hold_count),
        "no_signal_count": int(lock.no_signal_count),
        "valid_aoa_count": int(valid_count),
        "sector_votes": vote_text,
        "dominant_sector_count": int(dominant_count),
        "dominant_sector_ratio": float(dominant_ratio),
        "angle_median": float(np.median(angles)) if angles else "",
        "angle_spread": float(np.max(angles) - np.min(angles)) if len(angles) >= 2 else "",
        "median_coherence": float(np.median(cohs)) if cohs else "",
        "median_raw_p99": float(np.median(p99s)) if p99s else "",
        "reason": lock.reason,
    }


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None or value == "":
            return "n/a"
        v = float(value)
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}g}"
    except Exception:
        return str(value)


def build_overlay(
    state: ViewerState,
    raw: dict[str, Any],
    cnn_result: dict[str, Any],
    sector: dict[str, Any],
    raw_pass_count: int,
    cnn_drone_count: int,
    topk_count: int,
    profile_csv: Path,
) -> list[str]:
    return [
        f"[RF] idx={state.update_index} LIVE sector",
        f"cf={state.center_freq / 1e6:.3f}M sr={state.sample_rate / 1e6:.3f}M g={state.gain:.1f}",
        f"[TOPK] raw_pass={raw_pass_count} topk={topk_count} drone={cnn_drone_count}",
        f"[CNN] {cnn_result.get('cnn_raw_class_name', 'n/a')} conf={fmt(cnn_result.get('cnn_raw_confidence'), 3)}",
        f"[VOTE] {cnn_result.get('cnn_positive_votes', 0)}/{cnn_result.get('cnn_confirm_votes', 0)} confirmed={cnn_result.get('cnn_confirmed', False)}",
        f"[SECTOR] status={sector.get('sector_status')}",
        f"instant={sector.get('instant_sector_name') or 'None'} locked={sector.get('locked_sector_name') or 'None'}",
        f"validAoA={sector.get('valid_aoa_count')} votes={sector.get('sector_votes') or 'None'}",
        f"angle_med={fmt(sector.get('angle_median'), 3)} spread={fmt(sector.get('angle_spread'), 3)}",
        f"coh_med={fmt(sector.get('median_coherence'), 3)} raw_p99_med={fmt(sector.get('median_raw_p99'), 4)}",
        f"reason={sector.get('reason')}",
        f"save={profile_csv.name}",
        "q quit | p pause | [/] gain | s save sector",
    ]


def append_sector_profile(path: Path, row: dict[str, Any]) -> None:
    append_viewer_csv(path, row)


def run() -> int:
    args = apply_defaults(parse_args())

    state = ViewerState(
        mode="sector",
        gain=args.gain,
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        distance_m=args.distance_m,
        memo=args.memo,
        target_fps=args.target_fps,
    )

    raw_gate = RawNoiseGate(detect_config_path=Path(args.config_dir) / "detect.yaml")
    if bool(args.disable_raw_gate):
        raw_gate.enabled = False
        print("[DRY-RUN] raw noise gate disabled by --disable-raw-gate")

    receiver = build_receiver(args)
    cnn_runtime = build_cnn_runtime(args)
    aoa_runtime = build_aoa_runtime(args)

    renderer = OpenCVRenderer(
        window_name="AoA Sector Experiment",
        target_fps=args.target_fps,
        display_scale=args.display_scale,
        overlay_mode="right",
        overlay_width=args.overlay_panel_width,
    )

    out_dir = Path(args.sector_profile.get("output_dir", "outputs/aoa_sector_profiles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_csv = out_dir / f"{ts}_gain{args.gain:g}_cf{args.center_freq}_sector_profile.csv"

    lock = SectorLockState()
    last_image = np.zeros((128, 509), dtype=np.float32)
    last_row: dict[str, Any] | None = None

    try:
        while state.running:
            if state.paused:
                overlay = ["PAUSED", "q quit | p pause"]
                key = renderer.render(last_image, overlay)
                if key == "quit":
                    state.running = False
                elif key == "pause":
                    state.toggle_pause()
                continue

            blocks: list[np.ndarray] = []
            raws: list[dict[str, Any]] = []
            gate_results: list[Any] = []

            for _ in range(max(1, int(args.blocks_per_update))):
                try:
                    iq = receiver.read_block(args.block_size)
                except EOFError:
                    if hasattr(receiver, "reset"):
                        receiver.reset()
                        iq = receiver.read_block(args.block_size)
                    else:
                        raise

                if not args.disable_dc_offset_removal:
                    iq = remove_dc_offset(iq, axis=-1)

                raw = compute_raw_features(iq)
                gate = raw_gate.evaluate(iq, gain=state.gain)

                blocks.append(iq)
                raws.append(raw)
                gate_results.append(gate)

            state.mark_update()

            raw_pass_indices = [
                i for i, r in enumerate(gate_results)
                if bool((not r.enabled) or r.passed)
            ]
            raw_pass_count = len(raw_pass_indices)

            topk_idx = select_topk_indices(gate_results, args.top_k)

            topk_items: list[dict[str, Any]] = []
            for idx in topk_idx:
                image, cnn_raw = run_cnn_raw_no_history(cnn_runtime, blocks[idx])
                topk_items.append(
                    {
                        "index": idx,
                        "iq": blocks[idx],
                        "raw": raws[idx],
                        "gate": gate_results[idx],
                        "image": image,
                        "cnn": cnn_raw,
                    }
                )

            last_image = topk_items[0]["image"] if topk_items else last_image

            if raw_pass_count <= 0:
                if raw_gate.reset_cnn_history_on_fail():
                    cnn_runtime.reset_history()
                cnn_result = {
                    "cnn_raw_class_name": "RAW_GATE_BLOCKED",
                    "cnn_raw_confidence": 0.0,
                    "cnn_candidate": False,
                    "cnn_confirmed": False,
                    "cnn_positive_votes": 0,
                    "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
                    "cnn_skipped": True,
                    "cnn_skipped_reason": "raw_noise_gate_failed",
                }
                aoa_candidates: list[dict[str, Any]] = []
            else:
                vote_raw = pick_vote_cnn_result(topk_items)
                cnn_result = update_cnn_history_once(cnn_runtime, vote_raw) if vote_raw else {
                    "cnn_raw_class_name": "NO_CNN_RESULT",
                    "cnn_raw_confidence": 0.0,
                    "cnn_candidate": False,
                    "cnn_confirmed": False,
                    "cnn_positive_votes": 0,
                    "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
                }
                aoa_candidates = build_aoa_candidates(topk_items, aoa_runtime, args)

            cnn_drone_count = sum(
                1 for item in topk_items
                if bool(item["cnn"].get("cnn_candidate", False))
            )

            sector = update_sector_lock(
                lock=lock,
                candidates=aoa_candidates,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                args=args,
            )

            selected_raw = raws[topk_idx[0]] if topk_idx else {}
            selected_raw.update(
                {
                    "selected_block_index": int(topk_idx[0]) if topk_idx else -1,
                    "blocks_per_update": int(args.blocks_per_update),
                    "top_k": int(args.top_k),
                    "raw_gate_pass_count": int(raw_pass_count),
                    "cnn_topk_count": int(len(topk_items)),
                    "cnn_drone_count": int(cnn_drone_count),
                }
            )

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "update_idx": int(state.update_index),
                "gain": float(state.gain),
                "center_freq": int(state.center_freq),
                "sample_rate": int(state.sample_rate),
                "blocks_per_update": int(args.blocks_per_update),
                "top_k": int(args.top_k),
                "sector_preset": str(args.sector_preset_name),
                "distance_m": float(state.distance_m),
                "memo": str(state.memo),
                **selected_raw,
                **cnn_result,
                **sector,
            }
            last_row = row

            if not args.disable_cli_log and state.update_index % int(args.cli_log_every_n) == 0:
                print(
                    f"[SECTOR] idx={state.update_index} "
                    f"raw_pass={raw_pass_count} topk={len(topk_items)} drone={cnn_drone_count} "
                    f"status={sector['sector_status']} "
                    f"instant={sector['instant_sector_name'] or 'None'} "
                    f"locked={sector['locked_sector_name'] or 'None'} "
                    f"valid={sector['valid_aoa_count']} "
                    f"votes={sector['sector_votes'] or 'None'} "
                    f"reason={sector['reason']}",
                    flush=True,
                )

            overlay = build_overlay(
                state=state,
                raw=selected_raw,
                cnn_result=cnn_result,
                sector=sector,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                topk_count=len(topk_items),
                profile_csv=profile_csv,
            )

            key = renderer.render(last_image, overlay)

            if key == "quit":
                state.running = False
            elif key == "pause":
                state.toggle_pause()
            elif key == "gain_down":
                state.step_gain(-args.gain_step, args.min_gain, args.max_gain)
                apply_receiver_gain(receiver, state.gain)
                aoa_runtime.update_gain(state.gain)
                cnn_runtime.reset_history()
            elif key == "gain_up":
                state.step_gain(args.gain_step, args.min_gain, args.max_gain)
                apply_receiver_gain(receiver, state.gain)
                aoa_runtime.update_gain(state.gain)
                cnn_runtime.reset_history()
            elif key == "save_profile" and last_row is not None:
                append_sector_profile(profile_csv, last_row)
                print(f"[SAVE] sector profile appended -> {profile_csv}", flush=True)

    except KeyboardInterrupt:
        return 130
    finally:
        receiver.close()
        renderer.close()

    return 0


if __name__ == "__main__":
    sys.exit(run())
