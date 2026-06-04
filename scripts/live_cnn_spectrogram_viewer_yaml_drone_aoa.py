#!/usr/bin/env python3
"""
YAML-driven Live CNN Spectrogram Viewer with drone-gated AoA.

Purpose:
- Keep model path / class names / thresholds / temporal voting in configs/ml.yaml only.
- Keep raw safety / overload thresholds in configs/ml.yaml only.
- Reuse the existing live_cnn_spectrogram_viewer helper functions.
- Avoid command-line hardcoded model paths and threshold defaults.

Run:
    PYTHONPATH=. python scripts/live_cnn_spectrogram_viewer_yaml.py --decision-mode hybrid --gain 30
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from scripts.live_cnn_spectrogram_viewer import (
    CSV_COLUMNS,
    ViewerThresholds,
    append_csv_log,
    build_side_text,
    classify_signal_status,
    compute_cnn_input_spectrogram,
    compute_cnn_spec_features,
    compute_raw_features,
    empty_cnn_features,
    empty_feature_match_result,
    evaluate_feature_match,
    infer_receiver_value,
    infer_stft_params,
    load_viewer_configs,
    now_session_id,
    prepare_matplotlib,
    print_update,
    save_latest_image,
    select_representative_block,
    setup_gain_widgets,
    to_project_path,
    update_control_state_from_widgets,
    update_display,
    update_feature_widget_text,
    update_gain_profile_capture,
)
from src.core import load_all_configs
from src.ml.runtime_classifier_factory import build_runtime_cnn_classifier
from src.ml.runtime_decision import (
    load_runtime_decision_config,
    select_drone_threshold,
    update_temporal_decision,
)
from src.receiver import build_receiver
from src.viewer.aoa_runtime import AoARuntime
from src.runtime.raw_noise_gate import RawNoiseGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YAML-driven live viewer for RF Drone / NotDrone binary CNN."
    )

    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--center-freq", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--rf-bandwidth", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--rx-index", type=int, default=None)
    parser.add_argument("--distance-m", type=float, default=math.nan)
    parser.add_argument("--memo", default="")

    parser.add_argument("--update-interval-sec", type=float, default=1.0)
    parser.add_argument("--blocks-per-update", type=int, default=20)
    parser.add_argument("--select-policy", default="max_signal_ratio")
    parser.add_argument("--max-updates", type=int, default=None)

    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)

    # These can still be overridden for emergency debugging, but their defaults are YAML.
    parser.add_argument("--no-signal-ratio", type=float, default=None)
    parser.add_argument("--valid-signal-ratio", type=float, default=None)
    parser.add_argument("--overload-peak", type=float, default=None)
    parser.add_argument("--overload-clip-ratio", type=float, default=None)

    parser.add_argument("--stft-nperseg", type=int, default=None)
    parser.add_argument("--stft-noverlap", type=int, default=None)
    parser.add_argument("--stft-nfft", type=int, default=None)

    parser.add_argument("--log-dir", default="outputs/live_viewer/logs")
    parser.add_argument("--latest-image-dir", default="outputs/live_viewer/latest")
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--no-display", action="store_true")

    parser.add_argument(
        "--decision-mode",
        default="hybrid",
        choices=["none", "raw", "gain-aware", "temporal", "hybrid"],
        help="Default hybrid. Model path/threshold/voting are loaded from configs/ml.yaml.",
    )
    parser.add_argument("--reset-temporal-on-no-signal", action="store_true")
    parser.add_argument("--show-candidate-as-drone", action="store_true")

    # Drone-gated AoA options.
    # AoA is computed only when CNN temporal decision reaches confirmed Drone.
    parser.add_argument("--disable-aoa", action="store_true")
    parser.add_argument("--phase-gain-profile", default="outputs/calibration/phase_gain_by_gain_latest.json")
    parser.add_argument("--aoa-ref-channel", type=int, default=0)
    parser.add_argument("--aoa-target-channel", type=int, default=1)
    parser.add_argument("--aoa-antenna-spacing-m", type=float, default=None)
    parser.add_argument("--aoa-coherence-threshold", type=float, default=None)
    parser.add_argument("--aoa-energy-percentile", type=float, default=None)
    parser.add_argument("--aoa-disable-stft-coherence", action="store_true")

    return parser.parse_args()


RAW_GATE_EXTRA_CSV_COLUMNS = [
    "raw_gate_enabled",
    "raw_gate_passed",
    "raw_gate_label",
    "raw_gate_score_max",
    "raw_gate_score_median",
    "raw_gate_noise_floor",
    "raw_gate_threshold",
    "raw_gate_threshold_multiplier",
    "raw_gate_detection_ratio",
    "raw_gate_min_detection_ratio",
    "raw_gate_matched_gain",
    "raw_gate_matched_by",
    "raw_gate_reason",
]

AOA_EXTRA_CSV_COLUMNS = [
    "aoa_enabled",
    "aoa_gate_passed",
    "aoa_angle_deg",
    "aoa_valid",
    "aoa_skipped_reason",
    "aoa_stft_coherence",
    "aoa_stft_coherence_passed",
    "phase_diff_corrected_deg",
]


def _resolve_rx_index(ml_cfg: dict[str, Any], args: argparse.Namespace) -> int:
    if args.rx_index is not None:
        return int(args.rx_index)
    cnn_input_cfg = ml_cfg.get("cnn_input", {}) or {}
    return int(cnn_input_cfg.get("rx_index", 0))


def _ensure_csv_columns(row: dict[str, Any]) -> dict[str, Any]:
    columns = list(CSV_COLUMNS) + list(RAW_GATE_EXTRA_CSV_COLUMNS) + list(AOA_EXTRA_CSV_COLUMNS)
    return {key: row.get(key, "") for key in columns}


def _raw_safety_cfg(ml_cfg: dict[str, Any]) -> dict[str, Any]:
    return ml_cfg.get("raw_safety", {}) or {}


def _build_viewer_thresholds_from_yaml(
    ml_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> ViewerThresholds:
    safety = _raw_safety_cfg(ml_cfg)
    overload = safety.get("overload", {}) or {}

    no_signal_ratio = float(
        args.no_signal_ratio
        if args.no_signal_ratio is not None
        else safety.get("no_signal_ratio", 2.0)
    )
    valid_signal_ratio = float(
        args.valid_signal_ratio
        if args.valid_signal_ratio is not None
        else safety.get("valid_signal_ratio", 5.0)
    )
    overload_peak = float(
        args.overload_peak
        if args.overload_peak is not None
        else overload.get("raw_peak_overload", 30000.0)
    )
    overload_clip_ratio = float(
        args.overload_clip_ratio
        if args.overload_clip_ratio is not None
        else overload.get("clip_ratio_overload", 0.001)
    )

    return ViewerThresholds(
        no_signal_ratio=no_signal_ratio,
        valid_signal_ratio=valid_signal_ratio,
        overload_peak=overload_peak,
        overload_clip_ratio=overload_clip_ratio,
    )


def _initialize_display_window(
    plt: Any,
    image_handle: Any,
    text_handle: Any,
    nfft: int,
) -> tuple[Any, Any]:
    """Create a visible placeholder image before the first real RF frame arrives."""
    placeholder = np.zeros((int(nfft), 2), dtype=np.float32)
    image_handle, text_handle = update_display(
        plt,
        image_handle,
        text_handle,
        placeholder,
        "YAML Live CNN Spectrogram | starting...",
        "waiting for first RF block...",
    )
    plt.show(block=False)
    plt.pause(0.2)
    return image_handle, text_handle


def main() -> None:
    args = parse_args()
    receiver_cfg, ml_cfg = load_viewer_configs(args)
    all_cfg = load_all_configs(args.config_dir)
    ml_cfg = dict(all_cfg.get("ml", ml_cfg))

    decision_cfg = load_runtime_decision_config(ml_cfg)
    rx_index = _resolve_rx_index(ml_cfg, args)

    session_id = now_session_id()
    log_dir = to_project_path(args.log_dir)
    latest_image_dir = to_project_path(args.latest_image_dir)
    csv_path = log_dir / f"{session_id}_live_cnn_viewer_yaml_log.csv"

    thresholds = _build_viewer_thresholds_from_yaml(ml_cfg, args)

    block_size = int(infer_receiver_value(receiver_cfg, "block_size", args.block_size or 16384))
    center_freq = infer_receiver_value(receiver_cfg, "center_freq", args.center_freq)
    sample_rate = infer_receiver_value(receiver_cfg, "sample_rate", args.sample_rate)
    rf_bandwidth = infer_receiver_value(receiver_cfg, "rf_bandwidth", args.rf_bandwidth)
    gain = infer_receiver_value(receiver_cfg, "gain", args.gain)
    nperseg, noverlap, nfft = infer_stft_params(ml_cfg, args)

    cnn_model = None
    if args.decision_mode != "none":
        cnn_model = build_runtime_cnn_classifier(ml_cfg)

    voting_cfg = decision_cfg.temporal_voting
    temporal_history: deque[int] = deque(maxlen=int(voting_cfg.window_size))

    print("=== YAML-driven Live CNN Spectrogram Viewer ===")
    print(f"session_id         : {session_id}")
    print(f"csv_log            : {csv_path}")
    print(f"center_freq        : {center_freq}")
    print(f"sample_rate        : {sample_rate}")
    print(f"rf_bandwidth       : {rf_bandwidth}")
    print(f"gain               : {gain}")
    print(f"block_size         : {block_size}")
    print(f"rx_index           : {rx_index}")
    print(f"distance_m         : {args.distance_m}")
    print(f"memo               : {args.memo}")
    print(f"blocks_per_update  : {args.blocks_per_update}")
    print(f"stft               : nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}")
    print(f"decision_mode      : {args.decision_mode}")
    print(f"backend            : {decision_cfg.backend}")
    print(f"model_path         : {decision_cfg.model_path}")
    print(f"positive_class     : {decision_cfg.positive_class}")
    print(f"threshold default  : {decision_cfg.default_drone_threshold}")
    print(f"temporal           : window={voting_cfg.window_size}, candidate={voting_cfg.candidate_vote_k}, confirmed={voting_cfg.confirmed_vote_k}")
    print(f"raw_safety         : no_signal={thresholds.no_signal_ratio}, valid={thresholds.valid_signal_ratio}, overload_peak={thresholds.overload_peak}, clip_ratio={thresholds.overload_clip_ratio}")
    print("note               : model/threshold/voting/raw_safety are loaded from configs/ml.yaml")

    plt = prepare_matplotlib(args.no_display)
    backend = str(plt.get_backend())
    print(f"matplotlib_backend : {backend}")
    if "agg" in backend.lower() and not args.no_display:
        print("[DISPLAY WARN] Matplotlib backend is non-interactive. Use --save-latest or install/use a GUI backend.")
    print("Press Ctrl+C to stop.")

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
        class _TextHandle:
            def set_text(self, _text: str) -> None:
                return None
        text_handle = _TextHandle()

    receiver = build_receiver(receiver_cfg)

    raw_gate = RawNoiseGate(
        detect_config_path=f"{args.config_dir}/detect.yaml",
    )
    print("raw_noise_gate    :", "enabled" if raw_gate.enabled else "disabled")
    if raw_gate.enabled:
        print(f"raw_gate_profile  : {raw_gate.noise_profile_path}")
        print(f"raw_gate_method   : {raw_gate.detector_method}")
        print(f"raw_gate_frame    : frame_size={raw_gate.frame_size}, hop_size={raw_gate.hop_size}")

    aoa_cfg = all_cfg.get("aoa", {}) or {}
    aoa_runtime = None
    if not args.disable_aoa:
        aoa_spacing_m = (
            float(args.aoa_antenna_spacing_m)
            if args.aoa_antenna_spacing_m is not None
            else float(aoa_cfg.get("antenna_spacing_m", 0.06))
        )
        aoa_coherence_threshold = (
            float(args.aoa_coherence_threshold)
            if args.aoa_coherence_threshold is not None
            else float((aoa_cfg.get("coherence", {}) or {}).get("threshold", 0.6))
        )
        aoa_energy_percentile = (
            float(args.aoa_energy_percentile)
            if args.aoa_energy_percentile is not None
            else float((aoa_cfg.get("coherence", {}) or {}).get("energy_percentile", 75.0))
        )

        aoa_runtime = AoARuntime(
            carrier_freq=float(center_freq),
            sample_rate=float(sample_rate),
            antenna_spacing_m=aoa_spacing_m,
            phase_gain_profile_json=args.phase_gain_profile,
            ref_channel=int(args.aoa_ref_channel),
            target_channel=int(args.aoa_target_channel),
            coherence_threshold=aoa_coherence_threshold,
            energy_percentile=aoa_energy_percentile,
            nperseg=int(nperseg),
            noverlap=int(noverlap),
            nfft=int(nfft),
            window="hann",
            compute_coherence=not bool(args.aoa_disable_stft_coherence),
            gain=float(gain),
        )

        print("aoa               : enabled, gated by confirmed Drone")
        print(f"aoa_spacing_m     : {aoa_spacing_m}")
        print(f"phase_gain_profile: {args.phase_gain_profile}")
    else:
        print("aoa               : disabled")

    gain_state: dict[str, Any] | None = None
    if not args.no_display:
        gain_state = setup_gain_widgets(
            plt.gcf(),
            receiver,
            initial_gain=float(gain),
            initial_distance=args.distance_m,
            initial_memo=args.memo,
        )
        # setup_gain_widgets() creates many widget axes.
        # Force drawing back to the main spectrogram axis.
        plt.sca(ax)
        image_handle, text_handle = _initialize_display_window(
            plt,
            image_handle,
            text_handle,
            nfft=nfft,
        )

    update_index = 0

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
            raw_features_list = []

            for _ in range(args.blocks_per_update):
                block = receiver.read_block(block_size)
                features = compute_raw_features(
                    block,
                    rx_index=rx_index,
                    frame_size=args.frame_size,
                    hop_size=args.hop_size,
                    overload_peak=thresholds.overload_peak,
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

            raw_gate_result = raw_gate.evaluate(selected_block, gain=float(gain))
            raw_gate_passed = bool(
                (not raw_gate_result.enabled) or raw_gate_result.passed
            )
            raw_gate_text = raw_gate.status_text(raw_gate_result)

            raw_gate_row = {
                "raw_gate_enabled": raw_gate_result.enabled,
                "raw_gate_passed": raw_gate_result.passed,
                "raw_gate_label": raw_gate_result.label,
                "raw_gate_score_max": raw_gate_result.score_max,
                "raw_gate_score_median": raw_gate_result.score_median,
                "raw_gate_noise_floor": raw_gate_result.noise_floor,
                "raw_gate_threshold": raw_gate_result.threshold,
                "raw_gate_threshold_multiplier": raw_gate_result.threshold_multiplier,
                "raw_gate_detection_ratio": raw_gate_result.detection_ratio,
                "raw_gate_min_detection_ratio": raw_gate_result.min_detection_ratio,
                "raw_gate_matched_gain": raw_gate_result.matched_gain,
                "raw_gate_matched_by": raw_gate_result.matched_by,
                "raw_gate_reason": raw_gate_result.reason,
            }

            spec: np.ndarray | None = None
            if raw_gate_passed:
                spec = compute_cnn_input_spectrogram(
                    selected_block,
                    rx_index=rx_index,
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
                cnn_threshold = select_drone_threshold(decision_cfg, gain)

                if spec is not None and cnn_model is not None:
                    result = cnn_model.predict(spec)
                    class_names = list(getattr(cnn_model, "class_names", ml_cfg.get("class_names", [])))
                    if decision_cfg.positive_class in class_names:
                        pos_idx = class_names.index(decision_cfg.positive_class)
                    else:
                        pos_idx = len(class_names) - 1
                    cnn_prob_drone = float(result.probabilities[pos_idx])
                    raw_is_drone = int(cnn_prob_drone >= cnn_threshold)
                else:
                    raw_is_drone = 0

                cnn_raw_decision = "Drone" if raw_is_drone else "NotDrone"

                if (not raw_gate_passed) and raw_gate.reset_cnn_history_on_fail():
                    temporal_history.clear()

                if args.reset_temporal_on_no_signal and status == "NO_SIGNAL":
                    temporal_history.clear()

                if args.decision_mode in {"temporal", "hybrid"} and voting_cfg.enabled:
                    recent, vote_count, candidate_status, confirmed_status, final_decision = update_temporal_decision(
                        temporal_history,
                        raw_is_drone,
                        voting_cfg,
                    )
                    temporal_history_text = "".join(str(x) for x in recent)
                    if args.show_candidate_as_drone and candidate_status:
                        final_decision = "Drone Candidate"
                else:
                    final_decision = cnn_raw_decision
                    candidate_status = bool(raw_is_drone)
                    confirmed_status = bool(raw_is_drone)

            aoa_row: dict[str, Any] = {
                "aoa_enabled": aoa_runtime is not None,
                "aoa_gate_passed": False,
                "aoa_angle_deg": "",
                "aoa_valid": False,
                "aoa_skipped_reason": "",
                "aoa_stft_coherence": "",
                "aoa_stft_coherence_passed": "",
                "phase_diff_corrected_deg": "",
            }
            aoa_side_text = "AOA: disabled"

            if aoa_runtime is not None:
                aoa_runtime.carrier_freq = float(center_freq)
                aoa_runtime.update_gain(float(gain))

                if confirmed_status and raw_gate_passed:
                    try:
                        aoa_result = aoa_runtime.process(selected_block)
                        aoa_row.update(
                            {
                                "aoa_gate_passed": True,
                                "aoa_angle_deg": aoa_result.get("aoa_angle_deg", ""),
                                "aoa_valid": aoa_result.get("aoa_valid", False),
                                "aoa_stft_coherence": aoa_result.get("stft_coherence", ""),
                                "aoa_stft_coherence_passed": aoa_result.get("stft_coherence_passed", ""),
                                "phase_diff_corrected_deg": aoa_result.get("phase_diff_corrected_deg", ""),
                            }
                        )
                        aoa_side_text = (
                            f"AOA: angle={aoa_row['aoa_angle_deg']:.2f} deg "
                            f"valid={aoa_row['aoa_valid']} "
                            f"coh={aoa_row['aoa_stft_coherence']}"
                        )
                        print(
                            f"[DRONE_AOA] update={update_index} "
                            f"angle={aoa_row['aoa_angle_deg']} "
                            f"valid={aoa_row['aoa_valid']} "
                            f"coh={aoa_row['aoa_stft_coherence']}",
                            flush=True,
                        )
                    except Exception as exc:
                        aoa_row.update(
                            {
                                "aoa_gate_passed": True,
                                "aoa_valid": False,
                                "aoa_skipped_reason": f"aoa_error:{exc}",
                            }
                        )
                        aoa_side_text = f"AOA: error {exc}"
                else:
                    skip_reason = (
                        "raw_noise_gate_failed"
                        if not raw_gate_passed
                        else "cnn_not_confirmed"
                    )
                    aoa_row.update(
                        {
                            "aoa_gate_passed": False,
                            "aoa_valid": False,
                            "aoa_skipped_reason": skip_reason,
                        }
                    )
                    aoa_side_text = (
                        f"AOA: gated, {skip_reason} "
                        f"history={temporal_history_text or '-'}"
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
                "rx_index": rx_index,
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
                "cnn_model_path": decision_cfg.model_path,
                "cnn_prob_drone": cnn_prob_drone,
                "cnn_threshold": cnn_threshold,
                "cnn_raw_decision": cnn_raw_decision,
                "temporal_window": voting_cfg.window_size,
                "candidate_vote_k": voting_cfg.candidate_vote_k,
                "confirmed_vote_k": voting_cfg.confirmed_vote_k,
                "temporal_history": temporal_history_text,
                "candidate_status": candidate_status,
                "confirmed_status": confirmed_status,
                "final_decision": final_decision,
                "latency_sec": latency_sec,
                "processing_time_sec": processing_time_sec,
                **raw_gate_row,
                **aoa_row,
            }

            if gain_state is not None:
                match_result = evaluate_feature_match(row, gain_state)
                row.update(match_result)
                update_feature_widget_text(gain_state, row, match_result)
                update_gain_profile_capture(gain_state, row)
            else:
                row.update(empty_feature_match_result())

            append_csv_log(csv_path, _ensure_csv_columns(row))
            print_update(row)

            title = (
                f"YAML Live CNN Spectrogram | {raw_gate_result.label} | {final_decision} | "
                f"gain={gain} | d={current_distance_m}m | update={update_index}"
            )
            side_text = build_side_text(row)
            side_text = side_text + "\n" + raw_gate_text
            side_text = side_text + "\n" + aoa_side_text

            if not args.no_display:
                # Always draw spectrogram on the main large axis,
                # not on the TextBox/Button axes.
                plt.sca(ax)
                image_handle, text_handle = update_display(
                    plt, image_handle, text_handle, spec, title, side_text
                )
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.flush_events()
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
