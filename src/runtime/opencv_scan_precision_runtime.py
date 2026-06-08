from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.experimental.live_aoa_sector_dashboard import (
    SectorDashboardRenderer,
    load_dashboard_cfg,
)

from src.core.config import load_all_configs, load_yaml
from src.calibration import load_calibration_params
from src.receiver.factory import build_receiver
from src.scan.scan_policy import build_scan_freqs
from src.scan.precision_analyzer import PrecisionAnalyzer
from src.runtime.raw_noise_gate import RawNoiseGate
from src.ml.runtime_decision import load_runtime_decision_config
from src.viewer.state import ViewerState

# Reuse the same low-level receiver handling policy as scan_loop.py.
from src.runtime.scan_loop import (
    _get_receiver_gain,
    _receiver_sample_rate,
    _set_receiver_center_freq,
    _read_receiver_block,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _unwrap_scan_cfg(scan_cfg_raw: dict[str, Any]) -> dict[str, Any]:
    if "scan" in scan_cfg_raw and isinstance(scan_cfg_raw["scan"], dict):
        return scan_cfg_raw["scan"]
    return scan_cfg_raw


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _build_cnn_classifier_compat(
    *,
    ml_cfg: dict[str, Any],
    decision_cfg: Any,
):
    """
    runtime_classifier_factory.py의 함수 signature가 실험 중 바뀐 적이 있어서
    몇 가지 호출 방식을 순서대로 시도한다.
    """
    from src.ml.runtime_classifier_factory import build_runtime_cnn_classifier

    class_names = list(ml_cfg.get("class_names", ["NotDrone", "Drone"]))
    inference_cfg = ml_cfg.get("inference", {}) or {}

    attempts = [
        lambda: build_runtime_cnn_classifier(ml_cfg),
        lambda: build_runtime_cnn_classifier(decision_cfg),
        lambda: build_runtime_cnn_classifier(
            ml_cfg=ml_cfg,
            decision_cfg=decision_cfg,
        ),
        lambda: build_runtime_cnn_classifier(
            model_path=str(decision_cfg.model_path),
            class_names=class_names,
            device=str(decision_cfg.device),
            backend=str(decision_cfg.backend),
            general_threshold=float(decision_cfg.general_threshold),
            drone_threshold=float(decision_cfg.default_drone_threshold),
        ),
        lambda: build_runtime_cnn_classifier(
            model_path=str(inference_cfg.get("model_path", decision_cfg.model_path)),
            class_names=class_names,
            device=str(inference_cfg.get("device", decision_cfg.device)),
        ),
    ]

    last_error: Exception | None = None

    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    raise RuntimeError("failed to build runtime CNN classifier")


def _load_sector_args(
    *,
    config_dir: str | Path,
    configs: dict[str, dict[str, Any]],
    scan_freqs: list[float],
) -> Any:
    config_dir = Path(config_dir)

    aoa_sector_path = config_dir / "aoa_sector.yaml"
    if aoa_sector_path.exists():
        sector_yaml = load_yaml(aoa_sector_path)
    else:
        sector_yaml = {}

    sector_root = sector_yaml.get("aoa_sector", sector_yaml) or {}
    preset_name = str(sector_root.get("active_preset", "fixed_bins_7sector"))

    presets = sector_root.get("presets", {}) or {}
    sector_preset = presets.get(preset_name)

    if sector_preset is None:
        sector_preset = {
            "bins": [
                {"name": "LEFT_60_45", "min_deg": -60, "max_deg": -45, "label_deg": -52.5},
                {"name": "LEFT_45_30", "min_deg": -45, "max_deg": -30, "label_deg": -37.5},
                {"name": "LEFT_30_15", "min_deg": -30, "max_deg": -15, "label_deg": -22.5},
                {"name": "CENTER", "min_deg": -15, "max_deg": 15, "label_deg": 0.0},
                {"name": "RIGHT_15_30", "min_deg": 15, "max_deg": 30, "label_deg": 22.5},
                {"name": "RIGHT_30_45", "min_deg": 30, "max_deg": 45, "label_deg": 37.5},
                {"name": "RIGHT_45_60", "min_deg": 45, "max_deg": 60, "label_deg": 52.5},
            ]
        }

    receiver_cfg = configs.get("receiver", {}) or {}
    sdr_cfg = receiver_cfg.get("sdr", {}) or {}
    ml_cfg = configs.get("ml", {}) or {}
    aoa_cfg = configs.get("aoa", {}) or {}
    ui_cfg = configs.get("ui", {}) or {}
    live_cfg = ui_cfg.get("live_rf_viewer", {}) or {}
    sector_runtime = sector_root.get("runtime", {}) or {}

    return SimpleNamespace(
        config_dir=str(config_dir),
        window_name="RF SCAN + PRECISION Runtime",
        target_fps=float(ui_cfg.get("refresh_ms", 100)),
        sector_preset=sector_preset,
        sector_preset_name=preset_name,
        sector_root=sector_root,
        center_freq=int(sdr_cfg.get("center_freq", receiver_cfg.get("center_freq", scan_freqs[0]))),
        sample_rate=int(sdr_cfg.get("sample_rate", receiver_cfg.get("sample_rate", 5_000_000))),
        gain=float(sdr_cfg.get("gain", receiver_cfg.get("gain", 30.0))),
        block_size=int(sdr_cfg.get("block_size", receiver_cfg.get("block_size", ml_cfg.get("block_size", 16_384)))),
        top_k=int(sector_runtime.get("top_k", 5)),
        cli_log_every_n=int(live_cfg.get("cli_log_every_n", 1)),
        aoa_ref_channel=int(aoa_cfg.get("ref_channel", 0)),
        aoa_target_channel=int(aoa_cfg.get("target_channel", 1)),
        aoa_antenna_spacing_m=float(aoa_cfg.get("antenna_spacing_m", 0.06)),
    )


def _empty_sector(status: str = "scanning") -> dict[str, Any]:
    return {
        "sector_status": status,
        "locked_sector_name": "",
        "instant_sector_name": "",
        "median_angle_deg": "",
        "angle_spread": "",
        "median_coherence": "",
        "median_raw_p99": "",
        "dominant_sector_ratio": "",
        "valid_aoa_count": "",
        "votes": "None",
    }


def _empty_cnn(label: str = "WAITING") -> dict[str, Any]:
    return {
        "label": label,
        "class_name": label,
        "confidence": "",
        "probability": "",
    }


def _selected_raw_from_scan_event(event: Any | None) -> dict[str, Any]:
    if event is None:
        return {}

    score_max = _safe_float(getattr(event, "best_score_max", None), None)
    score_median = _safe_float(getattr(event, "best_score_median", None), None)

    return {
        "raw_abs_p99": score_max,
        "median_raw_p99": score_max,
        "raw_abs_p95": score_median,
        "raw_abs_mean": score_median,
        "raw_rms": math.sqrt(score_median) if score_median and score_median > 0 else score_median,
        "frame_power_p99": score_max,
    }


def _sector_from_precision(result: Any) -> dict[str, Any]:
    angle = _safe_float(
        getattr(result, "aoa_smoothed_angle_deg", None),
        _safe_float(getattr(result, "angle_deg", None), None),
    )

    coh = _safe_float(getattr(result, "coherence", None), None)
    raw_p99 = _safe_float(getattr(result, "raw_gate_score_max", None), None)

    sector_label = (
        getattr(result, "sector_label", None)
        or getattr(result, "sector_name", None)
        or ""
    )

    confirmed = bool(getattr(result, "confirmed_status", False))
    candidate = bool(getattr(result, "candidate_status", False))
    sector_valid = bool(getattr(result, "sector_valid", False))
    angle_valid = bool(getattr(result, "angle_valid", False))

    if confirmed or sector_valid:
        status = "trusted"
    elif candidate or angle_valid:
        status = "candidate"
    else:
        status = "precision"

    temporal_history = getattr(result, "temporal_history", None)
    votes_text = str(temporal_history) if temporal_history is not None else "None"

    return {
        "sector_status": status,
        "locked_sector_name": sector_label,
        "instant_sector_name": sector_label,
        "median_angle_deg": angle if angle is not None else "",
        "angle_spread": "",
        "median_coherence": coh if coh is not None else "",
        "median_raw_p99": raw_p99 if raw_p99 is not None else "",
        "dominant_sector_ratio": 1.0 if sector_label else "",
        "valid_aoa_count": 1 if angle_valid else 0,
        "votes": votes_text,
    }


def _cnn_from_precision(result: Any) -> dict[str, Any]:
    label = (
        getattr(result, "cnn_label", None)
        or getattr(result, "final_decision", None)
        or "n/a"
    )

    confidence = _safe_float(
        getattr(result, "cnn_score", None),
        _safe_float(getattr(result, "drone_probability", None), None),
    )

    return {
        "label": label,
        "class_name": label,
        "confidence": confidence if confidence is not None else "",
        "probability": _safe_float(getattr(result, "drone_probability", None), confidence),
    }


def _selected_raw_from_precision(result: Any) -> dict[str, Any]:
    score_max = _safe_float(getattr(result, "raw_gate_score_max", None), None)
    score_median = _safe_float(getattr(result, "raw_gate_score_median", None), None)

    if score_max is None:
        score_max = _safe_float(getattr(result, "selection_score", None), None)

    return {
        "raw_abs_p99": score_max,
        "median_raw_p99": score_max,
        "raw_abs_p95": score_median,
        "raw_abs_mean": score_median,
        "raw_rms": math.sqrt(score_median) if score_median and score_median > 0 else score_median,
        "frame_power_p99": score_max,
    }


def _build_precision_analyzer(
    *,
    configs: dict[str, dict[str, Any]],
    receiver: Any,
    scan_cfg: dict[str, Any],
    config_dir: str | Path,
    current_gain: float | None,
) -> PrecisionAnalyzer:
    receiver_cfg = configs.get("receiver", {}) or {}
    ml_cfg = configs.get("ml", {}) or {}
    aoa_cfg = configs.get("aoa", {}) or {}

    stft_cfg = ml_cfg.get("stft", {}) or {}
    cnn_input_cfg = ml_cfg.get("cnn_input", {}) or {}
    candidate_verify_cfg = (configs.get("detect", {}) or {}).get("candidate_verify", {}) or {}

    decision_cfg = load_runtime_decision_config(ml_cfg)
    cnn_enabled = bool(scan_cfg.get("cnn_enabled", True))

    cnn_classifier = None
    if cnn_enabled:
        cnn_classifier = _build_cnn_classifier_compat(
            ml_cfg=ml_cfg,
            decision_cfg=decision_cfg,
        )

    calibration = load_calibration_params(
        require_noise=False,
        require_phase_gain=False,
    )

    phase_offset_rad = float(aoa_cfg.get("phase_offset_rad", 0.0))
    if getattr(calibration, "phase_gain", None) is not None:
        phase_offset_rad = float(calibration.phase_gain.phase_offset_rad)

    precision_dir = PROJECT_ROOT / "outputs" / "runs" / "latest" / "opencv_scan_precision"
    precision_dir.mkdir(parents=True, exist_ok=True)

    coherence_cfg = aoa_cfg.get("coherence", {}) or {}

    return PrecisionAnalyzer(
        receiver=receiver,
        num_samples=int(scan_cfg.get("num_samples", receiver_cfg.get("num_samples", 16_384))),
        sample_rate=float(_receiver_sample_rate(receiver_cfg)),
        antenna_spacing_m=float(aoa_cfg.get("antenna_spacing_m", 0.06)),
        nperseg=int(stft_cfg.get("nperseg", 128)),
        noverlap=int(stft_cfg.get("noverlap", 96)),
        nfft=int(stft_cfg.get("nfft", 128)),
        window=str(stft_cfg.get("window", "hann")),
        coherence_threshold=float(coherence_cfg.get("threshold", scan_cfg.get("coherence_threshold", 0.6))),
        phase_offset_rad=float(phase_offset_rad),
        settle_sec=float(scan_cfg.get("settle_sec", 0.0)),
        precision_blocks=int(
            candidate_verify_cfg.get(
                "blocks_per_decision",
                scan_cfg.get("precision_blocks_per_candidate", 10),
            )
        ),
        save_dir=str(precision_dir),
        save_spectrogram=bool(scan_cfg.get("save_spectrogram", False)),
        save_stft=bool(scan_cfg.get("save_stft", False)),
        cnn_classifier=cnn_classifier,
        decision_cfg=decision_cfg,
        current_gain=current_gain,
        aoa_cfg=aoa_cfg,
        cnn_rx_index=int(cnn_input_cfg.get("rx_index", 0)),
    )


def _scan_one_frequency(
    *,
    receiver: Any,
    raw_gate: RawNoiseGate,
    center_freq: float,
    num_samples: int,
    scan_cfg: dict[str, Any],
    current_gain: float | None,
) -> dict[str, Any]:
    scan_candidate_cfg = ((load_yaml(PROJECT_ROOT / "configs" / "detect.yaml")).get("scan_candidate", {}) or {})

    blocks_per_freq = max(1, int(scan_candidate_cfg.get("blocks_per_freq", 8)))
    discard_blocks = max(0, int(scan_candidate_cfg.get("discard_blocks_after_tune", 4)))
    min_pass_count = max(1, int(scan_candidate_cfg.get("min_raw_gate_pass_count", 1)))

    if discard_blocks >= blocks_per_freq:
        discard_blocks = max(0, blocks_per_freq - 1)

    settle_sec = float(scan_cfg.get("settle_sec", 0.0))
    scan_gain = float(current_gain if current_gain is not None else 0.0)

    _set_receiver_center_freq(receiver, center_freq)
    if settle_sec > 0:
        time.sleep(settle_sec)

    pass_count = 0
    best_result = None
    best_score = float("-inf")
    scores: list[float] = []
    passed_blocks: list[int] = []

    for block_idx in range(blocks_per_freq):
        iq_block = _read_receiver_block(receiver, num_samples)

        if block_idx < discard_blocks:
            continue

        gate_result = raw_gate.evaluate(iq_block, gain=scan_gain)
        score = _safe_float(getattr(gate_result, "score_max", None), float("-inf"))
        scores.append(float(score))

        if score is not None and score > best_score:
            best_score = float(score)
            best_result = gate_result

        if bool((not getattr(gate_result, "enabled", True)) or getattr(gate_result, "passed", False)):
            pass_count += 1
            passed_blocks.append(block_idx)

    triggered = pass_count >= min_pass_count

    return {
        "center_freq": float(center_freq),
        "triggered": bool(triggered),
        "pass_count": int(pass_count),
        "scan_blocks": int(blocks_per_freq),
        "discard_blocks": int(discard_blocks),
        "usable_blocks": int(blocks_per_freq - discard_blocks),
        "best_score_max": _safe_float(getattr(best_result, "score_max", None), None),
        "best_score_median": _safe_float(getattr(best_result, "score_median", None), None),
        "threshold": _safe_float(getattr(best_result, "threshold", None), None),
        "noise_floor": _safe_float(getattr(best_result, "noise_floor", None), None),
        "threshold_multiplier": _safe_float(getattr(best_result, "threshold_multiplier", None), None),
        "matched_gain": _safe_float(getattr(best_result, "matched_gain", None), None),
        "matched_by": str(getattr(best_result, "matched_by", "")),
        "raw_gate_label": str(getattr(best_result, "label", "")),
        "raw_gate_reason": str(getattr(best_result, "reason", "")),
        "raw_gate_passed_blocks": passed_blocks,
        "raw_gate_scores": scores,
    }


def _render_scan(
    *,
    renderer: SectorDashboardRenderer,
    state: ViewerState,
    args: Any,
    dash_cfg: dict[str, Any],
    scan_freqs: list[float],
    current_freq: float,
    scan_event: dict[str, Any] | None = None,
) -> str | None:
    state.mode = "SCAN"
    state.center_freq = int(current_freq)
    state.mark_update()

    scan_rail = {
        "mode": "SCAN",
        "scan_freqs": scan_freqs,
        "current_freq": current_freq,
        "locked_freq": None,
        "candidate_freq": None,
        "status": "SWEEPING",
        "rail_width": 190,
    }

    selected_raw = _selected_raw_from_scan_event(SimpleNamespace(**scan_event)) if scan_event else {}

    return renderer.render(
        state=state,
        args=args,
        dash_cfg=dash_cfg,
        sector=_empty_sector("scanning"),
        selected_raw=selected_raw,
        cnn_result=_empty_cnn("WAITING"),
        raw_pass_count=int(scan_event.get("pass_count", 0)) if scan_event else 0,
        cnn_drone_count=0,
        topk_count=int(getattr(args, "top_k", 5)),
        paused=state.paused,
        scan_rail=scan_rail,
    )


def _render_precision(
    *,
    renderer: SectorDashboardRenderer,
    state: ViewerState,
    args: Any,
    dash_cfg: dict[str, Any],
    scan_freqs: list[float],
    locked_freq: float,
    result: Any,
) -> str | None:
    state.mode = "PRECISION"
    state.center_freq = int(locked_freq)
    state.mark_update()

    scan_rail = {
        "mode": "PRECISION",
        "scan_freqs": scan_freqs,
        "current_freq": locked_freq,
        "locked_freq": locked_freq,
        "candidate_freq": locked_freq,
        "status": "HANDOFF",
        "rail_width": 190,
    }

    raw_pass_count = 1 if bool(getattr(result, "raw_gate_passed", False)) else 0
    drone_votes = _safe_int(getattr(result, "drone_vote_count", 0), 0)

    return renderer.render(
        state=state,
        args=args,
        dash_cfg=dash_cfg,
        sector=_sector_from_precision(result),
        selected_raw=_selected_raw_from_precision(result),
        cnn_result=_cnn_from_precision(result),
        raw_pass_count=raw_pass_count,
        cnn_drone_count=drone_votes,
        topk_count=int(getattr(args, "top_k", 5)),
        paused=state.paused,
        scan_rail=scan_rail,
    )


def run_opencv_scan_precision_runtime(
    *,
    config_dir: str | Path = "configs",
    stop_key: str = "q",
    verbose: bool = True,
) -> int:
    config_dir = Path(config_dir)
    configs = load_all_configs(config_dir)

    scan_cfg = _unwrap_scan_cfg(configs.get("scan", {}) or {})
    receiver_cfg = configs.get("receiver", {}) or {}
    ui_cfg = configs.get("ui", {}) or {}

    start_freq = float(scan_cfg["start_freq"])
    stop_freq = float(scan_cfg["stop_freq"])
    step_freq = float(scan_cfg["step_freq"])
    num_samples = int(scan_cfg.get("num_samples", receiver_cfg.get("num_samples", 16_384)))

    scan_freqs = build_scan_freqs(
        start_freq=start_freq,
        stop_freq=stop_freq,
        step_freq=step_freq,
    )

    receiver = build_receiver(receiver_cfg)
    current_gain = _get_receiver_gain(receiver, receiver_cfg)

    raw_gate = RawNoiseGate(
        detect_config_path=PROJECT_ROOT / "configs" / "detect.yaml",
        project_root=PROJECT_ROOT,
    )

    analyzer = _build_precision_analyzer(
        configs=configs,
        receiver=receiver,
        scan_cfg=scan_cfg,
        config_dir=config_dir,
        current_gain=current_gain,
    )

    args = _load_sector_args(
        config_dir=config_dir,
        configs=configs,
        scan_freqs=scan_freqs,
    )

    dash_cfg = load_dashboard_cfg(args)

    renderer = SectorDashboardRenderer(
        window_name=args.window_name,
        target_fps=5.0,
        width=int(dash_cfg.get("canvas_width", 1320)),
        height=int(dash_cfg.get("canvas_height", 720)),
        blink_on_hold=bool(dash_cfg.get("blink_on_hold", True)),
        fade_on_signal_lost=bool(dash_cfg.get("fade_on_signal_lost", True)),
    )

    state = ViewerState(
        mode="SCAN",
        gain=float(current_gain if current_gain is not None else getattr(args, "gain", 0.0)),
        center_freq=int(scan_freqs[0]),
        sample_rate=int(_receiver_sample_rate(receiver_cfg)),
        target_fps=5.0,
    )

    hold_cfg = scan_cfg.get("precision_hold", {}) or {}
    min_hold_blocks = max(1, int(hold_cfg.get("min_hold_blocks", 7)))
    max_hold_blocks = max(min_hold_blocks, int(hold_cfg.get("max_hold_blocks", 100)))
    block_delay_sec = float(hold_cfg.get("block_delay_sec", 0.0))

    if verbose:
        print()
        print("=== OpenCV Real SCAN + PRECISION Runtime ===")
        print(f"scan freqs : {[round(f / 1e9, 3) for f in scan_freqs]}")
        print(f"gain       : {current_gain}")
        print(f"num_samples: {num_samples}")
        print("OpenCV key : q or ESC to return CLI")
        print()

    try:
        while state.running:
            for current_freq in scan_freqs:
                key = _render_scan(
                    renderer=renderer,
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    scan_freqs=scan_freqs,
                    current_freq=current_freq,
                )

                if key == "quit":
                    state.running = False
                    break
                if key == "pause":
                    state.toggle_pause()

                while state.paused and state.running:
                    key = _render_scan(
                        renderer=renderer,
                        state=state,
                        args=args,
                        dash_cfg=dash_cfg,
                        scan_freqs=scan_freqs,
                        current_freq=current_freq,
                    )
                    if key == "quit":
                        state.running = False
                        break
                    if key == "pause":
                        state.toggle_pause()

                if not state.running:
                    break

                scan_event = _scan_one_frequency(
                    receiver=receiver,
                    raw_gate=raw_gate,
                    center_freq=current_freq,
                    num_samples=num_samples,
                    scan_cfg=scan_cfg,
                    current_gain=current_gain,
                )

                key = _render_scan(
                    renderer=renderer,
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    scan_freqs=scan_freqs,
                    current_freq=current_freq,
                    scan_event=scan_event,
                )

                if key == "quit":
                    state.running = False
                    break
                if key == "pause":
                    state.toggle_pause()

                if not bool(scan_event.get("triggered", False)):
                    continue

                locked_freq = float(current_freq)

                if verbose:
                    print(
                        f"[CANDIDATE] cf={locked_freq / 1e9:.3f}GHz "
                        f"pass={scan_event.get('pass_count')}/{scan_event.get('usable_blocks')} "
                        f"score={scan_event.get('best_score_max')}"
                    )

                for hold_idx in range(max_hold_blocks):
                    result = analyzer.analyze(locked_freq)

                    key = _render_precision(
                        renderer=renderer,
                        state=state,
                        args=args,
                        dash_cfg=dash_cfg,
                        scan_freqs=scan_freqs,
                        locked_freq=locked_freq,
                        result=result,
                    )

                    if key == "quit":
                        state.running = False
                        break
                    if key == "pause":
                        state.toggle_pause()

                    while state.paused and state.running:
                        key = _render_precision(
                            renderer=renderer,
                            state=state,
                            args=args,
                            dash_cfg=dash_cfg,
                            scan_freqs=scan_freqs,
                            locked_freq=locked_freq,
                            result=result,
                        )
                        if key == "quit":
                            state.running = False
                            break
                        if key == "pause":
                            state.toggle_pause()

                    if not state.running:
                        break

                    raw_ok = bool(getattr(result, "raw_gate_passed", False))
                    candidate_ok = bool(getattr(result, "candidate_status", False))
                    confirmed_ok = bool(getattr(result, "confirmed_status", False))

                    # 최소 hold block 이후에는 신호/CNN 상태가 모두 약하면 SCAN 복귀
                    if hold_idx + 1 >= min_hold_blocks:
                        if not (raw_ok or candidate_ok or confirmed_ok):
                            if verbose:
                                print("[PRECISION -> SCAN] weak/no candidate")
                            break

                    if block_delay_sec > 0:
                        time.sleep(block_delay_sec)

            if not state.running:
                break

        return 0

    except KeyboardInterrupt:
        print()
        print("[STOP] OpenCV scan precision runtime stopped.")
        return 0

    finally:
        try:
            renderer.close()
        finally:
            close_fn = getattr(receiver, "close", None)
            if callable(close_fn):
                close_fn()


def main() -> int:
    return run_opencv_scan_precision_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
