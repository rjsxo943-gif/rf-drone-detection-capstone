from __future__ import annotations

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
from src.receiver.factory import build_receiver
from src.scan.scan_policy import build_scan_freqs
from src.runtime.raw_noise_gate import RawNoiseGate
from src.features.cnn_input import compute_runtime_cnn_spectrogram
from src.viewer.state import ViewerState

from src.runtime.scan_loop import (
    _get_receiver_gain,
    _receiver_sample_rate,
    _set_receiver_center_freq,
    _read_receiver_block,
)

from src.runtime.fixed2450_precision_runtime import run_fixed2450_precision_runtime

# 기존 더러운 scan/precision 상태머신은 쓰지 않는다.
# 다만 이미 검증된 UI/Analyzer 생성 helper만 재사용한다.
from src.runtime.opencv_scan_precision_runtime import (
    PROJECT_ROOT,
    _unwrap_scan_cfg,
    _safe_float,
    _safe_int,
    _build_precision_analyzer,
    _load_sector_args,
    _render_scan,
)


def _scan_one_frequency_activity(
    *,
    receiver: Any,
    raw_gate: RawNoiseGate,
    center_freq: float,
    num_samples: int,
    scan_cfg: dict[str, Any],
    current_gain: float | None,
) -> dict[str, Any]:
    """Noise calibration 기반 raw gate만으로 RF activity 후보를 찾는다."""

    detect_yaml = load_yaml(PROJECT_ROOT / "configs" / "detect.yaml")
    scan_candidate_cfg = detect_yaml.get("scan_candidate", {}) or {}

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

        passed = bool((not getattr(gate_result, "enabled", True)) or getattr(gate_result, "passed", False))
        if passed:
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


def _candidate_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    score = _safe_float(event.get("best_score_max"), 0.0) or 0.0
    threshold = _safe_float(event.get("threshold"), None)
    ratio = score / threshold if threshold and threshold > 0 else 0.0
    return (float(ratio), float(score))


def _verify_candidate_top5_cnn(
    *,
    analyzer: Any,
    center_freq: float,
    current_gain: float | None,
    verify_blocks: int = 20,
    cnn_top_m: int = 5,
    cnn_vote_required: int = 3,
    cnn_conf_min: float = 0.90,
    verbose: bool = True,
) -> dict[str, Any]:
    """한 후보 주파수 안에서 20블럭을 보고 raw score Top5를 CNN 검증한다."""

    if analyzer.cnn_classifier is None:
        return {
            "passed": False,
            "reason": "cnn_disabled",
            "votes": 0,
            "top_m": int(cnn_top_m),
            "results": [],
        }

    analyzer._set_center_freq(float(center_freq))

    settle_sec = float(getattr(analyzer, "settle_sec", 0.0))
    if settle_sec > 0:
        time.sleep(settle_sec)

    gain = current_gain
    if gain is None:
        try:
            gain = analyzer._get_gain()
        except Exception:
            gain = None
    if gain is None:
        gain = getattr(analyzer, "current_gain", 0.0)
    if gain is None:
        gain = 0.0

    raw_items: list[dict[str, Any]] = []

    for block_idx in range(max(1, int(verify_blocks))):
        iq_raw = analyzer._read_iq_block_like_live_viewer()
        gate = analyzer.raw_gate.evaluate(iq_raw, gain=float(gain))

        score = _safe_float(getattr(gate, "score_max", None), float("-inf"))
        threshold = _safe_float(getattr(gate, "threshold", None), None)
        passed = bool((not getattr(gate, "enabled", True)) or getattr(gate, "passed", False))

        if passed:
            raw_items.append(
                {
                    "block_idx": int(block_idx),
                    "iq": iq_raw,
                    "score": float(score),
                    "threshold": threshold,
                    "ratio": float(score / threshold) if threshold and threshold > 0 else None,
                    "gate": gate,
                }
            )

    if not raw_items:
        return {
            "passed": False,
            "reason": "no_raw_gate_pass",
            "votes": 0,
            "top_m": int(cnn_top_m),
            "results": [],
        }

    raw_items.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = raw_items[: max(1, int(cnn_top_m))]

    votes = 0
    results: list[dict[str, Any]] = []

    # scan 후보 검증은 후보 freq 내부 vote이므로 analyzer temporal history와 분리한다.
    try:
        analyzer.reset_temporal_history()
    except Exception:
        pass

    for rank, item in enumerate(selected, start=1):
        iq_raw = item["iq"]

        is_cw_tone = False
        cw_peak_ratio = None
        cw_occupied_bins = None
        try:
            is_cw_tone, cw_peak_ratio, cw_occupied_bins = analyzer._is_cw_tone_like(iq_raw)
        except Exception:
            is_cw_tone = False

        if is_cw_tone:
            label = "CW_TONE_REJECT"
            drone_prob = 0.0
            confidence = 0.0
            vote = 0
        else:
            cnn_spec = compute_runtime_cnn_spectrogram(
                iq_raw,
                rx_index=int(getattr(analyzer, "cnn_rx_index", 0)),
                nperseg=int(getattr(analyzer, "nperseg", 128)),
                noverlap=int(getattr(analyzer, "noverlap", 96)),
                nfft=int(getattr(analyzer, "nfft", 128)),
            )

            cnn_result = analyzer.cnn_classifier.predict(cnn_spec)
            label = str(getattr(cnn_result, "class_name", ""))
            confidence = float(getattr(cnn_result, "confidence", 0.0))
            drone_prob = float(analyzer._drone_probability(cnn_result))

            label_is_drone = label.strip().lower() == "drone"
            vote = int(label_is_drone and drone_prob >= float(cnn_conf_min))

        votes += int(vote)

        row = {
            "rank": int(rank),
            "block_idx": int(item["block_idx"]),
            "score": float(item["score"]),
            "threshold": item["threshold"],
            "ratio": item["ratio"],
            "label": label,
            "confidence": float(confidence),
            "drone_probability": float(drone_prob),
            "vote": int(vote),
            "cw_peak_ratio": cw_peak_ratio,
            "cw_occupied_bins": cw_occupied_bins,
        }
        results.append(row)

        if verbose:
            ratio_text = "" if row["ratio"] is None else f" ratio={row['ratio']:.2f}"
            print(
                f"  [CNN TOP{rank}] block={row['block_idx']} "
                f"score={row['score']:.3f}{ratio_text} "
                f"label={row['label']} prob={row['drone_probability']:.4f} "
                f"vote={row['vote']}",
                flush=True,
            )

    passed = votes >= int(cnn_vote_required)

    return {
        "passed": bool(passed),
        "reason": "top5_vote_pass" if passed else "top5_vote_fail",
        "votes": int(votes),
        "top_m": int(len(selected)),
        "vote_required": int(cnn_vote_required),
        "cnn_conf_min": float(cnn_conf_min),
        "results": results,
    }


def run_scan_activity_cnn_runtime(
    *,
    config_dir: str | Path = "configs",
    stop_key: str = "q",
    verbose: bool = True,
    handoff_to_precision: bool = False,
) -> int:
    """Clean SCAN mode.

    역할 분리:
    - scan: noise calibration raw gate로 RF activity 후보 찾기
    - cnn verify: 후보 freq 내부 20블럭 Top5 중 2표 이상 Drone 확인
    - precision: handoff_to_precision=True일 때만 fixed 2.450GHz runtime으로 handoff
    """

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

    clean_cfg = scan_cfg.get("scan_activity_cnn", {}) or {}
    candidate_top_k = max(1, int(clean_cfg.get("candidate_top_k", 3)))
    verify_blocks = max(1, int(clean_cfg.get("verify_blocks", 20)))
    cnn_top_m = max(1, int(clean_cfg.get("cnn_top_m", 5)))
    cnn_vote_required = max(1, int(clean_cfg.get("cnn_vote_required", 3)))
    cnn_conf_min = float(clean_cfg.get("cnn_conf_min", 0.90))
    block_delay_sec = float(clean_cfg.get("block_delay_sec", 0.0))
    precision_fixed_freq_hz = float(clean_cfg.get("precision_fixed_freq_hz", 2.450e9))

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
        window_name="RF Drone Detection Runtime",
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

    detected: dict[str, Any] | None = None

    if verbose:
        print()
        print("=== Clean SCAN Activity + CNN Verify Runtime ===")
        print(f"scan freqs       : {[round(f / 1e9, 3) for f in scan_freqs]}")
        print(f"gain             : {current_gain}")
        print(f"num_samples      : {num_samples}")
        print(f"candidate_top_k  : {candidate_top_k}")
        print(f"verify_blocks    : {verify_blocks}")
        print(f"cnn_top_m        : {cnn_top_m}")
        print(f"cnn_vote_required: {cnn_vote_required}")
        print(f"cnn_conf_min     : {cnn_conf_min}")
        print(f"precision fixed  : {precision_fixed_freq_hz / 1e9:.3f}GHz")
        print(f"handoff mode     : {'ON' if handoff_to_precision else 'OFF / observe only'}")
        print("OpenCV key       : q or ESC to return CLI")
        print()

    try:
        while state.running:
            sweep_candidates: list[dict[str, Any]] = []

            if verbose:
                print("[SCAN LOOP] begin sweep", flush=True)

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

                event = _scan_one_frequency_activity(
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
                    scan_event=event,
                )

                if key == "quit":
                    state.running = False
                    break
                if key == "pause":
                    state.toggle_pause()

                if verbose:
                    score = event.get("best_score_max")
                    thr = event.get("threshold")
                    ratio = (float(score) / float(thr)) if score is not None and thr not in (None, 0) else None
                    ratio_text = "" if ratio is None else f" ratio={ratio:.2f}"
                    print(
                        f"[SWEEP] cf={current_freq / 1e9:.3f}GHz "
                        f"triggered={event.get('triggered')} "
                        f"pass={event.get('pass_count')}/{event.get('usable_blocks')} "
                        f"score={score}{ratio_text}",
                        flush=True,
                    )

                if bool(event.get("triggered", False)):
                    sweep_candidates.append(event)

                if block_delay_sec > 0:
                    time.sleep(block_delay_sec)

            if not state.running:
                break

            if not sweep_candidates:
                if verbose:
                    print("[SCAN LOOP] no raw-gate candidate -> continue", flush=True)
                continue

            sweep_candidates.sort(key=_candidate_sort_key, reverse=True)
            candidates = sweep_candidates[:candidate_top_k]

            if verbose:
                print(
                    "[CANDIDATE LIST] "
                    + ", ".join(
                        f"{c['center_freq'] / 1e9:.3f}GHz(score={c.get('best_score_max')})"
                        for c in candidates
                    ),
                    flush=True,
                )

            for cand in candidates:
                cand_freq = float(cand["center_freq"])

                if verbose:
                    print(
                        f"[CNN VERIFY] cf={cand_freq / 1e9:.3f}GHz "
                        f"scan_pass={cand.get('pass_count')}/{cand.get('usable_blocks')} "
                        f"scan_score={cand.get('best_score_max')}",
                        flush=True,
                    )

                verify = _verify_candidate_top5_cnn(
                    analyzer=analyzer,
                    center_freq=cand_freq,
                    current_gain=current_gain,
                    verify_blocks=verify_blocks,
                    cnn_top_m=cnn_top_m,
                    cnn_vote_required=cnn_vote_required,
                    cnn_conf_min=cnn_conf_min,
                    verbose=verbose,
                )

                if verbose:
                    print(
                        f"[CNN VERIFY RESULT] cf={cand_freq / 1e9:.3f}GHz "
                        f"passed={verify.get('passed')} "
                        f"votes={verify.get('votes')}/{verify.get('top_m')} "
                        f"required={verify.get('vote_required')} "
                        f"reason={verify.get('reason')}",
                        flush=True,
                    )

                if bool(verify.get("passed", False)):
                    print(
                        f"[CNN BAND CANDIDATE] "
                        f"scan_cf={cand_freq / 1e9:.3f}GHz "
                        f"votes={verify.get('votes')}/{verify.get('top_m')} "
                        f"required={verify.get('vote_required')} "
                        f"-> 1차 Top5 CNN vote 통과, 같은 주파수 즉시 재검증",
                        flush=True,
                    )

                    verify_recheck = _verify_candidate_top5_cnn(
                        analyzer=analyzer,
                        center_freq=cand_freq,
                        current_gain=current_gain,
                        verify_blocks=verify_blocks,
                        cnn_top_m=cnn_top_m,
                        cnn_vote_required=cnn_vote_required,
                        cnn_conf_min=cnn_conf_min,
                        verbose=verbose,
                    )

                    if verbose:
                        print(
                            f"[CNN RECHECK RESULT] cf={cand_freq / 1e9:.3f}GHz "
                            f"passed={verify_recheck.get('passed')} "
                            f"votes={verify_recheck.get('votes')}/{verify_recheck.get('top_m')} "
                            f"required={verify_recheck.get('vote_required')} "
                            f"reason={verify_recheck.get('reason')}",
                            flush=True,
                        )

                    if bool(verify_recheck.get("passed", False)):
                        detected_now = {
                            "scan_detected_freq": cand_freq,
                            "precision_fixed_freq": precision_fixed_freq_hz,
                            "candidate": cand,
                            "verify": verify,
                            "verify_recheck": verify_recheck,
                        }

                        if handoff_to_precision:
                            detected = detected_now
                            state.running = False
                            break

                        print(
                            f"[DRONE BAND CONFIRMED / SCAN ONLY] "
                            f"scan_cf={cand_freq / 1e9:.3f}GHz "
                            f"first_votes={verify.get('votes')}/{verify.get('top_m')} "
                            f"recheck_votes={verify_recheck.get('votes')}/{verify_recheck.get('top_m')} "
                            f"-> 같은 후보 주파수에서 드론 RF 패턴 반복 확인, precision 진입 안 함",
                            flush=True,
                        )
                        # scan-only 모드에서는 정밀모드로 넘어가지 않고 계속 scan/CNN 검증을 반복한다.

                    else:
                        print(
                            f"[CNN BAND CANDIDATE REJECTED] "
                            f"scan_cf={cand_freq / 1e9:.3f}GHz "
                            f"first_votes={verify.get('votes')}/{verify.get('top_m')} "
                            f"recheck_votes={verify_recheck.get('votes')}/{verify_recheck.get('top_m')} "
                            f"-> 1차 후보였지만 즉시 재검증 실패",
                            flush=True,
                        )

        return_code = 0

    except KeyboardInterrupt:
        print()
        print("[STOP] clean scan activity cnn runtime stopped.")
        return_code = 0

    finally:
        try:
            # sf handoff 시에는 OpenCV window를 유지해서 fixed precision이 같은 창을 재사용하게 한다.
            if not (handoff_to_precision and detected is not None):
                renderer.close()
        finally:
            close_fn = getattr(receiver, "close", None)
            if callable(close_fn):
                close_fn()

    if detected is not None:
        scan_freq = float(detected["scan_detected_freq"])
        fixed_freq = float(detected["precision_fixed_freq"])

        print()
        print(
            f"[HANDOFF] CNN verified Drone candidate at scan_cf={scan_freq / 1e9:.3f}GHz "
            f"-> fixed precision {fixed_freq / 1e9:.3f}GHz"
        )
        print("[HANDOFF] SCAN receiver closed. Starting fixed 2.450GHz precision runtime.")
        print()

        import os as _os

        _sf_env_keys = (
            "RF_SF_AUTO_RETURN",
            "RF_SF_LOST_LIMIT",
            "RF_SF_WARMUP_UPDATES",
            "RF_SF_MIN_DRONE",
            "RF_SF_MIN_COH",
            "RF_SF_KEEP_WINDOW",
            "RF_SF_AUTO_RETURNING",
        )
        _sf_old_env = {k: _os.environ.get(k) for k in _sf_env_keys}

        try:
            _os.environ["RF_SF_AUTO_RETURN"] = "1"
            _os.environ["RF_SF_LOST_LIMIT"] = "5"
            _os.environ["RF_SF_WARMUP_UPDATES"] = "5"
            _os.environ["RF_SF_MIN_DRONE"] = "2"
            _os.environ["RF_SF_MIN_COH"] = "0.85"
            _os.environ["RF_SF_KEEP_WINDOW"] = "1"
            _os.environ["RF_SF_AUTO_RETURNING"] = "0"
            run_fixed2450_precision_runtime(
                config_dir=str(config_dir),
                center_freq_hz=fixed_freq,
            )

        except SystemExit as _e:
            if _e.code == 20:
                print("[HANDOFF] Fixed precision reported signal/AoA lost. Returning to SCAN mode.")
                return run_scan_activity_cnn_runtime(
                    handoff_to_precision=True,
                    config_dir=config_dir,
                    stop_key=stop_key,
                    verbose=verbose,
                )
            raise

        finally:
            for _k, _v in _sf_old_env.items():
                if _v is None:
                    _os.environ.pop(_k, None)
                else:
                    _os.environ[_k] = _v

    return int(return_code)


def main() -> int:
    return run_scan_activity_cnn_runtime(handoff_to_precision=False)


if __name__ == "__main__":
    raise SystemExit(main())
