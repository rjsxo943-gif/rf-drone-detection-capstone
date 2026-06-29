from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as:
#   PYTHONPATH=. python scripts/experimental/live_aoa_sector_dashboard.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experimental import live_aoa_sector_experiment_capture as base
from src.preprocess.dc_blocker import remove_dc_offset
from src.runtime.raw_noise_gate import RawNoiseGate
from src.viewer import ViewerState, compute_raw_features

# ============================================================
# SF auto-return print hook
# - Enabled only when RF_SF_AUTO_RETURN=1
# - Parses [DASH] log line directly
# - Does NOT use p99/raw_pass absolute strength
# - Return-to-scan when Drone or AoA/coherence is lost repeatedly
# ============================================================

def _install_sf_auto_return_print_hook() -> None:
    import os
    import re
    import builtins

    if str(os.environ.get("RF_SF_AUTO_RETURN", "0")).lower() not in {"1", "true", "yes", "on"}:
        return

    if getattr(builtins.print, "_sf_auto_return_hooked", False):
        return

    original_print = builtins.print

    state = {
        "seen": 0,
        "drone_lost": 0,
        "aoa_lost": 0,
    }

    limit = int(os.environ.get("RF_SF_LOST_LIMIT", "5"))
    warmup = int(os.environ.get("RF_SF_WARMUP_UPDATES", "5"))
    min_drone = int(os.environ.get("RF_SF_MIN_DRONE", "2"))
    min_coh = float(os.environ.get("RF_SF_MIN_COH", "0.85"))

    def _field(line: str, name: str):
        m = re.search(rf"{name}=([^ ]+)", line)
        return m.group(1).strip() if m else None

    def _to_int(x):
        try:
            return int(float(str(x)))
        except Exception:
            return None

    def _to_float(x):
        try:
            v = float(str(x))
            return v if math.isfinite(v) else None
        except Exception:
            return None

    def _none_like(x) -> bool:
        if x is None:
            return True
        return str(x).strip().lower() in {"", "none", "n/a", "na", "nan", "null"}

    def sf_print(*args, **kwargs):
        line = " ".join(str(a) for a in args)

        original_print(*args, **kwargs)

        if not line.startswith("[DASH]"):
            return

        drone = _to_int(_field(line, "drone"))
        status = str(_field(line, "status") or "").strip().lower()
        instant = _field(line, "instant")
        angle_med = _field(line, "angle_med")
        coh = _to_float(_field(line, "coh"))
        reason = str(_field(line, "reason") or "").strip().lower()

        state["seen"] += 1

        # precision 진입 직후 튀는 값 무시
        if state["seen"] <= warmup:
            state["drone_lost"] = 0
            state["aoa_lost"] = 0
            return

        # Drone lost: top-k drone vote 자체가 부족한 경우
        drone_ok = True if drone is None else (drone >= min_drone)

        # AoA lost:
        # - trusted가 아니거나
        # - instant sector 없음
        # - angle_med 없음
        # - coherence가 너무 낮음
        # - no valid / votes scattered 계열이면 lost로 봄
        instant_ok = not _none_like(instant)
        angle_ok = not _none_like(angle_med)
        coh_ok = True if coh is None else (coh >= min_coh)

        bad_reason = ("no valid" in reason) or ("votes scattered" in reason)
        status_ok = status == "trusted"

        aoa_ok = status_ok and instant_ok and angle_ok and coh_ok and not bad_reason

        if drone_ok:
            state["drone_lost"] = 0
        else:
            state["drone_lost"] += 1

        if aoa_ok:
            state["aoa_lost"] = 0
        else:
            state["aoa_lost"] += 1

        if state["drone_lost"] >= limit:
            original_print(
                f"[AUTO-RETURN] Drone lost {state['drone_lost']}/{limit} updates "
                f"(drone={drone}, status={status}, reason={reason}) -> return to SCAN"
            )
            __import__("os").environ["RF_SF_AUTO_RETURNING"] = "1"
            raise SystemExit(20)
        if state["aoa_lost"] >= limit:
            original_print(
                f"[AUTO-RETURN] AoA/coherence lost {state['aoa_lost']}/{limit} updates "
                f"(status={status}, instant={instant}, angle_med={angle_med}, coh={coh}, reason={reason}) -> return to SCAN"
            )
            raise SystemExit(20)

    sf_print._sf_auto_return_hooked = True
    builtins.print = sf_print


_install_sf_auto_return_print_hook()



from src.viewer.live_aoa_sector_dashboard import (
    SectorDashboardRenderer,
    load_dashboard_cfg,
)


def handle_key(
    key: str | None,
    *,
    state: ViewerState,
    args: Any,
    receiver: Any,
    aoa_runtime: Any,
    cnn_runtime: Any,
) -> None:
    if key is None:
        return

    if key == "save_profile":
        state.capture_active = True
        state.capture_saved_count = 0
        state.capture_target = max(1, int(args.capture_trusted_n))
        print(
            f"[CAPTURE] start trusted-only capture target={state.capture_target} "
            f"d={state.distance_m:.1f}m "
            f"angle={float(getattr(state, 'true_angle_deg', 0.0)):.1f}deg "
            f"-> {getattr(state, 'profile_csv_path', 'n/a')}",
            flush=True,
        )
        return

    if key == "cancel_capture":
        state.capture_active = False
        print(
            f"[CAPTURE] canceled at {int(getattr(state, 'capture_saved_count', 0))}/"
            f"{int(getattr(state, 'capture_target', 0))}",
            flush=True,
        )
        return

    if key == "quit":
        state.running = False
    elif key == "pause":
        state.toggle_pause()
    elif key == "gain_down":
        state.step_gain(-args.gain_step, args.min_gain, args.max_gain)
        base.apply_receiver_gain(receiver, state.gain)
        aoa_runtime.update_gain(state.gain)
        cnn_runtime.reset_history()
        print(f"[GAIN] gain -> {state.gain:.1f}", flush=True)
    elif key == "gain_up":
        state.step_gain(args.gain_step, args.min_gain, args.max_gain)
        base.apply_receiver_gain(receiver, state.gain)
        aoa_runtime.update_gain(state.gain)
        cnn_runtime.reset_history()
        print(f"[GAIN] gain -> {state.gain:.1f}", flush=True)
    elif key == "1":
        state.distance_m = max(0.0, float(state.distance_m) - float(args.distance_step_m))
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "2":
        state.distance_m = float(state.distance_m) + float(args.distance_step_m)
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "0":
        state.distance_m = 0.0
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "a":
        state.true_angle_deg = float(getattr(state, "true_angle_deg", 0.0)) - float(args.angle_step_deg)
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == "d":
        state.true_angle_deg = float(getattr(state, "true_angle_deg", 0.0)) + float(args.angle_step_deg)
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == "c":
        state.true_angle_deg = 0.0
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == ",":
        delta_deg = -1.0
        state.phase_offset_live_delta_deg = float(getattr(state, "phase_offset_live_delta_deg", 0.0)) + delta_deg
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(delta_deg))
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta={state.phase_offset_live_delta_deg:.1f}deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )
    elif key == ".":
        delta_deg = 1.0
        state.phase_offset_live_delta_deg = float(getattr(state, "phase_offset_live_delta_deg", 0.0)) + delta_deg
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(delta_deg))
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta={state.phase_offset_live_delta_deg:.1f}deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )
    elif key == "m":
        reset_deg = -float(getattr(state, "phase_offset_live_delta_deg", 0.0))
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(reset_deg))
        state.phase_offset_live_delta_deg = 0.0
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta reset -> 0.0deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )


def run() -> int:
    args = base.apply_defaults(base.parse_args())
    dash_cfg = load_dashboard_cfg(args)

    if not bool(dash_cfg.get("enabled", True)):
        print("[WARN] sector_dashboard.enabled=false, but script will still run.")

    state = ViewerState(
        mode="sector_dashboard",
        gain=args.gain,
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        distance_m=args.distance_m,
        memo=args.memo,
        target_fps=args.target_fps,
    )
    state.true_angle_deg = float(args.true_angle_deg)
    state.phase_offset_live_delta_deg = 0.0
    state.phase_offset_total_deg = 0.0

    raw_gate = RawNoiseGate(detect_config_path=Path(args.config_dir) / "detect.yaml")
    if bool(args.disable_raw_gate):
        raw_gate.enabled = False
        print("[DRY-RUN] raw noise gate disabled by --disable-raw-gate")

    receiver = base.build_receiver(args)
    cnn_runtime = base.build_cnn_runtime(args)
    aoa_runtime = base.build_aoa_runtime(args)
    state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))

    renderer = SectorDashboardRenderer(
        window_name="RF Drone Detection Runtime",
        target_fps=args.target_fps,
        width=int(dash_cfg.get("canvas_width", 1120)),
        height=int(dash_cfg.get("canvas_height", 720)),
        blink_on_hold=bool(dash_cfg.get("blink_on_hold", True)),
        fade_on_signal_lost=bool(dash_cfg.get("fade_on_signal_lost", True)),
    )

    lock = base.SectorLockState()

    out_dir = Path(args.sector_profile.get("output_dir", "outputs/aoa_sector_profiles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_csv = out_dir / f"{ts}_gain{args.gain:g}_cf{args.center_freq}_sector_profile.csv"

    state.capture_active = False
    state.capture_saved_count = 0
    state.capture_target = max(1, int(args.capture_trusted_n))
    state.profile_csv_name = profile_csv.name
    state.profile_csv_path = str(profile_csv)

    last_sector: dict[str, Any] = {
        "sector_status": "no_signal",
        "locked_sector_name": "",
        "instant_sector_name": "",
        "angle_median": "",
        "angle_spread": "",
        "median_coherence": "",
        "median_raw_p99": "",
        "valid_aoa_count": 0,
        "sector_votes": "",
        "reason": "initial",
    }
    last_selected_raw: dict[str, Any] = {}
    last_cnn_result: dict[str, Any] = {
        "cnn_raw_class_name": "n/a",
        "cnn_raw_confidence": 0.0,
        "cnn_positive_votes": 0,
        "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
    }
    last_raw_pass_count = 0
    last_cnn_drone_count = 0
    last_topk_count = 0

    try:
        while state.running:
            if state.paused:
                key = renderer.render(
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    sector=last_sector,
                    selected_raw=last_selected_raw,
                    cnn_result=last_cnn_result,
                    raw_pass_count=last_raw_pass_count,
                    cnn_drone_count=last_cnn_drone_count,
                    topk_count=last_topk_count,
                    paused=True,
                )
                handle_key(
                    key,
                    state=state,
                    args=args,
                    receiver=receiver,
                    aoa_runtime=aoa_runtime,
                    cnn_runtime=cnn_runtime,
                )
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

            topk_idx = base.select_topk_indices(gate_results, args.top_k)

            topk_items: list[dict[str, Any]] = []
            for idx in topk_idx:
                image, cnn_raw = base.run_cnn_raw_no_history(cnn_runtime, blocks[idx])
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
                vote_raw = base.pick_vote_cnn_result(topk_items)
                cnn_result = (
                    base.update_cnn_history_once(cnn_runtime, vote_raw)
                    if vote_raw
                    else {
                        "cnn_raw_class_name": "NO_CNN_RESULT",
                        "cnn_raw_confidence": 0.0,
                        "cnn_candidate": False,
                        "cnn_confirmed": False,
                        "cnn_positive_votes": 0,
                        "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
                    }
                )
                aoa_candidates = base.build_aoa_candidates(topk_items, aoa_runtime, args)

            cnn_drone_count = sum(
                1 for item in topk_items
                if bool(item["cnn"].get("cnn_candidate", False))
            )

            sector = base.update_sector_lock(
                lock=lock,
                candidates=aoa_candidates,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                args=args,
            )

            selected_raw = dict(raws[topk_idx[0]]) if topk_idx else {}
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
                "true_angle_deg": float(getattr(state, "true_angle_deg", 0.0)),
                "phase_offset_live_delta_deg": float(getattr(state, "phase_offset_live_delta_deg", 0.0)),
                "phase_offset_total_deg": float(getattr(state, "phase_offset_total_deg", 0.0)),
                "memo": str(state.memo),
                **selected_raw,
                **cnn_result,
                **sector,
            }

            if bool(getattr(state, "capture_active", False)):
                capture_ok = (
                    str(sector.get("sector_status", "")) == "trusted"
                    and int(sector.get("valid_aoa_count", 0) or 0) >= 2
                    and int(cnn_drone_count) >= 2
                    and str(sector.get("locked_sector_name", "")).strip() != ""
                )

                if capture_ok:
                    state.capture_saved_count = int(getattr(state, "capture_saved_count", 0)) + 1

                    capture_row = dict(row)
                    capture_row.update(
                        {
                            "capture_active": True,
                            "capture_saved_index": int(state.capture_saved_count),
                            "capture_target": int(state.capture_target),
                            "capture_done": bool(int(state.capture_saved_count) >= int(state.capture_target)),
                            "capture_source": "sector_dashboard",
                        }
                    )

                    base.append_sector_profile(profile_csv, capture_row)

                    print(
                        f"[CAPTURE] saved {int(state.capture_saved_count)}/{int(state.capture_target)} "
                        f"sector={sector.get('locked_sector_name')} "
                        f"d={state.distance_m:.1f}m "
                        f"angle={float(getattr(state, 'true_angle_deg', 0.0)):.1f}deg "
                        f"-> {profile_csv}",
                        flush=True,
                    )

                    if int(state.capture_saved_count) >= int(state.capture_target):
                        state.capture_active = False
                        print(f"[CAPTURE] done -> {profile_csv}", flush=True)

            last_sector = sector
            last_selected_raw = selected_raw
            last_cnn_result = cnn_result
            last_raw_pass_count = raw_pass_count
            last_cnn_drone_count = cnn_drone_count
            last_topk_count = len(topk_items)

            if not args.disable_cli_log and state.update_index % int(args.cli_log_every_n) == 0:
                print(
                    f"[DASH] idx={state.update_index} "
                    f"raw_pass={raw_pass_count} topk={len(topk_items)} drone={cnn_drone_count} "
                    f"status={sector['sector_status']} "
                    f"instant={sector['instant_sector_name'] or 'None'} "
                    f"locked={sector['locked_sector_name'] or 'None'} "
                    f"angle_med={base.fmt(sector.get('angle_median'), 3)} "
                    f"coh={base.fmt(sector.get('median_coherence'), 3)} "
                    f"p99={base.fmt(sector.get('median_raw_p99'), 4)} "
                    f"reason={sector['reason']}",
                    flush=True,
                )

            key = renderer.render(
                state=state,
                args=args,
                dash_cfg=dash_cfg,
                sector=sector,
                selected_raw=selected_raw,
                cnn_result=cnn_result,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                topk_count=len(topk_items),
                paused=False,
            )

            handle_key(
                key,
                state=state,
                args=args,
                receiver=receiver,
                aoa_runtime=aoa_runtime,
                cnn_runtime=cnn_runtime,
            )

    except KeyboardInterrupt:
        return 130
    finally:
        try:
            receiver.close()
        except Exception:
            pass
        _exc_type, _exc, _tb = __import__("sys").exc_info()
        _auto_return_exit = isinstance(_exc, SystemExit) and getattr(_exc, "code", None) == 20
        _keep_window = __import__("os").environ.get("RF_SF_KEEP_WINDOW") == "1"
        _auto_returning = __import__("os").environ.get("RF_SF_AUTO_RETURNING") == "1"

        # sf auto-return이면 precision 창을 닫지 않고 scan runtime이 같은 창 이름으로 재사용하게 둔다.
        if not (_keep_window and (_auto_returning or _auto_return_exit)):
            renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
