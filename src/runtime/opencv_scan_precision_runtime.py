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


def _sector_name_from_angle_for_dashboard(
    angle_deg: float | None,
    args: Any | None,
) -> str:
    """
    PrecisionAnalyzer의 sector_label은 'Far Right', 'Right 2' 같은 표시용 문자열일 수 있다.
    SectorDashboardRenderer는 aoa_sector.yaml의 bin name
    예: RIGHT_45_60, RIGHT_30_45, CENTER 를 기대한다.
    따라서 UI 표시는 angle을 기준으로 현재 preset bin name으로 다시 매핑한다.
    """
    if angle_deg is None:
        return ""

    try:
        angle = float(angle_deg)
    except Exception:
        return ""

    bins = []
    if args is not None:
        try:
            bins = list(getattr(args, "sector_preset", {}).get("bins", []) or [])
        except Exception:
            bins = []

    # fallback: configs/aoa_sector.yaml fixed_bins_7sector와 동일
    if not bins:
        bins = [
            {"name": "LEFT_60_45", "min_deg": -60.0, "max_deg": -45.0},
            {"name": "LEFT_45_30", "min_deg": -45.0, "max_deg": -30.0},
            {"name": "LEFT_30_15", "min_deg": -30.0, "max_deg": -15.0},
            {"name": "CENTER", "min_deg": -15.0, "max_deg": 15.0},
            {"name": "RIGHT_15_30", "min_deg": 15.0, "max_deg": 30.0},
            {"name": "RIGHT_30_45", "min_deg": 30.0, "max_deg": 45.0},
            {"name": "RIGHT_45_60", "min_deg": 45.0, "max_deg": 60.0},
        ]

    for b in bins:
        try:
            name = str(b.get("name", "") or "")
            mn = float(b.get("min_deg"))
            mx = float(b.get("max_deg"))
        except Exception:
            continue

        # 마지막 bin의 max 경계 포함
        if mn <= angle < mx or abs(angle - mx) < 1e-9:
            return name

    # AoA estimator가 ±60 밖으로 살짝 튀는 경우 UI에서는 끝 sector로 clamp
    try:
        left_bins = sorted(bins, key=lambda x: float(x.get("min_deg", 0.0)))
        if angle < float(left_bins[0].get("min_deg")):
            return str(left_bins[0].get("name", "") or "")
        if angle > float(left_bins[-1].get("max_deg")):
            return str(left_bins[-1].get("name", "") or "")
    except Exception:
        pass

    return ""


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



def _sector_has_display_target(sector: dict[str, Any]) -> bool:
    if not isinstance(sector, dict):
        return False

    status = str(sector.get("sector_status", "") or "").lower().strip()
    locked = str(sector.get("locked_sector_name", "") or "").strip()
    instant = str(sector.get("instant_sector_name", "") or "").strip()

    if status in {"", "none", "no_signal", "scanning", "precision"}:
        return False

    return bool(locked or instant)


def _reset_precision_ui_hold() -> None:
    _apply_precision_ui_hold._last_sector = None
    _apply_precision_ui_hold._miss_count = 0


def _apply_precision_ui_hold(
    sector: dict[str, Any],
    *,
    hold_max_updates: int = 8,
    no_signal_clear_updates: int = 10,
) -> dict[str, Any]:
    """
    SectorDashboardRenderer는 sector_status='hold...'이면 마지막 sector를 hold 색으로 유지한다.
    Runtime 쪽에서도 마지막 유효 sector를 저장해두고, 새 frame에서 AoA/sector가 누락되면
    바로 빈 화면으로 만들지 않고 hold 상태로 넘긴다.
    """
    hold_max_updates = max(1, int(hold_max_updates))
    no_signal_clear_updates = max(hold_max_updates, int(no_signal_clear_updates))

    if not hasattr(_apply_precision_ui_hold, "_last_sector"):
        _reset_precision_ui_hold()

    if sector is None:
        sector = _empty_sector("precision")

    # 새 frame에 표시 가능한 sector가 있으면 last를 갱신한다.
    if _sector_has_display_target(sector):
        _apply_precision_ui_hold._last_sector = dict(sector)
        _apply_precision_ui_hold._miss_count = 0
        return sector

    last = getattr(_apply_precision_ui_hold, "_last_sector", None)
    if not last:
        return sector

    miss_count = int(getattr(_apply_precision_ui_hold, "_miss_count", 0)) + 1
    _apply_precision_ui_hold._miss_count = miss_count

    # 너무 오래 누락되면 완전히 clear한다.
    if miss_count > no_signal_clear_updates:
        _reset_precision_ui_hold()
        return sector

    held = dict(last)

    if miss_count <= hold_max_updates:
        held["sector_status"] = "hold_no_valid_aoa"
        held["reason"] = "ui_hold_last_valid_sector"
    else:
        held["sector_status"] = "uncertain"
        held["reason"] = "ui_hold_expired_but_not_cleared"

    held["hold_count"] = miss_count
    held["no_signal_count"] = miss_count

    # locked가 비어 있으면 instant라도 locked로 승격해서 fan이 꺼지지 않게 한다.
    if not str(held.get("locked_sector_name", "") or "").strip():
        held["locked_sector_name"] = held.get("instant_sector_name", "") or ""

    if not str(held.get("instant_sector_name", "") or "").strip():
        held["instant_sector_name"] = held.get("locked_sector_name", "") or ""

    return held



def _handle_phase_offset_key(
    key: str | None,
    *,
    analyzer: Any,
    state: Any | None = None,
    step_deg: float = 1.0,
    verbose: bool = True,
) -> bool:
    """
    OpenCV start runtime live phase offset control.

    Keys:
      , : phase offset -step_deg
      . : phase offset +step_deg
      m : reset phase offset to 0 deg
    """
    if key not in {",", ".", "m"}:
        return False

    try:
        import numpy as _np
    except Exception:
        _np = None

    old_rad = float(getattr(analyzer, "phase_offset_rad", 0.0) or 0.0)

    if key == ",":
        delta_deg = -float(step_deg)
    elif key == ".":
        delta_deg = float(step_deg)
    else:
        delta_deg = None

    if key == "m":
        new_rad = 0.0
    else:
        if _np is not None:
            new_rad = old_rad + float(_np.deg2rad(delta_deg))
        else:
            import math as _math
            new_rad = old_rad + _math.radians(float(delta_deg))

    setattr(analyzer, "phase_offset_rad", float(new_rad))

    try:
        if state is not None:
            import math as _math
            state.phase_offset_total_deg = float(_math.degrees(new_rad))
    except Exception:
        pass

    if verbose:
        import math as _math
        print(
            f"[PHASE OFFSET] key={key!r} "
            f"offset={_math.degrees(new_rad):+.2f}deg",
            flush=True,
        )

    return True



def _coherence_hold_status(result: Any) -> str:
    """
    Precision confirmed 상태에서 AoA/coherence가 유지 가능한지 판단한다.

    return:
      - "ok"      : angle/coherence/sector가 모두 유효
      - "bad"     : confirmed인데 AoA/coherence가 무너짐
      - "neutral" : 아직 confirmed 전이거나 smoothing 준비 중
    """
    if result is None:
        return "neutral"

    confirmed = bool(getattr(result, "confirmed", False))
    if not confirmed:
        return "neutral"

    # confirmed 상태에서 현재 raw CNN label이 NotDrone이면 유지 신뢰도를 낮게 본다.
    try:
        if _is_cnn_not_drone_result(result):
            return "bad"
    except Exception:
        pass

    smooth_reject = getattr(result, "smooth_reject", None)

    # confirmed 직후 smoothing history가 부족한 건 정상적인 준비 단계라 bad로 치지 않는다.
    if smooth_reject == "not_enough_samples":
        return "neutral"

    # angle jump reject는 멀티패스/불안정 AoA 가능성이 높으므로 bad
    if smooth_reject == "angle_jump_rejected":
        return "bad"

    angle_valid = getattr(result, "angle_valid", None)
    coherence_passed = getattr(result, "coherence_passed", None)
    if coherence_passed is None:
        coherence_passed = getattr(result, "coh_passed", None)

    sector_valid = getattr(result, "sector_valid", None)

    if angle_valid is True and coherence_passed is True and sector_valid is True:
        return "ok"

    return "bad"



def _reset_precision_candidate_state(
    *,
    analyzer: Any,
    state: Any | None = None,
    verbose: bool = True,
) -> None:
    """
    새 scan candidate가 precision으로 넘어갈 때 이전 후보의 상태를 끊는다.

    Reset 대상:
    - CNN temporal vote history
    - AoA smoothing angle history
    - sector/UI hold state
    - precision payload hold state

    주의:
    - phase_offset_rad는 live 보정값이므로 reset하지 않는다.
    """
    # 1) CNN temporal voting reset
    reset_temporal = getattr(analyzer, "reset_temporal_history", None)
    if callable(reset_temporal):
        reset_temporal()
    else:
        vote_history = getattr(analyzer, "vote_history", None)
        if hasattr(vote_history, "clear"):
            vote_history.clear()

    # 2) AoA smoothing reset
    angle_history = getattr(analyzer, "angle_history", None)
    if hasattr(angle_history, "clear"):
        angle_history.clear()

    # 3) UI sector hold reset
    reset_ui_hold = globals().get("_reset_precision_ui_hold")
    if callable(reset_ui_hold):
        reset_ui_hold()

    # 4) 오른쪽 panel payload hold reset
    reset_payload_hold = globals().get("_reset_precision_payload_hold")
    if callable(reset_payload_hold):
        reset_payload_hold()

    # 5) state 쪽 표시용 값은 phase offset 빼고 초기화
    try:
        if state is not None:
            for attr in (
                "last_locked_sector_name",
                "last_instant_sector_name",
                "last_angle_deg",
                "last_smoothed_angle_deg",
                "last_coherence",
            ):
                if hasattr(state, attr):
                    setattr(state, attr, None)
    except Exception:
        pass

    if verbose:
        print(
            "[PRECISION RESET] new candidate -> "
            "cnn_vote_history cleared, aoa_smoothing cleared, ui_hold cleared",
            flush=True,
        )



def _is_cnn_not_drone_result(result: Any) -> bool:
    """
    현재 block의 CNN raw label 기준으로 NotDrone 여부를 판단한다.
    temporal confirmed_status가 True여도 현재 CNN label이 NotDrone이면 True.
    """
    label = str(
        getattr(result, "cnn_label", None)
        or getattr(result, "class_name", None)
        or getattr(result, "label", None)
        or ""
    ).strip().lower()

    compact = (
        label.replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
    )

    if compact in {"notdrone", "nondrone", "background", "noise", "none"}:
        return True

    # 혹시 label 문자열이 다른 형태로 들어올 경우 보조 판정
    if "not" in compact and "drone" in compact:
        return True

    return False



def _is_meaningful_ui_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    s = str(value).strip().lower()
    if s in {"", "n/a", "none", "nan"}:
        return False
    return True


def _payload_has_any_value(payload: dict[str, Any], keys: list[str]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in keys:
        if _is_meaningful_ui_value(payload.get(key)):
            return True
    return False


def _reset_precision_payload_hold() -> None:
    _apply_precision_payload_hold._last_sector = None
    _apply_precision_payload_hold._last_raw = None
    _apply_precision_payload_hold._last_cnn = None
    _apply_precision_payload_hold._last_raw_pass_count = 0
    _apply_precision_payload_hold._last_cnn_drone_count = 0
    _apply_precision_payload_hold._miss_count = 0


def _apply_precision_payload_hold(
    *,
    sector: dict[str, Any],
    selected_raw: dict[str, Any],
    cnn_result: dict[str, Any],
    raw_pass_count: int,
    cnn_drone_count: int,
    max_miss_updates: int = 30,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    """
    OpenCV dashboard는 매 frame 새 dict를 그린다.
    precision 분석 중 일부 frame에서 angle/CNN/raw 값이 비면 n/a로 덮이므로,
    마지막 유효 payload를 저장했다가 값 누락 시 그대로 유지한다.
    """
    max_miss_updates = max(1, int(max_miss_updates))

    if not hasattr(_apply_precision_payload_hold, "_last_sector"):
        _reset_precision_payload_hold()

    sector = dict(sector or {})
    selected_raw = dict(selected_raw or {})
    cnn_result = dict(cnn_result or {})

    sector_valid = _payload_has_any_value(
        sector,
        [
            "locked_sector_name",
            "instant_sector_name",
            "angle_median",
            "median_angle_deg",
            "median_coherence",
            "median_raw_p99",
            "sector_votes",
            "votes",
        ],
    )

    raw_valid = _payload_has_any_value(
        selected_raw,
        [
            "raw_abs_p99",
            "median_raw_p99",
            "raw_abs_p95",
            "raw_abs_mean",
            "raw_rms",
            "frame_power_p99",
        ],
    )

    cnn_valid = _payload_has_any_value(
        cnn_result,
        [
            "cnn_raw_class_name",
            "cnn_raw_confidence",
            "cnn_probability",
            "label",
            "class_name",
            "confidence",
            "probability",
        ],
    )

    any_valid = sector_valid or raw_valid or cnn_valid or raw_pass_count > 0 or cnn_drone_count > 0

    if any_valid:
        _apply_precision_payload_hold._miss_count = 0

        if sector_valid:
            _apply_precision_payload_hold._last_sector = dict(sector)
        if raw_valid:
            _apply_precision_payload_hold._last_raw = dict(selected_raw)
        if cnn_valid:
            _apply_precision_payload_hold._last_cnn = dict(cnn_result)

        if raw_pass_count > 0:
            _apply_precision_payload_hold._last_raw_pass_count = int(raw_pass_count)
        if cnn_drone_count > 0:
            _apply_precision_payload_hold._last_cnn_drone_count = int(cnn_drone_count)

        # 유효한 값은 현재 frame 우선
        out_sector = dict(_apply_precision_payload_hold._last_sector or sector)
        out_raw = dict(_apply_precision_payload_hold._last_raw or selected_raw)
        out_cnn = dict(_apply_precision_payload_hold._last_cnn or cnn_result)

        return (
            out_sector,
            out_raw,
            out_cnn,
            int(raw_pass_count if raw_pass_count > 0 else _apply_precision_payload_hold._last_raw_pass_count),
            int(cnn_drone_count if cnn_drone_count > 0 else _apply_precision_payload_hold._last_cnn_drone_count),
        )

    # 값이 전부 빈 frame이면 이전 payload 유지
    miss_count = int(getattr(_apply_precision_payload_hold, "_miss_count", 0)) + 1
    _apply_precision_payload_hold._miss_count = miss_count

    if miss_count > max_miss_updates:
        _reset_precision_payload_hold()
        return sector, selected_raw, cnn_result, int(raw_pass_count), int(cnn_drone_count)

    held_sector = dict(getattr(_apply_precision_payload_hold, "_last_sector", None) or sector)
    held_raw = dict(getattr(_apply_precision_payload_hold, "_last_raw", None) or selected_raw)
    held_cnn = dict(getattr(_apply_precision_payload_hold, "_last_cnn", None) or cnn_result)

    # sector가 hold 상태임을 UI에 알려 fan이 꺼지지 않게 함
    if held_sector:
        status = str(held_sector.get("sector_status", "") or "").lower()
        if status not in {"trusted", "candidate"}:
            held_sector["sector_status"] = "hold_no_valid_aoa"
        elif status == "trusted":
            held_sector["sector_status"] = "hold_last_valid"
        held_sector["reason"] = "payload_hold_last_valid"
        held_sector["hold_count"] = miss_count

    return (
        held_sector,
        held_raw,
        held_cnn,
        int(getattr(_apply_precision_payload_hold, "_last_raw_pass_count", raw_pass_count)),
        int(getattr(_apply_precision_payload_hold, "_last_cnn_drone_count", cnn_drone_count)),
    )



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


def _sector_from_precision(result: Any, args: Any | None = None) -> dict[str, Any]:
    angle = _safe_float(
        getattr(result, "aoa_smoothed_angle_deg", None),
        _safe_float(getattr(result, "angle_deg", None), None),
    )

    coh = _safe_float(getattr(result, "coherence", None), None)
    raw_p99 = _safe_float(getattr(result, "raw_gate_score_max", None), None)

    # UI dashboard는 'Far Right' 같은 표시용 label이 아니라
    # aoa_sector.yaml의 bin name 예: RIGHT_45_60 을 기대한다.
    dashboard_sector_label = _sector_name_from_angle_for_dashboard(angle, args)

    raw_sector_label = (
        getattr(result, "sector_label", None)
        or getattr(result, "sector_name", None)
        or ""
    )

    sector_label = dashboard_sector_label or str(raw_sector_label or "")

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
        "angle_median": angle if angle is not None else "",
        "angle_spread": "",
        "median_coherence": coh if coh is not None else "",
        "median_raw_p99": raw_p99 if raw_p99 is not None else "",
        "dominant_sector_ratio": 1.0 if sector_label else "",
        "valid_aoa_count": 1 if angle_valid else 0,
        "votes": votes_text,
        "sector_votes": votes_text,
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
    probability = _safe_float(getattr(result, "drone_probability", None), confidence)

    temporal_history = getattr(result, "temporal_history", None)

    positive_votes = _safe_int(getattr(result, "drone_vote_count", 0), 0)
    confirm_votes = _safe_int(getattr(result, "temporal_window", 0), 0)

    # temporal_window가 없으면 history 길이로 보조 계산
    try:
        if confirm_votes <= 0 and temporal_history is not None:
            confirm_votes = len(list(temporal_history))
    except Exception:
        pass

    # 혹시 drone_vote_count가 비어 있으면 history에서 Drone 계열만 count
    try:
        if positive_votes <= 0 and temporal_history is not None:
            hist = list(temporal_history)
            positive_votes = sum(
                1 for x in hist
                if str(x).strip().lower().replace("_", "").replace("-", "") in {"drone", "true", "1"}
            )
    except Exception:
        pass

    # dashboard 오른쪽 패널은 cnn_raw_* key를 읽는다.
    return {
        # 기존 호환 key
        "label": label,
        "class_name": label,
        "confidence": confidence if confidence is not None else "",
        "probability": probability if probability is not None else "",

        # live_aoa_sector_dashboard.py text panel용 key
        "cnn_raw_class_name": label,
        "cnn_raw_confidence": confidence if confidence is not None else "",
        "cnn_probability": probability if probability is not None else "",
        "cnn_positive_votes": int(positive_votes),
        "cnn_confirm_votes": int(confirm_votes),
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

    # UI fallback defaults to avoid unbound local variables.
    # Some frames, especially PRECISION_ANALYZING handoff frames, may not have
    # full raw/CNN/AoA payload yet. The renderer still needs stable values.
    selected_raw_for_ui = _selected_raw_from_precision(result)
    cnn_for_ui = _cnn_from_precision(result)
    raw_pass_count_for_ui = raw_pass_count
    drone_votes_for_ui = drone_votes

    # sector_for_ui는 이전 sector hold 패치가 있으면 이미 만들어져 있을 수 있다.
    # 없으면 여기서 만든다.
    try:
        sector_for_ui
    except Exception:
        try:
            sector_for_ui = _sector_from_precision(result, args)
        except TypeError:
            sector_for_ui = _sector_from_precision(result)

    selected_raw_for_ui = _selected_raw_from_precision(result)
    cnn_for_ui = _cnn_from_precision(result)

    sector_for_ui, selected_raw_for_ui, cnn_for_ui, raw_pass_count_for_ui, drone_votes_for_ui = (
        _apply_precision_payload_hold(
            sector=sector_for_ui,
            selected_raw=selected_raw_for_ui,
            cnn_result=cnn_for_ui,
            raw_pass_count=raw_pass_count_for_ui,
            cnn_drone_count=drone_votes_for_ui,
            max_miss_updates=30,
        )
    )

    hold_cfg = {}
    try:
        hold_cfg = dict(getattr(args, "sector_root", {}).get("hold", {}) or {})
    except Exception:
        hold_cfg = {}

    sector_for_ui = _sector_from_precision(result, args)
    sector_for_ui = _apply_precision_ui_hold(
        sector_for_ui,
        hold_max_updates=int(hold_cfg.get("hold_max_updates", 8)),
        no_signal_clear_updates=int(hold_cfg.get("no_signal_clear_updates", 10)),
    )

    return renderer.render(
        state=state,
        args=args,
        dash_cfg=dash_cfg,
        sector=sector_for_ui,
        selected_raw=selected_raw_for_ui,
        cnn_result=cnn_for_ui,
        raw_pass_count=raw_pass_count_for_ui,
        cnn_drone_count=drone_votes_for_ui,
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
            if verbose:
                print("[SCAN LOOP] begin sweep")
            sweep_events: list[dict[str, Any]] = []

            # 1) SCAN: 전체 scan_freqs를 한 바퀴 끝까지 돈다.
            for current_freq in scan_freqs:
                key = _render_scan(
                    renderer=renderer,
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    scan_freqs=scan_freqs,
                    current_freq=current_freq,
                )

                if verbose and key:
                    print(f"[KEY DEBUG][SCAN-BEFORE] key={key!r} cf={float(current_freq) / 1e9:.3f}GHz")

                if key == "quit":
                    print("[KEY DEBUG] renderer returned quit during scan")
                    state.running = False
                    break
                if _handle_phase_offset_key(key, analyzer=analyzer, state=state, verbose=verbose):
                    continue
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
                sweep_events.append(scan_event)

                key = _render_scan(
                    renderer=renderer,
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    scan_freqs=scan_freqs,
                    current_freq=current_freq,
                    scan_event=scan_event,
                )

                if verbose and key:
                    print(f"[KEY DEBUG][SCAN-AFTER] key={key!r} cf={float(current_freq) / 1e9:.3f}GHz")

                if key == "quit":
                    print("[KEY DEBUG] scan renderer returned quit after scan_event -> ignored once")
                    key = None
                if _handle_phase_offset_key(key, analyzer=analyzer, state=state, verbose=verbose):
                    continue
                if key == "pause":
                    state.toggle_pause()

                if verbose:
                    print(
                        f"[SWEEP] cf={float(current_freq) / 1e9:.3f}GHz "
                        f"triggered={bool(scan_event.get('triggered', False))} "
                        f"pass={scan_event.get('pass_count')}/{scan_event.get('usable_blocks')} "
                        f"score={scan_event.get('best_score_max')}"
                    )

            if not state.running:
                break

            # 2) 한 바퀴가 끝난 뒤에만 후보를 고른다.
            #
            # triggered=True만으로는 너무 약하다.
            # 노이즈 threshold를 살짝 넘은 정도의 후보는 드론 OFF 상태에서도 생기므로,
            # threshold 대비 score 비율과 pass_count를 한 번 더 본다.
            scan_candidate_score_ratio_min = float(scan_cfg.get("candidate_score_ratio_min", 10.0))
            preferred_candidate_score_ratio_min = float(
                scan_cfg.get("preferred_candidate_score_ratio_min", 3.0)
            )
            scan_candidate_pass_min = int(scan_cfg.get("candidate_pass_min", 3))
            preferred_candidate_pass_min = int(scan_cfg.get("preferred_candidate_pass_min", 1))

            preferred_candidate_freqs_hz = {
                2.450e9,
            }

            def _candidate_ratio(event: dict[str, Any]) -> float:
                score = _safe_float(event.get("best_score_max"), 0.0) or 0.0
                threshold = _safe_float(event.get("threshold"), None)

                if threshold is None or threshold <= 0:
                    return 999.0 if score > 0 else 0.0

                return float(score) / float(threshold)

            candidates: list[dict[str, Any]] = []
            rejected_candidates: list[tuple[dict[str, Any], int, float]] = []

            for event in sweep_events:
                if not bool(event.get("triggered", False)):
                    continue

                pass_count = int(event.get("pass_count", 0) or 0)
                ratio = _candidate_ratio(event)

                center_freq = float(event.get("center_freq", 0.0) or 0.0)
                # scan 간격은 5 MHz이므로 1 MHz tolerance로 preferred band 판정
                is_preferred_freq = any(
                    abs(center_freq - freq_hz) <= 1.0e6
                    for freq_hz in preferred_candidate_freqs_hz
                )

                required_ratio = (
                    preferred_candidate_score_ratio_min
                    if is_preferred_freq
                    else scan_candidate_score_ratio_min
                )
                required_pass = (
                    preferred_candidate_pass_min
                    if is_preferred_freq
                    else scan_candidate_pass_min
                )

                if pass_count >= required_pass and ratio >= required_ratio:
                    candidates.append(event)
                else:
                    event["_required_ratio"] = required_ratio
                    event["_required_pass"] = required_pass
                    event["_preferred_freq"] = is_preferred_freq
                    rejected_candidates.append((event, pass_count, ratio))

            if verbose and rejected_candidates:
                for event, pass_count, ratio in rejected_candidates:
                    print(
                        f"[SWEEP REJECT] "
                        f"cf={float(event.get('center_freq', 0.0)) / 1e9:.3f}GHz "
                        f"pass={pass_count}/{event.get('usable_blocks')} "
                        f"req_pass={int(event.get('_required_pass', scan_candidate_pass_min))} "
                        f"score={event.get('best_score_max')} "
                        f"thr={event.get('threshold')} "
                        f"ratio={ratio:.2f} "
                        f"req={float(event.get('_required_ratio', scan_candidate_score_ratio_min)):.2f} "
                        f"preferred={bool(event.get('_preferred_freq', False))} "
                        f"reason=weak_candidate"
                    )

            if not candidates:
                if verbose:
                    print("[SWEEP] no strong candidate -> continue scan loop")
                continue

            # 후보가 여러 개면 pass_count/score 경쟁이 아니라
            # 사용자가 정한 주파수 우선순위로 선택한다.
            #
            # 현재 우선순위:
            # 1) 2.450 GHz
            # 2) 2.455 GHz
            # 3) 2.460 GHz
            # 4) 그 외 주변 주파수
            preferred_freqs_hz = [
                2.450e9,
                2.455e9,
                2.460e9,
                2.445e9,
                2.465e9,
                2.440e9,
                2.435e9,
            ]

            def _is_same_freq(event: dict[str, Any], target_hz: float) -> bool:
                center_freq = _safe_float(event.get("center_freq", None), None)
                if center_freq is None:
                    return False
                # yaml/float 변환 오차 방지용. scan 간격은 5 MHz라 1 MHz tolerance면 안전함.
                return abs(float(center_freq) - float(target_hz)) <= 1.0e6

            best_event = None

            for preferred_freq_hz in preferred_freqs_hz:
                for event in candidates:
                    if _is_same_freq(event, preferred_freq_hz):
                        best_event = event
                        break
                if best_event is not None:
                    break

            # 혹시 우선순위표에 없는 주파수가 scan_freqs에 추가된 경우 fallback
            if best_event is None:
                best_event = candidates[0]

            locked_freq = float(best_event.get("center_freq"))

            if verbose:
                print(
                    f"[CANDIDATE SELECTED] cf={locked_freq / 1e9:.3f}GHz "
                    f"pass={best_event.get('pass_count')}/{best_event.get('usable_blocks')} "
                    f"score={best_event.get('best_score_max')}"
                )

            # 3) 선택된 best candidate 하나에 대해서만 PRECISION으로 넘어간다.
            _reset_precision_ui_hold()

            _reset_precision_candidate_state(
                analyzer=analyzer,
                state=state,
                verbose=verbose,
            )
            cnn_not_drone_streak = 0
            cnn_not_drone_limit = 5
            raw_only_streak = 0
            raw_only_grace_blocks = int(hold_cfg.get("raw_only_grace_blocks", 3))
            coherence_bad_streak = 0
            coherence_bad_grace_blocks = int(hold_cfg.get("coherence_bad_grace_blocks", 6))

            for hold_idx in range(max_hold_blocks):
                if verbose:
                    print(
                        f"[PRECISION ENTER] cf={locked_freq / 1e9:.3f}GHz "
                        f"hold={hold_idx + 1}/{max_hold_blocks}"
                    )

                # 후보가 선택되면 analyzer가 블록을 읽기 전에 먼저 UI를 PRECISION으로 전환한다.
                # analyzer.analyze()가 blocks_per_decision만큼 수신/분석하는 동안
                # 화면이 계속 SCAN에 머무르는 것처럼 보이는 문제를 막기 위한 handoff 렌더.
                analyzing_result = SimpleNamespace(
                    center_freq=float(locked_freq),
                    stft_done=False,
                    cnn_enabled=True,
                    cnn_label="PRECISION_ANALYZING",
                    cnn_score=None,
                    cnn_class_index=None,
                    cnn_probabilities=None,
                    coherence=None,
                    coherence_passed=None,
                    phase_diff_rad=None,
                    phase_diff_deg=None,
                    angle_deg=None,
                    angle_valid=None,
                    cnn_spectrogram_shape=None,
                    spectrogram_path=None,
                    rx0_stft_path=None,
                    rx1_stft_path=None,
                    sector_index=None,
                    sector_label=None,
                    sector_valid=None,
                    precision_blocks=None,
                    selected_block_index=None,
                    selection_score=None,
                    raw_gate_enabled=None,
                    raw_gate_passed=False,
                    raw_gate_label="ANALYZING",
                    raw_gate_score_max=None,
                    raw_gate_score_median=None,
                    raw_gate_noise_floor=None,
                    raw_gate_threshold=None,
                    raw_gate_threshold_multiplier=None,
                    raw_gate_detection_ratio=None,
                    raw_gate_min_detection_ratio=None,
                    raw_gate_matched_gain=None,
                    raw_gate_matched_by=None,
                    raw_gate_reason="precision_handoff",
                    representative_selection=None,
                    representative_policy=None,
                    drone_probability=None,
                    drone_threshold=None,
                    temporal_window=None,
                    drone_vote_count=0,
                    temporal_history=[],
                    candidate_status=False,
                    confirmed_status=False,
                    final_decision="PRECISION_ANALYZING",
                    aoa_skipped_reason="analyzing",
                    aoa_smoothed_angle_deg=None,
                    aoa_smoothing_valid=False,
                    aoa_smoothing_history_size=0,
                    aoa_smoothing_method=None,
                    aoa_smoothing_rejected_reason="analyzing",
                )

                key = _render_precision(
                    renderer=renderer,
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    scan_freqs=scan_freqs,
                    locked_freq=locked_freq,
                    result=analyzing_result,
                )

                if key == "quit":
                    state.running = False
                    break
                if _handle_phase_offset_key(key, analyzer=analyzer, state=state, verbose=verbose):
                    continue
                if key == "pause":
                    state.toggle_pause()

                if not state.running:
                    break

                result = analyzer.analyze(locked_freq)

                if verbose:
                    aoa_skip_reason = getattr(result, "aoa_skipped_reason", None)
                    angle = getattr(result, "angle_deg", None)
                    angle_valid = getattr(result, "angle_valid", None)
                    coh = getattr(result, "coherence", None)
                    coh_passed = getattr(result, "coherence_passed", None)
                    sector = getattr(result, "sector_label", None)
                    sector_valid = getattr(result, "sector_valid", None)
                    smooth_angle = getattr(result, "aoa_smoothed_angle_deg", None)
                    smooth_valid = getattr(result, "aoa_smoothing_valid", None)
                    smooth_reject = getattr(result, "aoa_smoothing_rejected_reason", None)

                    print(
                        f"[PRECISION RESULT] "
                        f"raw={bool(getattr(result, 'raw_gate_passed', False))} "
                        f"candidate={bool(getattr(result, 'candidate_status', False))} "
                        f"confirmed={bool(getattr(result, 'confirmed_status', False))} "
                        f"label={getattr(result, 'cnn_label', None)} "
                        f"prob={getattr(result, 'drone_probability', None)} "
                        f"angle={angle} "
                        f"angle_valid={angle_valid} "
                        f"coh={coh} "
                        f"coh_passed={coh_passed} "
                        f"sector={sector} "
                        f"sector_valid={sector_valid} "
                        f"smooth_angle={smooth_angle} "
                        f"smooth_valid={smooth_valid} "
                        f"smooth_reject={smooth_reject} "
                        f"aoa_skip_reason={aoa_skip_reason if aoa_skip_reason is not None else 'none'}"
                    )

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
                if _handle_phase_offset_key(key, analyzer=analyzer, state=state, verbose=verbose):
                    continue
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

                cnn_not_drone = _is_cnn_not_drone_result(result)
                if cnn_not_drone:
                    cnn_not_drone_streak += 1
                else:
                    cnn_not_drone_streak = 0

                if verbose:
                    print(
                        f"[PRECISION CNN HOLD] "
                        f"not_drone_streak={cnn_not_drone_streak}/{cnn_not_drone_limit}"
                    )

                if cnn_not_drone_streak >= cnn_not_drone_limit:
                    if verbose:
                        print(
                            f"[PRECISION -> SCAN] "
                            f"cnn_not_drone_streak={cnn_not_drone_streak}/{cnn_not_drone_limit}"
                        )
                    break

                raw_ok = bool(getattr(result, "raw_gate_passed", False))
                candidate_ok = bool(getattr(result, "candidate_status", False))
                confirmed_ok = bool(getattr(result, "confirmed_status", False))

                # 최소 hold block 이후에는 신호/CNN 상태가 모두 약하면 SCAN 복귀
                # Coherence/AoA 유지 정책:
                # CNN confirmed가 켜졌더라도 AoA/coherence가 계속 무너지면
                # 드론 OFF 간섭 또는 멀티패스성 오탐으로 보고 SCAN으로 복귀한다.
                coherence_hold_status = _coherence_hold_status(result)

                if confirmed_ok:
                    if coherence_hold_status == "ok":
                        coherence_bad_streak = 0
                    elif coherence_hold_status == "bad":
                        coherence_bad_streak += 1

                    if verbose:
                        coh_for_log = getattr(result, "coherence", None)
                        if coh_for_log is None:
                            coh_for_log = getattr(result, "coh", None)

                        print(
                            f"[PRECISION COH HOLD] "
                            f"status={coherence_hold_status} "
                            f"bad={coherence_bad_streak}/{coherence_bad_grace_blocks} "
                            f"angle_valid={getattr(result, 'angle_valid', None)} "
                            f"coh={coh_for_log} "
                            f"coh_passed={getattr(result, 'coherence_passed', getattr(result, 'coh_passed', None))} "
                            f"sector_valid={getattr(result, 'sector_valid', None)} "
                            f"smooth_reject={getattr(result, 'smooth_reject', None)}"
                        )

                    if coherence_bad_streak >= coherence_bad_grace_blocks:
                        if verbose:
                            print(
                                f"[PRECISION -> SCAN] coherence_bad_streak="
                                f"{coherence_bad_streak}/{coherence_bad_grace_blocks}"
                            )
                        break
                else:
                    coherence_bad_streak = 0

                # Precision 유지 정책:
                # - confirmed/candidate가 있으면 유지
                # - raw만 살아 있으면 짧은 grace만 허용
                # - raw도 없으면 바로 SCAN 복귀
                if candidate_ok or confirmed_ok:
                    raw_only_streak = 0
                elif raw_ok:
                    raw_only_streak += 1
                else:
                    raw_only_streak = raw_only_grace_blocks + 1

                if hold_idx + 1 >= min_hold_blocks:
                    if not (candidate_ok or confirmed_ok):
                        if raw_ok and raw_only_streak <= raw_only_grace_blocks:
                            if verbose:
                                print(
                                    f"[PRECISION HOLD] raw-only grace "
                                    f"{raw_only_streak}/{raw_only_grace_blocks}"
                                )
                        else:
                            if verbose:
                                print(
                                    f"[PRECISION -> SCAN] "
                                    f"cnn_lost raw_only_streak={raw_only_streak}/"
                                    f"{raw_only_grace_blocks} "
                                    f"raw={raw_ok} candidate={candidate_ok} confirmed={confirmed_ok}"
                                )
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
