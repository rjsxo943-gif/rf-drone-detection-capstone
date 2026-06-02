from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GainThresholdRule:
    name: str
    min_gain: float | None
    max_gain: float | None
    drone_threshold: float


@dataclass(frozen=True)
class TemporalVotingConfig:
    enabled: bool
    window_size: int
    candidate_vote_k: int
    confirmed_vote_k: int
    positive_class: str
    negative_class: str
    require_confirmed_before_aoa: bool
    reject_to_scan: bool


@dataclass(frozen=True)
class RuntimeDecisionConfig:
    backend: str
    model_path: str
    device: str
    positive_class: str
    negative_class: str
    default_drone_threshold: float
    general_threshold: float
    gain_threshold_enabled: bool
    gain_threshold_rules: list[GainThresholdRule]
    temporal_voting: TemporalVotingConfig


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_class_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def load_runtime_decision_config(ml_cfg: dict[str, Any]) -> RuntimeDecisionConfig:
    """
    Load all runtime CNN decision parameters from configs/ml.yaml.

    Rule:
    - model path, class names, thresholds, and temporal voting live in YAML.
    - runtime/viewer code should not hardcode model paths or class names.
    """
    inference_cfg = ml_cfg.get("inference", {}) or {}

    class_names = list(ml_cfg.get("class_names", ["NotDrone", "Drone"]))
    negative_class = str(ml_cfg.get("negative_class", class_names[0] if class_names else "NotDrone"))
    positive_class = str(ml_cfg.get("positive_class", class_names[-1] if class_names else "Drone"))

    default_drone_threshold = float(
        inference_cfg.get(
            "drone_threshold",
            inference_cfg.get("default_drone_threshold", 0.50),
        )
    )

    gain_policy = inference_cfg.get("gain_threshold_policy", {}) or {}
    gain_threshold_enabled = _as_bool(gain_policy.get("enabled", False), default=False)
    default_drone_threshold = float(
        gain_policy.get("default_drone_threshold", default_drone_threshold)
    )

    rules: list[GainThresholdRule] = []
    for raw_rule in gain_policy.get("rules", []) or []:
        if not isinstance(raw_rule, dict):
            continue
        rules.append(
            GainThresholdRule(
                name=str(raw_rule.get("name", "unnamed")),
                min_gain=_as_float_or_none(raw_rule.get("min_gain")),
                max_gain=_as_float_or_none(raw_rule.get("max_gain")),
                drone_threshold=float(raw_rule["drone_threshold"]),
            )
        )

    voting_cfg = inference_cfg.get("temporal_voting", {}) or {}
    temporal = TemporalVotingConfig(
        enabled=_as_bool(voting_cfg.get("enabled", False), default=False),
        window_size=int(voting_cfg.get("window_size", 5)),
        candidate_vote_k=int(voting_cfg.get("candidate_vote_k", 2)),
        confirmed_vote_k=int(voting_cfg.get("confirmed_vote_k", 3)),
        positive_class=str(voting_cfg.get("positive_class", positive_class)),
        negative_class=str(voting_cfg.get("negative_class", negative_class)),
        require_confirmed_before_aoa=_as_bool(
            voting_cfg.get("require_confirmed_before_aoa", True),
            default=True,
        ),
        reject_to_scan=_as_bool(voting_cfg.get("reject_to_scan", True), default=True),
    )

    return RuntimeDecisionConfig(
        backend=str(inference_cfg.get("backend", "dummy")).lower().strip(),
        model_path=str(inference_cfg.get("model_path", "")),
        device=str(inference_cfg.get("device", "cpu")),
        positive_class=positive_class,
        negative_class=negative_class,
        default_drone_threshold=default_drone_threshold,
        general_threshold=float(inference_cfg.get("general_threshold", 0.50)),
        gain_threshold_enabled=gain_threshold_enabled,
        gain_threshold_rules=rules,
        temporal_voting=temporal,
    )


def select_drone_threshold(
    decision_cfg: RuntimeDecisionConfig,
    gain: int | float | None,
) -> float:
    if not decision_cfg.gain_threshold_enabled:
        return float(decision_cfg.default_drone_threshold)

    try:
        gain_value = float(gain)
    except (TypeError, ValueError):
        return float(decision_cfg.default_drone_threshold)

    for rule in decision_cfg.gain_threshold_rules:
        if rule.min_gain is not None and gain_value < rule.min_gain:
            continue
        if rule.max_gain is not None and gain_value > rule.max_gain:
            continue
        return float(rule.drone_threshold)

    return float(decision_cfg.default_drone_threshold)


def get_positive_probability(
    cnn_result: Any,
    class_names: list[str],
    positive_class: str,
) -> float:
    if cnn_result is None:
        return 0.0

    positive_key = normalize_class_name(positive_class)
    normalized = [normalize_class_name(name) for name in class_names]

    if positive_key in normalized:
        idx = normalized.index(positive_key)
        if idx < len(cnn_result.probabilities):
            return float(cnn_result.probabilities[idx])

    result_key = normalize_class_name(getattr(cnn_result, "class_name", ""))
    if result_key == positive_key:
        return float(getattr(cnn_result, "confidence", 0.0))

    return 0.0


def update_temporal_decision(
    history: deque[int],
    raw_decision: int,
    voting_cfg: TemporalVotingConfig,
) -> tuple[list[int], int, bool, bool, str]:
    history.append(int(raw_decision))

    recent = list(history)[-int(voting_cfg.window_size):]
    vote_count = int(sum(recent))

    candidate = vote_count >= int(voting_cfg.candidate_vote_k)
    confirmed = vote_count >= int(voting_cfg.confirmed_vote_k)

    if confirmed:
        final_decision = "Confirmed Drone"
    elif candidate:
        final_decision = "Drone Candidate"
    else:
        final_decision = "NotDrone"

    return recent, vote_count, candidate, confirmed, final_decision
