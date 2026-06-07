# src/viewer/sector_range_estimator.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RANGE_LABELS_KO = {
    "WITHIN_9M": "9m 이내",
    "RANGE_9_TO_15M": "9~15m 구간",
    "SECTOR_ONLY": "섹터 전체 표시",
    "UNSTABLE": "거리 판단 불안정",
    "OUT_OF_PROFILE": "학습 범위 밖 가능성",
}


SECTOR_7_TO_5 = {
    "LEFT_60_45": "LEFT_OUTER",
    "LEFT_45_30": "LEFT_OUTER",
    "LEFT_30_15": "LEFT_INNER",
    "CENTER": "CENTER",
    "RIGHT_15_30": "RIGHT_INNER",
    "RIGHT_30_45": "RIGHT_OUTER",
    "RIGHT_45_60": "RIGHT_OUTER",
}


RELIABILITY_RANK = {
    "LOW": 0,
    "MID": 1,
    "HIGH": 2,
}


@dataclass
class SectorRangeEstimate:
    range_class: str
    range_label_ko: str

    # dashboard 표시 제어용
    # - range_bin   : WITHIN_9M 또는 RANGE_9_TO_15M 칸 하나만 점등
    # - sector_only : locked sector 전체 점등
    # - none        : 거리 overlay 표시 안 함
    display_mode: str
    sector_fill: bool

    confidence: str
    reliability: str

    score: float | None
    threshold: float | None
    margin: float | None

    features_used: list[str]
    enabled: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SectorRangeEstimator:
    """
    Sector-specific coarse range classifier.

    핵심 원칙:
    - 정확한 거리값 추정기가 아니다.
    - WITHIN_9M / RANGE_9_TO_15M 정도의 coarse range class만 판단한다.
    - 거리 구분이 애매해도 sector가 locked 되어 있으면 SECTOR_ONLY를 반환한다.
    - SECTOR_ONLY는 dashboard에서 해당 sector 전체를 점등하기 위한 상태다.
    """

    def __init__(
        self,
        profile_path: str | Path | None,
        *,
        min_reliability: str = "LOW",
        min_margin_for_range: float = 0.25,
        eps: float = 1e-9,
    ) -> None:
        self.profile_path = Path(profile_path).expanduser() if profile_path else None
        self.min_reliability = str(min_reliability).upper()
        self.min_margin_for_range = float(min_margin_for_range)
        self.eps = float(eps)
        self.profile: dict[str, Any] = {}

        if self.profile_path is not None:
            self.profile = self._load_profile(self.profile_path)

    def estimate(
        self,
        sector5: str | None,
        features: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        sector5:
            locked sector 이름.
            7-sector 이름이 들어와도 내부에서 5-sector로 변환한다.

        features:
            runtime에서 얻은 feature dict.
            raw_abs_p99, raw_abs_mean, raw_rms, frame_power_p99 등.

        Returns
        -------
        dict:
            dashboard에서 바로 쓸 수 있는 결과 dict.
        """

        # 1. sector 자체가 없으면 표시할 방향도 없으므로 none
        if not sector5:
            return self._none(
                reason="missing_sector",
            ).to_dict()

        sector_name = normalize_sector_to_5sector(sector5)

        # 2. profile이 없으면 거리 구분은 못 하지만 sector는 살린다.
        if not self.profile:
            return self._sector_only(
                reliability="LOW",
                reason="profile_not_loaded",
            ).to_dict()

        runtime_features = build_runtime_features(features or {}, eps=self.eps)

        sector_profiles = self.profile.get("sectors", {})
        sector_profile = sector_profiles.get(sector_name)

        # 3. 해당 sector profile이 없으면 sector 전체 표시
        if sector_profile is None:
            return self._sector_only(
                reliability="LOW",
                reason=f"sector_not_in_profile:{sector_name}",
            ).to_dict()

        # 4. profile에서 꺼진 sector면 sector 전체 표시
        if not bool(sector_profile.get("enabled", True)):
            return self._sector_only(
                reliability=str(sector_profile.get("reliability", "LOW")).upper(),
                reason=f"sector_disabled:{sector_name}",
            ).to_dict()

        reliability = str(sector_profile.get("reliability", "LOW")).upper()

        # 5. 신뢰도 기준 미달이면 거리 칸 확정 X, sector 전체 표시
        if not reliability_passes(reliability, self.min_reliability):
            return self._sector_only(
                reliability=reliability,
                reason=f"reliability_below_min:{reliability}<{self.min_reliability}",
            ).to_dict()

        required_features = list(sector_profile.get("features", []))
        missing = [name for name in required_features if name not in runtime_features]

        # 6. feature 부족이면 거리 칸 확정 X, sector 전체 표시
        if missing:
            return self._sector_only(
                reliability=reliability,
                features_used=required_features,
                reason=f"missing_feature:{','.join(missing)}",
            ).to_dict()

        # 7. score 계산
        try:
            score = compute_profile_score(
                sector_profile=sector_profile,
                features=runtime_features,
                eps=self.eps,
            )
        except Exception as exc:
            return self._sector_only(
                reliability=reliability,
                features_used=required_features,
                reason=f"score_error:{type(exc).__name__}",
            ).to_dict()

        threshold = float(sector_profile.get("threshold", 0.0))
        direction = str(sector_profile.get("direction", "high_is_within_9m"))

        if direction == "high_is_within_9m":
            range_class = "WITHIN_9M" if score >= threshold else "RANGE_9_TO_15M"
        elif direction == "low_is_within_9m":
            range_class = "WITHIN_9M" if score <= threshold else "RANGE_9_TO_15M"
        else:
            return self._sector_only(
                reliability=reliability,
                score=score,
                threshold=threshold,
                features_used=required_features,
                reason=f"unknown_direction:{direction}",
            ).to_dict()

        margin = abs(score - threshold)

        # 8. threshold 근처면 억지로 거리 칸을 고르지 않고 sector 전체 표시
        if margin < self.min_margin_for_range:
            return self._sector_only(
                reliability=reliability,
                score=score,
                threshold=threshold,
                margin=margin,
                features_used=required_features,
                reason="range_ambiguous_sector_locked",
            ).to_dict()

        confidence = estimate_confidence_from_margin(
            margin=margin,
            reliability=reliability,
        )

        return SectorRangeEstimate(
            range_class=range_class,
            range_label_ko=RANGE_LABELS_KO[range_class],
            display_mode="range_bin",
            sector_fill=False,
            confidence=confidence,
            reliability=reliability,
            score=score,
            threshold=threshold,
            margin=margin,
            features_used=required_features,
            enabled=True,
            reason="score_threshold_passed",
        ).to_dict()

    def _load_profile(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        if not isinstance(profile, dict):
            raise ValueError("sector range profile must be a JSON object")

        return profile

    def _sector_only(
        self,
        *,
        reliability: str,
        reason: str,
        score: float | None = None,
        threshold: float | None = None,
        margin: float | None = None,
        features_used: list[str] | None = None,
    ) -> SectorRangeEstimate:
        return SectorRangeEstimate(
            range_class="SECTOR_ONLY",
            range_label_ko=RANGE_LABELS_KO["SECTOR_ONLY"],
            display_mode="sector_only",
            sector_fill=True,
            confidence="LOW",
            reliability=reliability,
            score=score,
            threshold=threshold,
            margin=margin,
            features_used=features_used or [],
            enabled=True,
            reason=reason,
        )

    def _none(
        self,
        *,
        reason: str,
    ) -> SectorRangeEstimate:
        return SectorRangeEstimate(
            range_class="UNSTABLE",
            range_label_ko=RANGE_LABELS_KO["UNSTABLE"],
            display_mode="none",
            sector_fill=False,
            confidence="LOW",
            reliability="LOW",
            score=None,
            threshold=None,
            margin=None,
            features_used=[],
            enabled=False,
            reason=reason,
        )


def normalize_sector_to_5sector(sector_name: str) -> str:
    name = str(sector_name).strip().upper()
    return SECTOR_7_TO_5.get(name, name)


def reliability_passes(reliability: str, min_reliability: str) -> bool:
    r = RELIABILITY_RANK.get(str(reliability).upper(), 0)
    m = RELIABILITY_RANK.get(str(min_reliability).upper(), 0)
    return r >= m


def build_runtime_features(
    features: dict[str, Any],
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    """
    runtime feature dict를 float dict로 정리하고,
    필요한 ratio feature를 자동 생성한다.
    """

    out: dict[str, float] = {}

    for key, value in features.items():
        val = to_finite_float(value)
        if val is not None:
            out[str(key)] = val

    raw_abs_p99 = out.get("raw_abs_p99")
    raw_abs_p95 = out.get("raw_abs_p95")
    raw_abs_mean = out.get("raw_abs_mean")
    raw_rms = out.get("raw_rms")
    frame_power_p99 = out.get("frame_power_p99")

    if raw_abs_p99 is not None and raw_rms is not None:
        out["ratio_p99_to_rms"] = safe_div(raw_abs_p99, raw_rms, eps)

    if raw_abs_p95 is not None and raw_rms is not None:
        out["ratio_p95_to_rms"] = safe_div(raw_abs_p95, raw_rms, eps)

    if raw_abs_p99 is not None and raw_abs_mean is not None:
        out["ratio_p99_to_mean"] = safe_div(raw_abs_p99, raw_abs_mean, eps)

    if frame_power_p99 is not None and raw_rms is not None:
        out["ratio_framepower_to_rms2"] = safe_div(
            frame_power_p99,
            raw_rms * raw_rms,
            eps,
        )

    return out


def compute_profile_score(
    *,
    sector_profile: dict[str, Any],
    features: dict[str, float],
    eps: float = 1e-9,
) -> float:
    feature_names = list(sector_profile.get("features", []))
    means = list(sector_profile.get("mean", []))
    stds = list(sector_profile.get("std", []))
    weights = list(sector_profile.get("weights", []))

    n = len(feature_names)

    if not (len(means) == len(stds) == len(weights) == n):
        raise ValueError("features, mean, std, weights length mismatch")

    score = 0.0

    for name, mean, std, weight in zip(feature_names, means, stds, weights):
        x = float(features[name])
        mu = float(mean)
        sigma = max(abs(float(std)), eps)
        w = float(weight)

        z = (x - mu) / sigma
        score += w * z

    return float(score)


def estimate_confidence_from_margin(
    *,
    margin: float,
    reliability: str,
) -> str:
    reliability = reliability.upper()

    if margin >= 0.75:
        raw_conf = "HIGH"
    elif margin >= 0.25:
        raw_conf = "MID"
    else:
        raw_conf = "LOW"

    if reliability == "LOW":
        return "LOW"

    if reliability == "MID" and raw_conf == "HIGH":
        return "MID"

    return raw_conf


def safe_div(a: float, b: float, eps: float) -> float:
    denom = b if abs(b) > eps else eps
    return float(a / denom)


def to_finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x
