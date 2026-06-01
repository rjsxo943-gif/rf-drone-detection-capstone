from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_FEATURES = [
    "raw_abs_p99",
    "frame_power_p99",
    "raw_rms",
]


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(x):
        return math.nan
    return x


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.array([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def summarize_feature_rows(
    rows: list[dict[str, Any]],
    *,
    gain: float,
    distance_m: float = math.nan,
    memo: str = "",
    method: str = "median",
) -> dict[str, Any]:
    """
    여러 block row에서 gain별 대표 feature profile을 만든다.

    기본 대표값:
    - median
    - mean
    - std
    - min
    - max
    - p25
    - p75

    비교용 대표값은 *_median을 우선 사용한다.
    """
    if len(rows) == 0:
        raise ValueError("rows is empty.")

    profile: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gain": float(gain),
        "distance_m": _safe_float(distance_m),
        "memo": str(memo),
        "method": method,
        "num_blocks": int(len(rows)),
    }

    valid_counts = []

    for feature in PROFILE_FEATURES:
        values = _finite_values(rows, feature)
        valid_counts.append(len(values))

        if len(values) == 0:
            profile[f"{feature}_valid_blocks"] = 0
            profile[f"{feature}_median"] = math.nan
            profile[f"{feature}_mean"] = math.nan
            profile[f"{feature}_std"] = math.nan
            profile[f"{feature}_min"] = math.nan
            profile[f"{feature}_max"] = math.nan
            profile[f"{feature}_p25"] = math.nan
            profile[f"{feature}_p75"] = math.nan
            continue

        profile[f"{feature}_valid_blocks"] = int(len(values))
        profile[f"{feature}_median"] = float(np.median(values))
        profile[f"{feature}_mean"] = float(np.mean(values))
        profile[f"{feature}_std"] = float(np.std(values))
        profile[f"{feature}_min"] = float(np.min(values))
        profile[f"{feature}_max"] = float(np.max(values))
        profile[f"{feature}_p25"] = float(np.percentile(values, 25))
        profile[f"{feature}_p75"] = float(np.percentile(values, 75))

    profile["min_valid_blocks"] = int(min(valid_counts)) if valid_counts else 0

    return profile


def feature_ratio_db(
    current: float,
    target: float,
    *,
    power_like: bool = False,
) -> float:
    """
    current/target 차이를 dB로 계산.

    amplitude 계열: 20log10
    power 계열: 10log10
    """
    current = _safe_float(current)
    target = _safe_float(target)

    if current <= 0 or target <= 0:
        return math.nan

    scale = 10.0 if power_like else 20.0
    return float(scale * np.log10(current / target))


def compare_profiles_db(
    current_profile: dict[str, Any],
    target_profile: dict[str, Any],
) -> dict[str, Any]:
    """
    두 profile의 median feature를 dB 기준으로 비교한다.
    """
    errors = {}

    for feature in PROFILE_FEATURES:
        key = f"{feature}_median"
        current = _safe_float(current_profile.get(key))
        target = _safe_float(target_profile.get(key))
        power_like = feature == "frame_power_p99"

        err_db = feature_ratio_db(
            current,
            target,
            power_like=power_like,
        )
        errors[f"{feature}_error_db"] = err_db
        errors[f"{feature}_abs_error_db"] = abs(err_db) if math.isfinite(err_db) else math.nan

    finite_abs = [
        float(v)
        for k, v in errors.items()
        if k.endswith("_abs_error_db") and math.isfinite(float(v))
    ]

    errors["max_abs_error_db"] = max(finite_abs) if finite_abs else math.nan
    errors["mean_abs_error_db"] = float(np.mean(finite_abs)) if finite_abs else math.nan

    return errors


def append_gain_profile_csv(csv_path: str | Path, profile: dict[str, Any]) -> None:
    """
    gain profile을 CSV에 누적 저장한다.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "gain",
        "distance_m",
        "memo",
        "method",
        "num_blocks",
        "min_valid_blocks",
    ]

    for feature in PROFILE_FEATURES:
        fieldnames.extend(
            [
                f"{feature}_valid_blocks",
                f"{feature}_median",
                f"{feature}_mean",
                f"{feature}_std",
                f"{feature}_min",
                f"{feature}_max",
                f"{feature}_p25",
                f"{feature}_p75",
            ]
        )

    exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow({key: profile.get(key, "") for key in fieldnames})


def save_gain_profiles_json(
    json_path: str | Path,
    profiles_by_gain: dict[str, dict[str, Any]],
) -> None:
    """
    최신 gain별 profile table을 JSON으로 저장한다.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "table_type": "gain_feature_profiles",
        "features": PROFILE_FEATURES,
        "profiles": profiles_by_gain,
    }

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_profile_one_line(profile: dict[str, Any]) -> str:
    """
    viewer 하단 표시용 한 줄 요약.
    """
    gain = _safe_float(profile.get("gain"))
    n = int(profile.get("num_blocks", 0))

    raw = _safe_float(profile.get("raw_abs_p99_median"))
    power = _safe_float(profile.get("frame_power_p99_median"))
    rms = _safe_float(profile.get("raw_rms_median"))

    return (
        f"Saved gain={gain:.1f}, n={n}, "
        f"raw_p99_med={raw:.3g}, "
        f"power_p99_med={power:.3g}, "
        f"rms_med={rms:.3g}"
    )
