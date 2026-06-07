#!/usr/bin/env python3
# scripts/experimental/build_sector_range_profile.py
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SECTOR_7_TO_5 = {
    "LEFT_60_45": "LEFT_OUTER",
    "LEFT_45_30": "LEFT_OUTER",
    "LEFT_30_15": "LEFT_INNER",
    "CENTER": "CENTER",
    "RIGHT_15_30": "RIGHT_INNER",
    "RIGHT_30_45": "RIGHT_OUTER",
    "RIGHT_45_60": "RIGHT_OUTER",
}


RANGE_CLASSES = {
    "WITHIN_9M": {
        "label_ko": "9m 이내",
        "source_distances_m": [6, 9],
    },
    "RANGE_9_TO_15M": {
        "label_ko": "9~15m 구간",
        "source_distances_m": [12, 15],
    },
}


BASE_FEATURES = [
    "raw_abs_mean",
    "raw_abs_p50",
    "raw_abs_p95",
    "raw_abs_p99",
    "raw_abs_max",
    "raw_rms",
    "frame_power_p99",
    "median_raw_p99",
    "angle_spread",
    "median_coherence",
    "dominant_sector_ratio",
    "valid_aoa_count",
]


RATIO_FEATURES = [
    "ratio_p99_to_rms",
    "ratio_p95_to_rms",
    "ratio_p99_to_mean",
    "ratio_framepower_to_rms2",
]


# 너무 많은 feature를 한 번에 넣지 않기 위해,
# sector별로 설명 가능한 추천 조합만 후보로 둔다.
SECTOR_FEATURE_CANDIDATES = {
    "LEFT_OUTER": [
        ["median_raw_p99"],
        ["raw_abs_p99"],
        ["raw_abs_p95"],
    ],
    "LEFT_INNER": [
        ["frame_power_p99", "ratio_framepower_to_rms2"],
        ["raw_abs_p95"],
        ["median_raw_p99"],
    ],
    "CENTER": [
        ["raw_abs_p99", "raw_rms", "median_raw_p99"],
        ["frame_power_p99"],
        ["raw_abs_mean", "raw_rms"],
        ["median_raw_p99"],
    ],
    "RIGHT_INNER": [
        ["median_raw_p99", "ratio_p99_to_mean"],
        ["median_raw_p99"],
        ["raw_abs_p99"],
    ],
    "RIGHT_OUTER": [
        ["raw_abs_mean", "raw_abs_p99", "ratio_framepower_to_rms2"],
        ["raw_abs_p99"],
        ["median_raw_p99"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build experimental sector-specific coarse range profile JSON."
    )

    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="Input sector_profile.csv files. If omitted, outputs/aoa_sector_profiles/*sector_profile*.csv is used.",
    )
    parser.add_argument(
        "--out",
        default="outputs/sector_range_profiles/gain35_cf2450000000_nearfar_profile.json",
        help="Output profile JSON path.",
    )
    parser.add_argument("--gain", type=float, default=35.0)
    parser.add_argument("--center-freq", type=int, default=2450000000)
    parser.add_argument("--min-bacc", type=float, default=0.65)
    parser.add_argument("--min-conditions", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-9)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_paths = resolve_csv_paths(args.csv_paths)
    if not csv_paths:
        raise FileNotFoundError(
            "No input CSV files found. Pass CSV paths or check outputs/aoa_sector_profiles."
        )

    print("[Input CSV]")
    for path in csv_paths:
        print(f"  - {path}")

    df = load_csvs(csv_paths)
    df = prepare_dataframe(df, eps=args.eps)

    condition_df = make_condition_level_table(df)

    if condition_df.empty:
        raise RuntimeError("Condition-level table is empty.")

    profile = build_profile(
        condition_df=condition_df,
        csv_paths=csv_paths,
        gain=args.gain,
        center_freq=args.center_freq,
        min_bacc=args.min_bacc,
        min_conditions=args.min_conditions,
        eps=args.eps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"[Saved] {out_path}")
    print_profile_summary(profile)


def resolve_csv_paths(csv_paths: list[str]) -> list[Path]:
    if csv_paths:
        return [Path(p).expanduser() for p in csv_paths]

    default_dir = Path("outputs/aoa_sector_profiles")
    return sorted(default_dir.glob("*sector_profile*.csv"))


def load_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_dataframe(df: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    df = df.copy()

    if "locked_sector_name" not in df.columns:
        raise KeyError("CSV must contain locked_sector_name column.")

    if "distance_m" not in df.columns:
        raise KeyError("CSV must contain distance_m column.")

    if "true_angle_deg" not in df.columns:
        df["true_angle_deg"] = np.nan

    if "phase_offset_live_delta_deg" not in df.columns:
        df["phase_offset_live_delta_deg"] = 0.0

    df["locked_sector_name"] = df["locked_sector_name"].astype(str).str.strip().str.upper()
    df["sector5"] = df["locked_sector_name"].map(SECTOR_7_TO_5).fillna(df["locked_sector_name"])

    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["true_angle_deg"] = pd.to_numeric(df["true_angle_deg"], errors="coerce")
    df["phase_offset_live_delta_deg"] = pd.to_numeric(
        df["phase_offset_live_delta_deg"],
        errors="coerce",
    ).fillna(0.0)

    for col in BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    add_ratio_features(df, eps=eps)

    # 이번 profile은 6/9m vs 12/15m만 사용한다.
    df = df[df["distance_m"].isin([6, 9, 12, 15])].copy()

    df["range_class"] = np.where(
        df["distance_m"].isin([6, 9]),
        "WITHIN_9M",
        "RANGE_9_TO_15M",
    )

    valid_sectors = set(SECTOR_FEATURE_CANDIDATES.keys())
    df = df[df["sector5"].isin(valid_sectors)].copy()

    return df


def add_ratio_features(df: pd.DataFrame, *, eps: float) -> None:
    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        denom = b.abs().where(b.abs() > eps, eps)
        return a / denom

    if {"raw_abs_p99", "raw_rms"}.issubset(df.columns):
        df["ratio_p99_to_rms"] = safe_div(df["raw_abs_p99"], df["raw_rms"])

    if {"raw_abs_p95", "raw_rms"}.issubset(df.columns):
        df["ratio_p95_to_rms"] = safe_div(df["raw_abs_p95"], df["raw_rms"])

    if {"raw_abs_p99", "raw_abs_mean"}.issubset(df.columns):
        df["ratio_p99_to_mean"] = safe_div(df["raw_abs_p99"], df["raw_abs_mean"])

    if {"frame_power_p99", "raw_rms"}.issubset(df.columns):
        df["ratio_framepower_to_rms2"] = safe_div(
            df["frame_power_p99"],
            df["raw_rms"] * df["raw_rms"],
        )


def make_condition_level_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col for col in BASE_FEATURES + RATIO_FEATURES
        if col in df.columns
    ]

    group_cols = [
        "source_file",
        "distance_m",
        "true_angle_deg",
        "phase_offset_live_delta_deg",
        "sector5",
        "range_class",
    ]

    agg = (
        df[group_cols + feature_cols]
        .groupby(group_cols, dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )

    return agg


def build_profile(
    *,
    condition_df: pd.DataFrame,
    csv_paths: list[Path],
    gain: float,
    center_freq: int,
    min_bacc: float,
    min_conditions: int,
    eps: float,
) -> dict[str, Any]:
    sectors: dict[str, Any] = {}

    for sector_name in SECTOR_FEATURE_CANDIDATES:
        sector_df = condition_df[condition_df["sector5"] == sector_name].copy()

        sector_profile = build_sector_profile(
            sector_name=sector_name,
            sector_df=sector_df,
            min_bacc=min_bacc,
            min_conditions=min_conditions,
            eps=eps,
        )

        sectors[sector_name] = sector_profile

    return {
        "profile_type": "sector_range_within9m_vs_9to15m",
        "experimental": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gain": gain,
        "center_freq": center_freq,
        "sector_mode": "5sector",
        "range_classes": RANGE_CLASSES,
        "source_csv_files": [p.name for p in csv_paths],
        "aggregation": {
            "level": "condition_median",
            "group_by": [
                "source_file",
                "distance_m",
                "true_angle_deg",
                "phase_offset_live_delta_deg",
                "sector5",
            ],
        },
        "decision": {
            "target": "WITHIN_9M_vs_RANGE_9_TO_15M",
            "within_9m_distances": [6, 9],
            "range_9_to_15m_distances": [12, 15],
            "min_bacc": min_bacc,
            "min_conditions": min_conditions,
            "note": "Experimental coarse range profile. Not an exact distance estimator.",
        },
        "sectors": sectors,
    }


def build_sector_profile(
    *,
    sector_name: str,
    sector_df: pd.DataFrame,
    min_bacc: float,
    min_conditions: int,
    eps: float,
) -> dict[str, Any]:
    n_total = int(len(sector_df))
    n_within = int((sector_df["range_class"] == "WITHIN_9M").sum()) if n_total else 0
    n_far = int((sector_df["range_class"] == "RANGE_9_TO_15M").sum()) if n_total else 0

    base = {
        "enabled": False,
        "reliability": "LOW",
        "features": [],
        "mean": [],
        "std": [],
        "weights": [],
        "threshold": 0.0,
        "direction": "high_is_within_9m",
        "validation_bacc": 0.0,
        "n_conditions": n_total,
        "n_within_9m": n_within,
        "n_range_9_to_15m": n_far,
        "reason": "",
    }

    if n_total < min_conditions:
        base["reason"] = f"not_enough_conditions:{n_total}<{min_conditions}"
        return base

    if n_within == 0 or n_far == 0:
        base["reason"] = "missing_one_class"
        return base

    best: dict[str, Any] | None = None

    for features in SECTOR_FEATURE_CANDIDATES[sector_name]:
        if not all(f in sector_df.columns for f in features):
            continue

        candidate_df = sector_df[["range_class"] + features].dropna().copy()
        if len(candidate_df) < min_conditions:
            continue

        if (candidate_df["range_class"] == "WITHIN_9M").sum() == 0:
            continue

        if (candidate_df["range_class"] == "RANGE_9_TO_15M").sum() == 0:
            continue

        result = evaluate_feature_combo(
            df=candidate_df,
            features=features,
            eps=eps,
        )

        if best is None or result["validation_bacc"] > best["validation_bacc"]:
            best = result

    if best is None:
        base["reason"] = "no_valid_feature_combo"
        return base

    reliability = reliability_from_bacc(best["validation_bacc"])

    enabled = bool(best["validation_bacc"] >= min_bacc)

    best.update(
        {
            "enabled": enabled,
            "reliability": reliability,
            "n_conditions": n_total,
            "n_within_9m": n_within,
            "n_range_9_to_15m": n_far,
            "reason": "ok" if enabled else "bacc_below_min",
        }
    )

    return best


def evaluate_feature_combo(
    *,
    df: pd.DataFrame,
    features: list[str],
    eps: float,
) -> dict[str, Any]:
    y = (df["range_class"] == "WITHIN_9M").to_numpy(dtype=bool)
    X = df[features].to_numpy(dtype=float)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(np.abs(std) > eps, std, 1.0)

    Z = (X - mean) / std

    within_mean = np.nanmean(Z[y], axis=0)
    far_mean = np.nanmean(Z[~y], axis=0)

    effect = within_mean - far_mean

    if np.all(np.abs(effect) < eps):
        weights = np.ones(len(features), dtype=float) / max(len(features), 1)
    else:
        weights = effect / np.sum(np.abs(effect))

    scores = Z @ weights

    threshold, bacc, acc = find_best_threshold(scores=scores, y=y)

    return {
        "enabled": False,
        "reliability": "LOW",
        "features": features,
        "mean": [to_py_float(v) for v in mean],
        "std": [to_py_float(v) for v in std],
        "weights": [to_py_float(v) for v in weights],
        "threshold": to_py_float(threshold),
        "direction": "high_is_within_9m",
        "validation_bacc": to_py_float(bacc),
        "validation_acc": to_py_float(acc),
    }


def find_best_threshold(
    *,
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=bool)

    unique_scores = np.unique(scores)

    if len(unique_scores) == 1:
        threshold_candidates = unique_scores
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        threshold_candidates = np.concatenate(
            [
                [unique_scores[0] - 1e-6],
                mids,
                [unique_scores[-1] + 1e-6],
            ]
        )

    best_threshold = float(threshold_candidates[0])
    best_bacc = -1.0
    best_acc = -1.0

    for threshold in threshold_candidates:
        pred = scores >= threshold

        bacc = balanced_accuracy(y_true=y, y_pred=pred)
        acc = float((pred == y).mean())

        if bacc > best_bacc:
            best_bacc = bacc
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold, best_bacc, best_acc


def balanced_accuracy(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true
    neg = ~y_true

    tpr = float((y_pred[pos] == True).mean()) if pos.any() else 0.0
    tnr = float((y_pred[neg] == False).mean()) if neg.any() else 0.0

    return 0.5 * (tpr + tnr)


def reliability_from_bacc(bacc: float) -> str:
    if bacc >= 0.85:
        return "HIGH"
    if bacc >= 0.70:
        return "MID"
    return "LOW"


def to_py_float(value: Any) -> float:
    x = float(value)
    if not math.isfinite(x):
        return 0.0
    return x


def print_profile_summary(profile: dict[str, Any]) -> None:
    print()
    print("[Sector Summary]")

    for sector_name, sector in profile["sectors"].items():
        enabled = sector.get("enabled")
        reliability = sector.get("reliability")
        features = sector.get("features")
        bacc = sector.get("validation_bacc")
        n_total = sector.get("n_conditions")
        n_w = sector.get("n_within_9m")
        n_f = sector.get("n_range_9_to_15m")
        reason = sector.get("reason")

        print(
            f"  {sector_name:12s} "
            f"enabled={str(enabled):5s} "
            f"rel={reliability:4s} "
            f"bacc={bacc:.3f} "
            f"n={n_total} "
            f"within={n_w} "
            f"far={n_f} "
            f"features={features} "
            f"reason={reason}"
        )


if __name__ == "__main__":
    main()
