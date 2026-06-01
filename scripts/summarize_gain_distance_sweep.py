from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def to_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def safe_median(values):
    clean = [v for v in values if isinstance(v, float) and not math.isnan(v)]
    return median(clean) if clean else float("nan")


def ratio_true(values):
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


def pick(row: dict, *names: str, default=""):
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return default


def classify_decision(summary: dict) -> str:
    overload_ratio = summary["overload_ratio"]
    cnn_confirm_ratio = summary["cnn_confirm_ratio"]
    coherence_median = summary["coherence_median"]

    if overload_ratio >= 0.05:
        return "too_strong"

    if cnn_confirm_ratio >= 0.70 and coherence_median >= 0.75:
        return "good"

    if cnn_confirm_ratio >= 0.50 and coherence_median >= 0.60:
        return "usable"

    if cnn_confirm_ratio > 0.10 or coherence_median >= 0.45:
        return "weak"

    return "lost"


def fmt(x, digits=4):
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize drone controller distance-gain sweep logs."
    )
    parser.add_argument(
        "--input",
        default="outputs/viewer/drone_controller_gain_distance_sweep.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/viewer/drone_controller_gain_distance_summary.csv",
    )
    parser.add_argument(
        "--output-md",
        default="docs/experiments/drone_controller_gain_distance_summary.md",
    )
    parser.add_argument("--reference-distance", type=float, default=2.0)
    parser.add_argument("--reference-gain", type=float, default=30.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input CSV not found: {input_path}")

    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        distance = to_float(pick(row, "distance_m", "distance"))
        gain = to_float(pick(row, "gain"))
        memo = pick(row, "memo", default="")

        if math.isnan(distance) or math.isnan(gain):
            continue

        key = (round(distance, 3), round(gain, 3), memo)
        groups[key].append(row)

    summaries = []
    for (distance, gain, memo), group_rows in sorted(groups.items()):
        raw_p99 = [to_float(pick(r, "raw_abs_p99")) for r in group_rows]
        raw_rms = [to_float(pick(r, "raw_rms")) for r in group_rows]
        raw_max = [to_float(pick(r, "raw_abs_max")) for r in group_rows]
        frame_p99 = [to_float(pick(r, "frame_power_p99")) for r in group_rows]

        overloads = [to_bool(pick(r, "overloaded", "overload", default="False")) for r in group_rows]

        cnn_conf = [
            to_float(pick(r, "cnn_smoothed_confidence", "cnn_raw_confidence"))
            for r in group_rows
        ]
        cnn_confirmed = [
            to_bool(pick(r, "cnn_confirmed", default="False"))
            for r in group_rows
        ]

        coherence = [
            to_float(pick(r, "stft_coherence", "coherence_like", "coherence"))
            for r in group_rows
        ]
        angle = [
            to_float(pick(r, "aoa_angle_deg", "angle_deg"))
            for r in group_rows
        ]

        raw_class_values = [
            pick(r, "cnn_smoothed_class_name", "cnn_raw_class_name", default="")
            for r in group_rows
            if pick(r, "cnn_smoothed_class_name", "cnn_raw_class_name", default="")
        ]
        raw_class = raw_class_values[-1] if raw_class_values else ""

        summary = {
            "distance_m": distance,
            "gain_db": gain,
            "source": "Drone controller",
            "num_rows": len(group_rows),
            "raw_abs_p99_median": safe_median(raw_p99),
            "raw_rms_median": safe_median(raw_rms),
            "raw_abs_max_median": safe_median(raw_max),
            "frame_power_p99_median": safe_median(frame_p99),
            "overload_ratio": ratio_true(overloads),
            "cnn_class_last": raw_class,
            "cnn_conf_median": safe_median(cnn_conf),
            "cnn_confirm_ratio": ratio_true(cnn_confirmed),
            "coherence_median": safe_median(coherence),
            "aoa_angle_median": safe_median(angle),
            "memo": memo,
        }
        summary["decision"] = classify_decision(summary)
        summaries.append(summary)

    # reference feature ratios
    ref = None
    for s in summaries:
        if (
            abs(s["distance_m"] - args.reference_distance) < 1e-6
            and abs(s["gain_db"] - args.reference_gain) < 1e-6
        ):
            ref = s
            break

    for s in summaries:
        if ref and ref["raw_abs_p99_median"] and not math.isnan(ref["raw_abs_p99_median"]):
            s["raw_p99_ratio_to_ref"] = s["raw_abs_p99_median"] / ref["raw_abs_p99_median"]
        else:
            s["raw_p99_ratio_to_ref"] = float("nan")

        if ref and ref["raw_rms_median"] and not math.isnan(ref["raw_rms_median"]):
            s["raw_rms_ratio_to_ref"] = s["raw_rms_median"] / ref["raw_rms_median"]
        else:
            s["raw_rms_ratio_to_ref"] = float("nan")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "distance_m",
        "gain_db",
        "source",
        "num_rows",
        "raw_abs_p99_median",
        "raw_p99_ratio_to_ref",
        "raw_rms_median",
        "raw_rms_ratio_to_ref",
        "raw_abs_max_median",
        "frame_power_p99_median",
        "overload_ratio",
        "cnn_class_last",
        "cnn_conf_median",
        "cnn_confirm_ratio",
        "coherence_median",
        "aoa_angle_median",
        "decision",
        "memo",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Drone Controller Distance–Gain Sweep Summary")
    lines.append("")
    lines.append(f"Reference condition: {args.reference_distance} m / gain {args.reference_gain} dB")
    lines.append("")
    lines.append("| Distance (m) | Gain (dB) | Rows | raw_p99 med | p99/ref | raw_rms med | rms/ref | overload | CNN conf | CNN confirmed | Coherence | AoA angle | Decision | Memo |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    for s in summaries:
        lines.append(
            "| "
            f"{fmt(s['distance_m'], 1)} | "
            f"{fmt(s['gain_db'], 0)} | "
            f"{s['num_rows']} | "
            f"{fmt(s['raw_abs_p99_median'])} | "
            f"{fmt(s['raw_p99_ratio_to_ref'], 3)} | "
            f"{fmt(s['raw_rms_median'])} | "
            f"{fmt(s['raw_rms_ratio_to_ref'], 3)} | "
            f"{fmt(s['overload_ratio'], 3)} | "
            f"{fmt(s['cnn_conf_median'], 3)} | "
            f"{fmt(s['cnn_confirm_ratio'], 3)} | "
            f"{fmt(s['coherence_median'], 3)} | "
            f"{fmt(s['aoa_angle_median'], 2)} | "
            f"{s['decision']} | "
            f"{s['memo']} |"
        )

    lines.append("")
    lines.append("## Decision 기준")
    lines.append("")
    lines.append("- `too_strong`: overload ratio가 높음")
    lines.append("- `good`: CNN confirmed 비율과 coherence가 모두 높음")
    lines.append("- `usable`: CNN/coherence가 사용 가능한 수준")
    lines.append("- `weak`: 약하지만 일부 신호 확인 가능")
    lines.append("- `lost`: 탐지/추적 어려움")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved CSV: {output_csv}")
    print(f"saved MD : {output_md}")
    print(f"groups   : {len(summaries)}")


if __name__ == "__main__":
    main()
