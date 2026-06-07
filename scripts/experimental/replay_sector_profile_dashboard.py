#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# 기존 dashboard renderer 재사용
from scripts.experimental import live_aoa_sector_dashboard as dash


RAW_FEATURE_KEYS = [
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
    "ratio_p99_to_rms",
    "ratio_p95_to_rms",
    "ratio_p99_to_mean",
    "ratio_framepower_to_rms2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay saved AoA sector_profile.csv rows on the OpenCV sector dashboard."
    )

    p.add_argument(
        "--csv",
        default=None,
        help="Input sector_profile.csv. If omitted, latest outputs/aoa_sector_profiles/*sector_profile*.csv is used.",
    )
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--only-trusted", action="store_true")
    p.add_argument("--target-fps", type=float, default=10.0)
    p.add_argument("--window-name", default="RF Sector Dashboard - CSV Replay")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv)
    rows = load_rows(csv_path)

    if args.only_trusted:
        rows = [
            r for r in rows
            if str(r.get("sector_status", "")).strip().lower() == "trusted"
        ]

    if args.start > 0:
        rows = rows[args.start:]

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    if not rows:
        raise RuntimeError("No rows to replay.")

    cfg_args = SimpleNamespace(config_dir=args.config_dir)
    dash_cfg = dash.load_dashboard_cfg(cfg_args)

    # CSV replay는 실제 데이터 확인용이므로 demo cycle은 무조건 끈다.
    dash_cfg["demo_cycle"] = False

    renderer = dash.SectorDashboardRenderer(
        window_name=args.window_name,
        target_fps=args.target_fps,
        width=int(dash_cfg.get("canvas_width", 1320)),
        height=int(dash_cfg.get("canvas_height", 720)),
        blink_on_hold=bool(dash_cfg.get("blink_on_hold", True)),
        fade_on_signal_lost=bool(dash_cfg.get("fade_on_signal_lost", True)),
    )

    print("=== CSV Replay Dashboard ===")
    print(f"csv        : {csv_path}")
    print(f"rows       : {len(rows)}")
    print(f"fps        : {args.fps}")
    print(f"loop       : {args.loop}")
    print(f"distance   : {dash_cfg.get('distance')}")
    print("============================")

    frame_delay = 1.0 / max(float(args.fps), 0.1)

    try:
        while True:
            for idx, row in enumerate(rows):
                state = make_state(row=row, csv_path=csv_path, idx=idx)
                sector = make_sector(row)
                selected_raw = make_selected_raw(row)
                cnn_result = make_cnn_result(row)

                raw_pass_count = int_or_default(row.get("raw_pass_count"), 0)
                if raw_pass_count <= 0:
                    raw_pass_count = int_or_default(row.get("raw_pass"), 0)

                cnn_drone_count = int_or_default(row.get("cnn_drone_count"), 0)
                if cnn_drone_count <= 0:
                    cnn_drone_count = int_or_default(row.get("drone_topk_count"), 0)

                topk_count = int_or_default(row.get("top_k"), 0)
                if topk_count <= 0:
                    topk_count = int_or_default(row.get("topk_count"), 0)

                key = renderer.render(
                    state=state,
                    args=SimpleNamespace(),
                    dash_cfg=dash_cfg,
                    sector=sector,
                    selected_raw=selected_raw,
                    cnn_result=cnn_result,
                    raw_pass_count=raw_pass_count,
                    cnn_drone_count=cnn_drone_count,
                    topk_count=topk_count,
                    paused=False,
                )

                locked = sector.get("locked_sector_name", "None")
                dist = row.get("distance_m", "")
                true_angle = row.get("true_angle_deg", "")
                p99 = sector.get("median_raw_p99", "")

                print(
                    f"[REPLAY] {idx + 1:04d}/{len(rows):04d} "
                    f"sector={locked} dist={dist} angle={true_angle} p99={p99}",
                    flush=True,
                )

                if key == "quit":
                    return 0

                time.sleep(frame_delay)

            if not args.loop:
                print("[DONE] replay finished.")
                time.sleep(0.7)
                return 0

    finally:
        renderer.close()


def resolve_csv_path(csv_path: str | None) -> Path:
    if csv_path:
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(Path("outputs/aoa_sector_profiles").glob("*sector_profile*.csv"))

    if not candidates:
        raise FileNotFoundError(
            "No CSV found in outputs/aoa_sector_profiles. Pass --csv manually."
        )

    return candidates[-1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def make_state(*, row: dict[str, str], csv_path: Path, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        update_idx=idx,
        gain=float_or_default(row.get("gain"), 35.0),
        center_freq=int_or_default(row.get("center_freq"), 2450000000),
        sample_rate=int_or_default(row.get("sample_rate"), 5000000),
        blocks_per_update=int_or_default(row.get("blocks_per_update"), 20),
        top_k=int_or_default(row.get("top_k"), 5),
        distance_m=float_or_default(row.get("distance_m"), 0.0),
        true_angle_deg=float_or_default(row.get("true_angle_deg"), 0.0),
        phase_offset_live_delta_deg=float_or_default(
            row.get("phase_offset_live_delta_deg"),
            0.0,
        ),
        capture_active=False,
        capture_requested=False,
        capture_saved_count=0,
        capture_target_n=0,
        capture_remaining=0,
        profile_csv=str(csv_path.name),
        profile_csv_path=str(csv_path),
    )


def make_sector(row: dict[str, str]) -> dict[str, Any]:
    locked = first_nonempty(
        row,
        [
            "locked_sector_name",
            "locked_sector",
            "sector5",
            "sector_name",
        ],
        default="",
    )

    instant = first_nonempty(
        row,
        [
            "instant_sector_name",
            "instant_sector",
            "locked_sector_name",
        ],
        default=locked,
    )

    status = first_nonempty(
        row,
        [
            "sector_status",
            "status",
        ],
        default="trusted" if locked else "no_signal",
    )

    votes = first_nonempty(
        row,
        [
            "votes",
            "sector_votes",
        ],
        default=f"{locked}=csv" if locked else "",
    )

    return {
        "sector_status": status,
        "locked_sector_name": locked,
        "instant_sector_name": instant,
        "angle_median": float_or_empty(row.get("angle_median")),
        "angle_spread": float_or_empty(row.get("angle_spread")),
        "median_coherence": float_or_empty(row.get("median_coherence")),
        "median_raw_p99": float_or_empty(
            first_nonempty_value(row, ["median_raw_p99", "raw_abs_p99"])
        ),
        "valid_aoa_count": int_or_default(row.get("valid_aoa_count"), 0),
        "votes": votes,
        "reason": first_nonempty(row, ["reason", "sector_reason"], default="csv_replay"),
    }


def make_selected_raw(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in RAW_FEATURE_KEYS:
        if key in row and row[key] not in ("", None):
            out[key] = float_or_empty(row[key])

    return out


def make_cnn_result(row: dict[str, str]) -> dict[str, Any]:
    return {
        "cnn_raw_class_name": first_nonempty(
            row,
            ["cnn_raw_class_name", "cnn_class_name", "class_name"],
            default="Drone",
        ),
        "cnn_raw_confidence": float_or_empty(
            first_nonempty_value(
                row,
                ["cnn_raw_confidence", "cnn_confidence", "confidence"],
                default="1.0",
            )
        ),
        "cnn_positive_votes": int_or_default(
            first_nonempty_value(row, ["cnn_positive_votes", "cnn_vote"], default="0"),
            0,
        ),
        "cnn_confirm_votes": int_or_default(
            first_nonempty_value(row, ["cnn_confirm_votes", "cnn_confirm"], default="0"),
            0,
        ),
    }


def first_nonempty(
    row: dict[str, str],
    keys: list[str],
    *,
    default: str = "",
) -> str:
    return str(first_nonempty_value(row, keys, default=default))


def first_nonempty_value(
    row: dict[str, str],
    keys: list[str],
    *,
    default: Any = "",
) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def float_or_empty(value: Any) -> float | str:
    try:
        return float(value)
    except Exception:
        return ""


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
