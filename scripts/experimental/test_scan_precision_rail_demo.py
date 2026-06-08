from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.experimental.live_aoa_sector_dashboard import (
    SectorDashboardRenderer,
    load_dashboard_cfg,
)
from src.aoa.sector_quantizer import sector_index_to_label
from src.scan.scan_policy import build_scan_freqs
from src.viewer.state import ViewerState


def load_scan_freqs() -> list[float]:
    with Path("configs/scan.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scan = cfg.get("scan", cfg)

    return build_scan_freqs(
        start_freq=float(scan["start_freq"]),
        stop_freq=float(scan["stop_freq"]),
        step_freq=float(scan["step_freq"]),
    )


def main() -> int:
    scan_freqs = load_scan_freqs()

    args = SimpleNamespace(
        config_dir="configs",
        window_name="SCAN + PRECISION UI Demo",
        target_fps=5.0,
        sector_preset={
            "bins": [
                {"name": "LEFT_60_45", "min_deg": -60, "max_deg": -45, "label_deg": -52.5},
                {"name": "LEFT_45_30", "min_deg": -45, "max_deg": -30, "label_deg": -37.5},
                {"name": "LEFT_30_15", "min_deg": -30, "max_deg": -15, "label_deg": -22.5},
                {"name": "CENTER", "min_deg": -15, "max_deg": 15, "label_deg": 0},
                {"name": "RIGHT_15_30", "min_deg": 15, "max_deg": 30, "label_deg": 22.5},
                {"name": "RIGHT_30_45", "min_deg": 30, "max_deg": 45, "label_deg": 37.5},
                {"name": "RIGHT_45_60", "min_deg": 45, "max_deg": 60, "label_deg": 52.5},
            ]
        },
    )

    dash_cfg = load_dashboard_cfg(args)

    renderer = SectorDashboardRenderer(
        window_name=args.window_name,
        target_fps=args.target_fps,
        width=int(dash_cfg.get("canvas_width", 1320)),
        height=int(dash_cfg.get("canvas_height", 720)),
        blink_on_hold=bool(dash_cfg.get("blink_on_hold", True)),
        fade_on_signal_lost=bool(dash_cfg.get("fade_on_signal_lost", True)),
    )

    state = ViewerState(
        mode="demo",
        gain=35,
        center_freq=int(scan_freqs[0]),
        sample_rate=5_000_000,
        target_fps=5.0,
    )

    locked_freq = None
    precision_active = False
    idx = 0

    try:
        while True:
            current_freq = scan_freqs[idx % len(scan_freqs)]

            # demo: 몇 바퀴 돌다가 가운데 근처 주파수에서 lock
            if idx > len(scan_freqs) + 2:
                locked_freq = scan_freqs[len(scan_freqs) // 2]
                precision_active = True

            state.update_index = idx
            state.center_freq = int(locked_freq or current_freq)

            if precision_active:
                sector = {
                    "sector_status": "trusted",
                    "locked_sector_name": "RIGHT_15_30",
                    "instant_sector_name": "RIGHT_15_30",
                    "median_angle_deg": 24.0,
                    "median_coherence": 0.91,
                    "median_raw_p99": 12.4,
                }
                cnn_result = {
                    "label": "Drone",
                    "confidence": 0.94,
                    "probability": 0.94,
                }
                scan_rail = {
                    "mode": "PRECISION",
                    "scan_freqs": scan_freqs,
                    "current_freq": current_freq,
                    "locked_freq": locked_freq,
                    "candidate_freq": locked_freq,
                    "status": "HANDOFF",
                    "rail_width": 190,
                }
            else:
                sector = {
                    "sector_status": "no_signal",
                    "locked_sector_name": "",
                    "instant_sector_name": "",
                    "median_angle_deg": "",
                    "median_coherence": "",
                    "median_raw_p99": "",
                }
                cnn_result = {
                    "label": "WAITING",
                    "confidence": 0.0,
                    "probability": 0.0,
                }
                scan_rail = {
                    "mode": "SCAN",
                    "scan_freqs": scan_freqs,
                    "current_freq": current_freq,
                    "locked_freq": None,
                    "candidate_freq": None,
                    "status": "SWEEPING",
                    "rail_width": 190,
                }

            selected_raw = {
                "raw_abs_p99": sector.get("median_raw_p99", ""),
                "median_raw_p99": sector.get("median_raw_p99", ""),
            }

            key = renderer.render(
                state=state,
                args=args,
                dash_cfg=dash_cfg,
                sector=sector,
                selected_raw=selected_raw,
                cnn_result=cnn_result,
                raw_pass_count=1 if precision_active else 0,
                cnn_drone_count=3 if precision_active else 0,
                topk_count=5,
                paused=False,
                scan_rail=scan_rail,
            )

            if key == "quit":
                return 0

            idx += 1
            time.sleep(0.05)

    finally:
        renderer.close()


if __name__ == "__main__":
    raise SystemExit(main())
