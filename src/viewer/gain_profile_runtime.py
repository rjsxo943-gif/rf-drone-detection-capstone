from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROFILE_FIELDS = (
    "raw_abs_mean",
    "raw_abs_p50",
    "raw_abs_p95",
    "raw_abs_p99",
    "raw_abs_max",
    "raw_rms",
    "frame_power_p99",
)


@dataclass
class GainProfileRuntime:
    """Collect and persist N-block feature profiles for one gain setting."""

    blocks: int = 20
    csv_path: str | Path = "outputs/viewer/gain_feature_profiles.csv"
    json_path: str | Path = "outputs/viewer/gain_feature_profiles_latest.json"
    feature_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS
    active: bool = False
    requested: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    capture_meta: dict[str, Any] = field(default_factory=dict)

    def request_capture(self, gain: float, distance_m: float = 0.0, memo: str = "") -> None:
        self.active = True
        self.requested = False
        self.rows = []
        self.capture_meta = {
            "capture_started_at": datetime.now().isoformat(timespec="seconds"),
            "gain": float(gain),
            "distance_m": float(distance_m),
            "memo": str(memo),
            "target_blocks": int(self.blocks),
        }

    def update(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if not self.active:
            return None

        merged = dict(self.capture_meta)
        merged.update(row)
        merged["capture_block_index"] = len(self.rows)
        self.rows.append(merged)

        if len(self.rows) < int(self.blocks):
            return None

        summary = self._build_summary()
        self._append_csv(summary)
        self._write_latest_json(summary)
        self.active = False
        self.rows = []
        return summary

    def status_text(self) -> str:
        if self.active:
            return f"PROFILE capturing {len(self.rows)}/{int(self.blocks)}"
        return "PROFILE idle - press s to capture"

    def _build_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            **self.capture_meta,
            "captured_blocks": len(self.rows),
        }

        for field_name in self.feature_fields:
            values = np.asarray(
                [float(row[field_name]) for row in self.rows if field_name in row],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            summary[f"{field_name}_mean"] = float(np.mean(values))
            summary[f"{field_name}_std"] = float(np.std(values))
            summary[f"{field_name}_median"] = float(np.median(values))
            summary[f"{field_name}_p25"] = float(np.percentile(values, 25))
            summary[f"{field_name}_p75"] = float(np.percentile(values, 75))
            summary[f"{field_name}_min"] = float(np.min(values))
            summary[f"{field_name}_max"] = float(np.max(values))

        return summary

    def _append_csv(self, summary: dict[str, Any]) -> None:
        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()

        fieldnames = list(summary.keys())
        if exists:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header:
                for key in existing_header:
                    if key not in fieldnames:
                        fieldnames.append(key)
                for key in summary.keys():
                    if key not in existing_header:
                        existing_header.append(key)
                fieldnames = existing_header

        rows: list[dict[str, Any]] = []
        if exists:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        rows.append(summary)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow({key: item.get(key, "") for key in fieldnames})

    def _write_latest_json(self, summary: dict[str, Any]) -> None:
        path = Path(self.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
