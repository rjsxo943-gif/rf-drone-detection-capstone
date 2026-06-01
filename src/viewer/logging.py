from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_viewer_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append one live-viewer row to CSV while preserving old columns.

    New experiment fields may appear as the viewer modes evolve. This helper
    rewrites the CSV header when new columns are introduced so full-mode logs
    can combine raw/CNN/AoA/profile values without a rigid schema.
    """

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = _clean_row(row)
    if "logged_at" not in cleaned:
        cleaned["logged_at"] = datetime.now().isoformat(timespec="seconds")

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cleaned.keys()))
            writer.writeheader()
            writer.writerow(cleaned)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        old_rows = list(reader)

    fieldnames = list(old_fieldnames)
    for key in cleaned.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    old_rows.append(cleaned)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in old_rows:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _to_csv_value(value) for key, value in row.items()}


def _to_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist(), ensure_ascii=False)
    except Exception:
        pass
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
