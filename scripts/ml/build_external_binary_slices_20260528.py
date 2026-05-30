from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

SRC = Path("data/processed/external_binary_20260528_manifest.csv")
OUT_DIR = Path("data/processed/external_slices_20260528")

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def write_rows(name: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.csv"

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_drone = sum(r["label"] == "Drone" for r in rows)
    n_non = sum(r["label"] == "NonDrone" for r in rows)

    print(f"{name:35s} total={len(rows):4d} Drone={n_drone:4d} NonDrone={n_non:4d} -> {out}")

def contains(row: dict[str, str], keyword: str) -> bool:
    text = (row["path"] + " " + row["session"]).lower()
    return keyword.lower() in text

def main() -> None:
    rows = read_rows(SRC)
    fieldnames = list(rows[0].keys())

    all_drone = [r for r in rows if r["label"] == "Drone"]
    all_non = [r for r in rows if r["label"] == "NonDrone"]

    slices: dict[str, list[dict[str, str]]] = {}

    # 1. 전체
    slices["all_external"] = rows

    # 2. gain별
    slices["gain_g25"] = [r for r in rows if "_g25_" in (r["path"] + " " + r["session"]).lower()]
    slices["gain_g30"] = [r for r in rows if "_g30_" in (r["path"] + " " + r["session"]).lower()]

    # 3. 드론 상태별: 특정 Drone + 전체 NonDrone
    drone_conditions = {
        "drone_controller_far_vs_all_non": "controller_far",
        "drone_motor_off_vs_all_non": "motor_off",
        "drone_motor_on_no_stick_vs_all_non": "motor_on_no_stick",
    }

    for name, key in drone_conditions.items():
        selected_drone = [r for r in all_drone if contains(r, key)]
        slices[name] = selected_drone + all_non

    # 4. NonDrone 종류별: 전체 Drone + 특정 NonDrone
    non_conditions = {
        "all_drone_vs_background": "background",
        "all_drone_vs_wifi": "wifi",
        "all_drone_vs_bluetooth": "bluetooth",
    }

    for name, key in non_conditions.items():
        selected_non = [r for r in all_non if contains(r, key)]
        slices[name] = all_drone + selected_non

    for name, selected in slices.items():
        write_rows(name, selected, fieldnames)

if __name__ == "__main__":
    main()
