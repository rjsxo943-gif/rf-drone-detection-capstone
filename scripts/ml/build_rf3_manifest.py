
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path


LABEL_MAP = {
    "background": "Background",
    "bluetooth": "Bluetooth",
    "wifi": "WiFi",
}


def normalize_label(folder_name: str) -> str | None:
    return LABEL_MAP.get(folder_name.lower())


def has_selected_dir(path: Path) -> bool:
    """
    경로 중간에 selected로 시작하는 폴더가 있으면 True.
    예:
    selected_meaningful
    selected_bluetooth_meaningful
    selected_background_clean
    """
    return any(part.lower().startswith("selected") for part in path.parts)


def find_selected_dir(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("selected"):
            return part
    return ""


def parse_center_freq_mhz(text: str) -> str:
    """
    폴더명/파일명에서 2437, 2450, 2460 같은 2.4GHz 중심주파수 후보 추출.
    없으면 빈 문자열.
    """
    matches = re.findall(r"(?<!\d)(24[0-8][0-9])(?!\d)", text)
    return matches[0] if matches else ""


def parse_gain(text: str) -> str:
    match = re.search(r"gain(\d+)", text.lower())
    return match.group(1) if match else ""


def parse_distance(text: str) -> str:
    """
    0.4m, 1m 같은 거리 정보 추출.
    Background처럼 거리 정보가 없으면 빈 문자열.
    """
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)m(?![a-zA-Z])", text.lower())
    return match.group(1) + "m" if match else ""


def infer_session(label_dir: Path, npy_path: Path) -> str:
    """
    label 폴더 바로 아래 폴더를 session으로 사용.

    예:
    WIFI/home_wifihot_ch6_on_2437_gain10_1m/part1/selected_meaningful/a.npy
    -> session = home_wifihot_ch6_on_2437_gain10_1m

    Bluetooth/home_bt_audio_on_2450_gain10_0.4m/20260509_xxxxxx/selected.../a.npy
    -> session = home_bt_audio_on_2450_gain10_0.4m
    """
    rel_parts = npy_path.relative_to(label_dir).parts
    if len(rel_parts) <= 1:
        return "unknown_session"
    return rel_parts[0]


def infer_group(label_dir: Path, npy_path: Path) -> str:
    """
    session 아래의 part 또는 날짜 폴더를 group으로 기록.
    WiFi면 part1, part2...
    Bluetooth/Background면 20260509_xxxxxx 같은 날짜 폴더가 들어갈 가능성이 큼.
    """
    rel_parts = npy_path.relative_to(label_dir).parts
    if len(rel_parts) <= 2:
        return ""
    return rel_parts[1]


def collect_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = normalize_label(label_dir.name)
        if label is None:
            continue

        for npy_path in sorted(label_dir.rglob("*.npy")):
            if not has_selected_dir(npy_path):
                continue

            session = infer_session(label_dir, npy_path)
            group = infer_group(label_dir, npy_path)
            full_text = str(npy_path)

            rows.append(
                {
                    "filepath": str(npy_path),
                    "label": label,
                    "session": session,
                    "group": group,
                    "selected_dir": find_selected_dir(npy_path),
                    "center_freq_mhz": parse_center_freq_mhz(full_text),
                    "gain": parse_gain(full_text),
                    "distance": parse_distance(full_text),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filepath",
        "label",
        "session",
        "group",
        "selected_dir",
        "center_freq_mhz",
        "gain",
        "distance",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], title: str) -> None:
    print()
    print(f"=== {title} ===")
    print(f"total: {len(rows)}")

    by_label: dict[str, int] = defaultdict(int)
    by_session: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        label = row["label"]
        session = row["session"]
        by_label[label] += 1
        by_session[(label, session)] += 1

    print()
    print("[label counts]")
    for label in sorted(by_label):
        print(f"{label}: {by_label[label]}")

    print()
    print("[session counts]")
    for (label, session), count in sorted(by_session.items()):
        print(f"{label:10s} | {session:45s} | {count}")


def sample_balanced_by_label_and_session(
    rows: list[dict[str, str]],
    target_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    """
    각 라벨에서 target_per_label개만 뽑는다.
    가능하면 session별로 고르게 뽑는다.
    """
    rng = random.Random(seed)

    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    selected_all: list[dict[str, str]] = []

    for label in sorted(rows_by_label):
        label_rows = rows_by_label[label]

        if len(label_rows) <= target_per_label:
            selected_all.extend(label_rows)
            continue

        rows_by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in label_rows:
            rows_by_session[row["session"]].append(row)

        sessions = sorted(rows_by_session.keys())
        for session in sessions:
            rng.shuffle(rows_by_session[session])

        quota = target_per_label // len(sessions)
        remainder = target_per_label % len(sessions)

        selected: list[dict[str, str]] = []
        leftovers: list[dict[str, str]] = []

        for idx, session in enumerate(sessions):
            session_rows = rows_by_session[session]
            take = quota + (1 if idx < remainder else 0)

            selected.extend(session_rows[:take])
            leftovers.extend(session_rows[take:])

        if len(selected) < target_per_label:
            rng.shuffle(leftovers)
            selected.extend(leftovers[: target_per_label - len(selected)])

        selected_all.extend(selected[:target_per_label])

    selected_all.sort(key=lambda x: (x["label"], x["session"], x["filepath"]))
    return selected_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/processed/rf_dataset_v1",
        help="RF dataset root directory",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=500,
        help="Number of samples per label for balanced manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    out_dir = root / "manifests"
    manifest_all_path = out_dir / "manifest_all.csv"
    manifest_balanced_path = out_dir / "manifest_rf3_balanced_v1.csv"

    rows = collect_rows(root)
    write_csv(manifest_all_path, rows)

    balanced_rows = sample_balanced_by_label_and_session(
        rows=rows,
        target_per_label=args.target_per_label,
        seed=args.seed,
    )
    write_csv(manifest_balanced_path, balanced_rows)

    print_summary(rows, "manifest_all")
    print_summary(balanced_rows, "manifest_rf3_balanced_v1")

    print()
    print(f"[OK] saved: {manifest_all_path}")
    print(f"[OK] saved: {manifest_balanced_path}")


if __name__ == "__main__":
    main()