from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


BLOCK_RE = re.compile(r"_block_(\d+)\.npy$")


def extract_block_index(path: Path) -> int:
    match = BLOCK_RE.search(path.name)
    if not match:
        return -1
    return int(match.group(1))


def remove_block_suffix(path: Path) -> str:
    """
    예:
    home_wifi_block_0003.npy
    -> home_wifi
    """
    return BLOCK_RE.sub("", path.name)


def compute_stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_value = float(np.max(arr))
    burst_score = p99 - median

    return {
        "median": median,
        "p95": p95,
        "p99": p99,
        "max": max_value,
        "burst_score": burst_score,
    }


def is_meaningful(
    stats: dict[str, float],
    burst_threshold: float,
    p99_threshold: float,
    max_allowed: float,
) -> tuple[bool, str]:
    """
    의미 있는 Wi-Fi 신호 판단 기준.

    1. burst_score가 크면 순간적으로 강한 Wi-Fi burst가 있다고 판단
    2. p99가 높으면 전체적으로 active한 Wi-Fi 블록이라고 판단
    3. max가 너무 크면 비정상 포화 가능성이 있으므로 제외 가능
    """
    if stats["max"] > max_allowed:
        return False, "too_high_max"

    if stats["burst_score"] >= burst_threshold:
        return True, "burst"

    if stats["p99"] >= p99_threshold:
        return True, "active_high_p99"

    return False, "weak_or_background"


def copy_companion_images(src_npy: Path, dst_stem: Path) -> None:
    """
    같은 stem의 png/jpg/jpeg가 있으면 같이 복사.
    capture 스크립트가 이미지도 저장하는 경우를 대비.
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        src_img = src_npy.with_suffix(ext)
        if src_img.exists():
            dst_img = dst_stem.with_suffix(ext)
            shutil.copy2(src_img, dst_img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="capture 결과 폴더")
    parser.add_argument("--out-dir", default=None, help="선별 결과 저장 폴더")
    parser.add_argument("--burst-threshold", type=float, default=20.0)
    parser.add_argument("--p99-threshold", type=float, default=20.0)
    parser.add_argument("--max-allowed", type=float, default=80.0)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    out_dir = Path(args.out_dir) if args.out_dir else folder / "selected_meaningful"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(folder.glob("*.npy"), key=extract_block_index)

    selected_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []

    for npy_path in npy_files:
        block_index = extract_block_index(npy_path)
        if block_index < 0:
            continue

        arr = np.load(npy_path)
        stats = compute_stats(arr)

        selected, reason = is_meaningful(
            stats=stats,
            burst_threshold=args.burst_threshold,
            p99_threshold=args.p99_threshold,
            max_allowed=args.max_allowed,
        )

        base_name = remove_block_suffix(npy_path)
        session_id = folder.name

        new_name = f"{block_index:04d}__{session_id}__{base_name}.npy"
        dst_path = out_dir / new_name

        row = {
            "block_index": block_index,
            "session_id": session_id,
            "src_file": npy_path.name,
            "dst_file": new_name if selected else "",
            "selected": selected,
            "reason": reason,
            **stats,
        }

        all_rows.append(row)

        if selected:
            selected_rows.append(row)

            if not args.dry_run:
                shutil.copy2(npy_path, dst_path)

                if args.copy_images:
                    copy_companion_images(npy_path, dst_path.with_suffix(""))
        else:
            rejected_rows.append(row)

    def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if not args.dry_run:
        save_csv(out_dir / "selected_summary.csv", selected_rows)
        save_csv(out_dir / "rejected_summary.csv", rejected_rows)
        save_csv(out_dir / "all_summary.csv", all_rows)

    print("=== Meaningful Capture Block Selection ===")
    print(f"folder          : {folder}")
    print(f"out_dir         : {out_dir}")
    print(f"num_files       : {len(npy_files)}")
    print(f"selected        : {len(selected_rows)}")
    print(f"rejected        : {len(rejected_rows)}")
    print(f"burst_threshold : {args.burst_threshold}")
    print(f"p99_threshold   : {args.p99_threshold}")
    print(f"max_allowed     : {args.max_allowed}")

    selected_reason_counts = Counter(row["reason"] for row in selected_rows)
    rejected_reason_counts = Counter(row["reason"] for row in rejected_rows)

    print()
    print("Selected reason counts:")
    if selected_reason_counts:
        for reason, count in selected_reason_counts.items():
            print(f"  {reason}: {count}")
    else:
        print("  none")

    print()
    print("Rejected reason counts:")
    if rejected_reason_counts:
        for reason, count in rejected_reason_counts.items():
            print(f"  {reason}: {count}")
    else:
        print("  none")

    print()
    print("Selected blocks:")
    for row in selected_rows[:30]:
        print(
            f"block_{int(row['block_index']):04d} | "
            f"reason={row['reason']} | "
            f"median={row['median']:.2f}, "
            f"p99={row['p99']:.2f}, "
            f"max={row['max']:.2f}, "
            f"burst={row['burst_score']:.2f}"
        )

    if len(selected_rows) > 30:
        print(f"... and {len(selected_rows) - 30} more")


if __name__ == "__main__":
    main()
