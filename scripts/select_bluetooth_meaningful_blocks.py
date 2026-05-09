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
    home_bt_audio_on_2450_gain10_1m_block_0003.npy
    -> home_bt_audio_on_2450_gain10_1m
    """
    return BLOCK_RE.sub("", path.name)


def compute_stats(arr: np.ndarray) -> dict[str, float]:
    """
    저장된 spectrogram 또는 IQ magnitude 배열에서 블록별 통계값을 계산한다.

    Bluetooth는 주파수 hopping 때문에 짧고 강한 burst 형태가 많이 나타나므로,
    단순 평균보다 median, p99, burst_score를 중심으로 판단한다.
    """
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


def is_meaningful_bluetooth(
    stats: dict[str, float],
    burst_threshold: float,
    p99_threshold: float,
    max_allowed: float,
    weak_burst_threshold: float,
    keep_weak: bool,
) -> tuple[bool, str]:
    """
    의미 있는 Bluetooth 신호 판단 기준.

    Bluetooth는 Wi-Fi처럼 한 채널에 계속 머무는 신호가 아니라
    2.4 GHz 대역에서 짧게 hopping하는 burst 형태가 많다.

    기본 clean 기준:
    - burst_score >= 30 dB
    - p99 >= 5 dB

    넓게 포함하는 usable 기준:
    - burst_score >= 25 dB
    - keep_weak 옵션을 켰을 때만 포함

    max가 너무 크면 SDR 포화나 비정상 spike 가능성이 있으므로 제외한다.
    """
    if stats["max"] > max_allowed:
        return False, "too_high_max_possible_saturation"

    burst_score = stats["burst_score"]
    p99 = stats["p99"]

    # 가장 추천하는 Bluetooth clean sample 기준
    if burst_score >= burst_threshold and p99 >= p99_threshold:
        return True, "clean_bt_burst"

    # p99는 낮지만 burst_score가 매우 크면 hopping burst 후보로 인정
    if burst_score >= burst_threshold:
        return True, "bt_burst_high_score"

    # 학습 데이터를 넓히고 싶을 때만 약한 Bluetooth 후보 포함
    if keep_weak and burst_score >= weak_burst_threshold:
        return True, "weak_bt_candidate"

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
    parser = argparse.ArgumentParser(
        description="Select meaningful Bluetooth burst blocks from captured .npy files."
    )
    parser.add_argument("--folder", required=True, help="capture 결과 폴더")
    parser.add_argument("--out-dir", default=None, help="선별 결과 저장 폴더")

    # Bluetooth 기본 추천값
    parser.add_argument(
        "--burst-threshold",
        type=float,
        default=30.0,
        help="Bluetooth clean sample 기준 burst_score 임계값. 기본값: 30 dB",
    )
    parser.add_argument(
        "--p99-threshold",
        type=float,
        default=5.0,
        help="Bluetooth clean sample 기준 p99 임계값. 기본값: 5 dB",
    )
    parser.add_argument(
        "--weak-burst-threshold",
        type=float,
        default=25.0,
        help="넓게 포함할 Bluetooth 후보 기준. --keep-weak 사용 시 적용. 기본값: 25 dB",
    )
    parser.add_argument(
        "--max-allowed",
        type=float,
        default=80.0,
        help="이 값보다 max가 크면 포화/이상치로 보고 제외. 기본값: 80 dB",
    )
    parser.add_argument(
        "--keep-weak",
        action="store_true",
        help="burst_score가 weak_burst_threshold 이상인 약한 Bluetooth 후보도 포함",
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    out_dir = Path(args.out_dir) if args.out_dir else folder / "selected_bluetooth_meaningful"
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

        selected, reason = is_meaningful_bluetooth(
            stats=stats,
            burst_threshold=args.burst_threshold,
            p99_threshold=args.p99_threshold,
            max_allowed=args.max_allowed,
            weak_burst_threshold=args.weak_burst_threshold,
            keep_weak=args.keep_weak,
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

    print("=== Bluetooth Meaningful Capture Block Selection ===")
    print(f"folder               : {folder}")
    print(f"out_dir              : {out_dir}")
    print(f"num_files            : {len(npy_files)}")
    print(f"selected             : {len(selected_rows)}")
    print(f"rejected             : {len(rejected_rows)}")
    print(f"burst_threshold      : {args.burst_threshold}")
    print(f"p99_threshold        : {args.p99_threshold}")
    print(f"weak_burst_threshold : {args.weak_burst_threshold}")
    print(f"keep_weak            : {args.keep_weak}")
    print(f"max_allowed          : {args.max_allowed}")

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
