from __future__ import annotations

from pathlib import Path
import argparse
import csv
import numpy as np


def analyze_file(path: Path) -> dict:
    x = np.load(path)

    median = float(np.median(x))
    p90 = float(np.percentile(x, 90))
    p95 = float(np.percentile(x, 95))
    p99 = float(np.percentile(x, 99))
    max_v = float(np.max(x))
    mean = float(np.mean(x))

    # burst 판단용 간단 지표
    burst_score = p99 - median

    return {
        "file": path.name,
        "shape": str(tuple(x.shape)),
        "mean": mean,
        "median": median,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": max_v,
        "burst_score_p99_minus_median": burst_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="분석할 .npy 폴더")
    parser.add_argument("--out", default=None, help="저장할 csv 경로")
    args = parser.parse_args()

    folder = Path(args.folder)
    paths = sorted(folder.glob("*.npy"))

    if not paths:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    rows = [analyze_file(p) for p in paths]

    if args.out is None:
        out_path = folder / "summary.csv"
    else:
        out_path = Path(args.out)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    medians = np.array([r["median"] for r in rows])
    p95s = np.array([r["p95"] for r in rows])
    p99s = np.array([r["p99"] for r in rows])
    maxs = np.array([r["max"] for r in rows])
    burst_scores = np.array([r["burst_score_p99_minus_median"] for r in rows])

    print("=== Folder Summary ===")
    print("folder:", folder)
    print("num_files:", len(rows))
    print()
    print("median mean:", float(medians.mean()))
    print("p95 mean   :", float(p95s.mean()))
    print("p99 mean   :", float(p99s.mean()))
    print("max mean   :", float(maxs.mean()))
    print("max highest:", float(maxs.max()))
    print()
    print("burst_score mean:", float(burst_scores.mean()))
    print("burst_score max :", float(burst_scores.max()))
    print()
    print("Top 10 burst files:")
    top = sorted(rows, key=lambda r: r["burst_score_p99_minus_median"], reverse=True)[:10]
    for r in top:
        print(
            f'{r["file"]} | '
            f'median={r["median"]:.2f}, '
            f'p99={r["p99"]:.2f}, '
            f'max={r["max"]:.2f}, '
            f'burst_score={r["burst_score_p99_minus_median"]:.2f}'
        )

    print()
    print("saved csv:", out_path)


if __name__ == "__main__":
    main()
