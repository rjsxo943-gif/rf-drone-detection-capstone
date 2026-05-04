from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.spectrogram import compute_stft_branch


DEFAULT_MAT_PATH = Path("data/external/mavic/Mavic_0.mat")
DEFAULT_OUT_DIR = Path("data/processed/mavic_auto")

ORIG_FS = 100_000_000
TARGET_FS = 5_000_000
BLOCK_SIZE = 16_384

DECIM = ORIG_FS // TARGET_FS
WIDE_BLOCK_SIZE = BLOCK_SIZE * DECIM

NPERSEG = 512
NOVERLAP = 384
NFFT = 512
WINDOW = "hann"


def safe_offset_name(offset_mhz: float) -> str:
    sign = "p" if offset_mhz >= 0 else "m"
    value = abs(offset_mhz)
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return f"{sign}{text}"


def load_complex_iq_from_mat(dset, start_sample: int, num_samples: int) -> np.ndarray:
    end_sample = start_sample + num_samples

    if dset.shape[0] == 1:
        raw = dset[0, start_sample:end_sample]
    else:
        raw = dset[start_sample:end_sample, 0]

    if raw.dtype.fields is None:
        raise ValueError(f"지원하지 않는 dtype입니다: {raw.dtype}")

    if "real" not in raw.dtype.fields or "imag" not in raw.dtype.fields:
        raise ValueError(f"real/imag field가 없습니다: {raw.dtype.fields}")

    iq = raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)
    return iq.astype(np.complex64)


def build_band_masks(fs: int, band_width: int, fft_size: int):
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
    edges = np.arange(-fs / 2, fs / 2 + band_width, band_width)

    bands = []
    masks = []

    for i in range(len(edges) - 1):
        f0 = float(edges[i])
        f1 = float(edges[i + 1])
        fc = (f0 + f1) / 2.0
        mask = (freqs >= f0) & (freqs < f1)

        bands.append(
            {
                "band_index": i,
                "f_start_hz": f0,
                "f_end_hz": f1,
                "f_center_hz": fc,
                "f_start_mhz": f0 / 1e6,
                "f_end_mhz": f1 / 1e6,
                "f_center_mhz": fc / 1e6,
            }
        )
        masks.append(mask)

    return bands, masks


def compute_band_powers(wide_iq: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    x = wide_iq.astype(np.complex64)
    x = x - np.mean(x)

    window = np.hanning(len(x)).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(x * window))
    power = np.abs(spec) ** 2

    values = []
    for mask in masks:
        values.append(float(np.mean(power[mask])))

    return np.array(values, dtype=np.float64)


def extract_subband(
    wide_iq: np.ndarray,
    wide_start_sample: int,
    offset_hz: float,
) -> np.ndarray:
    x = wide_iq.astype(np.complex64)
    x = x - np.mean(x)

    n = np.arange(wide_start_sample, wide_start_sample + len(x), dtype=np.float64)

    mixer = np.exp(
        -1j * 2.0 * np.pi * offset_hz * n / ORIG_FS
    ).astype(np.complex64)

    shifted = x * mixer
    narrow = resample_poly(shifted, up=1, down=DECIM).astype(np.complex64)

    if len(narrow) != BLOCK_SIZE:
        raise RuntimeError(f"narrow block length mismatch: {len(narrow)} != {BLOCK_SIZE}")

    return narrow


def compute_quality_metrics(spec: np.ndarray, power_ratio: float) -> dict:
    s = spec.astype(np.float32)

    spec_min = float(np.min(s))
    spec_max = float(np.max(s))
    spec_mean = float(np.mean(s))
    spec_std = float(np.std(s))

    if spec_max - spec_min < 1e-8:
        active_mask = np.zeros_like(s, dtype=bool)
    else:
        active_mask = s >= 0.55

    active_ratio = float(np.mean(active_mask))

    time_active = np.any(active_mask, axis=0)
    time_activity_ratio = float(np.mean(time_active))

    freq_active = np.any(active_mask, axis=1)
    freq_occupancy_ratio = float(np.mean(freq_active))

    if active_ratio < 0.002:
        structure_score = 0.0
    elif active_ratio > 0.95:
        structure_score = 0.1
    else:
        structure_score = (
            0.4 * min(time_activity_ratio, 1.0)
            + 0.4 * min(freq_occupancy_ratio, 1.0)
            + 0.2 * min(spec_std * 4.0, 1.0)
        )

    quality_score = float(power_ratio * structure_score)

    return {
        "spec_min": spec_min,
        "spec_max": spec_max,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "active_ratio": active_ratio,
        "time_activity_ratio": time_activity_ratio,
        "freq_occupancy_ratio": freq_occupancy_ratio,
        "quality_score": quality_score,
    }


def save_spectrogram_png(spec: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.colorbar(label="normalized magnitude")
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def make_contact_sheet(
    records: list[dict],
    save_path: Path,
    max_images: int = 30,
    cols: int = 5,
) -> None:
    if len(records) == 0:
        return

    records = sorted(records, key=lambda r: r["quality_score"], reverse=True)
    records = records[:max_images]

    rows = int(np.ceil(len(records) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for i, record in enumerate(records):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        spec = np.load(record["spectrogram_npy"])
        ax.imshow(spec, aspect="auto", origin="lower")
        ax.set_title(
            f"blk {record['wide_block_index']}\n"
            f"{record['f_center_mhz']:+.1f} MHz | q={record['quality_score']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto scan Mavic wideband IQ and export useful CNN spectrograms."
    )

    parser.add_argument("--mat-path", type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument(
        "--start-wide-block",
        type=int,
        default=0,
        help="시작 wide block index",
    )

    parser.add_argument(
        "--max-wide-blocks",
        type=int,
        default=20,
        help="처리할 wide block 수. 0이면 파일 끝까지 처리",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="각 wide block마다 최대 몇 개 후보 대역을 볼지",
    )

    parser.add_argument(
        "--power-ratio",
        type=float,
        default=1.2,
        help="band_power / median_band_power가 이 값 이상이면 후보",
    )

    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="이 quality_score 이상만 최종 저장",
    )

    parser.add_argument(
        "--save-top-if-none",
        action="store_true",
        help="조건 통과 후보가 없어도 가장 센 대역 1개는 저장 후보로 봄",
    )

    parser.add_argument(
        "--save-png",
        action="store_true",
        help="spectrogram PNG도 저장",
    )

    parser.add_argument(
        "--save-iq",
        action="store_true",
        help="선택된 5 MSPS IQ block도 저장",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="drone_like_candidate",
        help="저장 metadata에 들어갈 label",
    )

    parser.add_argument(
        "--contact-sheet-count",
        type=int,
        default=30,
        help="상위 몇 개 spectrogram을 contact sheet로 묶을지",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {args.mat_path}")

    out_dir = args.out_dir
    spec_npy_dir = out_dir / "spectrogram_npy"
    spec_png_dir = out_dir / "spectrogram_png"
    iq_dir = out_dir / "iq_blocks"
    meta_dir = out_dir / "meta"

    spec_npy_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        spec_png_dir.mkdir(parents=True, exist_ok=True)

    if args.save_iq:
        iq_dir.mkdir(parents=True, exist_ok=True)

    bands, masks = build_band_masks(
        fs=ORIG_FS,
        band_width=TARGET_FS,
        fft_size=WIDE_BLOCK_SIZE,
    )

    scan_rows = []
    export_rows = []
    avg_power_sum = np.zeros(len(bands), dtype=np.float64)
    avg_power_count = 0

    print("=== Auto Mavic Pipeline ===")
    print(f"mat_path          : {args.mat_path}")
    print(f"out_dir           : {out_dir}")
    print(f"orig_fs           : {ORIG_FS}")
    print(f"target_fs         : {TARGET_FS}")
    print(f"block_size        : {BLOCK_SIZE}")
    print(f"wide_block_size   : {WIDE_BLOCK_SIZE}")
    print(f"top_k             : {args.top_k}")
    print(f"power_ratio       : {args.power_ratio}")
    print(f"min_quality_score : {args.min_quality_score}")
    print(f"save_png          : {args.save_png}")
    print(f"save_iq           : {args.save_iq}")
    print()

    with h5py.File(args.mat_path, "r") as f:
        if "uhd_samps" not in f:
            raise KeyError(f"'uhd_samps' dataset not found. keys={list(f.keys())}")

        dset = f["uhd_samps"]

        if dset.shape[0] == 1:
            total_samples = dset.shape[1]
        else:
            total_samples = dset.shape[0]

        total_wide_blocks = total_samples // WIDE_BLOCK_SIZE

        start_block = int(args.start_wide_block)
        if args.max_wide_blocks == 0:
            end_block = total_wide_blocks
        else:
            end_block = min(total_wide_blocks, start_block + int(args.max_wide_blocks))

        print(f"dataset shape     : {dset.shape}")
        print(f"dataset dtype      : {dset.dtype}")
        print(f"total samples      : {total_samples}")
        print(f"total wide blocks  : {total_wide_blocks}")
        print(f"process range      : {start_block} ~ {end_block - 1}")
        print()

        for wide_block_idx in range(start_block, end_block):
            wide_start = wide_block_idx * WIDE_BLOCK_SIZE
            wide_end = wide_start + WIDE_BLOCK_SIZE

            wide_iq = load_complex_iq_from_mat(
                dset=dset,
                start_sample=wide_start,
                num_samples=WIDE_BLOCK_SIZE,
            )

            band_powers = compute_band_powers(wide_iq, masks)
            median_power = float(np.median(band_powers))
            ratios = band_powers / (median_power + 1e-30)

            avg_power_sum += band_powers
            avg_power_count += 1

            order = np.argsort(band_powers)[::-1]

            candidates = [
                int(i)
                for i in order
                if ratios[i] >= args.power_ratio
            ][: args.top_k]

            if len(candidates) == 0 and args.save_top_if_none and args.top_k > 0:
                candidates = [int(order[0])]

            candidate_text = ", ".join(
                f"{bands[i]['f_center_mhz']:+.1f}MHz(r={ratios[i]:.2f})"
                for i in candidates
            )

            print(
                f"[wide {wide_block_idx:06d}] "
                f"median={median_power:.3e}, candidates={len(candidates)} "
                f"{candidate_text}"
            )

            for i, b in enumerate(bands):
                scan_rows.append(
                    {
                        "wide_block_index": wide_block_idx,
                        "wide_start_sample": wide_start,
                        "wide_end_sample": wide_end,
                        "band_index": b["band_index"],
                        "f_start_mhz": b["f_start_mhz"],
                        "f_end_mhz": b["f_end_mhz"],
                        "f_center_mhz": b["f_center_mhz"],
                        "band_power": float(band_powers[i]),
                        "median_power": median_power,
                        "power_ratio": float(ratios[i]),
                        "selected_by_power": int(i in candidates),
                    }
                )

            for band_idx in candidates:
                b = bands[band_idx]
                offset_hz = float(b["f_center_hz"])
                offset_mhz = float(b["f_center_mhz"])
                offset_name = safe_offset_name(offset_mhz)

                narrow_iq = extract_subband(
                    wide_iq=wide_iq,
                    wide_start_sample=wide_start,
                    offset_hz=offset_hz,
                )

                narrow_iq = narrow_iq - np.mean(narrow_iq)

                branch = compute_stft_branch(
                    iq_block=narrow_iq,
                    sample_rate=TARGET_FS,
                    nperseg=NPERSEG,
                    noverlap=NOVERLAP,
                    nfft=NFFT,
                    window=WINDOW,
                )

                spec = branch.cnn_spectrogram.astype(np.float32)

                metrics = compute_quality_metrics(
                    spec=spec,
                    power_ratio=float(ratios[band_idx]),
                )

                if metrics["quality_score"] < args.min_quality_score:
                    continue

                base_name = (
                    f"{args.label}_wide_{wide_block_idx:06d}_"
                    f"offset_{offset_name}MHz"
                )

                offset_spec_npy_dir = spec_npy_dir / f"offset_{offset_name}MHz"
                offset_meta_dir = meta_dir / f"offset_{offset_name}MHz"

                offset_spec_npy_dir.mkdir(parents=True, exist_ok=True)
                offset_meta_dir.mkdir(parents=True, exist_ok=True)

                spec_npy_path = offset_spec_npy_dir / f"{base_name}.npy"
                np.save(spec_npy_path, spec)

                iq_path = ""
                if args.save_iq:
                    offset_iq_dir = iq_dir / f"offset_{offset_name}MHz"
                    offset_iq_dir.mkdir(parents=True, exist_ok=True)

                    iq_path_obj = offset_iq_dir / f"{base_name}_iq.npy"
                    np.save(iq_path_obj, narrow_iq.astype(np.complex64))
                    iq_path = str(iq_path_obj)

                png_path = ""
                if args.save_png:
                    offset_spec_png_dir = spec_png_dir / f"offset_{offset_name}MHz"
                    offset_spec_png_dir.mkdir(parents=True, exist_ok=True)

                    png_path_obj = offset_spec_png_dir / f"{base_name}.png"
                    save_spectrogram_png(
                        spec=spec,
                        save_path=png_path_obj,
                        title=(
                            f"{args.label} | wide {wide_block_idx} | "
                            f"offset {offset_mhz:+.1f} MHz | "
                            f"q={metrics['quality_score']:.2f}"
                        ),
                    )
                    png_path = str(png_path_obj)

                record = {
                    "label": args.label,
                    "source_mat_path": str(args.mat_path),
                    "wide_block_index": wide_block_idx,
                    "wide_start_sample": wide_start,
                    "wide_end_sample": wide_end,
                    "original_sample_rate_hz": ORIG_FS,
                    "target_sample_rate_hz": TARGET_FS,
                    "block_size": BLOCK_SIZE,
                    "band_index": int(band_idx),
                    "f_start_mhz": float(b["f_start_mhz"]),
                    "f_end_mhz": float(b["f_end_mhz"]),
                    "f_center_mhz": offset_mhz,
                    "band_power": float(band_powers[band_idx]),
                    "median_power": median_power,
                    "power_ratio": float(ratios[band_idx]),
                    "nperseg": NPERSEG,
                    "noverlap": NOVERLAP,
                    "nfft": NFFT,
                    "window": WINDOW,
                    "spectrogram_shape": list(spec.shape),
                    "spectrogram_npy": str(spec_npy_path),
                    "spectrogram_png": png_path,
                    "iq_npy": iq_path,
                    **metrics,
                }

                meta_path = offset_meta_dir / f"{base_name}.json"
                with meta_path.open("w", encoding="utf-8") as fp:
                    json.dump(record, fp, indent=2, ensure_ascii=False)

                export_rows.append(record)

    if len(scan_rows) == 0:
        raise RuntimeError("처리된 scan row가 없습니다.")

    scan_csv_path = out_dir / "wideband_scan_all_bands.csv"
    with scan_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=scan_rows[0].keys())
        writer.writeheader()
        writer.writerows(scan_rows)

    export_csv_path = out_dir / "exported_candidates.csv"
    if len(export_rows) > 0:
        fieldnames = list(export_rows[0].keys())
        with export_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_rows)

    quality_csv_path = out_dir / "quality_report.csv"
    if len(export_rows) > 0:
        quality_rows = sorted(
            export_rows,
            key=lambda r: r["quality_score"],
            reverse=True,
        )

        fieldnames = [
            "label",
            "wide_block_index",
            "f_center_mhz",
            "power_ratio",
            "quality_score",
            "active_ratio",
            "time_activity_ratio",
            "freq_occupancy_ratio",
            "spec_mean",
            "spec_std",
            "spectrogram_npy",
            "spectrogram_png",
        ]

        with quality_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in quality_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    avg_band_power = avg_power_sum / max(1, avg_power_count)

    avg_rows = []
    for i, b in enumerate(bands):
        avg_rows.append(
            {
                "band_index": b["band_index"],
                "f_start_mhz": b["f_start_mhz"],
                "f_end_mhz": b["f_end_mhz"],
                "f_center_mhz": b["f_center_mhz"],
                "avg_band_power": float(avg_band_power[i]),
            }
        )

    avg_csv_path = out_dir / "average_band_power.csv"
    with avg_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=avg_rows[0].keys())
        writer.writeheader()
        writer.writerows(avg_rows)

    plt.figure(figsize=(12, 5))
    plt.bar(
        [r["f_center_mhz"] for r in avg_rows],
        [r["avg_band_power"] for r in avg_rows],
        width=4.5,
    )
    plt.xlabel("Frequency offset from file center (MHz)")
    plt.ylabel("Average band power")
    plt.title("Average 5 MHz band power")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    avg_png_path = out_dir / "average_band_power.png"
    plt.savefig(avg_png_path, dpi=150)
    plt.close()

    contact_sheet_path = out_dir / "top_quality_contact_sheet.png"
    if len(export_rows) > 0:
        make_contact_sheet(
            records=export_rows,
            save_path=contact_sheet_path,
            max_images=args.contact_sheet_count,
            cols=5,
        )

    summary = {
        "mat_path": str(args.mat_path),
        "out_dir": str(out_dir),
        "orig_fs": ORIG_FS,
        "target_fs": TARGET_FS,
        "block_size": BLOCK_SIZE,
        "wide_block_size": WIDE_BLOCK_SIZE,
        "top_k": args.top_k,
        "power_ratio": args.power_ratio,
        "min_quality_score": args.min_quality_score,
        "num_scan_rows": len(scan_rows),
        "num_exported_candidates": len(export_rows),
        "save_iq": args.save_iq,
        "save_png": args.save_png,
        "scan_csv": str(scan_csv_path),
        "export_csv": str(export_csv_path),
        "quality_csv": str(quality_csv_path),
        "average_band_power_csv": str(avg_csv_path),
        "average_band_power_png": str(avg_png_path),
        "top_quality_contact_sheet": str(contact_sheet_path),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print()
    print("=== Done ===")
    print(f"scan csv       : {scan_csv_path}")
    print(f"export csv     : {export_csv_path}")
    print(f"quality csv    : {quality_csv_path}")
    print(f"avg power png  : {avg_png_path}")
    print(f"contact sheet  : {contact_sheet_path}")
    print(f"summary        : {summary_path}")
    print(f"exported       : {len(export_rows)}")


if __name__ == "__main__":
    main()
