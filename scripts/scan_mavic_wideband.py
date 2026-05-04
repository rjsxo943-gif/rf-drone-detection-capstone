from pathlib import Path
import csv

import h5py
import numpy as np
import matplotlib.pyplot as plt


MAT_PATH = Path("data/external/mavic/Mavic_0.mat")
OUT_DIR = Path("outputs/runs/latest/mavic_wideband_scan")

ORIG_FS = 100_000_000          # 원본 sample rate: 100 MSPS라고 가정
TARGET_BW = 5_000_000          # 5 MHz 단위로 power 확인
TARGET_BLOCK_SIZE = 16_384     # 네 pipeline 기준 block size
DECIM = ORIG_FS // TARGET_BW   # 20

# 5 MSPS에서 16384 samples와 같은 시간 길이를 원본 100 MSPS에서 읽기
WIDE_BLOCK_SIZE = TARGET_BLOCK_SIZE * DECIM  # 327680 samples

# 너무 많이 돌리면 느리니까 처음은 20개 block만 확인
NUM_WIDE_BLOCKS = 20
START_SAMPLE = 0


def load_wide_block(dset, start_sample: int, num_samples: int) -> np.ndarray:
    raw = dset[0, start_sample:start_sample + num_samples]

    if raw.dtype.fields is not None:
        iq = raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)
    else:
        raise ValueError(f"지원하지 않는 dtype입니다: {raw.dtype}")

    return iq.astype(np.complex64)


def compute_5mhz_band_power(iq: np.ndarray, fs: int, band_width: int):
    n = len(iq)

    # DC offset 제거
    iq = iq - np.mean(iq)

    # leakage 줄이기
    window = np.hanning(n).astype(np.float32)
    x = iq * window

    spec = np.fft.fftshift(np.fft.fft(x))
    power = np.abs(spec) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))

    # -50~+50 MHz를 5 MHz씩 나눔
    edges = np.arange(-fs / 2, fs / 2 + band_width, band_width)

    rows = []
    for i in range(len(edges) - 1):
        f0 = edges[i]
        f1 = edges[i + 1]
        fc = (f0 + f1) / 2

        mask = (freqs >= f0) & (freqs < f1)
        band_power_mean = float(np.mean(power[mask]))
        band_power_max = float(np.max(power[mask]))

        rows.append({
            "band_index": i,
            "f_start_hz": f0,
            "f_end_hz": f1,
            "f_center_hz": fc,
            "mean_power": band_power_mean,
            "max_power": band_power_max,
        })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Mavic wideband scan ===")
    print(f"MAT_PATH: {MAT_PATH}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"ORIG_FS: {ORIG_FS}")
    print(f"WIDE_BLOCK_SIZE: {WIDE_BLOCK_SIZE}")
    print(f"NUM_WIDE_BLOCKS: {NUM_WIDE_BLOCKS}")

    all_band_mean = None
    all_band_max = None
    band_info = None
    used_blocks = 0

    with h5py.File(MAT_PATH, "r") as f:
        dset = f["uhd_samps"]
        total_samples = dset.shape[1]

        print(f"dataset shape: {dset.shape}")
        print(f"total samples: {total_samples}")

        for block_idx in range(NUM_WIDE_BLOCKS):
            start = START_SAMPLE + block_idx * WIDE_BLOCK_SIZE
            end = start + WIDE_BLOCK_SIZE

            if end > total_samples:
                break

            iq = load_wide_block(dset, start, WIDE_BLOCK_SIZE)
            rows = compute_5mhz_band_power(iq, ORIG_FS, TARGET_BW)

            mean_arr = np.array([r["mean_power"] for r in rows], dtype=np.float64)
            max_arr = np.array([r["max_power"] for r in rows], dtype=np.float64)

            if all_band_mean is None:
                all_band_mean = mean_arr
                all_band_max = max_arr
                band_info = rows
            else:
                all_band_mean += mean_arr
                all_band_max = np.maximum(all_band_max, max_arr)

            used_blocks += 1

            print(f"[block {block_idx:04d}] scanned samples {start} ~ {end}")

    if used_blocks == 0:
        raise RuntimeError("읽은 block이 없습니다. 파일 경로 또는 sample 범위를 확인하세요.")

    avg_band_mean = all_band_mean / used_blocks

    summary_rows = []
    for i, info in enumerate(band_info):
        summary_rows.append({
            "band_index": info["band_index"],
            "f_start_mhz": info["f_start_hz"] / 1e6,
            "f_end_mhz": info["f_end_hz"] / 1e6,
            "f_center_mhz": info["f_center_hz"] / 1e6,
            "avg_mean_power": float(avg_band_mean[i]),
            "max_power": float(all_band_max[i]),
        })

    csv_path = OUT_DIR / "wideband_5mhz_power.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # power 기준 상위 대역 출력
    sorted_rows = sorted(summary_rows, key=lambda r: r["avg_mean_power"], reverse=True)

    print()
    print("=== Top 5 bands by avg_mean_power ===")
    for r in sorted_rows[:5]:
        print(
            f"band {r['band_index']:02d}: "
            f"{r['f_start_mhz']:+.1f} ~ {r['f_end_mhz']:+.1f} MHz "
            f"(center {r['f_center_mhz']:+.1f} MHz), "
            f"avg_power={r['avg_mean_power']:.3e}, "
            f"max_power={r['max_power']:.3e}"
        )

    # 그래프 저장
    centers = np.array([r["f_center_mhz"] for r in summary_rows])
    powers = np.array([r["avg_mean_power"] for r in summary_rows])

    plt.figure(figsize=(12, 5))
    plt.bar(centers, powers, width=4.5)
    plt.xlabel("Frequency offset from file center (MHz)")
    plt.ylabel("Average power")
    plt.title("Mavic wideband scan: 5 MHz band power")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    fig_path = OUT_DIR / "wideband_5mhz_power.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print()
    print(f"saved csv: {csv_path}")
    print(f"saved plot: {fig_path}")


if __name__ == "__main__":
    main()