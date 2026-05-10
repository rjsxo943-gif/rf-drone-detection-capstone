from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def ensure_2d_iq(data) -> np.ndarray:
    """
    pyadi-iio rx() 결과를 shape=(channels, samples)로 맞춘다.
    """
    if isinstance(data, list):
        arr = np.stack(data, axis=0)
    else:
        arr = np.asarray(data)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr.astype(np.complex64)


def frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(x) < frame_size:
        raise ValueError("signal length is shorter than frame_size")

    starts = np.arange(0, len(x) - frame_size + 1, hop_size)
    frames = np.stack([x[s:s + frame_size] for s in starts], axis=0)
    return frames


def compute_frame_energy(
    x: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: str = "hann",
) -> np.ndarray:
    frames = frame_signal(x, frame_size, hop_size)

    if window == "hann":
        w = np.hanning(frame_size).astype(np.float32)
        frames = frames * w[np.newaxis, :]
    elif window == "rect":
        pass
    else:
        raise ValueError(f"unsupported window: {window}")

    energy = np.mean(np.abs(frames) ** 2, axis=1)
    return energy.astype(np.float32)


def build_center_freqs(
    band_start: int,
    band_end: int,
    channel_bw: int,
) -> np.ndarray:
    """
    예:
    band_start=2.400GHz, band_end=2.485GHz, channel_bw=5MHz이면

    2.400~2.405GHz 조각의 중심: 2.4025GHz
    2.405~2.410GHz 조각의 중심: 2.4075GHz
    ...
    """
    num_bins = int(np.floor((band_end - band_start) / channel_bw))
    centers = band_start + channel_bw / 2 + np.arange(num_bins) * channel_bw
    return centers.astype(np.int64)


def configure_sdr(
    uri: str,
    sample_rate: int,
    rf_bandwidth: int,
    buffer_size: int,
    gain: float,
):
    sdr = adi.ad9361(uri=uri)

    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(rf_bandwidth)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.rx_enabled_channels = [0, 1]

    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = gain
    sdr.rx_hardwaregain_chan1 = gain

    return sdr


def retune_sdr(sdr, center_freq: int, retune_warmup_reads: int) -> None:
    sdr.rx_lo = int(center_freq)

    # retune 직후 기존 buffer 제거 시도
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    for _ in range(retune_warmup_reads):
        _ = sdr.rx()


def plot_sweep_score(
    results: list[dict],
    out_dir: Path,
    channel_bw: int,
) -> None:
    center_mhz = np.array([r["center_freq"] / 1e6 for r in results])
    score_db = np.array([r["score_db"] for r in results])
    ratio = np.array([r["max_detection_ratio"] for r in results])

    plt.figure(figsize=(14, 5))
    plt.bar(center_mhz, score_db, width=(channel_bw / 1e6) * 0.8)
    plt.title("Sweep Energy Score by Frequency")
    plt.xlabel("Center Frequency [MHz]")
    plt.ylabel("Peak / Noise Floor [dB]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_score_by_frequency.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.bar(center_mhz, ratio, width=(channel_bw / 1e6) * 0.8)
    plt.title("Max Detection Ratio by Frequency")
    plt.xlabel("Center Frequency [MHz]")
    plt.ylabel("Max block detection ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_detection_ratio_by_frequency.png", dpi=150)
    plt.close()


def plot_block_spectrogram_with_marker(
    iq_block: np.ndarray,
    detected_mask: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    title: str,
    out_png: Path,
) -> None:
    ch0 = iq_block[0]
    ch0 = ch0 - np.mean(ch0)

    f, t, Sxx = spectrogram(
        ch0,
        fs=sample_rate,
        nperseg=512,
        noverlap=384,
        return_onesided=False,
        mode="magnitude",
    )

    f_shift = np.fft.fftshift(f)
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 20 * np.log10(Sxx_shift + 1e-12)

    frame_starts = np.arange(len(detected_mask)) * hop_size / sample_rate
    frame_ends = frame_starts + frame_size / sample_rate

    fig, ax = plt.subplots(figsize=(10, 5))

    pcm = ax.pcolormesh(
        t,
        f_shift / 1e6,
        Sxx_db,
        shading="gouraud",
    )

    # 중요:
    # 전체 주파수축을 칠하지 않고, 아래쪽 얇은 막대로만 표시한다.
    # 의미: "이 시간 구간이 energy detector에 걸렸다"
    overlay_labeled = False
    for is_detected, start_t, end_t in zip(
        detected_mask,
        frame_starts,
        frame_ends,
    ):
        if is_detected:
            if not overlay_labeled:
                ax.axvspan(
                    start_t,
                    end_t,
                    ymin=0.00,
                    ymax=0.06,
                    color="red",
                    alpha=0.75,
                    label="Detected time window",
                )
                overlay_labeled = True
            else:
                ax.axvspan(
                    start_t,
                    end_t,
                    ymin=0.00,
                    ymax=0.06,
                    color="red",
                    alpha=0.75,
                )

    ax.set_title(title)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Baseband Frequency [MHz]")

    if overlay_labeled:
        ax.legend(loc="upper right")

    fig.colorbar(pcm, ax=ax, label="Magnitude [dB]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--uri", default="ip:192.168.2.1")

    parser.add_argument("--band-start", type=int, default=2_400_000_000)
    parser.add_argument("--band-end", type=int, default=2_485_000_000)
    parser.add_argument("--channel-bw", type=int, default=5_000_000)

    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=5_000_000)
    parser.add_argument("--block-size", type=int, default=16_384)

    parser.add_argument("--scan-blocks", type=int, default=30)
    parser.add_argument("--gain", type=float, default=40.0)

    parser.add_argument("--initial-warmup-reads", type=int, default=5)
    parser.add_argument("--retune-warmup-reads", type=int, default=3)

    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--threshold-multiplier", type=float, default=5.0)
    parser.add_argument("--min-detection-ratio", type=float, default=0.05)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-top-iq", action="store_true")

    parser.add_argument("--precision-blocks", type=int, default=200)
    parser.add_argument("--precision-top-k", type=int, default=5)

    parser.add_argument(
        "--out-dir",
        default="outputs/figures/sdr_energy_sweep_test",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center_freqs = build_center_freqs(
        band_start=args.band_start,
        band_end=args.band_end,
        channel_bw=args.channel_bw,
    )

    print("[INFO] Sweep frequencies:")
    for cf in center_freqs:
        f0 = cf - args.channel_bw // 2
        f1 = cf + args.channel_bw // 2
        print(f"  {f0 / 1e9:.4f} ~ {f1 / 1e9:.4f} GHz, center={cf / 1e9:.4f} GHz")

    print("[INFO] Configure SDR")
    sdr = configure_sdr(
        uri=args.uri,
        sample_rate=args.sample_rate,
        rf_bandwidth=args.rf_bandwidth,
        buffer_size=args.block_size,
        gain=args.gain,
    )

    print("[INFO] Initial warmup")
    for _ in range(args.initial_warmup_reads):
        _ = sdr.rx()

    results: list[dict] = []
    strongest_blocks: list[dict] = []

    print("[INFO] Start sweep")

    for freq_idx, cf in enumerate(center_freqs):
        print(f"\n[SCAN] {freq_idx + 1}/{len(center_freqs)} center={cf / 1e9:.4f} GHz")

        retune_sdr(
            sdr=sdr,
            center_freq=int(cf),
            retune_warmup_reads=args.retune_warmup_reads,
        )

        block_iqs = []
        block_energies = []

        for block_idx in range(args.scan_blocks):
            data = sdr.rx()
            iq = ensure_2d_iq(data)

            ch0 = iq[0]
            ch0 = ch0 - np.mean(ch0)

            energy = compute_frame_energy(
                ch0,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                window="hann",
            )

            block_iqs.append(iq)
            block_energies.append(energy)

        block_iqs = np.stack(block_iqs, axis=0)
        block_energies = np.stack(block_energies, axis=0)

        # 주파수 조각별 noise floor
        noise_floor = float(np.median(block_energies))
        threshold = noise_floor * args.threshold_multiplier

        detected_frames = block_energies > threshold
        block_detection_ratio = detected_frames.mean(axis=1)
        detected_blocks = block_detection_ratio >= args.min_detection_ratio

        # score는 peak energy가 noise floor보다 몇 dB 큰지
        max_energy_per_block = np.max(block_energies, axis=1)
        strongest_block_local_idx = int(np.argmax(max_energy_per_block))
        max_energy = float(max_energy_per_block[strongest_block_local_idx])

        score_db = float(10 * np.log10((max_energy + 1e-12) / (noise_floor + 1e-12)))
        threshold_db = float(10 * np.log10(threshold + 1e-12))

        result = {
            "freq_idx": freq_idx,
            "center_freq": int(cf),
            "freq_start": int(cf - args.channel_bw // 2),
            "freq_end": int(cf + args.channel_bw // 2),
            "noise_floor": noise_floor,
            "threshold": float(threshold),
            "threshold_db": threshold_db,
            "max_energy": max_energy,
            "score_db": score_db,
            "max_detection_ratio": float(np.max(block_detection_ratio)),
            "num_detected_blocks": int(detected_blocks.sum()),
            "strongest_block_local_idx": strongest_block_local_idx,
        }

        results.append(result)

        strongest_blocks.append(
            {
                "center_freq": int(cf),
                "iq": block_iqs[strongest_block_local_idx],
                "detected_mask": detected_frames[strongest_block_local_idx],
                "detection_ratio": float(block_detection_ratio[strongest_block_local_idx]),
                "score_db": score_db,
                "local_block_idx": strongest_block_local_idx,
                "noise_floor": noise_floor,
                "threshold": float(threshold),
            }
        )

        print(
            f"  score={score_db:.2f} dB, "
            f"max_ratio={result['max_detection_ratio']:.3f}, "
            f"detected_blocks={result['num_detected_blocks']}/{args.scan_blocks}, "
            f"strongest_block={strongest_block_local_idx}"
        )

    # ------------------------------------------------------------
    # 전체 sweep 결과 저장
    # ------------------------------------------------------------
    plot_sweep_score(
        results=results,
        out_dir=out_dir,
        channel_bw=args.channel_bw,
    )

    with open(out_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------------
    # Top-K frequency spectrogram 저장
    # ------------------------------------------------------------
    top_dir = out_dir / "top_frequencies"
    top_dir.mkdir(parents=True, exist_ok=True)

    score_arr = np.array([r["score_db"] for r in results])
    top_indices = np.argsort(score_arr)[::-1][: args.top_k]

    top_summary = []

    print("\n[RESULT] Top frequencies")
    for rank, result_idx in enumerate(top_indices, start=1):
        r = results[int(result_idx)]
        b = strongest_blocks[int(result_idx)]

        cf = r["center_freq"]

        print(
            f"  Top {rank}: center={cf / 1e9:.4f} GHz, "
            f"score={r['score_db']:.2f} dB, "
            f"max_ratio={r['max_detection_ratio']:.3f}"
        )

        title = (
            f"Top {rank} Sweep Spectrogram "
            f"@ {cf / 1e9:.4f} GHz "
            f"(score={r['score_db']:.2f} dB, ratio={b['detection_ratio']:.3f})"
        )

        out_png = top_dir / f"top_{rank:02d}_{int(cf / 1e6)}MHz_spectrogram.png"

        plot_block_spectrogram_with_marker(
            iq_block=b["iq"],
            detected_mask=b["detected_mask"],
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            title=title,
            out_png=out_png,
        )

        if args.save_top_iq:
            out_npz = top_dir / f"top_{rank:02d}_{int(cf / 1e6)}MHz_iq.npz"
            np.savez_compressed(
                out_npz,
                iq=b["iq"],
                center_freq=cf,
                sample_rate=args.sample_rate,
                rf_bandwidth=args.rf_bandwidth,
                block_size=args.block_size,
                local_block_idx=b["local_block_idx"],
                detection_ratio=b["detection_ratio"],
                score_db=b["score_db"],
                noise_floor=b["noise_floor"],
                threshold=b["threshold"],
            )

        top_summary.append(
            {
                "rank": rank,
                "center_freq": cf,
                "center_freq_ghz": cf / 1e9,
                "score_db": r["score_db"],
                "max_detection_ratio": r["max_detection_ratio"],
                "num_detected_blocks": r["num_detected_blocks"],
                "strongest_block_local_idx": r["strongest_block_local_idx"],
            }
        )

        # ------------------------------------------------------------
    # Precision Analysis
    # - sweep 결과 중 max_detection_ratio가 가장 높은 주파수 하나 선택
    # - 그 주파수에 고정해서 precision_blocks만큼 추가 수집
    # ------------------------------------------------------------
    precision_dir = out_dir / "precision_selected_frequency"
    precision_dir.mkdir(parents=True, exist_ok=True)

    ratio_arr = np.array([r["max_detection_ratio"] for r in results])
    best_result_idx = int(np.argmax(ratio_arr))
    best_result = results[best_result_idx]
    best_freq = int(best_result["center_freq"])

    print("\n[PRECISION] Selected frequency by max_detection_ratio")
    print(
        f"  center={best_freq / 1e9:.4f} GHz, "
        f"max_ratio={best_result['max_detection_ratio']:.3f}, "
        f"score={best_result['score_db']:.2f} dB"
    )

    retune_sdr(
        sdr=sdr,
        center_freq=best_freq,
        retune_warmup_reads=args.retune_warmup_reads,
    )

    precision_iqs = []
    precision_energies = []

    print(f"[PRECISION] Capture {args.precision_blocks} blocks")

    for block_idx in range(args.precision_blocks):
        data = sdr.rx()
        iq = ensure_2d_iq(data)

        ch0 = iq[0]
        ch0 = ch0 - np.mean(ch0)

        energy = compute_frame_energy(
            ch0,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            window="hann",
        )

        precision_iqs.append(iq)
        precision_energies.append(energy)

        if block_idx % 50 == 0:
            print(f"  captured precision block {block_idx:04d}/{args.precision_blocks}")

    precision_iqs = np.stack(precision_iqs, axis=0)
    precision_energies = np.stack(precision_energies, axis=0)

    precision_noise_floor = float(np.median(precision_energies))
    precision_threshold = precision_noise_floor * args.threshold_multiplier
    precision_threshold_db = float(10 * np.log10(precision_threshold + 1e-12))

    precision_detected_frames = precision_energies > precision_threshold
    precision_block_detection_ratio = precision_detected_frames.mean(axis=1)
    precision_detected_blocks = (
        precision_block_detection_ratio >= args.min_detection_ratio
    )

    precision_max_energy_per_block = np.max(precision_energies, axis=1)

    print("[PRECISION RESULT]")
    print(f"  noise_floor={precision_noise_floor:.6e}")
    print(f"  threshold={precision_threshold:.6e}")
    print(f"  threshold_db={precision_threshold_db:.2f} dB")
    print(
        f"  detected_blocks={int(precision_detected_blocks.sum())}/"
        f"{args.precision_blocks}"
    )

    # ------------------------------------------------------------
    # Precision 1) energy timeline 저장
    # ------------------------------------------------------------
    num_precision_blocks, num_frames = precision_energies.shape

    frame_duration = args.hop_size / args.sample_rate
    total_frames = num_precision_blocks * num_frames
    time_axis = np.arange(total_frames) * frame_duration

    precision_energy_db_flat = (
        10 * np.log10(precision_energies.reshape(-1) + 1e-12)
    )
    precision_detected_flat = precision_detected_frames.reshape(-1)

    plt.figure(figsize=(14, 5))
    plt.plot(
        time_axis,
        precision_energy_db_flat,
        linewidth=0.8,
        label="Frame energy",
    )
    plt.axhline(
        precision_threshold_db,
        linestyle="--",
        label="Threshold",
    )
    plt.scatter(
        time_axis[precision_detected_flat],
        precision_energy_db_flat[precision_detected_flat],
        s=8,
        label="Detected frames",
    )
    plt.title(
        f"Precision Energy Timeline @ {best_freq / 1e9:.4f} GHz"
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Energy [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(precision_dir / "precision_energy_timeline.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Precision 2) block detection ratio 저장
    # ------------------------------------------------------------
    block_time_axis = (
        np.arange(num_precision_blocks) * args.block_size / args.sample_rate
    )

    plt.figure(figsize=(14, 4))
    plt.plot(
        block_time_axis,
        precision_block_detection_ratio,
        marker="o",
        markersize=3,
    )
    plt.axhline(
        args.min_detection_ratio,
        linestyle="--",
        label="min_detection_ratio",
    )
    plt.title(
        f"Precision Detection Ratio by Block @ {best_freq / 1e9:.4f} GHz"
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Detected frame ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(precision_dir / "precision_block_detection_ratio.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Precision 3) Top-K block spectrogram 저장
    # 정렬 기준:
    # 1순위: detection_ratio 높은 block
    # 2순위: max_energy 높은 block
    # ------------------------------------------------------------
    precision_top_dir = precision_dir / "top_detected_blocks"
    precision_top_dir.mkdir(parents=True, exist_ok=True)

    precision_block_score = (
        precision_block_detection_ratio * 1_000_000
        + precision_max_energy_per_block
    )

    if np.any(precision_detected_blocks):
        candidate_indices = np.where(precision_detected_blocks)[0]
    else:
        candidate_indices = np.arange(num_precision_blocks)

    precision_top_indices = candidate_indices[
        np.argsort(precision_block_score[candidate_indices])[::-1]
    ][: args.precision_top_k]

    precision_top_summary = []

    print("[PRECISION] Top detected blocks")
    for rank, block_idx in enumerate(precision_top_indices, start=1):
        block_idx = int(block_idx)

        ratio = float(precision_block_detection_ratio[block_idx])
        max_energy = float(precision_max_energy_per_block[block_idx])

        print(
            f"  Top {rank}: block={block_idx}, "
            f"ratio={ratio:.3f}, max_energy={max_energy:.6e}"
        )

        title = (
            f"Precision Top {rank} Block Spectrogram "
            f"@ {best_freq / 1e9:.4f} GHz "
            f"(block={block_idx}, ratio={ratio:.3f})"
        )

        out_png = (
            precision_top_dir
            / f"top_{rank:02d}_block_{block_idx:04d}_spectrogram.png"
        )

        plot_block_spectrogram_with_marker(
            iq_block=precision_iqs[block_idx],
            detected_mask=precision_detected_frames[block_idx],
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            title=title,
            out_png=out_png,
        )

        if args.save_top_iq:
            out_npz = (
                precision_top_dir
                / f"top_{rank:02d}_block_{block_idx:04d}_iq.npz"
            )
            np.savez_compressed(
                out_npz,
                iq=precision_iqs[block_idx],
                center_freq=best_freq,
                sample_rate=args.sample_rate,
                rf_bandwidth=args.rf_bandwidth,
                block_size=args.block_size,
                block_idx=block_idx,
                detection_ratio=ratio,
                max_energy=max_energy,
                noise_floor=precision_noise_floor,
                threshold=precision_threshold,
                threshold_db=precision_threshold_db,
            )

        precision_top_summary.append(
            {
                "rank": rank,
                "block_idx": block_idx,
                "detection_ratio": ratio,
                "max_energy": max_energy,
            }
        )

    precision_summary = {
        "selected_by": "max_detection_ratio",
        "center_freq": best_freq,
        "center_freq_ghz": best_freq / 1e9,
        "sweep_max_detection_ratio": best_result["max_detection_ratio"],
        "sweep_score_db": best_result["score_db"],
        "precision_blocks": args.precision_blocks,
        "noise_floor": precision_noise_floor,
        "threshold": float(precision_threshold),
        "threshold_db": precision_threshold_db,
        "num_detected_blocks": int(precision_detected_blocks.sum()),
        "top_blocks": precision_top_summary,
    }

    with open(
        precision_dir / "precision_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(precision_summary, f, indent=2)

    summary = {
        "uri": args.uri,
        "band_start": args.band_start,
        "band_end": args.band_end,
        "channel_bw": args.channel_bw,
        "sample_rate": args.sample_rate,
        "rf_bandwidth": args.rf_bandwidth,
        "block_size": args.block_size,
        "scan_blocks": args.scan_blocks,
        "gain": args.gain,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "threshold_multiplier": args.threshold_multiplier,
        "min_detection_ratio": args.min_detection_ratio,
        "num_freqs": len(center_freqs),
        "top_summary": top_summary,
        "precision_summary": precision_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] {out_dir}")


if __name__ == "__main__":
    main()