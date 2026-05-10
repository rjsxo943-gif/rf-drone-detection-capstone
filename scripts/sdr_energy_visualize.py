from __future__ import annotations

import argparse
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
    """
    1D IQ 신호를 frame 단위로 자른다.
    output shape = (num_frames, frame_size)
    """
    if len(x) < frame_size:
        raise ValueError("signal length is shorter than frame_size")

    starts = np.arange(0, len(x) - frame_size + 1, hop_size)
    frames = np.stack([x[s:s + frame_size] for s in starts], axis=0)
    return frames


def compute_frame_energy_db(
    x: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    frame별 mean(|x[n]|^2)를 dB로 계산한다.
    """
    frames = frame_signal(x, frame_size, hop_size)

    if window == "hann":
        w = np.hanning(frame_size).astype(np.float32)
        frames = frames * w[np.newaxis, :]
    elif window == "rect":
        pass
    else:
        raise ValueError(f"unsupported window: {window}")

    energy = np.mean(np.abs(frames) ** 2, axis=1)
    energy_db = 10 * np.log10(energy + 1e-12)

    return energy, energy_db


def configure_sdr(
    uri: str,
    sample_rate: int,
    center_freq: int,
    rf_bandwidth: int,
    buffer_size: int,
    gain: float,
):
    sdr = adi.ad9361(uri=uri)

    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(rf_bandwidth)
    sdr.rx_lo = int(center_freq)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.rx_enabled_channels = [0, 1]

    # manual gain 설정
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = gain
    sdr.rx_hardwaregain_chan1 = gain

    return sdr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, default=2_412_000_000)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=5_000_000)
    parser.add_argument("--block-size", type=int, default=16_384)
    parser.add_argument("--num-blocks", type=int, default=200)
    parser.add_argument("--gain", type=float, default=40.0)
    parser.add_argument("--warmup-reads", type=int, default=5)

    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--threshold-multiplier", type=float, default=5.0)
    parser.add_argument("--calib-blocks", type=int, default=20)
    parser.add_argument("--min-detection-ratio", type=float, default=0.05)

    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument("--save-detected-iq", action="store_true")

    parser.add_argument(
        "--out-dir",
        default="outputs/figures/sdr_energy_detect_test",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Configure SDR")
    sdr = configure_sdr(
        uri=args.uri,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        rf_bandwidth=args.rf_bandwidth,
        buffer_size=args.block_size,
        gain=args.gain,
    )

    print("[INFO] Warmup")
    for _ in range(args.warmup_reads):
        _ = sdr.rx()

    all_iq = []
    all_energy = []
    all_energy_db = []
    block_detection_ratio = []

    print("[INFO] Capture blocks")
    for block_idx in range(args.num_blocks):
        data = sdr.rx()
        iq = ensure_2d_iq(data)

        # 우선 RX0 기준으로 energy detection
        ch0 = iq[0]

        # DC 제거
        ch0 = ch0 - np.mean(ch0)

        energy, energy_db = compute_frame_energy_db(
            ch0,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            window="hann",
        )

        all_iq.append(iq)
        all_energy.append(energy)
        all_energy_db.append(energy_db)

        if block_idx % 20 == 0:
            print(f"[INFO] captured block {block_idx:04d}/{args.num_blocks}")

    all_iq = np.stack(all_iq, axis=0)
    all_energy = np.stack(all_energy, axis=0)
    all_energy_db = np.stack(all_energy_db, axis=0)

    num_blocks, num_frames = all_energy.shape

    # 처음 calib_blocks를 noise 기준으로 사용
    calib_blocks = min(args.calib_blocks, num_blocks)
    calib_energy = all_energy[:calib_blocks].reshape(-1)

    noise_floor = float(np.median(calib_energy))
    threshold = noise_floor * args.threshold_multiplier
    threshold_db = 10 * np.log10(threshold + 1e-12)

    detected_frames = all_energy > threshold
    block_detection_ratio = detected_frames.mean(axis=1)
    detected_blocks = block_detection_ratio >= args.min_detection_ratio

    strongest_block_idx = int(np.argmax(block_detection_ratio))
    strongest_ch0 = all_iq[strongest_block_idx, 0]
    strongest_ch0 = strongest_ch0 - np.mean(strongest_ch0)

    print("[RESULT]")
    print(f"noise_floor     : {noise_floor:.6e}")
    print(f"threshold       : {threshold:.6e}")
    print(f"threshold_db    : {threshold_db:.2f} dB")
    print(f"detected_blocks : {int(detected_blocks.sum())}/{num_blocks}")
    print(f"strongest_block : {strongest_block_idx}")

    # ------------------------------------------------------------
    # 1) 전체 frame energy timeline
    # ------------------------------------------------------------
    frame_duration = args.hop_size / args.sample_rate
    total_frames = num_blocks * num_frames
    time_axis = np.arange(total_frames) * frame_duration

    energy_db_flat = all_energy_db.reshape(-1)
    detected_flat = detected_frames.reshape(-1)

    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, energy_db_flat, linewidth=0.8, label="Frame energy")
    plt.axhline(threshold_db, linestyle="--", label="Threshold")

    plt.scatter(
        time_axis[detected_flat],
        energy_db_flat[detected_flat],
        s=8,
        label="Detected frames",
    )

    plt.title(
        f"SDR Energy Detection Timeline @ {args.center_freq / 1e9:.3f} GHz"
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Energy [dB]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "energy_timeline.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 2) block별 detection ratio
    # ------------------------------------------------------------
    block_time_axis = (
        np.arange(num_blocks) * args.block_size / args.sample_rate
    )

    plt.figure(figsize=(14, 4))
    plt.plot(block_time_axis, block_detection_ratio, marker="o", markersize=3)
    plt.axhline(
        args.min_detection_ratio,
        linestyle="--",
        label="min_detection_ratio",
    )
    plt.title("Detection Ratio by Block")
    plt.xlabel("Time [sec]")
    plt.ylabel("Detected frame ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "block_detection_ratio.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 3) strongest block의 IQ magnitude
    # ------------------------------------------------------------
    sample_time = np.arange(args.block_size) / args.sample_rate

    plt.figure(figsize=(14, 4))
    plt.plot(sample_time, np.abs(strongest_ch0), linewidth=0.8)
    plt.title(f"Strongest Block RX0 Magnitude - block {strongest_block_idx}")
    plt.xlabel("Time [sec]")
    plt.ylabel("|IQ|")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "strongest_block_magnitude.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 4) strongest block의 spectrogram
    # ------------------------------------------------------------
    f, t, Sxx = spectrogram(
        strongest_ch0,
        fs=args.sample_rate,
        nperseg=512,
        noverlap=384,
        return_onesided=False,
        mode="magnitude",
    )

    f_shift = np.fft.fftshift(f)
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 20 * np.log10(Sxx_shift + 1e-12)

    strongest_detect_mask = detected_frames[strongest_block_idx]
    strongest_frame_starts = (
        np.arange(len(strongest_detect_mask)) * args.hop_size / args.sample_rate
    )
    strongest_frame_ends = (
        strongest_frame_starts + args.frame_size / args.sample_rate
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(
        t,
        f_shift / 1e6,
        Sxx_db,
        shading="gouraud",
    )

    overlay_labeled = False
    for is_detected, start_t, end_t in zip(
        strongest_detect_mask,
        strongest_frame_starts,
        strongest_frame_ends,
    ):
        if is_detected:
            if not overlay_labeled:
                ax.axvspan(
                    start_t,
                    end_t,
                    color="red",
                    alpha=0.18,
                    label="Detected frame",
                )
                overlay_labeled = True
            else:
                ax.axvspan(
                    start_t,
                    end_t,
                    color="red",
                    alpha=0.18,
                )

    ax.set_title(f"Strongest Block Spectrogram - block {strongest_block_idx}")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Baseband Frequency [MHz]")

    if overlay_labeled:
        ax.legend(loc="upper right")

    fig.colorbar(pcm, ax=ax, label="Magnitude [dB]")
    fig.tight_layout()
    fig.savefig(out_dir / "strongest_block_spectrogram.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------
    # 5) detected block 중 상위 Top-K spectrogram 저장
    # ------------------------------------------------------------
    top_dir = out_dir / "top_detected_blocks"
    top_dir.mkdir(parents=True, exist_ok=True)

    detected_indices = np.where(detected_blocks)[0]

    if len(detected_indices) > 0:
        sorted_indices = detected_indices[
            np.argsort(block_detection_ratio[detected_indices])[::-1]
        ]

        top_indices = sorted_indices[: args.save_top_k]

        print("[INFO] Save top detected block spectrograms")
        print(f"top_indices: {top_indices.tolist()}")

        for rank, block_idx in enumerate(top_indices, start=1):
            ch0 = all_iq[block_idx, 0]
            ch0 = ch0 - np.mean(ch0)

            f, t, Sxx = spectrogram(
                ch0,
                fs=args.sample_rate,
                nperseg=512,
                noverlap=384,
                return_onesided=False,
                mode="magnitude",
            )

            f_shift = np.fft.fftshift(f)
            Sxx_shift = np.fft.fftshift(Sxx, axes=0)
            Sxx_db = 20 * np.log10(Sxx_shift + 1e-12)

            detected_mask = detected_frames[block_idx]
            frame_starts = (
                np.arange(len(detected_mask)) * args.hop_size / args.sample_rate
            )
            frame_ends = frame_starts + args.frame_size / args.sample_rate

            fig, ax = plt.subplots(figsize=(10, 5))
            pcm = ax.pcolormesh(
                t,
                f_shift / 1e6,
                Sxx_db,
                shading="gouraud",
            )

            overlay_labeled = False
            for is_detected, start_t, end_t in zip(
                detected_mask, frame_starts, frame_ends
            ):
                if is_detected:
                    if not overlay_labeled:
                        ax.axvspan(
                            start_t,
                            end_t,
                            color="red",
                            alpha=0.18,
                            label="Detected frame",
                        )
                        overlay_labeled = True
                    else:
                        ax.axvspan(
                            start_t,
                            end_t,
                            color="red",
                            alpha=0.18,
                        )

            ax.set_title(
                f"Top {rank} Detected Block Spectrogram "
                f"- block {block_idx}, ratio={block_detection_ratio[block_idx]:.3f}"
            )
            ax.set_xlabel("Time [sec]")
            ax.set_ylabel("Baseband Frequency [MHz]")

            if overlay_labeled:
                ax.legend(loc="upper right")

            fig.colorbar(pcm, ax=ax, label="Magnitude [dB]")
            fig.tight_layout()

            out_png = top_dir / f"top_{rank:02d}_block_{block_idx:04d}_spectrogram.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

            if args.save_detected_iq:
                out_npz = top_dir / f"top_{rank:02d}_block_{block_idx:04d}_iq.npz"
                np.savez_compressed(
                    out_npz,
                    iq=all_iq[block_idx],
                    sample_rate=args.sample_rate,
                    center_freq=args.center_freq,
                    rf_bandwidth=args.rf_bandwidth,
                    block_size=args.block_size,
                    block_idx=block_idx,
                    detection_ratio=block_detection_ratio[block_idx],
                    threshold=threshold,
                    threshold_db=threshold_db,
                )
    else:
        print("[INFO] No detected blocks to save.")

    summary = {
        "uri": args.uri,
        "center_freq": args.center_freq,
        "sample_rate": args.sample_rate,
        "rf_bandwidth": args.rf_bandwidth,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "gain": args.gain,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "threshold_multiplier": args.threshold_multiplier,
        "calib_blocks": calib_blocks,
        "min_detection_ratio": args.min_detection_ratio,
        "noise_floor": noise_floor,
        "threshold": threshold,
        "threshold_db": threshold_db,
        "num_detected_blocks": int(detected_blocks.sum()),
        "strongest_block_idx": strongest_block_idx,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()