from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from src.core import load_all_configs, get_block_size
from src.receiver import build_receiver


def make_spectrogram_128(
    x: np.ndarray,
    sample_rate: int,
    nperseg: int = 128,
    noverlap: int = 96,
    normalize: bool = False,
    remove_dc_bin: bool = True,
) -> np.ndarray:
    """
    단일 채널 IQ 1 block을 STFT spectrogram[dB]으로 변환한다.

    기본 설정:
    - nperseg = 128
    - noverlap = 96
    - hop = 32
    - block_size = 16384일 때 shape ≈ (128, 509)

    normalize=False:
    - OFF/ON 비교용
    - 실제 수신 세기 차이를 어느 정도 유지

    normalize=True:
    - CNN 입력 통일용 후보
    - 단, 학습 데이터 전체에서 같은 방식으로 써야 함
    """

    x = np.asarray(x)

    # 시간 영역 DC offset 제거
    x = x - np.mean(x)

    # 비교 실험에서는 기본적으로 끔
    if normalize:
        x = x / (np.max(np.abs(x)) + 1e-12)

    _, _, zxx = stft(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    spec = np.abs(np.fft.fftshift(zxx, axes=0))
    spec_db = 20 * np.log10(spec + 1e-12)

    # 중앙 DC/LO leakage 줄이기
    if remove_dc_bin:
        center = spec_db.shape[0] // 2
        median_value = np.median(spec_db)
        spec_db[center - 1:center + 2, :] = median_value

    return spec_db.astype(np.float32)


def save_spectrogram_png(
    spec_db: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Magnitude [dB]")
    plt.title(title)
    plt.xlabel("Time frame")
    plt.ylabel("Frequency bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real Pluto+ Wi-Fi/hotspot spectrograms with STFT 128/96."
    )

    parser.add_argument(
        "--label",
        required=True,
        help="저장 라벨. 예: library_wifi_off, wifihot_on",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=20,
        help="저장할 block 개수",
    )
    parser.add_argument(
        "--center-freq",
        type=int,
        default=2_437_000_000,
        help="수신 중심 주파수. 기본값 2.437GHz",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=35.0,
        help="SDR manual gain. 도서관에서는 35~40 추천",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="사용할 RX channel index. 현재 잘 보이면 0 유지, 안 보이면 1 테스트",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="켜면 peak normalization 적용. OFF/ON 비교용이면 보통 끄는 것을 추천",
    )
    parser.add_argument(
        "--keep-dc-bin",
        action="store_true",
        help="켜면 중앙 DC bin 제거를 하지 않음",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-90.0,
        help="PNG color scale lower bound",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-20.0,
        help="PNG color scale upper bound",
    )

    args = parser.parse_args()

    nperseg = 128
    noverlap = 96
    hop = nperseg - noverlap

    configs = load_all_configs("configs")
    receiver_cfg = configs["receiver"]
    block_size = get_block_size(configs)

    # SDR 설정을 명령어 인자로 덮어쓰기
    receiver_cfg["source_type"] = "sdr"
    receiver_cfg["center_freq"] = args.center_freq
    receiver_cfg["sample_rate"] = 5_000_000
    receiver_cfg["block_size"] = block_size
    receiver_cfg["num_samples"] = block_size
    receiver_cfg["num_channels"] = 1

    sdr_cfg = receiver_cfg.setdefault("sdr", {})
    sdr_cfg["center_freq"] = args.center_freq
    sdr_cfg["sample_rate"] = 5_000_000
    sdr_cfg["rf_bandwidth"] = 5_000_000
    sdr_cfg["channels"] = [args.channel]
    sdr_cfg["gain_control_mode"] = "manual"
    sdr_cfg["gain"] = args.gain
    sdr_cfg["num_samples"] = block_size
    sdr_cfg["block_size"] = block_size

    sample_rate = int(sdr_cfg["sample_rate"])
    center_freq = int(sdr_cfg["center_freq"])
    channels = sdr_cfg["channels"]
    gain = sdr_cfg["gain"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/processed/cnn_capture") / args.label / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Wi-Fi / Hotspot Capture 128 ===")
    print("label       :", args.label)
    print("source_type :", receiver_cfg.get("source_type"))
    print("center_freq :", center_freq)
    print("sample_rate :", sample_rate)
    print("block_size  :", block_size)
    print("channels    :", channels)
    print("gain        :", gain)
    print("nperseg     :", nperseg)
    print("noverlap    :", noverlap)
    print("hop         :", hop)
    print("normalize   :", args.normalize)
    print("remove_dc   :", not args.keep_dc_bin)
    print("vmin/vmax   :", args.vmin, args.vmax)
    print("out_dir     :", out_dir)
    print()

    rx = build_receiver(receiver_cfg)
    saved = 0

    try:
        for block_idx in range(args.blocks):
            iq = rx.read_block(block_size)
            iq = np.asarray(iq)

            if iq.ndim == 1:
                x = iq
            elif iq.ndim == 2:
                x = iq[0]
            else:
                raise ValueError(f"Unexpected IQ shape: {iq.shape}")

            iq_mean_abs = float(np.abs(x).mean())
            iq_max_abs = float(np.abs(x).max())

            spec_db = make_spectrogram_128(
                x=x,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                normalize=args.normalize,
                remove_dc_bin=not args.keep_dc_bin,
            )

            npy_path = out_dir / f"{args.label}_block_{block_idx:04d}.npy"
            png_path = out_dir / f"{args.label}_block_{block_idx:04d}.png"

            np.save(npy_path, spec_db)

            title = (
                f"{args.label} | {center_freq / 1e9:.4f} GHz | "
                f"gain={gain} | STFT 128/96 | block={block_idx:04d}"
            )

            save_spectrogram_png(
                spec_db=spec_db,
                out_path=png_path,
                title=title,
                vmin=args.vmin,
                vmax=args.vmax,
            )

            print(
                f"[{block_idx:04d}] "
                f"iq_shape={iq.shape}, "
                f"iq_mean_abs={iq_mean_abs:.3f}, "
                f"iq_max_abs={iq_max_abs:.3f}, "
                f"spec_shape={spec_db.shape}, "
                f"spec_min={spec_db.min():.2f}, "
                f"spec_max={spec_db.max():.2f}, "
                f"saved={png_path}"
            )

            saved += 1

    finally:
        if hasattr(rx, "close"):
            rx.close()

    metadata = {
        "label": args.label,
        "saved": saved,
        "blocks": args.blocks,
        "source_type": receiver_cfg.get("source_type"),
        "center_freq": center_freq,
        "sample_rate": sample_rate,
        "rf_bandwidth": sdr_cfg.get("rf_bandwidth"),
        "block_size": block_size,
        "channels": channels,
        "gain": gain,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "hop": hop,
        "normalize": args.normalize,
        "remove_dc_bin": not args.keep_dc_bin,
        "expected_shape": [128, 509],
        "vmin": args.vmin,
        "vmax": args.vmax,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=== Capture Finished ===")
    print("saved     :", saved)
    print("output_dir:", out_dir)


if __name__ == "__main__":
    main()
