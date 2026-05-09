from __future__ import annotations

from pathlib import Path
from datetime import datetime
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
    단일 채널 IQ 1 block을 128-bin STFT spectrogram[dB]으로 변환한다.

    normalize=False:
        수신 상태 확인용. 핫스팟 ON/OFF 비교에 더 적합하다.

    normalize=True:
        CNN 입력용 후보. 단, 데이터셋 전체에서 방식 통일 필요.
    """

    # 시간영역 DC offset 제거
    x = x - np.mean(x)

    # 시각화/비교용에서는 기본적으로 peak normalization 끔
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
        spec_db[center - 1:center + 2, :] = np.median(spec_db)

    return spec_db.astype(np.float32)


def save_spectrogram_png(
    spec_db: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float = -90.0,
    vmax: float = -20.0,
) -> None:
    """
    비교가 가능하도록 color scale을 고정해서 저장한다.
    """

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
    # =========================
    # 설정
    # =========================
    label = "wifihot"       # 나중에 off / wifihot / school_wifi 등으로 바꿔도 됨
    num_blocks = 20         # 저장할 block 개수

    nperseg = 128
    noverlap = 96
    hop = nperseg - noverlap

    configs = load_all_configs("configs")
    receiver_cfg = configs["receiver"]
    block_size = get_block_size(configs)

    sdr_cfg = receiver_cfg.get("sdr", {})
    sample_rate = int(sdr_cfg.get("sample_rate", receiver_cfg.get("sample_rate", 5_000_000)))
    center_freq = int(sdr_cfg.get("center_freq", receiver_cfg.get("center_freq", 2_437_000_000)))
    channels = sdr_cfg.get("channels", receiver_cfg.get("channels", [0]))
    gain = sdr_cfg.get("gain", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/processed/cnn_capture") / label / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Hotspot Spectrogram Capture 128 ===")
    print("receiver source_type:", receiver_cfg.get("source_type"))
    print("center_freq:", center_freq)
    print("sample_rate:", sample_rate)
    print("block_size:", block_size)
    print("channels:", channels)
    print("gain:", gain)
    print("nperseg:", nperseg)
    print("noverlap:", noverlap)
    print("hop:", hop)
    print("out_dir:", out_dir)
    print()

    rx = build_receiver(receiver_cfg)

    saved = 0

    try:
        for block_idx in range(num_blocks):
            iq = rx.read_block(block_size)
            iq = np.asarray(iq)

            if iq.ndim == 1:
                x = iq
            elif iq.ndim == 2:
                # 현재 RX 단일 채널이면 shape=(1, 16384)이므로 iq[0] 사용
                x = iq[0]
            else:
                raise ValueError(f"unexpected iq shape: {iq.shape}")

            spec_db = make_spectrogram_128(
                x=x,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                normalize=False,
                remove_dc_bin=True,
            )

            npy_path = out_dir / f"{label}_block_{block_idx:04d}.npy"
            png_path = out_dir / f"{label}_block_{block_idx:04d}.png"

            np.save(npy_path, spec_db)

            title = (
                f"{label} | {center_freq / 1e9:.4f} GHz | "
                f"nperseg={nperseg}, noverlap={noverlap} | "
                f"block={block_idx:04d}"
            )

            save_spectrogram_png(
                spec_db=spec_db,
                out_path=png_path,
                title=title,
                vmin=-90,
                vmax=-20,
            )

            print(
                f"[{block_idx:04d}] "
                f"iq_shape={iq.shape}, "
                f"spec_shape={spec_db.shape}, "
                f"min={spec_db.min():.2f} dB, "
                f"max={spec_db.max():.2f} dB, "
                f"saved={png_path}"
            )

            saved += 1

    finally:
        if hasattr(rx, "close"):
            rx.close()

    metadata = {
        "label": label,
        "num_blocks": num_blocks,
        "saved": saved,
        "source_type": receiver_cfg.get("source_type"),
        "center_freq": center_freq,
        "sample_rate": sample_rate,
        "block_size": block_size,
        "channels": channels,
        "gain": gain,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "hop": hop,
        "expected_shape": [128, 509],
        "color_scale": {
            "vmin": -90,
            "vmax": -20,
        },
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=== Capture Finished ===")
    print("saved:", saved)
    print("output_dir:", out_dir)


if __name__ == "__main__":
    main()
