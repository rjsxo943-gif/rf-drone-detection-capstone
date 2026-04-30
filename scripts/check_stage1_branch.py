# check_stage1_branch.py
from __future__ import annotations

import numpy as np

from src.core.stage1_artifact_store import save_stage1_artifacts
from src.features.spectrogram import compute_dual_channel_stft_branch
from src.ui.result_plotter import save_spectrogram_image


def make_dual_channel_test_iq(
    sample_rate: float,
    block_size: int,
    tone_freq: float = 500_000,
    phase_offset_rad: float = 0.7,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RX0, RX1 테스트용 complex IQ 생성.
    RX1에는 고정 phase offset을 준다.
    """

    rng = np.random.default_rng(42)

    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2.0 * np.pi * tone_freq * t)
    rx1 = np.exp(1j * (2.0 * np.pi * tone_freq * t + phase_offset_rad))

    rx0_noise = noise_std * (
        rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)
    )
    rx1_noise = noise_std * (
        rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)
    )

    rx0 = rx0 + rx0_noise
    rx1 = rx1 + rx1_noise

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def main() -> None:
    sample_rate = 5_000_000
    center_freq = 2_437_000_000
    block_size = 16_384
    block_index = 0

    nperseg = 512   
    noverlap = 384
    hop_length = nperseg - noverlap

    rx0_iq, rx1_iq = make_dual_channel_test_iq(
        sample_rate=sample_rate,
        block_size=block_size,
    )

    result = compute_dual_channel_stft_branch(
        rx0_iq=rx0_iq,
        rx1_iq=rx1_iq,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=512,
        window="hann",
        cnn_source="rx0",
    )

    artifact_path = save_stage1_artifacts(
        output_path="outputs/runs/latest/stage1/block_000000.npz",
        block_index=block_index,
        sample_rate=sample_rate,
        center_freq=center_freq,
        cnn_spectrogram=result.cnn_spectrogram,
        block_size=block_size,
        nperseg=nperseg,
        noverlap=noverlap,
        hop_length=hop_length,
        rx0_complex_stft=result.rx0.complex_stft,
        rx1_complex_stft=result.rx1.complex_stft,
        rx0_phase=result.rx0.phase,
        rx1_phase=result.rx1.phase,
        rx0_log_magnitude=result.rx0.log_magnitude,
        rx1_log_magnitude=result.rx1.log_magnitude,
        save_complex_stft=True,
        save_phase=True,
        save_log_magnitude=True,
    )

    image_path = save_spectrogram_image(
        spectrogram=result.cnn_spectrogram,
        save_path="outputs/runs/latest/stage1/cnn_spectrogram.png",
        title="Stage 1 CNN Spectrogram",
    )

    print("=== Stage 1 Branch Check ===")
    print(f"RX0 IQ shape          : {rx0_iq.shape}")
    print(f"RX1 IQ shape          : {rx1_iq.shape}")
    print(f"sample_rate           : {sample_rate}")
    print(f"center_freq           : {center_freq}")
    print()

    print("[RX0]")
    print(f"complex STFT shape    : {result.rx0.complex_stft.shape}")
    print(f"phase shape           : {result.rx0.phase.shape}")
    print(f"log magnitude shape   : {result.rx0.log_magnitude.shape}")
    print(f"cnn spectrogram shape : {result.rx0.cnn_spectrogram.shape}")
    print()

    print("[RX1]")
    print(f"complex STFT shape    : {result.rx1.complex_stft.shape}")
    print(f"phase shape           : {result.rx1.phase.shape}")
    print(f"log magnitude shape   : {result.rx1.log_magnitude.shape}")
    print(f"cnn spectrogram shape : {result.rx1.cnn_spectrogram.shape}")
    print()

    print("[CNN Input]")
    print(f"cnn_spectrogram shape : {result.cnn_spectrogram.shape}")
    print(f"cnn_spectrogram dtype : {result.cnn_spectrogram.dtype}")
    print(f"cnn_spectrogram min   : {result.cnn_spectrogram.min():.6f}")
    print(f"cnn_spectrogram max   : {result.cnn_spectrogram.max():.6f}")
    print()

    print("[Saved]")
    print(f"artifact_path         : {artifact_path}")
    print(f"spectrogram_image     : {image_path}")
    print()

    expected_freq_bins = 512

    shape_ok = (
        result.rx0.complex_stft.shape[0] == expected_freq_bins
        and result.rx1.complex_stft.shape[0] == expected_freq_bins
        and result.cnn_spectrogram.shape[0] == expected_freq_bins
    )

    dtype_ok = result.cnn_spectrogram.dtype == np.float32

    range_ok = (
        np.isfinite(result.cnn_spectrogram).all()
        and result.cnn_spectrogram.min() >= 0.0
        and result.cnn_spectrogram.max() <= 1.0
    )

    print("=== Check Result ===")
    print(f"shape_ok : {shape_ok}")
    print(f"dtype_ok : {dtype_ok}")
    print(f"range_ok : {range_ok}")

    if shape_ok and dtype_ok and range_ok:
        print("OK: Stage 1 branch check success")
    else:
        raise RuntimeError("Stage 1 branch check failed.")


if __name__ == "__main__":
    main()