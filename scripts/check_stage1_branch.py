from __future__ import annotations

import numpy as np

from src.features.spectrogram import compute_dual_channel_stft_branch


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

    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2 * np.pi * tone_freq * t)

    rx1 = np.exp(1j * (2 * np.pi * tone_freq * t + phase_offset_rad))

    rx0_noise = noise_std * (
        np.random.randn(block_size) + 1j * np.random.randn(block_size)
    )

    rx1_noise = noise_std * (
        np.random.randn(block_size) + 1j * np.random.randn(block_size)
    )

    rx0 = rx0 + rx0_noise
    rx1 = rx1 + rx1_noise

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def main() -> None:
    sample_rate = 5_000_000
    block_size = 16_384

    rx0_iq, rx1_iq = make_dual_channel_test_iq(
        sample_rate=sample_rate,
        block_size=block_size,
    )

    result = compute_dual_channel_stft_branch(
        rx0_iq=rx0_iq,
        rx1_iq=rx1_iq,
        sample_rate=sample_rate,
        nperseg=512,
        noverlap=384,
        nfft=512,
        window="hann",
        cnn_source="rx0",
    )

    print("=== Stage 1 Branch Check ===")
    print(f"RX0 IQ shape: {rx0_iq.shape}")
    print(f"RX1 IQ shape: {rx1_iq.shape}")
    print()

    print("[RX0]")
    print(f"complex STFT shape: {result.rx0.complex_stft.shape}")
    print(f"phase shape: {result.rx0.phase.shape}")
    print(f"log magnitude shape: {result.rx0.log_magnitude.shape}")
    print(f"cnn spectrogram shape: {result.rx0.cnn_spectrogram.shape}")
    print()

    print("[RX1]")
    print(f"complex STFT shape: {result.rx1.complex_stft.shape}")
    print(f"phase shape: {result.rx1.phase.shape}")
    print(f"log magnitude shape: {result.rx1.log_magnitude.shape}")
    print(f"cnn spectrogram shape: {result.rx1.cnn_spectrogram.shape}")
    print()

    print("[CNN Input]")
    print(f"cnn_spectrogram shape: {result.cnn_spectrogram.shape}")
    print(f"cnn_spectrogram dtype: {result.cnn_spectrogram.dtype}")
    print(f"cnn_spectrogram min: {result.cnn_spectrogram.min():.6f}")
    print(f"cnn_spectrogram max: {result.cnn_spectrogram.max():.6f}")

    expected_shape = (512, 125)

    print()
    if result.rx0.complex_stft.shape == expected_shape:
        print("OK: complex STFT shape is 512 × 125")
    else:
        print(f"WARNING: expected {expected_shape}, got {result.rx0.complex_stft.shape}")

    if result.cnn_spectrogram.shape == expected_shape:
        print("OK: CNN spectrogram shape is 512 × 125")
    else:
        print(f"WARNING: expected {expected_shape}, got {result.cnn_spectrogram.shape}")


if __name__ == "__main__":
    main()