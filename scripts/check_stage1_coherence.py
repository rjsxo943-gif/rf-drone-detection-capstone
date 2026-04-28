from __future__ import annotations

import numpy as np

from src.features.spectrogram import compute_dual_channel_stft_branch
from src.aoa.coherence import coherence_gate


def make_coherent_dual_iq(
    sample_rate: float,
    block_size: int,
    tone_freq: float = 500_000,
    phase_offset_rad: float = 0.7,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 채널이 같은 신호를 보고 있고,
    RX1에만 일정한 phase offset이 있는 테스트 신호.
    coherence가 높게 나와야 정상이다.
    """

    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2 * np.pi * tone_freq * t)
    rx1 = np.exp(1j * (2 * np.pi * tone_freq * t + phase_offset_rad))

    rx0 += noise_std * (np.random.randn(block_size) + 1j * np.random.randn(block_size))
    rx1 += noise_std * (np.random.randn(block_size) + 1j * np.random.randn(block_size))

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def make_incoherent_dual_iq(
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 채널이 서로 관련 없는 랜덤 노이즈인 경우.
    coherence가 낮게 나와야 정상이다.
    """

    rx0 = np.random.randn(block_size) + 1j * np.random.randn(block_size)
    rx1 = np.random.randn(block_size) + 1j * np.random.randn(block_size)

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def run_case(name: str, rx0_iq: np.ndarray, rx1_iq: np.ndarray) -> None:
    sample_rate = 5_000_000

    branch = compute_dual_channel_stft_branch(
        rx0_iq=rx0_iq,
        rx1_iq=rx1_iq,
        sample_rate=sample_rate,
        nperseg=512,
        noverlap=384,
        nfft=512,
        window="hann",
        cnn_source="rx0",
    )

    result = coherence_gate(
        z0=branch.rx0.complex_stft,
        z1=branch.rx1.complex_stft,
        threshold=0.6,
        energy_percentile=75.0,
    )

    print(f"=== {name} ===")
    print(f"complex STFT shape: {branch.rx0.complex_stft.shape}")
    print(f"coherence: {result.coherence:.4f}")
    print(f"threshold: {result.threshold:.4f}")
    print(f"used_bins: {result.used_bins}")
    print(f"gate passed: {result.passed}")
    print()


def main() -> None:
    sample_rate = 5_000_000
    block_size = 16_384

    coherent_rx0, coherent_rx1 = make_coherent_dual_iq(
        sample_rate=sample_rate,
        block_size=block_size,
    )

    incoherent_rx0, incoherent_rx1 = make_incoherent_dual_iq(
        block_size=block_size,
    )

    run_case("Coherent signal case", coherent_rx0, coherent_rx1)
    run_case("Incoherent noise case", incoherent_rx0, incoherent_rx1)


if __name__ == "__main__":
    main()