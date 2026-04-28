# check_stage1_coherence.py
from __future__ import annotations

import numpy as np

from src.aoa.angle_estimator import phase_diff_to_angle
from src.aoa.coherence import coherence_gate
from src.aoa.phase_diff import estimate_phase_diff
from src.features.spectrogram import compute_dual_channel_stft_branch


def make_coherent_dual_iq(
    sample_rate: float,
    block_size: int,
    tone_freq: float = 500_000,
    phase_offset_rad: float = 0.7,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    n = np.arange(block_size)
    t = n / sample_rate

    rx0 = np.exp(1j * 2.0 * np.pi * tone_freq * t)
    rx1 = np.exp(1j * (2.0 * np.pi * tone_freq * t + phase_offset_rad))

    rx0 += noise_std * (
        rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)
    )
    rx1 += noise_std * (
        rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)
    )

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def make_incoherent_dual_iq(
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)

    rx0 = rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)
    rx1 = rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)

    return rx0.astype(np.complex64), rx1.astype(np.complex64)


def run_case(
    name: str,
    rx0_iq: np.ndarray,
    rx1_iq: np.ndarray,
    expected_pass: bool,
) -> None:
    sample_rate = 5_000_000
    carrier_freq = 2_400_000_000
    antenna_spacing_m = 0.0625

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

    coherence_result = coherence_gate(
        z0=branch.rx0.complex_stft,
        z1=branch.rx1.complex_stft,
        threshold=0.6,
        energy_percentile=75.0,
    )

    iq_block = np.stack([rx0_iq, rx1_iq], axis=0)

    phase_result = estimate_phase_diff(
        iq_block=iq_block,
        ref_channel=0,
        target_channel=1,
    )

    angle_result = phase_diff_to_angle(
        phase_diff_rad=phase_result.phase_diff_rad,
        carrier_freq=carrier_freq,
        antenna_spacing_m=antenna_spacing_m,
        phase_offset_rad=0.0,
        clip_input=True,
    )

    pass_ok = coherence_result.passed == expected_pass

    print(f"=== {name} ===")
    print(f"complex STFT shape : {branch.rx0.complex_stft.shape}")
    print(f"coherence          : {coherence_result.coherence:.4f}")
    print(f"threshold          : {coherence_result.threshold:.4f}")
    print(f"used_bins          : {coherence_result.used_bins}")
    print(f"gate passed        : {coherence_result.passed}")
    print(f"expected_pass      : {expected_pass}")
    print(f"pass_ok            : {pass_ok}")
    print()
    print("[Phase]")
    print(f"phase_diff_rad     : {phase_result.phase_diff_rad:.6f}")
    print(f"phase_diff_deg     : {phase_result.phase_diff_deg:.2f}")
    print(f"coherence_like     : {phase_result.coherence_like:.4f}")
    print()
    print("[Angle]")
    print(f"angle_deg          : {angle_result.angle_deg:.2f}")
    print(f"arcsin_input       : {angle_result.arcsin_input:.6f}")
    print(f"valid              : {angle_result.valid}")
    print()

    if not pass_ok:
        raise RuntimeError(f"{name} coherence gate result is not expected.")


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

    run_case(
        name="Coherent signal case",
        rx0_iq=coherent_rx0,
        rx1_iq=coherent_rx1,
        expected_pass=True,
    )

    run_case(
        name="Incoherent noise case",
        rx0_iq=incoherent_rx0,
        rx1_iq=incoherent_rx1,
        expected_pass=False,
    )

    print("OK: Stage 1 coherence / phase / angle check success")


if __name__ == "__main__":
    main()