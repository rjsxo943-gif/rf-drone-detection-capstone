"""
Feature extraction package for RF IQ blocks.

현재 프로젝트 기준:
- FFT: 탐색 / 디버깅 / Energy Detector 보조용
- STFT Spectrogram: CNN 입력 생성용
- Window: FFT/STFT 공통 window 생성용
"""

from src.features.window import (
    get_window,
    apply_window,
)

from src.features.fft import (
    compute_fft_magnitude,
    compute_fft_power,
    compute_block_fft_magnitude,
    compute_block_fft_power,
)

from src.features.spectrogram import (
    StftBranchOutput,
    DualChannelStftOutput,
    normalize_spectrogram,
    compute_stft_branch,
    compute_dual_channel_stft_branch,
)

__all__ = [
    # Window
    "get_window",
    "apply_window",

    # FFT
    "compute_fft_magnitude",
    "compute_fft_power",
    "compute_block_fft_magnitude",
    "compute_block_fft_power",

    # STFT / Spectrogram
    "StftBranchOutput",
    "DualChannelStftOutput",
    "normalize_spectrogram",
    "compute_stft_branch",
    "compute_dual_channel_stft_branch",
]