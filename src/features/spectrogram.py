from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft



@dataclass
class StftBranchOutput:
    """
    한 채널 IQ block에서 나온 STFT 분기 결과.

    complex_stft:
        AoA / coherence 계산용 complex STFT

    phase:
        phase 확인용. 단, AoA 계산은 phase만 쓰기보다
        complex_stft를 이용하는 것이 더 안전하다.

    log_magnitude:
        CNN 입력 전 단계의 log magnitude spectrogram

    cnn_spectrogram:
        CNN 입력용 0~1 normalized spectrogram
    """

    freqs: np.ndarray
    times: np.ndarray
    complex_stft: np.ndarray
    phase: np.ndarray
    log_magnitude: np.ndarray
    cnn_spectrogram: np.ndarray


@dataclass
class DualChannelStftOutput:
    """
    RX0, RX1 두 채널 STFT 분기 결과.

    rx0, rx1:
        각 채널의 complex STFT, phase, magnitude 정보를 모두 포함한다.

    cnn_spectrogram:
        CNN에 넣을 대표 spectrogram.
        기본은 RX0 기준으로 사용한다.
    """

    rx0: StftBranchOutput
    rx1: StftBranchOutput
    cnn_spectrogram: np.ndarray


def _validate_iq_block(iq_block: np.ndarray, name: str = "iq_block") -> np.ndarray:
    iq_block = np.asarray(iq_block)

    if iq_block.ndim != 1:
        raise ValueError(f"{name} must be 1-D complex IQ array. got shape={iq_block.shape}")

    if not np.iscomplexobj(iq_block):
        iq_block = iq_block.astype(np.complex64)

    return iq_block.astype(np.complex64)


def normalize_spectrogram(
    spec: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    CNN 입력용 spectrogram을 0~1 범위로 정규화한다.
    Stage 1에서는 단순 min-max normalization을 사용한다.
    """

    spec = np.asarray(spec, dtype=np.float32)

    min_val = np.min(spec)
    max_val = np.max(spec)

    if max_val - min_val < eps:
        return np.zeros_like(spec, dtype=np.float32)

    normalized = (spec - min_val) / (max_val - min_val + eps)
    return normalized.astype(np.float32)


def compute_stft_branch(
    iq_block: np.ndarray,
    sample_rate: float,
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 512,
    window: str = "hann",
    apply_fftshift: bool = True,
) -> StftBranchOutput:
    """
    단일 채널 complex IQ block을 받아서 다음 두 branch를 동시에 만든다.

    1. AoA / coherence용 complex STFT branch
    2. CNN용 log-magnitude spectrogram branch

    입력:
        iq_block.shape = (16384,)

    출력:
        complex_stft.shape = (512, 125)
        phase.shape = (512, 125)
        cnn_spectrogram.shape = (512, 125)
    """

    iq_block = _validate_iq_block(iq_block)

    if len(iq_block) < nperseg:
        raise ValueError(
            f"iq_block length must be >= nperseg. "
            f"got len={len(iq_block)}, nperseg={nperseg}"
        )

    freqs, times, complex_stft = stft(
        iq_block,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    complex_stft = complex_stft.astype(np.complex64)

    # 주파수축을 보기 좋게 -fs/2 ~ +fs/2 순서로 정렬
    if apply_fftshift:
        freqs = np.fft.fftshift(freqs)
        complex_stft = np.fft.fftshift(complex_stft, axes=0)

    # AoA 확인용 phase branch
    phase = np.angle(complex_stft).astype(np.float32)

    # CNN용 magnitude branch
    magnitude = np.abs(complex_stft)
    log_magnitude = np.log1p(magnitude).astype(np.float32)
    cnn_spectrogram = normalize_spectrogram(log_magnitude)

    return StftBranchOutput(
        freqs=freqs.astype(np.float32),
        times=times.astype(np.float32),
        complex_stft=complex_stft,
        phase=phase,
        log_magnitude=log_magnitude,
        cnn_spectrogram=cnn_spectrogram,
    )


def compute_dual_channel_stft_branch(
    rx0_iq: np.ndarray,
    rx1_iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 512,
    window: str = "hann",
    cnn_source: str = "rx0",
) -> DualChannelStftOutput:
    """
    RX0, RX1 두 채널 IQ block을 받아서 CNN branch와 AoA branch를 동시에 만든다.

    cnn_source:
        "rx0"      -> RX0 spectrogram을 CNN 입력으로 사용
        "rx1"      -> RX1 spectrogram을 CNN 입력으로 사용
        "mean_mag" -> RX0/RX1 log magnitude 평균을 CNN 입력으로 사용

    Stage 1에서는 기본값 "rx0"를 추천한다.
    """

    rx0_iq = _validate_iq_block(rx0_iq, name="rx0_iq")
    rx1_iq = _validate_iq_block(rx1_iq, name="rx1_iq")

    if rx0_iq.shape != rx1_iq.shape:
        raise ValueError(
            f"rx0_iq and rx1_iq must have same shape. "
            f"got rx0={rx0_iq.shape}, rx1={rx1_iq.shape}"
        )

    rx0 = compute_stft_branch(
        iq_block=rx0_iq,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
    )

    rx1 = compute_stft_branch(
        iq_block=rx1_iq,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
    )

    if rx0.complex_stft.shape != rx1.complex_stft.shape:
        raise ValueError(
            f"RX0 and RX1 STFT shapes must match. "
            f"got rx0={rx0.complex_stft.shape}, rx1={rx1.complex_stft.shape}"
        )

    if cnn_source == "rx0":
        cnn_spectrogram = rx0.cnn_spectrogram

    elif cnn_source == "rx1":
        cnn_spectrogram = rx1.cnn_spectrogram

    elif cnn_source == "mean_mag":
        mean_log_mag = 0.5 * (rx0.log_magnitude + rx1.log_magnitude)
        cnn_spectrogram = normalize_spectrogram(mean_log_mag)

    else:
        raise ValueError(
            f"Unsupported cnn_source={cnn_source}. "
            f"Use 'rx0', 'rx1', or 'mean_mag'."
        )

    return DualChannelStftOutput(
        rx0=rx0,
        rx1=rx1,
        cnn_spectrogram=cnn_spectrogram.astype(np.float32),
    )