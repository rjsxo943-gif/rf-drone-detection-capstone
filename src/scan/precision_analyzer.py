from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocess import remove_dc_offset, normalize_iq
from src.features.spectrogram import compute_dual_channel_stft_branch
from src.aoa.coherence import coherence_gate
from src.aoa.phase_diff import estimate_phase_diff
from src.aoa.angle_estimator import phase_diff_to_angle


@dataclass
class PrecisionAnalysisResult:
    center_freq: float
    stft_done: bool
    cnn_enabled: bool
    cnn_label: str | None
    cnn_score: float | None
    coherence: float | None
    coherence_passed: bool | None
    phase_diff_rad: float | None
    phase_diff_deg: float | None
    angle_deg: float | None
    angle_valid: bool | None
    cnn_spectrogram_shape: list[int] | None


class PrecisionAnalyzer:
    """
    Trigger된 주파수 대역에서 정밀 분석을 수행하는 클래스.

    현재 역할:
    - IQ 재수집
    - DC offset 제거
    - normalize
    - STFT 생성
    - coherence 검사
    - phase_diff 계산
    - AoA 계산

    CNN은 아직 연결하지 않고 자리만 만들어둔다.
    """

    def __init__(
        self,
        receiver,
        num_samples: int,
        sample_rate: float,
        antenna_spacing_m: float,
        nperseg: int = 512,
        noverlap: int = 384,
        nfft: int = 512,
        coherence_threshold: float = 0.6,
    ) -> None:
        self.receiver = receiver
        self.num_samples = int(num_samples)
        self.sample_rate = float(sample_rate)
        self.antenna_spacing_m = float(antenna_spacing_m)

        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.nfft = int(nfft)
        self.coherence_threshold = float(coherence_threshold)

    def analyze(self, center_freq: float) -> PrecisionAnalysisResult:
        iq = self.receiver.read_samples(self.num_samples)
        iq = remove_dc_offset(iq)
        iq = normalize_iq(iq)

        if iq.ndim != 2 or iq.shape[0] < 2:
            return PrecisionAnalysisResult(
                center_freq=float(center_freq),
                stft_done=False,
                cnn_enabled=False,
                cnn_label=None,
                cnn_score=None,
                coherence=None,
                coherence_passed=None,
                phase_diff_rad=None,
                phase_diff_deg=None,
                angle_deg=None,
                angle_valid=None,
                cnn_spectrogram_shape=None,
            )

        branch = compute_dual_channel_stft_branch(
            rx0_iq=iq[0],
            rx1_iq=iq[1],
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            window="hann",
            cnn_source="rx0",
        )

        coherence_result = coherence_gate(
            z0=branch.rx0.complex_stft,
            z1=branch.rx1.complex_stft,
            threshold=self.coherence_threshold,
            energy_percentile=75.0,
        )

        phase_result = estimate_phase_diff(
            iq_block=iq,
            ref_channel=0,
            target_channel=1,
        )

        angle_result = phase_diff_to_angle(
            phase_diff_rad=phase_result.phase_diff_rad,
            carrier_freq=float(center_freq),
            antenna_spacing_m=self.antenna_spacing_m,
            phase_offset_rad=0.0,
            clip_input=True,
        )

        return PrecisionAnalysisResult(
            center_freq=float(center_freq),
            stft_done=True,
            cnn_enabled=False,
            cnn_label=None,
            cnn_score=None,
            coherence=float(coherence_result.coherence),
            coherence_passed=bool(coherence_result.passed),
            phase_diff_rad=float(phase_result.phase_diff_rad),
            phase_diff_deg=float(phase_result.phase_diff_deg),
            angle_deg=float(angle_result.angle_deg),
            angle_valid=bool(angle_result.valid),
            cnn_spectrogram_shape=list(branch.cnn_spectrogram.shape),
        )