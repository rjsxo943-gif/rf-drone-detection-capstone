from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    spectrogram_path: str | None
    rx0_stft_path: str | None
    rx1_stft_path: str | None


class PrecisionAnalyzer:
    """
    Trigger된 주파수 대역에서 정밀 분석을 수행하는 클래스.

    현재 역할:
    - IQ 재수집
    - DC offset 제거
    - normalize
    - STFT 생성
    - CNN 입력용 spectrogram 생성
    - 선택적으로 spectrogram/STFT 저장
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
        save_dir: str | None = None,
        save_spectrogram: bool = False,
        save_stft: bool = False,
    ) -> None:
        self.receiver = receiver
        self.num_samples = int(num_samples)
        self.sample_rate = float(sample_rate)
        self.antenna_spacing_m = float(antenna_spacing_m)

        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.nfft = int(nfft)
        self.coherence_threshold = float(coherence_threshold)

        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_spectrogram = bool(save_spectrogram)
        self.save_stft = bool(save_stft)

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

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
                spectrogram_path=None,
                rx0_stft_path=None,
                rx1_stft_path=None,
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

        spectrogram_path = None
        rx0_stft_path = None
        rx1_stft_path = None

        if self.save_dir is not None:
            freq_tag = str(int(center_freq))

            if self.save_spectrogram:
                spectrogram_path = self.save_dir / f"{freq_tag}_cnn_spectrogram.npy"
                np.save(spectrogram_path, branch.cnn_spectrogram)

            if self.save_stft:
                rx0_stft_path = self.save_dir / f"{freq_tag}_rx0_complex_stft.npy"
                rx1_stft_path = self.save_dir / f"{freq_tag}_rx1_complex_stft.npy"

                np.save(rx0_stft_path, branch.rx0.complex_stft)
                np.save(rx1_stft_path, branch.rx1.complex_stft)

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
            spectrogram_path=str(spectrogram_path) if spectrogram_path is not None else None,
            rx0_stft_path=str(rx0_stft_path) if rx0_stft_path is not None else None,
            rx1_stft_path=str(rx1_stft_path) if rx1_stft_path is not None else None,
        )