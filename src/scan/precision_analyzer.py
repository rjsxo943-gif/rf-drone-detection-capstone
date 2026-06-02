from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess import remove_dc_offset, normalize_iq
from src.features.cnn_input import compute_runtime_cnn_spectrogram
from src.features.spectrogram import compute_dual_channel_stft_branch
from src.aoa.coherence import coherence_gate
from src.aoa.phase_diff import estimate_phase_diff
from src.aoa.angle_estimator import phase_diff_to_angle
from src.aoa.sector_quantizer import quantize_front_angle_to_sector
from src.ml.runtime_decision import (
    RuntimeDecisionConfig,
    get_positive_probability,
    select_drone_threshold,
    update_temporal_decision,
)


@dataclass
class PrecisionAnalysisResult:
    center_freq: float
    stft_done: bool

    cnn_enabled: bool
    cnn_label: str | None
    cnn_score: float | None
    cnn_class_index: int | None
    cnn_probabilities: list[float] | None

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

    sector_index: int | None
    sector_label: str | None
    sector_valid: bool | None

    precision_blocks: int | None = None
    selected_block_index: int | None = None
    selection_score: float | None = None

    drone_probability: float | None = None
    drone_threshold: float | None = None
    temporal_window: int | None = None
    drone_vote_count: int | None = None
    temporal_history: list[int] | None = None
    candidate_status: bool | None = None
    confirmed_status: bool | None = None
    final_decision: str | None = None
    aoa_skipped_reason: str | None = None

    aoa_smoothed_angle_deg: float | None = None
    aoa_smoothing_valid: bool | None = None
    aoa_smoothing_history_size: int | None = None
    aoa_smoothing_method: str | None = None
    aoa_smoothing_rejected_reason: str | None = None


class PrecisionAnalyzer:
    """
    Trigger된 주파수 대역에서 정밀 분석을 수행한다.

    핵심:
    - scan trigger 후 해당 center_freq로 다시 tuning
    - precision block 여러 개 수집
    - CNN binary Drone/NotDrone 결과를 YAML threshold로 평가
    - temporal voting으로 candidate/confirmed 판단
    - confirmed일 때만 AoA/sector 계산
    - AoA geometry/coherence/smoothing/sector는 configs/aoa.yaml 기준
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
        window: str = "hann",
        coherence_threshold: float = 0.6,
        phase_offset_rad: float = 0.0,
        settle_sec: float = 0.0,
        precision_blocks: int = 1,
        save_dir: str | None = None,
        save_spectrogram: bool = False,
        save_stft: bool = False,
        cnn_classifier=None,
        decision_cfg: RuntimeDecisionConfig | None = None,
        current_gain: int | float | None = None,
        aoa_cfg: dict[str, Any] | None = None,
        cnn_rx_index: int = 0,
    ) -> None:
        self.receiver = receiver
        self.num_samples = int(num_samples)
        self.sample_rate = float(sample_rate)
        self.antenna_spacing_m = float(antenna_spacing_m)

        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.nfft = int(nfft)
        self.window = str(window)
        self.cnn_rx_index = int(cnn_rx_index)

        self.aoa_cfg = aoa_cfg or {}
        self.coherence_cfg = self.aoa_cfg.get("coherence", {}) or {}
        self.smoothing_cfg = self.aoa_cfg.get("smoothing", {}) or {}
        self.sector_cfg = self.aoa_cfg.get("sector", {}) or {}
        self.runtime_cfg = self.aoa_cfg.get("runtime", {}) or {}

        self.ref_channel = int(self.aoa_cfg.get("ref_channel", 0))
        self.target_channel = int(self.aoa_cfg.get("target_channel", 1))
        self.clip_angle_input = bool(self.aoa_cfg.get("clip_arcsin_input", True))
        self.energy_percentile = float(self.coherence_cfg.get("energy_percentile", 75.0))
        self.require_coherence_for_valid_angle = bool(
            self.coherence_cfg.get("require_passed_for_valid_angle", True)
        )

        self.coherence_threshold = float(
            self.coherence_cfg.get("threshold", coherence_threshold)
        )
        self.phase_offset_rad = float(phase_offset_rad)
        self.settle_sec = float(settle_sec)
        self.precision_blocks = max(1, int(precision_blocks))

        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_spectrogram = bool(save_spectrogram)
        self.save_stft = bool(save_stft)

        self.cnn_classifier = cnn_classifier
        self.decision_cfg = decision_cfg
        self.current_gain = current_gain

        self.smoothing_enabled = bool(self.smoothing_cfg.get("enabled", False))
        self.smoothing_method = str(self.smoothing_cfg.get("method", "median")).lower().strip()
        self.smoothing_window = max(1, int(self.smoothing_cfg.get("window_size", 5)))
        self.smoothing_min_valid_samples = max(1, int(self.smoothing_cfg.get("min_valid_samples", 1)))
        self.smoothing_require_angle_valid = bool(self.smoothing_cfg.get("require_angle_valid", True))
        self.smoothing_require_coherence_passed = bool(
            self.smoothing_cfg.get("require_coherence_passed", True)
        )
        self.smoothing_max_angle_jump_deg = self.smoothing_cfg.get("max_angle_jump_deg", None)
        self.angle_history: deque[float] = deque(maxlen=self.smoothing_window)
        self.reset_smoothing_on_not_confirmed = bool(
            self.runtime_cfg.get("reset_smoothing_on_not_confirmed", True)
        )

        # IMPORTANT:
        # Temporal voting must persist across analyze() calls during precision hold.
        # Otherwise blocks_per_step=1 always produces votes=1/5 and never confirmed=True.
        if self.decision_cfg is not None:
            vote_window = int(self.decision_cfg.temporal_voting.window_size)
        else:
            vote_window = int(self.precision_blocks)
        self.vote_history: deque[int] = deque(maxlen=max(1, vote_window))

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _set_center_freq(self, center_freq: float) -> None:
        if hasattr(self.receiver, "set_center_freq"):
            self.receiver.set_center_freq(center_freq)
            return

        if hasattr(self.receiver, "center_freq"):
            self.receiver.center_freq = center_freq
            return

        if hasattr(self.receiver, "sdr") and hasattr(self.receiver.sdr, "rx_lo"):
            self.receiver.sdr.rx_lo = int(center_freq)
            return

        raise AttributeError("receiver에서 center frequency를 설정할 방법을 찾지 못했습니다.")

    def _fft_selection_score(self, cnn_iq: np.ndarray) -> float:
        x = np.asarray(cnn_iq).reshape(-1)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft(x)))
        return float(np.max(fft_mag ** 2))

    def _get_gain(self) -> float | None:
        get_gain = getattr(self.receiver, "get_gain", None)
        if callable(get_gain):
            try:
                return float(get_gain())
            except Exception:
                pass

        if hasattr(self.receiver, "gain"):
            try:
                return float(self.receiver.gain)
            except Exception:
                pass

        if self.current_gain is not None:
            return float(self.current_gain)

        return None

    def _drone_probability(self, cnn_result) -> float:
        if cnn_result is None:
            return 0.0

        if self.decision_cfg is None:
            class_names = getattr(self.cnn_classifier, "class_names", [])
            return get_positive_probability(cnn_result, list(class_names), "Drone")

        class_names = getattr(self.cnn_classifier, "class_names", [])
        return get_positive_probability(
            cnn_result,
            list(class_names),
            self.decision_cfg.positive_class,
        )

    def _update_smoothed_angle(
        self,
        angle_deg: float | None,
        angle_valid: bool | None,
        coherence_passed: bool | None,
    ) -> tuple[float | None, bool, int, str | None]:
        if not self.smoothing_enabled:
            return None, False, len(self.angle_history), "disabled"

        if angle_deg is None or not np.isfinite(float(angle_deg)):
            return None, False, len(self.angle_history), "angle_nan"

        if self.smoothing_require_angle_valid and not bool(angle_valid):
            return None, False, len(self.angle_history), "angle_invalid"

        if self.smoothing_require_coherence_passed and not bool(coherence_passed):
            return None, False, len(self.angle_history), "coherence_failed"

        value = float(angle_deg)

        if self.angle_history and self.smoothing_max_angle_jump_deg is not None:
            last_value = float(self.angle_history[-1])
            max_jump = float(self.smoothing_max_angle_jump_deg)
            if abs(value - last_value) > max_jump:
                return None, False, len(self.angle_history), "angle_jump_rejected"

        self.angle_history.append(value)

        if len(self.angle_history) < self.smoothing_min_valid_samples:
            return None, False, len(self.angle_history), "not_enough_samples"

        values = np.asarray(list(self.angle_history), dtype=float)
        if self.smoothing_method == "mean":
            smoothed = float(np.mean(values))
        else:
            smoothed = float(np.median(values))

        return smoothed, True, len(self.angle_history), None

    def reset_temporal_history(self) -> None:
        """Reset CNN temporal voting history.

        Called before a new CNN screening sequence starts.
        During precision hold, this history is intentionally preserved block-to-block.
        """
        self.vote_history.clear()

    def _empty_result(self, center_freq: float) -> PrecisionAnalysisResult:
        temporal_window = None
        if self.decision_cfg is not None:
            temporal_window = int(self.decision_cfg.temporal_voting.window_size)

        return PrecisionAnalysisResult(
            center_freq=float(center_freq),
            stft_done=False,
            sector_index=None,
            sector_label=None,
            sector_valid=None,
            cnn_enabled=self.cnn_classifier is not None,
            cnn_label=None,
            cnn_score=None,
            cnn_class_index=None,
            cnn_probabilities=None,
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
            precision_blocks=self.precision_blocks,
            selected_block_index=None,
            selection_score=None,
            drone_probability=None,
            drone_threshold=None,
            temporal_window=temporal_window,
            drone_vote_count=0,
            temporal_history=[],
            candidate_status=False,
            confirmed_status=False,
            final_decision="No valid precision block",
            aoa_skipped_reason="no_valid_precision_block",
            aoa_smoothed_angle_deg=None,
            aoa_smoothing_valid=False,
            aoa_smoothing_history_size=len(self.angle_history),
            aoa_smoothing_method=self.smoothing_method,
            aoa_smoothing_rejected_reason="no_valid_precision_block",
        )

    def analyze(self, center_freq: float) -> PrecisionAnalysisResult:
        self._set_center_freq(float(center_freq))

        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

        current_gain = self._get_gain()
        drone_threshold = None
        if self.decision_cfg is not None:
            drone_threshold = select_drone_threshold(self.decision_cfg, current_gain)

        best: dict[str, Any] | None = None
        # Use persistent voting history.
        # This is required for precision hold with blocks_per_step=1.
        vote_history = self.vote_history

        recent: list[int] = []
        vote_count = 0
        candidate_status = False
        confirmed_status = False
        final_decision = "NotDrone"

        for block_index in range(self.precision_blocks):
            iq_raw = self.receiver.read_samples(self.num_samples)
            iq_dc = remove_dc_offset(iq_raw)

            iq = normalize_iq(iq_dc)

            if iq.ndim != 2 or iq.shape[0] < 2:
                continue

            branch = compute_dual_channel_stft_branch(
                rx0_iq=iq[self.ref_channel],
                rx1_iq=iq[self.target_channel],
                sample_rate=self.sample_rate,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                window=self.window,
                cnn_source="rx0",
            )

            cnn_spectrogram = compute_runtime_cnn_spectrogram(
                iq_raw,
                rx_index=self.cnn_rx_index,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
            )

            cnn_result = None

            if self.cnn_classifier is not None:
                cnn_result = self.cnn_classifier.predict(cnn_spectrogram)
                selection_score = self._drone_probability(cnn_result)
            else:
                selection_score = self._fft_selection_score(cnn_spectrogram)

            raw_vote = 0
            if self.cnn_classifier is not None and drone_threshold is not None:
                raw_vote = int(float(selection_score) >= float(drone_threshold))

            if self.decision_cfg is not None and self.decision_cfg.temporal_voting.enabled:
                recent, vote_count, candidate_status, confirmed_status, final_decision = update_temporal_decision(
                    vote_history,
                    raw_vote,
                    self.decision_cfg.temporal_voting,
                )
            else:
                vote_history.append(raw_vote)
                recent = list(vote_history)
                vote_count = int(sum(recent))
                candidate_status = bool(raw_vote)
                confirmed_status = bool(raw_vote)
                final_decision = "Confirmed Drone" if raw_vote else "NotDrone"

            if best is None or selection_score > best["selection_score"]:
                best = {
                    "block_index": block_index,
                    "selection_score": float(selection_score),
                    "iq": iq,
                    "branch": branch,
                    "cnn_spectrogram": cnn_spectrogram,
                    "cnn_result": cnn_result,
                }

        if best is None:
            return self._empty_result(center_freq)

        iq = best["iq"]
        branch = best["branch"]
        cnn_spectrogram = best["cnn_spectrogram"]
        cnn_result = best["cnn_result"]

        cnn_label = None
        cnn_score = None
        cnn_class_index = None
        cnn_probabilities = None

        if cnn_result is not None:
            cnn_label = cnn_result.class_name
            cnn_score = float(cnn_result.confidence)
            cnn_class_index = int(cnn_result.class_index)
            cnn_probabilities = [float(p) for p in cnn_result.probabilities]

        spectrogram_path = None
        rx0_stft_path = None
        rx1_stft_path = None

        if self.save_dir is not None:
            freq_tag = str(int(center_freq))

            if self.save_spectrogram:
                spectrogram_path = self.save_dir / f"{freq_tag}_cnn_spectrogram.npy"
                np.save(spectrogram_path, cnn_spectrogram)

            if self.save_stft:
                rx0_stft_path = self.save_dir / f"{freq_tag}_rx0_complex_stft.npy"
                rx1_stft_path = self.save_dir / f"{freq_tag}_rx1_complex_stft.npy"

                np.save(rx0_stft_path, branch.rx0.complex_stft)
                np.save(rx1_stft_path, branch.rx1.complex_stft)

        require_confirmed = True
        if self.decision_cfg is not None:
            require_confirmed = bool(self.decision_cfg.temporal_voting.require_confirmed_before_aoa)

        should_run_aoa = (not require_confirmed) or bool(confirmed_status)
        aoa_skipped_reason = None

        coherence_value = None
        coherence_passed = None
        phase_diff_rad = None
        phase_diff_deg = None
        angle_deg = None
        angle_valid = None
        sector_index = None
        sector_label = None
        sector_valid = None
        aoa_smoothed_angle_deg = None
        aoa_smoothing_valid = False
        aoa_smoothing_rejected_reason = None

        if should_run_aoa:
            coherence_result = coherence_gate(
                z0=branch.rx0.complex_stft,
                z1=branch.rx1.complex_stft,
                threshold=self.coherence_threshold,
                energy_percentile=self.energy_percentile,
            )

            phase_result = estimate_phase_diff(
                iq_block=iq,
                ref_channel=self.ref_channel,
                target_channel=self.target_channel,
            )

            angle_result = phase_diff_to_angle(
                phase_diff_rad=phase_result.phase_diff_rad,
                carrier_freq=float(center_freq),
                antenna_spacing_m=self.antenna_spacing_m,
                phase_offset_rad=self.phase_offset_rad,
                clip_input=self.clip_angle_input,
            )

            coherence_value = float(coherence_result.coherence)
            coherence_passed = bool(coherence_result.passed)
            phase_diff_rad = float(phase_result.phase_diff_rad)
            phase_diff_deg = float(phase_result.phase_diff_deg)
            angle_deg = float(angle_result.angle_deg)
            angle_valid = bool(angle_result.valid)

            if self.require_coherence_for_valid_angle and not coherence_passed:
                angle_valid = False

            aoa_smoothed_angle_deg, aoa_smoothing_valid, _, aoa_smoothing_rejected_reason = self._update_smoothed_angle(
                angle_deg=angle_deg,
                angle_valid=angle_valid,
                coherence_passed=coherence_passed,
            )

            angle_for_sector = aoa_smoothed_angle_deg if aoa_smoothing_valid else angle_deg
            sector_result = quantize_front_angle_to_sector(
                angle_for_sector if angle_valid and coherence_passed else None,
                num_sectors=int(self.sector_cfg.get("num_sectors", 8)),
                min_angle=float(self.sector_cfg.get("min_angle_deg", -90.0)),
                max_angle=float(self.sector_cfg.get("max_angle_deg", 90.0)),
            )

            sector_index = sector_result.sector_index
            sector_label = sector_result.sector_label
            sector_valid = sector_result.valid
        else:
            aoa_skipped_reason = "not_confirmed"
            aoa_smoothing_rejected_reason = "not_confirmed"
            if self.reset_smoothing_on_not_confirmed:
                self.angle_history.clear()

        return PrecisionAnalysisResult(
            center_freq=float(center_freq),
            stft_done=True,
            cnn_enabled=self.cnn_classifier is not None,
            cnn_label=cnn_label,
            cnn_score=cnn_score,
            cnn_class_index=cnn_class_index,
            cnn_probabilities=cnn_probabilities,
            coherence=coherence_value,
            coherence_passed=coherence_passed,
            phase_diff_rad=phase_diff_rad,
            phase_diff_deg=phase_diff_deg,
            angle_deg=angle_deg,
            angle_valid=angle_valid,
            cnn_spectrogram_shape=list(cnn_spectrogram.shape),
            spectrogram_path=str(spectrogram_path) if spectrogram_path is not None else None,
            rx0_stft_path=str(rx0_stft_path) if rx0_stft_path is not None else None,
            rx1_stft_path=str(rx1_stft_path) if rx1_stft_path is not None else None,
            sector_index=sector_index,
            sector_label=sector_label,
            sector_valid=sector_valid,
            precision_blocks=self.precision_blocks,
            selected_block_index=int(best["block_index"]),
            selection_score=float(best["selection_score"]),
            drone_probability=float(best["selection_score"]),
            drone_threshold=drone_threshold,
            temporal_window=int(vote_history.maxlen or len(recent)),
            drone_vote_count=int(vote_count),
            temporal_history=[int(x) for x in recent],
            candidate_status=bool(candidate_status),
            confirmed_status=bool(confirmed_status),
            final_decision=final_decision,
            aoa_skipped_reason=aoa_skipped_reason,
            aoa_smoothed_angle_deg=aoa_smoothed_angle_deg,
            aoa_smoothing_valid=bool(aoa_smoothing_valid),
            aoa_smoothing_history_size=len(self.angle_history),
            aoa_smoothing_method=self.smoothing_method,
            aoa_smoothing_rejected_reason=aoa_smoothing_rejected_reason,
        )
