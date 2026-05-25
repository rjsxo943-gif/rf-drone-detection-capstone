from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocess import remove_dc_offset
from src.scan.scan_policy import build_scan_freqs, is_energy_passed, is_candidate


@dataclass
class ScanEvent:
    center_freq: float
    max_fft_power: float
    mean_fft_power: float
    threshold: float
    pass_count: int
    triggered: bool


def compute_fft_scan_score(iq: np.ndarray) -> tuple[float, float]:
    """
    스캔용 가벼운 FFT score 계산.
    반환:
        max_fft_power:
            FFT bin 중 가장 큰 power
        mean_fft_power:
            전체 FFT power 평균
    """
    if iq.ndim == 2:
        x = iq[0]
    else:
        x = iq

    x = np.asarray(x, dtype=np.complex64)

    spectrum = np.fft.fft(x)
    power = np.abs(spectrum) ** 2

    return float(np.max(power)), float(np.mean(power))


class FrequencyScanner:
    def __init__(
        self,
        receiver,
        start_freq: float,
        stop_freq: float,
        step_freq: float,
        num_samples: int,
        threshold: float,
        scan_blocks: int = 3,
        min_pass_blocks: int = 2,
    ) -> None:
        self.receiver = receiver
        self.scan_freqs = build_scan_freqs(start_freq, stop_freq, step_freq)
        self.num_samples = int(num_samples)
        self.threshold = float(threshold)
        self.scan_blocks = int(scan_blocks)
        self.min_pass_blocks = int(min_pass_blocks)

        print()
        print("=== FREQUENCY SCANNER DEBUG ===")
        print(f"start_freq     : {start_freq}")
        print(f"stop_freq      : {stop_freq}")
        print(f"step_freq      : {step_freq}")
        print(f"num_scan_freqs : {len(self.scan_freqs)}")
        print(f"scan_freqs MHz : {[round(f / 1e6, 3) for f in self.scan_freqs]}")
        print("===============================")
        print()

    def _set_center_freq(self, center_freq: float) -> None:
        """
        receiver 종류에 따라 center frequency 설정.
        PlutoReceiver에 set_center_freq가 있으면 그걸 사용하고,
        없으면 내부 객체 속성을 최대한 찾아서 설정한다.
        """
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

    def scan_once(self) -> list[ScanEvent]:
        events: list[ScanEvent] = []

        for center_freq in self.scan_freqs:
            self._set_center_freq(center_freq)

            pass_count = 0
            max_scores = []
            mean_scores = []

            for _ in range(self.scan_blocks):
                iq = self.receiver.read_samples(self.num_samples)
                iq = remove_dc_offset(iq)

                max_power, mean_power = compute_fft_scan_score(iq)
                max_scores.append(max_power)
                mean_scores.append(mean_power)

                if is_energy_passed(max_power, self.threshold):
                    pass_count += 1

            triggered = is_candidate(pass_count, self.min_pass_blocks)

            event = ScanEvent(
                center_freq=float(center_freq),
                max_fft_power=float(max(max_scores)),
                mean_fft_power=float(np.mean(mean_scores)),
                threshold=float(self.threshold),
                pass_count=int(pass_count),
                triggered=bool(triggered),
            )

            events.append(event)

        return events