from __future__ import annotations

import numpy as np


def build_scan_freqs(
    start_freq: float,
    stop_freq: float,
    step_freq: float,
) -> list[float]:
    """
    start_freq부터 stop_freq까지 step_freq 간격으로 center frequency 리스트 생성.
    stop_freq도 포함되도록 약간 여유를 둔다.
    """
    if step_freq <= 0:
        raise ValueError("step_freq must be positive")

    if stop_freq < start_freq:
        raise ValueError("stop_freq must be greater than or equal to start_freq")

    freqs = np.arange(start_freq, stop_freq + step_freq * 0.5, step_freq)
    return [float(f) for f in freqs]


def is_energy_passed(
    score: float,
    threshold: float,
) -> bool:
    """
    FFT/PSD 기반 score가 threshold를 넘었는지 판단.
    """
    return float(score) >= float(threshold)


def is_candidate(
    pass_count: int,
    min_pass_blocks: int,
) -> bool:
    """
    여러 block 중 threshold를 넘은 횟수가 충분한지 판단.
    예: scan_blocks=3, min_pass_blocks=2이면 3번 중 2번 이상 통과해야 후보.
    """
    if min_pass_blocks <= 0:
        raise ValueError("min_pass_blocks must be positive")

    return int(pass_count) >= int(min_pass_blocks)