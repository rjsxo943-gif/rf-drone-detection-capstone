from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewerState:
    """Mutable state shared by the real-time RF viewer modes."""

    mode: str
    gain: float
    center_freq: int
    sample_rate: int
    distance_m: float = 0.0
    memo: str = ""
    paused: bool = False
    update_index: int = 0
    target_fps: float = 10.0
    running: bool = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def step_gain(self, delta: float, min_gain: float = 0.0, max_gain: float = 73.0) -> float:
        self.gain = max(float(min_gain), min(float(max_gain), float(self.gain) + float(delta)))
        return self.gain

    def mark_update(self) -> int:
        self.update_index += 1
        return self.update_index
