import numpy as np


class EnergyDetector:
    def __init__(self, threshold_multiplier: float = 3.0) -> None:
        self.threshold_multiplier = threshold_multiplier
        self.noise_floor = None
        self.threshold = None

    def fit(self, frame_energies: np.ndarray) -> None:
        self.noise_floor = float(np.median(frame_energies))
        self.threshold = float(self.noise_floor * self.threshold_multiplier)

    def detect(self, frame_energies: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            self.fit(frame_energies)
        return frame_energies > self.threshold
