import numpy as np
from .base import BaseReceiver


class HackRFReceiver(BaseReceiver):
    def __init__(self, *args, **kwargs):
        self.sample_rate = kwargs.get("sample_rate")
        self.center_freq = kwargs.get("center_freq")

    def read_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError("HackRF backend is not implemented yet.")
