from abc import ABC, abstractmethod
import numpy as np


class BaseReceiver(ABC):
    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None
