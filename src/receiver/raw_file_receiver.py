# src/receiver/file_receiver.py

from pathlib import Path
import numpy as np


class FileReceiver:
    def __init__(self, filepath: str, sample_rate: float = 2.4e6):
        self.filepath = Path(filepath)
        self.sample_rate = sample_rate

    def receive(self) -> np.ndarray:
        if not self.filepath.exists():
            raise FileNotFoundError(f"파일이 없음: {self.filepath}")

        raw = np.load(self.filepath)

        if raw.size == 0:
            raise ValueError(f"파일이 비어있거나 읽기 실패: {self.filepath}")

        return raw.astype(np.complex64, copy=False)