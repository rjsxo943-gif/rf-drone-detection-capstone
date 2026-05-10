from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml.rf3_labels import label_to_id


def read_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_manifest_path(path_text: str, project_root: str | Path) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return Path(project_root) / path


def compute_spectrogram_mean_std(
    rows: list[dict[str, str]],
    project_root: str | Path,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """
    train set 기준 mean/std 계산.
    모든 파일을 한 번에 메모리에 올리지 않고 누적합으로 계산한다.
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for row in rows:
        path = resolve_manifest_path(row["filepath"], project_root)
        x = np.load(path).astype(np.float64)

        total_sum += float(x.sum())
        total_sq_sum += float((x * x).sum())
        total_count += int(x.size)

    if total_count == 0:
        raise ValueError("No samples found while computing mean/std.")

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean * mean
    std = math.sqrt(max(var, eps))

    return float(mean), float(std)


class RFSpectrogramDataset(Dataset):
    """
    RF 3분류용 spectrogram dataset.

    입력 파일:
    - .npy
    - shape = (128, 509)
    - dtype = float32

    반환:
    - x: torch.Tensor, shape = (1, 128, 509)
    - y: torch.Tensor, scalar long
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        project_root: str | Path,
        mean: float,
        std: float,
        expected_shape: tuple[int, int] = (128, 509),
    ) -> None:
        self.rows = rows
        self.project_root = Path(project_root)
        self.mean = float(mean)
        self.std = float(std)
        self.expected_shape = expected_shape

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]

        path = resolve_manifest_path(row["filepath"], self.project_root)
        label = row["label"]

        x = np.load(path).astype(np.float32)

        if x.shape != self.expected_shape:
            raise ValueError(
                f"Unexpected spectrogram shape: {x.shape}, "
                f"expected={self.expected_shape}, path={path}"
            )

        x = (x - self.mean) / (self.std + 1e-8)

        x_tensor = torch.from_numpy(x).unsqueeze(0).float()
        y_tensor = torch.tensor(label_to_id(label), dtype=torch.long)

        return x_tensor, y_tensor
