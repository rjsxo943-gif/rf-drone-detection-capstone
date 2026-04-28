from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_energy_plot(
    energies: Sequence[float] | np.ndarray,
    threshold: float,
    save_path: str | Path,
    detections: Sequence[bool] | np.ndarray | None = None,
    title: str = "Frame Energy",
) -> Path:
    """
    energy detector 결과를 그래프로 저장한다.

    표시 내용:
    - frame/block energy
    - threshold line
    - detection point
    """

    energies = np.asarray(energies, dtype=float)

    if energies.size == 0:
        raise ValueError("energies is empty.")

    x = np.arange(len(energies))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(x, energies, label="energy")
    plt.axhline(float(threshold), linestyle="--", label="threshold")

    if detections is not None:
        detections = np.asarray(detections).astype(bool)

        if len(detections) != len(energies):
            raise ValueError(
                f"detections length must match energies length. "
                f"got detections={len(detections)}, energies={len(energies)}"
            )

        plt.scatter(
            x[detections],
            energies[detections],
            s=12,
            label="detections",
        )

    plt.xlabel("frame index")
    plt.ylabel("energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def save_spectrogram_image(
    spectrogram: np.ndarray,
    save_path: str | Path,
    title: str = "Spectrogram",
) -> Path:
    """
    spectrogram 또는 CNN 입력 이미지를 PNG로 저장한다.

    입력:
    - shape = (freq_bins, time_frames)
    - 또는 shape = (H, W)
    """

    spectrogram = np.asarray(spectrogram)

    if spectrogram.size == 0:
        raise ValueError("spectrogram is empty.")

    if spectrogram.ndim != 2:
        raise ValueError(
            f"spectrogram must be 2-D. got shape={spectrogram.shape}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(label="magnitude")
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def save_aoa_plot(
    block_indices: Sequence[int] | np.ndarray,
    angles_deg: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "AoA Estimate",
) -> Path:
    """
    block별 AoA 추정 각도 변화를 그래프로 저장한다.
    """

    block_indices = np.asarray(block_indices, dtype=int)
    angles_deg = np.asarray(angles_deg, dtype=float)

    if block_indices.size == 0 or angles_deg.size == 0:
        raise ValueError("block_indices and angles_deg must not be empty.")

    if block_indices.shape != angles_deg.shape:
        raise ValueError(
            f"block_indices and angles_deg must have same shape. "
            f"got block_indices={block_indices.shape}, angles_deg={angles_deg.shape}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(block_indices, angles_deg, marker="o", label="angle")
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel("block index")
    plt.ylabel("angle [deg]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path