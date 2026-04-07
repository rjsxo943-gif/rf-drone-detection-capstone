from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_energy_plot(
    energies,
    threshold,
    save_path,
    detections=None,
    title="Frame Energy",
):
    energies = np.asarray(energies, dtype=float)
    x = np.arange(len(energies))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(x, energies, label="energy")
    plt.axhline(float(threshold), linestyle="--", label="threshold")

    if detections is not None:
        detections = np.asarray(detections).astype(bool)
        if len(detections) == len(energies):
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
