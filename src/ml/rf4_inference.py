from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.ml import RF3SmallCNN


DEFAULT_RF4_CLASS_NAMES = [
    "Background",
    "WiFi",
    "Bluetooth",
    "Drone-like",
]


@dataclass(frozen=True)
class RF4Result:
    class_id: int
    class_name: str
    confidence: float
    final_class: str
    probabilities: dict[str, float]
    threshold: float


class RF4Classifier:
    """
    RF4 CNN classifier for spectrogram input.

    현재 프로젝트 기준:
    - input spectrogram shape: (128, 509)
    - model input shape: (1, 1, 128, 509)
    - classes: Background / WiFi / Bluetooth / Drone-like
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        threshold: float = 0.70,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = float(threshold)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.mean = float(self.checkpoint["mean"])
        self.std = float(self.checkpoint["std"])

        self.input_shape = self.checkpoint.get("input_shape", [1, 128, 509])
        self.num_classes = int(self.checkpoint.get("num_classes", 4))
        self.class_names = self.checkpoint.get(
            "class_names",
            DEFAULT_RF4_CLASS_NAMES,
        )

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length mismatch: "
                f"len={len(self.class_names)}, num_classes={self.num_classes}"
            )

        self.expected_shape = (
            int(self.input_shape[1]),
            int(self.input_shape[2]),
        )

        self.model = RF3SmallCNN(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def predict_file(self, npy_path: str | Path) -> RF4Result:
        path = Path(npy_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        spec = np.load(path).astype(np.float32)
        return self.predict_array(spec)

    @torch.no_grad()
    def predict_array(self, spectrogram: np.ndarray) -> RF4Result:
        spec = np.asarray(spectrogram, dtype=np.float32)

        if spec.shape != self.expected_shape:
            raise ValueError(
                f"Unexpected spectrogram shape: {spec.shape}, "
                f"expected: {self.expected_shape}"
            )

        x = (spec - self.mean) / (self.std + 1e-8)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        x_tensor = x_tensor.to(self.device)

        logits = self.model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = str(self.class_names[class_id])
        final_class = class_name if confidence >= self.threshold else "Unknown"

        probabilities = {
            str(class_name): float(prob)
            for class_name, prob in zip(self.class_names, probs)
        }

        return RF4Result(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            final_class=final_class,
            probabilities=probabilities,
            threshold=self.threshold,
        )
