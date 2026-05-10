from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.ml.rf3_labels import ID_TO_LABEL, LABEL_TO_ID, num_rf3_classes
from src.ml.rf3_model import RF3SmallCNN


@dataclass(frozen=True)
class RF3Result:
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    input_shape: tuple[int, int]
    model_path: str
    valid: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "input_shape": list(self.input_shape),
            "model_path": self.model_path,
            "valid": self.valid,
            "reason": self.reason,
        }


class RF3Classifier:
    """
    Runtime-compatible RF3 classifier.

    현재 RF3 모델 기준:
    - 입력 spectrogram shape: (128, 509)
    - 입력 dtype: float32
    - 클래스: Background / Bluetooth / WiFi

    이 클래스는 학습된 best_model.pt를 한 번 로드한 뒤,
    predict_array() 또는 predict_file()로 반복 호출하는 용도다.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"RF3 model not found: {self.model_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.mean = float(checkpoint["mean"])
        self.std = float(checkpoint["std"])

        input_shape = checkpoint.get("input_shape", [1, 128, 509])
        self.expected_shape = (int(input_shape[-2]), int(input_shape[-1]))

        self.model = RF3SmallCNN(num_classes=num_rf3_classes()).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_file(self, path: str | Path) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"RF3 input file not found: {path}")

        x = np.load(path).astype(np.float32)
        return x

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        # 허용: (128, 509)
        if x.shape == self.expected_shape:
            return x

        # 허용: (1, 128, 509)
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[1:] == self.expected_shape:
            return x[0]

        raise ValueError(
            f"Unexpected RF3 input shape: {x.shape}, "
            f"expected {self.expected_shape} or (1, {self.expected_shape[0]}, {self.expected_shape[1]})"
        )

    @torch.no_grad()
    def predict_array(self, x: np.ndarray) -> RF3Result:
        x = self._normalize_input(x)

        x_norm = (x - self.mean) / (self.std + 1e-8)

        # Conv2d input: batch x channel x height x width
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).float()
        x_tensor = x_tensor.to(self.device)

        logits = self.model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        class_name = ID_TO_LABEL[pred_id]
        confidence = float(probs[pred_id])

        probabilities = {
            "Background": float(probs[LABEL_TO_ID["Background"]]),
            "Bluetooth": float(probs[LABEL_TO_ID["Bluetooth"]]),
            "WiFi": float(probs[LABEL_TO_ID["WiFi"]]),
        }

        return RF3Result(
            class_name=class_name,
            confidence=confidence,
            probabilities=probabilities,
            input_shape=tuple(x.shape),
            model_path=str(self.model_path),
            valid=True,
        )

    def predict_file(self, path: str | Path) -> RF3Result:
        x = self._load_file(path)
        return self.predict_array(x)

    def predict_with_threshold(
        self,
        x: np.ndarray,
        confidence_threshold: float = 0.70,
    ) -> RF3Result:
        result = self.predict_array(x)

        if result.confidence >= confidence_threshold:
            return result

        return RF3Result(
            class_name="Unknown",
            confidence=result.confidence,
            probabilities=result.probabilities,
            input_shape=result.input_shape,
            model_path=result.model_path,
            valid=False,
            reason=f"confidence below threshold: {result.confidence:.4f} < {confidence_threshold:.4f}",
        )

    def predict_file_with_threshold(
        self,
        path: str | Path,
        confidence_threshold: float = 0.70,
    ) -> RF3Result:
        x = self._load_file(path)
        return self.predict_with_threshold(
            x=x,
            confidence_threshold=confidence_threshold,
        )
