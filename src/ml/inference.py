#src/ml/inference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.ml.transforms import ensure_cnn_input_shape, add_batch_dimension


@dataclass
class CNNResult:
    """
    block 하나에 대한 CNN 분류 결과.
    """

    class_name: str
    class_index: int
    confidence: float
    probabilities: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class DummyCNNClassifier:
    """
    실제 CNN 모델이 없을 때 사용하는 가짜 classifier.

    목적:
    - 전체 pipeline 연결 테스트
    - AoA gate 동작 테스트
    - result schema 확인
    """

    def __init__(
        self,
        class_names: list[str],
        dummy_class_name: str = "Background",
        dummy_confidence: float = 0.0,
    ) -> None:
        self.class_names = class_names
        self.dummy_class_name = dummy_class_name
        self.dummy_confidence = float(dummy_confidence)

        if dummy_class_name not in class_names:
            raise ValueError(
                f"dummy_class_name={dummy_class_name} is not in class_names={class_names}"
            )

    def predict(self, spectrogram: np.ndarray) -> CNNResult:
        class_index = self.class_names.index(self.dummy_class_name)

        probabilities = [0.0 for _ in self.class_names]
        probabilities[class_index] = self.dummy_confidence

        return CNNResult(
            class_name=self.dummy_class_name,
            class_index=class_index,
            confidence=self.dummy_confidence,
            probabilities=probabilities,
        )


class KerasCNNClassifier:
    """
    나중에 실제 .keras 모델이 생기면 사용하는 classifier.

    현재는 구조만 만들어둔다.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        input_shape: tuple[int, int, int] = (512, 125, 1),
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names
        self.input_shape = input_shape

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow가 설치되어 있지 않습니다. "
                "Keras 모델을 사용하려면 tensorflow를 설치해야 합니다."
            ) from exc

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, spectrogram: np.ndarray) -> CNNResult:
        x = ensure_cnn_input_shape(spectrogram, expected_shape=self.input_shape)
        x = add_batch_dimension(x)

        pred = self.model.predict(x, verbose=0)[0]
        pred = np.asarray(pred, dtype=np.float32)

        class_index = int(np.argmax(pred))
        confidence = float(pred[class_index])
        class_name = self.class_names[class_index]

        return CNNResult(
            class_name=class_name,
            class_index=class_index,
            confidence=confidence,
            probabilities=pred.tolist(),
        )


def build_cnn_classifier(ml_cfg: dict[str, Any]):
    """
    ml.yaml 설정을 보고 CNN classifier를 생성한다.

    inference.backend:
    - dummy: 실제 모델 없이 고정 결과 반환
    - keras: models/checkpoints/model.keras 로드
    """
    class_names = list(ml_cfg["class_names"])

    inference_cfg = ml_cfg.get("inference", {})
    backend = str(inference_cfg.get("backend", "dummy")).lower().strip()

    if backend == "dummy":
        return DummyCNNClassifier(
            class_names=class_names,
            dummy_class_name=inference_cfg.get("dummy_class_name", "Background"),
            dummy_confidence=float(inference_cfg.get("dummy_confidence", 0.0)),
        )

    if backend == "keras":
        model_cfg = ml_cfg.get("model", {})
        spec_cfg = ml_cfg.get("spectrogram", {})

        input_shape = tuple(spec_cfg.get("input_shape", [512, 125, 1]))

        return KerasCNNClassifier(
            model_path=model_cfg.get("model_path", "models/checkpoints/model.keras"),
            class_names=class_names,
            input_shape=input_shape,
        )

    raise ValueError(
        f"Unsupported CNN inference backend: {backend}. "
        "Expected 'dummy' or 'keras'."
    )