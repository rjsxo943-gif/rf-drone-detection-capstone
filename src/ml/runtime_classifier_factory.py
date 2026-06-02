from __future__ import annotations

from typing import Any

from src.ml.inference import BinaryFlatCNNClassifier, build_cnn_classifier


BINARY_BACKENDS = {
    "binary_flat",
    "binary",
    "rf4_binary",
    "drone_binary",
}


def build_runtime_cnn_classifier(ml_cfg: dict[str, Any]):
    """
    Runtime 전용 CNN classifier factory.

    기존 build_cnn_classifier()는 4-class RF4/torch/keras 호환을 유지한다.
    통합 runtime에서는 Drone / NotDrone binary 모델도 바로 물릴 수 있어야 하므로
    binary_flat backend를 여기서 먼저 처리한다.
    """
    inference_cfg = ml_cfg.get("inference", {}) or {}
    backend = str(inference_cfg.get("backend", "dummy")).lower().strip()

    if backend in BINARY_BACKENDS:
        model_path = inference_cfg.get("model_path")
        if model_path in [None, "", "null", "None"]:
            raise ValueError("inference.model_path is required for binary_flat backend")

        class_names = list(ml_cfg.get("class_names", ["NotDrone", "Drone"]))

        return BinaryFlatCNNClassifier(
            model_path=str(model_path),
            class_names=class_names,
            device=inference_cfg.get("device", "cpu"),
        )

    return build_cnn_classifier(ml_cfg)
