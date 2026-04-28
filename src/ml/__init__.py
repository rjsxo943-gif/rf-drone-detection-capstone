#src/ml/__init__.py

"""
ML package for 2D CNN-based RF signal classification.

현재 역할:
- STFT spectrogram을 CNN 입력 형태로 맞춤
- dummy 또는 keras classifier를 생성
- CNN 결과를 class_name / confidence 형태로 반환

현재 프로젝트 기준:
- CNN 입력 shape: (512, 125, 1)
- class: Background / WiFi / Bluetooth / Drone-like
"""

from src.ml.inference import (
    CNNResult,
    DummyCNNClassifier,
    KerasCNNClassifier,
    build_cnn_classifier,
)

from src.ml.transforms import (
    ensure_cnn_input_shape,
    add_batch_dimension,
)

__all__ = [
    "CNNResult",
    "DummyCNNClassifier",
    "KerasCNNClassifier",
    "build_cnn_classifier",
    "ensure_cnn_input_shape",
    "add_batch_dimension",
]