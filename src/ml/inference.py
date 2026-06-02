from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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



class LegacyRF4Live2450Factory:
    """
    outputs/ml/rf4_cnn_live2450_v2/best_model.pt 호환용 legacy CNN.

    checkpoint key 구조:
    - features.0.net.*
    - features.1.net.*
    - features.2.net.*
    - features.3.net.*
    - classifier.2.*
    """

    @staticmethod
    def build(num_classes: int):
        import torch.nn as nn

        class ConvBlock(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                stride: int = 1,
            ) -> None:
                super().__init__()

                self.net = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        class LegacyRF4Live2450CNN(nn.Module):
            def __init__(self, num_classes: int = 4) -> None:
                super().__init__()

                self.features = nn.Sequential(
                    ConvBlock(1, 16, stride=1),
                    ConvBlock(16, 32, stride=2),
                    ConvBlock(32, 64, stride=2),
                    ConvBlock(64, 128, stride=2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                logits = self.classifier(x)
                return logits

        return LegacyRF4Live2450CNN(num_classes=num_classes)


class RF4CNNClassifierAdapter:
    """
    RF4Classifier를 기존 PrecisionAnalyzer가 기대하는 CNNResult 형식으로 감싸는 adapter.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        device: str = "cpu",
        general_threshold: float = 0.50,
        drone_threshold: float = 0.70,
    ) -> None:
        from src.ml.rf4_inference import RF4Classifier

        self.class_names = class_names
        self.rf4 = RF4Classifier(
            checkpoint_path=model_path,
            general_threshold=general_threshold,
            drone_threshold=drone_threshold,
            device=device,
        )

    def predict(self, spectrogram):
        import numpy as np

        x = np.asarray(spectrogram, dtype=np.float32)

        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]

        result = self.rf4.predict_array(x)

        probabilities = [
            float(result.probabilities[name])
            for name in self.rf4.class_names
        ]

        return CNNResult(
            class_name=result.class_name,
            class_index=int(result.class_id),
            confidence=float(result.confidence),
            probabilities=probabilities,
        )

class TorchCNNClassifier:
    """
    PyTorch 기반 SpectrogramCNN classifier.

    현재 프로젝트의 src/ml/model_2dcnn.py 모델을 사용한다.

    model_path가 None이면 랜덤 초기화 모델을 사용한다.
    이 경우 예측 결과는 의미 없고, pipeline 연결 테스트용이다.
    """

    def __init__(
        self,
        model_path: str | None,
        class_names: list[str],
        device: str = "cpu",
        model_arch: str = "current",
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names
        self.device_name = device
        self.mean = 0.0
        self.std = 1.0

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch가 설치되어 있지 않습니다. "
                "Torch 모델을 사용하려면 torch를 설치해야 합니다."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model_arch = str(model_arch).lower().strip()

        if self.model_arch in ["legacy_rf4_live2450", "legacy", "rf4_live2450_v2"]:
            self.model = LegacyRF4Live2450Factory.build(
                num_classes=len(class_names),
            )
        else:
            from src.ml.model_2dcnn import build_model
            self.model = build_model(num_classes=len(class_names))

        self.model.to(self.device)
        self.model.eval()

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Torch model checkpoint not found: {path}")

        checkpoint = self.torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "mean" in checkpoint:
                self.mean = float(checkpoint["mean"])
            if "std" in checkpoint:
                self.std = float(checkpoint["std"])

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, spectrogram: np.ndarray) -> CNNResult:
        x = np.asarray(spectrogram, dtype=np.float32)

        # (512, 125, 1) 형태로 들어오면 (512, 125)로 변환
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]

        if x.ndim != 2:
            raise ValueError(f"Expected spectrogram shape (H, W), got {x.shape}")

        x = (x - self.mean) / max(self.std, 1e-8)

        x_tensor = self.torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # (H, W) -> (B, C, H, W)
        x_tensor = x_tensor.to(self.device)

        with self.torch.no_grad():
            logits = self.model(x_tensor)
            probs = self.torch.softmax(logits, dim=1)

        pred = probs.squeeze(0).cpu().numpy().astype(np.float32)

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
    - keras: .keras 모델 로드
    - torch: PyTorch SpectrogramCNN 모델 로드
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
    
    if backend == "rf4":
        model_path = inference_cfg.get("model_path")

        if model_path in [None, "", "null", "None"]:
            raise ValueError("inference.model_path is required for backend='rf4'")

        return RF4CNNClassifierAdapter(
            model_path=model_path,
            class_names=class_names,
            device=inference_cfg.get("device", "cpu"),
            general_threshold=float(inference_cfg.get("general_threshold", 0.50)),
            drone_threshold=float(inference_cfg.get("drone_threshold", 0.70)),
        )

    if backend == "torch":
        model_cfg = ml_cfg.get("model", {})

        model_path = inference_cfg.get(
            "model_path",
            model_cfg.get("torch_model_path", None),
        )

        if model_path in ["", "null", "None"]:
            model_path = None

        return TorchCNNClassifier(
            model_path=model_path,
            class_names=class_names,
            device=inference_cfg.get("device", "cpu"),
            model_arch=inference_cfg.get("model_arch", "current"),
        )

    raise ValueError(
        f"Unsupported CNN inference backend: {backend}. "
        "Expected 'dummy', 'keras', 'torch', or 'rf4'." 
    )
class BinaryFlatCNNClassifier:
    """
    rf4_binary_20260530_no_controller_lr1e3_v2/best_model.pt 전용 binary CNN loader.

    구조:
    Conv-BN-ReLU-MaxPool x3
    Conv-BN-ReLU
    AdaptiveAvgPool2d((1, 1))
    Flatten
    Dropout(0.30)
    Linear(128, 2)

    class_names:
    0 = NotDrone
    1 = Drone
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path)
        self.device_name = str(device)
        self.mean = 0.0
        self.std = 1.0

        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PyTorch is required for BinaryFlatCNNClassifier.") from exc

        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)

        from pathlib import Path
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Binary CNN checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected checkpoint dict, got {type(checkpoint)}")

        self.checkpoint = checkpoint
        self.class_names = list(checkpoint.get("class_names", class_names or ["NotDrone", "Drone"]))
        self.num_classes = int(checkpoint.get("num_classes", len(self.class_names)))

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length mismatch: len={len(self.class_names)}, "
                f"num_classes={self.num_classes}"
            )

        if "mean" in checkpoint:
            self.mean = float(checkpoint["mean"])
        if "std" in checkpoint:
            self.std = float(checkpoint["std"])

        self.model = self._build_model(num_classes=self.num_classes)
        self.model.to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint missing model_state_dict/state_dict")

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _build_model(self, num_classes: int):
        nn = self.nn

        class BinaryCNN(nn.Module):
            def __init__(self, num_classes: int = 2) -> None:
                super().__init__()

                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),

                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),

                    nn.AdaptiveAvgPool2d((1, 1)),
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.30),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                logits = self.classifier(x)
                return logits

        return BinaryCNN(num_classes=num_classes)

    def predict(self, spectrogram):
        import numpy as np

        x = np.asarray(spectrogram, dtype=np.float32)

        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]

        if x.ndim != 2:
            raise ValueError(f"Expected spectrogram shape (H, W), got {x.shape}")

        x = (x - self.mean) / max(self.std, 1e-8)

        x_tensor = self.torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
        x_tensor = x_tensor.to(self.device)

        with self.torch.no_grad():
            logits = self.model(x_tensor)
            probs = self.torch.softmax(logits, dim=1)

        pred = probs.squeeze(0).cpu().numpy().astype(np.float32)
        class_index = int(np.argmax(pred))
        confidence = float(pred[class_index])
        class_name = self.class_names[class_index]

        return CNNResult(
            class_name=class_name,
            class_index=class_index,
            confidence=confidence,
            probabilities=pred.tolist(),
        )
