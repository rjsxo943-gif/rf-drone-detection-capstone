from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.features.spectrogram import compute_stft_branch
from src.ml.inference import DummyCNNClassifier, KerasCNNClassifier, TorchCNNClassifier, BinaryFlatCNNClassifier


DEFAULT_CLASS_NAMES = ("Background", "WiFi", "Bluetooth", "Drone-like")
DEFAULT_POSITIVE_CLASS_NAMES = ("Drone-like", "Drone", "drone")


@dataclass
class CNNRuntime:
    """Runtime wrapper for live STFT spectrogram generation and CNN inference.

    The viewer keeps raw IQ features separate from the CNN branch. This class
    receives raw IQ, selects one RX channel, generates the normalized STFT
    spectrogram, runs inference, and applies lightweight temporal smoothing.
    """

    model_path: str | Path | None = None
    backend: str = "torch"
    device: str = "cpu"
    class_names: Iterable[str] = DEFAULT_CLASS_NAMES
    rx_index: int = 0
    sample_rate: float = 5_000_000.0
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 512
    window: str = "hann"
    confidence_threshold: float = 0.5
    smooth_window: int = 5
    confirm_votes: int = 3
    positive_class_names: Iterable[str] = DEFAULT_POSITIVE_CLASS_NAMES
    dummy_class_name: str = "Background"
    dummy_confidence: float = 0.0
    history: deque[dict[str, Any]] = field(init=False)
    classifier: Any = field(init=False)

    def __post_init__(self) -> None:
        self.class_names = tuple(str(name) for name in self.class_names)
        self.positive_class_names = tuple(str(name) for name in self.positive_class_names)
        self.smooth_window = max(1, int(self.smooth_window))
        self.confirm_votes = max(1, int(self.confirm_votes))
        self.history = deque(maxlen=self.smooth_window)
        self.classifier = self._build_classifier()

    def process(self, iq: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return (spectrogram_image, cnn_result_dict) for one IQ block."""

        iq_1d = self._select_iq_channel(iq)
        stft = compute_stft_branch(
            iq_block=iq_1d,
            sample_rate=float(self.sample_rate),
            nperseg=int(self.nperseg),
            noverlap=int(self.noverlap),
            nfft=int(self.nfft),
            window=self.window,
        )
        spectrogram = stft.cnn_spectrogram

        result = self.classifier.predict(spectrogram)
        raw = result.to_dict()
        raw_class = str(raw["class_name"])
        raw_confidence = float(raw["confidence"])
        candidate = self._is_positive_candidate(raw_class, raw_confidence)

        self.history.append(
            {
                "class_name": raw_class,
                "confidence": raw_confidence,
                "candidate": candidate,
            }
        )
        smooth = self._build_smoothing_result()

        metrics: dict[str, Any] = {
            "cnn_raw_class_name": raw_class,
            "cnn_raw_class_index": int(raw["class_index"]),
            "cnn_raw_confidence": raw_confidence,
            "cnn_probabilities": list(raw.get("probabilities", [])),
            "cnn_candidate": bool(candidate),
            "cnn_confirmed": bool(smooth["confirmed"]),
            "cnn_smoothed_class_name": smooth["smoothed_class_name"],
            "cnn_smoothed_confidence": float(smooth["smoothed_confidence"]),
            "cnn_positive_votes": int(smooth["positive_votes"]),
            "cnn_history_size": int(len(self.history)),
            "cnn_smooth_window": int(self.smooth_window),
            "cnn_confirm_votes": int(self.confirm_votes),
            "cnn_confidence_threshold": float(self.confidence_threshold),
            "cnn_backend": str(self.backend),
            "cnn_model_path": str(self.model_path) if self.model_path else "",
            "cnn_rx_index": int(self.rx_index),
        }
        return spectrogram, metrics

    def status_text(self) -> str:
        return (
            f"CNN backend={self.backend} model={self.model_path or 'none'} "
            f"window={self.smooth_window} votes={self.confirm_votes}"
        )

    def reset_history(self) -> None:
        self.history.clear()

    def _build_classifier(self) -> Any:
        backend = str(self.backend).lower().strip()
        if backend == "dummy":
            return DummyCNNClassifier(
                class_names=list(self.class_names),
                dummy_class_name=str(self.dummy_class_name),
                dummy_confidence=float(self.dummy_confidence),
            )
        if backend == "binary":
            if self.model_path in (None, "", "None", "null"):
                raise ValueError("CNN binary backend requires --model PATH. Use --cnn-backend dummy for dry-run.")
            return BinaryFlatCNNClassifier(
                model_path=str(self.model_path),
                class_names=list(self.class_names),
                device=str(self.device),
            )
        if backend == "torch":
            if self.model_path in (None, "", "None", "null"):
                raise ValueError("CNN torch backend requires --model PATH. Use --cnn-backend dummy for dry-run.")
            return TorchCNNClassifier(
                model_path=str(self.model_path),
                class_names=list(self.class_names),
                device=str(self.device),
            )
        if backend == "keras":
            if self.model_path in (None, "", "None", "null"):
                raise ValueError("CNN keras backend requires --model PATH. Use --cnn-backend dummy for dry-run.")
            return KerasCNNClassifier(
                model_path=str(self.model_path),
                class_names=list(self.class_names),
            )
        raise ValueError(f"Unsupported CNN backend={self.backend}. Use binary, torch, keras, or dummy.")

    def _select_iq_channel(self, iq: np.ndarray) -> np.ndarray:
        arr = np.asarray(iq)
        if arr.ndim == 1:
            return arr.astype(np.complex64, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected IQ shape (channels, samples), got {arr.shape}")
        if self.rx_index < 0 or self.rx_index >= arr.shape[0]:
            raise IndexError(f"rx_index={self.rx_index} out of range for IQ shape {arr.shape}")
        return arr[self.rx_index].astype(np.complex64, copy=False)

    def _is_positive_candidate(self, class_name: str, confidence: float) -> bool:
        class_norm = class_name.strip().lower()
        positive = {name.strip().lower() for name in self.positive_class_names}
        return class_norm in positive and float(confidence) >= float(self.confidence_threshold)

    def _build_smoothing_result(self) -> dict[str, Any]:
        if not self.history:
            return {
                "smoothed_class_name": "Unknown",
                "smoothed_confidence": 0.0,
                "positive_votes": 0,
                "confirmed": False,
            }

        class_counter = Counter(str(item["class_name"]) for item in self.history)
        smoothed_class_name, _ = class_counter.most_common(1)[0]
        matching_confidences = [
            float(item["confidence"])
            for item in self.history
            if str(item["class_name"]) == smoothed_class_name
        ]
        smoothed_confidence = float(np.mean(matching_confidences)) if matching_confidences else 0.0
        positive_votes = sum(1 for item in self.history if bool(item.get("candidate", False)))
        confirmed = positive_votes >= int(self.confirm_votes)

        return {
            "smoothed_class_name": smoothed_class_name,
            "smoothed_confidence": smoothed_confidence,
            "positive_votes": int(positive_votes),
            "confirmed": bool(confirmed),
        }
