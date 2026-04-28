#src/ml/evaluate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClassificationReport:
    """
    CNN 분류 평가 결과.
    """

    accuracy: float
    confusion_matrix: np.ndarray
    class_names: list[str]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_names": self.class_names,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
        }


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    전체 accuracy 계산.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return 0.0

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape. "
            f"Got {y_true.shape} and {y_pred.shape}"
        )

    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    confusion matrix 계산.

    행: 실제 label
    열: 예측 label
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape. "
            f"Got {y_true.shape} and {y_pred.shape}"
        )

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1

    return cm


def precision_recall_f1_from_cm(
    cm: np.ndarray,
    class_names: list[str],
    eps: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    confusion matrix에서 class별 precision, recall, f1 계산.
    """
    cm = np.asarray(cm, dtype=np.float64)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape {cm.shape}")

    if cm.shape[0] != len(class_names):
        raise ValueError(
            f"Number of classes in cm and class_names mismatch. "
            f"cm={cm.shape}, class_names={len(class_names)}"
        )

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        p = float(tp / (tp + fp + eps))
        r = float(tp / (tp + fn + eps))
        f = float(2.0 * p * r / (p + r + eps))

        precision[name] = p
        recall[name] = r
        f1[name] = f

    return precision, recall, f1


def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> ClassificationReport:
    """
    y_true, y_pred로 전체 평가 report 생성.
    """
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)

    precision, recall, f1 = precision_recall_f1_from_cm(
        cm,
        class_names=class_names,
    )

    return ClassificationReport(
        accuracy=acc,
        confusion_matrix=cm,
        class_names=class_names,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
    )


def predict_labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    모델 출력 확률 배열에서 class index를 뽑는다.

    입력:
    - probabilities shape = (N, num_classes)

    출력:
    - y_pred shape = (N,)
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be 2D array. Got shape {probabilities.shape}"
        )

    return np.argmax(probabilities, axis=1).astype(np.int64)