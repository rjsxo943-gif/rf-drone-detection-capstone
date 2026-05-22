from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.ml.rf3_labels import ID_TO_LABEL


def build_confusion_matrix(
    preds: list[int],
    targets: list[int],
    num_classes: int = 3,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for target, pred in zip(targets, preds):
        cm[target, pred] += 1

    return cm


def save_confusion_matrix_csv(path: str | Path, cm: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + labels)

        for i, label in enumerate(labels):
            writer.writerow([label] + cm[i].tolist())


def save_confusion_matrix_png(path: str | Path, cm: np.ndarray) -> None:
    path = Path(path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skip confusion_matrix.png")
        return

    labels = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("RF 3-Class Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_classification_report_text(cm: np.ndarray) -> str:
    lines: list[str] = []
    labels = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    total_correct = int(np.trace(cm))
    total = int(cm.sum())
    accuracy = total_correct / total if total > 0 else 0.0

    lines.append("RF 4-Class Classification Report")
    lines.append("=" * 40)
    lines.append(f"accuracy: {accuracy:.4f}")
    lines.append("")

    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(cm[i, :].sum())

        lines.append(
            f"{label:10s} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f} "
            f"support={support}"
        )

    lines.append("")
    lines.append("confusion_matrix:")
    lines.append(str(cm))

    return "\n".join(lines)


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
