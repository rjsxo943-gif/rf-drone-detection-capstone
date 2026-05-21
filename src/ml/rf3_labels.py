from __future__ import annotations

"""
RF label definitions.

주의:
- 파일명은 기존 호환성 때문에 rf3_labels.py로 유지한다.
- 현재 프로젝트 기준 라벨은 RF4이다.
- 클래스 순서는 학습/추론/Confusion Matrix에서 반드시 동일해야 한다.
"""

RF3_CLASS_NAMES = [
    "Background",
    "WiFi",
    "Bluetooth",
    "Drone-like",
]

# 호환용 alias
DEFAULT_CLASS_NAMES = RF3_CLASS_NAMES
CLASS_NAMES = RF3_CLASS_NAMES

LABEL_TO_ID = {label: idx for idx, label in enumerate(RF3_CLASS_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def num_rf3_classes() -> int:
    return len(RF3_CLASS_NAMES)


def label_to_id(label: str) -> int:
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_TO_ID[label]


def id_to_label(class_id: int) -> str:
    if class_id not in ID_TO_LABEL:
        raise ValueError(f"Unknown class_id: {class_id}")
    return ID_TO_LABEL[class_id]


def get_class_names() -> list[str]:
    return list(RF3_CLASS_NAMES)
