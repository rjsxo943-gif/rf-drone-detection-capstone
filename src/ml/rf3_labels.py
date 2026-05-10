from __future__ import annotations

LABEL_TO_ID = {
    "Background": 0,
    "Bluetooth": 1,
    "WiFi": 2,
}

ID_TO_LABEL = {
    0: "Background",
    1: "Bluetooth",
    2: "WiFi",
}


def label_to_id(label: str) -> int:
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_TO_ID[label]


def id_to_label(idx: int) -> str:
    if idx not in ID_TO_LABEL:
        raise ValueError(f"Unknown label id: {idx}")
    return ID_TO_LABEL[idx]


def num_rf3_classes() -> int:
    return len(LABEL_TO_ID)

