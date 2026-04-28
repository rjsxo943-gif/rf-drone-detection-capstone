#src/ml/dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np


SUPPORTED_EXTENSIONS = {".npy", ".npz"}


def list_spectrogram_files(root: str | Path) -> list[Path]:
    """
    root 아래의 .npy / .npz spectrogram 파일을 모두 찾는다.

    예:
    data/processed/Background/*.npy
    data/processed/WiFi/*.npz
    """
    root = Path(root)

    if not root.exists():
        return []

    files: list[Path] = []

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))

    return sorted(files)


def load_spectrogram(path: str | Path, key: str = "spectrogram") -> np.ndarray:
    """
    spectrogram 파일을 읽는다.

    지원:
    - .npy: 배열 자체 저장
    - .npz: 기본 key='spectrogram' 또는 'cnn_spectrogram' 사용

    반환:
    - shape 예: (512, 125) 또는 (512, 125, 1)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Spectrogram file not found: {path}")

    if path.suffix == ".npy":
        arr = np.load(path)

    elif path.suffix == ".npz":
        data = np.load(path)

        if key in data:
            arr = data[key]
        elif "cnn_spectrogram" in data:
            arr = data["cnn_spectrogram"]
        else:
            raise KeyError(
                f"No spectrogram key found in {path}. "
                f"Expected '{key}' or 'cnn_spectrogram'. "
                f"Available keys: {list(data.keys())}"
            )

    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    if arr.ndim != 3:
        raise ValueError(
            f"Spectrogram must have shape (H, W) or (H, W, C), got {arr.shape}"
        )

    return arr.astype(np.float32, copy=False)


def infer_label_from_parent(
    path: str | Path,
    class_names: list[str],
) -> int:
    """
    파일의 부모 폴더 이름으로 class label을 추정한다.

    예:
    data/processed/WiFi/block_000001.npz
    → parent name = WiFi
    → label index = class_names.index("WiFi")
    """
    path = Path(path)
    label_name = path.parent.name

    if label_name not in class_names:
        raise ValueError(
            f"Cannot infer label from parent folder '{label_name}'. "
            f"Expected one of {class_names}"
        )

    return class_names.index(label_name)


def load_dataset_from_folders(
    root: str | Path,
    class_names: list[str],
    expected_shape: tuple[int, int, int] = (512, 125, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """
    클래스별 폴더 구조에서 CNN 학습용 X, y를 만든다.

    예:
    root/
    ├── Background/
    ├── WiFi/
    ├── Bluetooth/
    └── Drone-like/

    Returns:
        X: shape = (N, 512, 125, 1)
        y: shape = (N,)
    """
    files = list_spectrogram_files(root)

    if len(files) == 0:
        raise FileNotFoundError(f"No spectrogram files found under: {root}")

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for path in files:
        spectrogram = load_spectrogram(path)

        if spectrogram.shape != expected_shape:
            raise ValueError(
                f"Unexpected spectrogram shape in {path}: {spectrogram.shape}. "
                f"Expected {expected_shape}."
            )

        label = infer_label_from_parent(path, class_names)

        x_list.append(spectrogram)
        y_list.append(label)

    X = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    간단한 train/validation split.

    Returns:
        X_train, y_train, X_val, y_val
    """
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    num_samples = len(X)

    if num_samples != len(y):
        raise ValueError("X and y must have the same length.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(num_samples * validation_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]