from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso(timespec: str = "seconds") -> str:
    return datetime.now().isoformat(timespec=timespec)


def format_block_filename(
    block_index: int,
    prefix: str = "block",
    suffix: str = ".npz",
) -> str:
    if not suffix.startswith("."):
        suffix = "." + suffix

    return f"{prefix}_{int(block_index):06d}{suffix}"


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)

    if not suffix.startswith("."):
        suffix = "." + suffix

    if path.suffix != suffix:
        path = path.with_suffix(suffix)

    return path


def to_complex64_1d(
    x: np.ndarray,
    name: str = "array",
    require_complex: bool = True,
) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D array. got shape={x.shape}")

    if require_complex and not np.iscomplexobj(x):
        raise ValueError(f"{name} must be complex IQ array. got dtype={x.dtype}")

    return x.astype(np.complex64, copy=False)


def to_float32_array(x: np.ndarray, name: str = "array") -> np.ndarray:
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError(f"{name} is empty.")

    return x.astype(np.float32, copy=False)


def check_same_shape(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have same shape. "
            f"got {name_a}={a.shape}, {name_b}={b.shape}"
        )


def check_non_empty_array(x: np.ndarray, name: str = "array") -> None:
    if np.asarray(x).size == 0:
        raise ValueError(f"{name} is empty.")


def get_sample_range(block_index: int, block_size: int) -> tuple[int, int]:
    block_index = int(block_index)
    block_size = int(block_size)

    if block_index < 0:
        raise ValueError(f"block_index must be non-negative. got {block_index}")

    if block_size <= 0:
        raise ValueError(f"block_size must be positive. got {block_size}")

    sample_start = block_index * block_size
    sample_end = sample_start + block_size

    return sample_start, sample_end


def dumps_json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def loads_json(data: str | bytes | np.ndarray) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        data = str(data.item())

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    if data == "":
        return {}

    loaded = json.loads(str(data))

    if not isinstance(loaded, dict):
        raise ValueError("JSON data must represent a dict.")

    return loaded


def save_json(path: str | Path, data: dict[str, Any], indent: int = 2) -> Path:
    path = ensure_suffix(path, ".json")
    ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be dict. got {type(data).__name__}")

    return data