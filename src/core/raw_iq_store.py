from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_complex64(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D complex IQ array. got shape={x.shape}")

    return x.astype(np.complex64)


def create_raw_iq_session(
    root_dir: str | Path,
    label: str,
    session_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    raw IQ 저장용 실험 세션 폴더를 만든다.

    예:
    data/raw_iq/pluto/drone/drone_20260426_153000/
    """

    root_dir = Path(root_dir)
    session_id = session_name or f"{label}_{_now_string()}"

    session_dir = root_dir / label / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_meta = {
        "session_id": session_id,
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Raw complex IQ capture session",
    }

    if metadata:
        session_meta.update(metadata)

    meta_path = session_dir / "session_meta.json"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=2, ensure_ascii=False)

    return session_dir


def save_raw_iq_block(
    session_dir: str | Path,
    block_index: int,
    rx0_iq: np.ndarray,
    rx1_iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    label: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    RX0/RX1 raw complex IQ block만 저장한다.

    저장 파일:
    block_000000.npz

    저장 내용:
    - rx0_iq
    - rx1_iq
    - sample_rate
    - center_freq
    - label
    - block_index
    - timestamp
    - metadata_json

    주의:
    STFT, spectrogram, phase는 저장하지 않는다.
    나중에 raw IQ에서 다시 생성한다.
    """

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    rx0_iq = _to_complex64(rx0_iq, "rx0_iq")
    rx1_iq = _to_complex64(rx1_iq, "rx1_iq")

    if rx0_iq.shape != rx1_iq.shape:
        raise ValueError(
            f"rx0_iq and rx1_iq must have same shape. "
            f"got rx0={rx0_iq.shape}, rx1={rx1_iq.shape}"
        )

    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    output_path = session_dir / f"block_{block_index:06d}.npz"

    # raw IQ는 압축해도 용량이 크게 줄지 않고 저장 속도만 느려질 수 있어서
    # np.savez_compressed 대신 np.savez 사용
    np.savez(
        output_path,
        rx0_iq=rx0_iq,
        rx1_iq=rx1_iq,
        block_index=np.array(block_index, dtype=np.int64),
        sample_rate=np.array(sample_rate, dtype=np.float64),
        center_freq=np.array(center_freq, dtype=np.float64),
        label=np.array(label),
        timestamp=np.array(datetime.now().isoformat(timespec="milliseconds")),
        metadata_json=np.array(metadata_json),
    )

    return output_path


def load_raw_iq_block(input_path: str | Path) -> dict[str, Any]:
    """
    저장된 raw IQ block을 다시 불러온다.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    data = np.load(input_path, allow_pickle=False)

    metadata_json = str(data["metadata_json"].item())

    return {
        "rx0_iq": data["rx0_iq"],
        "rx1_iq": data["rx1_iq"],
        "block_index": int(data["block_index"]),
        "sample_rate": float(data["sample_rate"]),
        "center_freq": float(data["center_freq"]),
        "label": str(data["label"].item()),
        "timestamp": str(data["timestamp"].item()),
        "metadata": json.loads(metadata_json),
    }