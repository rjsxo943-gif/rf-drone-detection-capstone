from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import (
    check_same_shape,
    dumps_json,
    format_block_filename,
    loads_json,
    now_iso,
    now_string,
    save_json,
    to_complex64_1d,
)


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
    session_id = session_name or f"{label}_{now_string()}"

    session_dir = root_dir / label / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_meta = {
        "session_id": session_id,
        "label": label,
        "created_at": now_iso(timespec="seconds"),
        "description": "Raw complex IQ capture session",
    }

    if metadata:
        session_meta.update(metadata)

    save_json(session_dir / "session_meta.json", session_meta)

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

    block_index = int(block_index)

    if block_index < 0:
        raise ValueError(f"block_index must be non-negative. got {block_index}")

    rx0_iq = to_complex64_1d(rx0_iq, "rx0_iq")
    rx1_iq = to_complex64_1d(rx1_iq, "rx1_iq")

    check_same_shape(rx0_iq, rx1_iq, "rx0_iq", "rx1_iq")

    metadata_json = dumps_json(metadata)

    output_path = session_dir / format_block_filename(block_index)

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
        timestamp=np.array(now_iso(timespec="milliseconds")),
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

    with np.load(input_path, allow_pickle=False) as data:
        metadata = loads_json(data["metadata_json"])

        return {
            "rx0_iq": data["rx0_iq"].astype(np.complex64, copy=False),
            "rx1_iq": data["rx1_iq"].astype(np.complex64, copy=False),
            "block_index": int(data["block_index"]),
            "sample_rate": float(data["sample_rate"]),
            "center_freq": float(data["center_freq"]),
            "label": str(data["label"].item()),
            "timestamp": str(data["timestamp"].item()),
            "metadata": metadata,
        }