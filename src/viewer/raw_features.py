from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"iq must be 1-D or 2-D complex array. got shape={arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    return arr.astype(np.complex64, copy=False)


def _frame_power_p99(iq_1d: np.ndarray, frame_size: int, hop_size: int) -> float:
    if iq_1d.size < frame_size:
        return float(np.mean(np.abs(iq_1d) ** 2))

    starts = range(0, iq_1d.size - frame_size + 1, hop_size)
    powers = [float(np.mean(np.abs(iq_1d[start:start + frame_size]) ** 2)) for start in starts]
    if not powers:
        return float(np.mean(np.abs(iq_1d) ** 2))
    return float(np.percentile(np.asarray(powers, dtype=np.float32), 99))


def compute_raw_features(
    iq: np.ndarray,
    *,
    frame_size: int = 512,
    hop_size: int = 512,
    overload_abs_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute gain/distance features from raw or minimally processed IQ.

    Do not feed normalized CNN spectrograms into this function. The values are
    intended to preserve gain and distance information before the CNN branch
    applies spectrogram normalization.
    """

    iq_2d = _ensure_2d_iq(iq)
    flat = iq_2d.reshape(-1)
    abs_iq = np.abs(flat).astype(np.float32)

    raw_abs_max = float(np.max(abs_iq)) if abs_iq.size else 0.0
    overloaded = bool(
        overload_abs_threshold is not None and raw_abs_max >= float(overload_abs_threshold)
    )

    # Frame power is computed from the mean signal across available channels.
    iq_for_power = np.mean(iq_2d, axis=0).astype(np.complex64, copy=False)

    return {
        "num_channels": int(iq_2d.shape[0]),
        "num_samples": int(iq_2d.shape[1]),
        "raw_abs_mean": float(np.mean(abs_iq)) if abs_iq.size else 0.0,
        "raw_abs_p50": float(np.percentile(abs_iq, 50)) if abs_iq.size else 0.0,
        "raw_abs_p95": float(np.percentile(abs_iq, 95)) if abs_iq.size else 0.0,
        "raw_abs_p99": float(np.percentile(abs_iq, 99)) if abs_iq.size else 0.0,
        "raw_abs_max": raw_abs_max,
        "raw_rms": float(np.sqrt(np.mean(abs_iq ** 2))) if abs_iq.size else 0.0,
        "frame_power_p99": _frame_power_p99(
            iq_for_power,
            frame_size=int(frame_size),
            hop_size=int(hop_size),
        ),
        "overloaded": overloaded,
        "overload_abs_threshold": overload_abs_threshold,
    }
