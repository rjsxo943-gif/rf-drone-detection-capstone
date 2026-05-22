from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core import get_block_size, load_all_configs
from src.features.spectrogram import compute_stft_branch
from src.ml import RF4Classifier
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver import build_receiver


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)

    if iq.ndim == 1:
        return iq.reshape(1, -1)

    if iq.ndim == 2:
        return iq

    raise ValueError(f"iq must be 1D or 2D, got shape={iq.shape}")


def _read_block(receiver: Any, block_size: int) -> np.ndarray:
    if hasattr(receiver, "read_block"):
        return _ensure_2d_iq(receiver.read_block(block_size))

    if hasattr(receiver, "read_samples"):
        return _ensure_2d_iq(receiver.read_samples(block_size))

    raise AttributeError("receiver has neither read_block nor read_samples")


def _close_receiver(receiver: Any) -> None:
    if hasattr(receiver, "close"):
        try:
            receiver.close()
        except Exception:
            pass


def _set_receiver_center_freq(receiver: Any, center_freq: int) -> None:
    center_freq = int(center_freq)

    if hasattr(receiver, "center_freq"):
        try:
            receiver.center_freq = center_freq
        except Exception:
            pass

    if hasattr(receiver, "sdr"):
        sdr = getattr(receiver, "sdr")
        if sdr is not None and hasattr(sdr, "rx_lo"):
            sdr.rx_lo = center_freq
            return

    if hasattr(receiver, "_set_center_freq"):
        receiver._set_center_freq(center_freq)
        return


def _get_block_size_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        _safe_get(configs, "receiver", "block_size"),
        _safe_get(configs, "receiver", "num_samples"),
        _safe_get(configs, "sdr", "block_size"),
        _safe_get(configs, "sdr", "num_samples"),
        configs.get("block_size") if isinstance(configs, dict) else None,
    ]:
        if value is not None:
            return int(value)

    try:
        return int(get_block_size(configs))
    except Exception:
        pass

    try:
        return int(get_block_size(configs.get("receiver", {})))
    except Exception:
        pass

    return 16_384


def _get_sample_rate_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        _safe_get(configs, "receiver", "sample_rate"),
        _safe_get(configs, "receiver", "sdr", "sample_rate"),
        _safe_get(configs, "sdr", "sample_rate"),
        configs.get("sample_rate") if isinstance(configs, dict) else None,
    ]:
        if value is not None:
            return int(value)

    return 5_000_000


def _build_runtime_receiver(configs: dict[str, Any]) -> Any:
    receiver_cfg = configs.get("receiver", configs)

    try:
        return build_receiver(receiver_cfg)
    except TypeError:
        return build_receiver(configs)


def _compute_rf4_spectrogram(
    iq_1d: np.ndarray,
    sample_rate: int,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
    window: str = "hann",
) -> np.ndarray:
    stft_out = compute_stft_branch(
        iq_1d,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
    )

    return stft_out.cnn_spectrogram.astype(np.float32)


def run_rf4_single_block_action(
    model_path: str = "outputs/ml/rf4_cnn_baseline_v2/best_model.pt",
    center_freq: int = 2_437_000_000,
    rx_index: int = 0,
    general_threshold: float = 0.50,
    drone_threshold: float = 0.70,
    warmup_reads: int = 3,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
    window: str = "hann",
) -> int:
    configs = load_all_configs()

    block_size = _get_block_size_from_configs(configs)
    sample_rate = _get_sample_rate_from_configs(configs)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"RF4 model not found: {model_path_obj}")

    classifier = RF4Classifier(
        checkpoint_path=model_path_obj,
        general_threshold=general_threshold,
        drone_threshold=drone_threshold,
    )

    receiver = _build_runtime_receiver(configs)

    try:
        _set_receiver_center_freq(receiver, center_freq)

        for _ in range(max(0, int(warmup_reads))):
            _read_block(receiver, block_size)

        iq = _read_block(receiver, block_size)
        iq = remove_dc_offset(iq, axis=-1)
        iq = _ensure_2d_iq(iq)

        if rx_index < 0 or rx_index >= iq.shape[0]:
            raise ValueError(f"rx_index out of range: {rx_index}, num_channels={iq.shape[0]}")

        iq_1d = iq[rx_index]

        spec = _compute_rf4_spectrogram(
            iq_1d=iq_1d,
            sample_rate=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
        )

        result = classifier.predict_array(spec)

        print()
        print("=== RF4 Runtime Inference Result ===")
        print(f"model             : {model_path_obj}")
        print(f"center_freq        : {center_freq} Hz ({center_freq / 1e6:.3f} MHz)")
        print(f"sample_rate        : {sample_rate}")
        print(f"block_size         : {block_size}")
        print(f"rx_index           : {rx_index}")
        print(f"spectrogram shape  : {spec.shape}")
        print(f"raw class          : {result.class_name}")
        print(f"confidence         : {result.confidence:.4f}")
        print(f"final class        : {result.final_class}")
        print(f"applied threshold  : {result.applied_threshold:.2f}")
        print()
        print("[probabilities]")
        for class_name, prob in result.probabilities.items():
            print(f"{class_name:10s}: {prob:.6f}")

        return 0

    finally:
        _close_receiver(receiver)
