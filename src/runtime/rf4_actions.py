from __future__ import annotations

import json
import time
from collections import Counter
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
    vmin: float = -40.0,
    vmax: float = 40.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    RF4 canonical01 spectrogram 생성.

    기존 WiFi/Bluetooth/Background 학습 데이터는 dB scale 기반이었고,
    canonical01 모델은 dB spectrogram을 fixed [-40, 40] 기준으로 0~1 변환해 학습했다.

    따라서 runtime에서도 compute_stft_branch()의 per-block min-max 결과를 쓰지 않고,
    동일하게 fixed dB range 기준 0~1 spectrogram을 만든다.
    """
    _ = sample_rate  # 현재 계산에는 직접 사용하지 않지만 인터페이스 유지용

    x = np.asarray(iq_1d).reshape(-1)

    if x.size < nperseg:
        raise ValueError(f"iq length {x.size} is shorter than nperseg {nperseg}")

    hop_size = nperseg - noverlap
    if hop_size <= 0:
        raise ValueError(f"invalid hop_size={hop_size}; nperseg={nperseg}, noverlap={noverlap}")

    if window == "hann":
        win = np.hanning(nperseg).astype(np.float32)
    elif window == "hamming":
        win = np.hamming(nperseg).astype(np.float32)
    else:
        win = np.ones(nperseg, dtype=np.float32)

    cols: list[np.ndarray] = []
    for start in range(0, x.size - nperseg + 1, hop_size):
        frame = x[start:start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        # FFT magnitude는 window 길이/합에 비례해서 커지므로,
        # 기존 dB spectrogram 스케일과 맞추기 위해 window coherent gain으로 정규화한다.
        mag = np.abs(spec) / (float(np.sum(win)) + eps)
        cols.append(mag.astype(np.float32))

    mag_spec = np.stack(cols, axis=1)

    db_spec = 20.0 * np.log10(mag_spec + eps)
    db_spec = np.clip(db_spec, vmin, vmax)

    norm_spec = (db_spec - vmin) / (vmax - vmin)
    return np.clip(norm_spec, 0.0, 1.0).astype(np.float32)


def _frame_signal_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)

    if x.size < frame_size:
        return x.reshape(1, -1)

    starts = range(0, x.size - frame_size + 1, hop_size)
    frames = [x[start : start + frame_size] for start in starts]
    return np.stack(frames, axis=0)


def _compute_frame_energies(
    iq_1d: np.ndarray,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> np.ndarray:
    frames = _frame_signal_1d(iq_1d, frame_size=frame_size, hop_size=hop_size)

    if frames.shape[-1] == frame_size:
        win = np.hanning(frame_size).astype(np.float32)
        frames = frames * win

    energies = np.mean(np.abs(frames) ** 2, axis=-1)
    return energies.astype(np.float32)


def _load_noise_threshold(
    path: str = "outputs/calibration/noise_latest.json",
    default: float = 0.0,
) -> float:
    p = Path(path)

    if not p.exists():
        return float(default)

    data = json.loads(p.read_text(encoding="utf-8"))

    for key in ["threshold", "noise_threshold"]:
        if key in data:
            return float(data[key])

    return float(default)


def _compute_detection_ratio(frame_energies: np.ndarray, threshold: float) -> float:
    frame_energies = np.asarray(frame_energies)

    if frame_energies.size == 0:
        return 0.0

    return float(np.mean(frame_energies >= float(threshold)))


def _empty_prob_dict(class_names: list[str]) -> dict[str, float]:
    return {name: 0.0 for name in class_names}


def run_rf4_single_block_action(
    model_path: str = "outputs/ml/rf4_cnn_live2450_v2/best_model.pt",
    center_freq: int = 2_437_000_000,
    rx_index: int = 0,
    general_threshold: float = 0.50,
    drone_threshold: float = 0.70,
    warmup_reads: int = 3,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
    window: str = "hann",
    require_signal_gate: bool = True,
    gate_frame_size: int = 1024,
    gate_hop_size: int = 512,
    min_detection_ratio: float = 0.01,
    num_blocks: int = 10,
    min_drone_votes: int = 3,
    block_delay_sec: float = 0.02,
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

    block_results: list[dict[str, Any]] = []
    class_names = list(classifier.class_names)

    try:
        _set_receiver_center_freq(receiver, center_freq)

        for _ in range(max(0, int(warmup_reads))):
            _read_block(receiver, block_size)

        print()
        print("=== RF4 Multi-Block Runtime Inference ===")
        print(f"model             : {model_path_obj}")
        print(f"center_freq        : {center_freq} Hz ({center_freq / 1e6:.3f} MHz)")
        print(f"sample_rate        : {sample_rate}")
        print(f"block_size         : {block_size}")
        print(f"rx_index           : {rx_index}")
        print(f"num_blocks         : {num_blocks}")
        print(f"min_drone_votes    : {min_drone_votes}")
        print(f"general_threshold  : {general_threshold:.2f}")
        print(f"drone_threshold    : {drone_threshold:.2f}")
        print(f"min_signal_ratio   : {min_detection_ratio:.4f}")
        print()

        noise_threshold = _load_noise_threshold()

        for block_idx in range(1, int(num_blocks) + 1):
            iq = _read_block(receiver, block_size)
            iq = remove_dc_offset(iq, axis=-1)
            iq = _ensure_2d_iq(iq)

            if rx_index < 0 or rx_index >= iq.shape[0]:
                raise ValueError(
                    f"rx_index out of range: {rx_index}, num_channels={iq.shape[0]}"
                )

            iq_1d = iq[rx_index]

            frame_energies = _compute_frame_energies(
                iq_1d,
                frame_size=gate_frame_size,
                hop_size=gate_hop_size,
            )
            detection_ratio = _compute_detection_ratio(
                frame_energies,
                threshold=noise_threshold,
            )

            energy_median = float(np.median(frame_energies))
            energy_p95 = float(np.percentile(frame_energies, 95))
            energy_max = float(np.max(frame_energies))

            if require_signal_gate and detection_ratio < min_detection_ratio:
                result_row = {
                    "block": block_idx,
                    "gate": "NO_SIGNAL",
                    "raw_class": "NoSignal",
                    "final_class": "NoSignal",
                    "confidence": 0.0,
                    "detection_ratio": detection_ratio,
                    "probabilities": _empty_prob_dict(class_names),
                }

                block_results.append(result_row)

                print(
                    f"[{block_idx:02d}/{num_blocks}] "
                    f"gate=NO_SIGNAL "
                    f"ratio={detection_ratio:.4f} "
                    f"emax={energy_max:.4g} "
                    f"p95={energy_p95:.4g}"
                )

                time.sleep(block_delay_sec)
                continue

            spec = _compute_rf4_spectrogram(
                iq_1d=iq_1d,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                window=window,
            )

            result = classifier.predict_array(spec)

            result_row = {
                "block": block_idx,
                "gate": "SIGNAL",
                "raw_class": result.class_name,
                "final_class": result.final_class,
                "confidence": result.confidence,
                "detection_ratio": detection_ratio,
                "probabilities": result.probabilities,
            }

            block_results.append(result_row)

            print(
                f"[{block_idx:02d}/{num_blocks}] "
                f"gate=SIGNAL "
                f"ratio={detection_ratio:.4f} "
                f"raw={result.class_name:10s} "
                f"final={result.final_class:10s} "
                f"conf={result.confidence:.4f}"
            )

            time.sleep(block_delay_sec)

        final_classes = [row["final_class"] for row in block_results]
        raw_classes = [row["raw_class"] for row in block_results]

        final_counts = Counter(final_classes)
        raw_counts = Counter(raw_classes)

        drone_votes = final_counts.get("Drone-like", 0)
        signal_votes = sum(1 for row in block_results if row["gate"] == "SIGNAL")
        no_signal_votes = final_counts.get("NoSignal", 0)

        avg_probs = {name: 0.0 for name in class_names}
        signal_rows = [row for row in block_results if row["gate"] == "SIGNAL"]

        if signal_rows:
            for row in signal_rows:
                for name in class_names:
                    avg_probs[name] += float(row["probabilities"].get(name, 0.0))

            for name in class_names:
                avg_probs[name] /= len(signal_rows)

        if drone_votes >= min_drone_votes:
            final_decision = "Drone-like"
            decision_reason = f"drone_votes >= min_drone_votes ({drone_votes} >= {min_drone_votes})"
        elif signal_votes == 0:
            final_decision = "Background"
            decision_reason = "no signal blocks"
        else:
            non_nosignal_counts = Counter(
                cls for cls in final_classes if cls != "NoSignal"
            )
            if non_nosignal_counts:
                final_decision = non_nosignal_counts.most_common(1)[0][0]
                decision_reason = "majority vote among signal blocks"
            else:
                final_decision = "Background"
                decision_reason = "no confident signal class"

        print()
        print("=== RF4 Multi-Block Summary ===")
        print(f"signal blocks     : {signal_votes}/{num_blocks}")
        print(f"no signal blocks  : {no_signal_votes}/{num_blocks}")
        print(f"drone votes       : {drone_votes}/{num_blocks}")
        print(f"final decision    : {final_decision}")
        print(f"reason            : {decision_reason}")

        print()
        print("[final class counts]")
        for name, count in final_counts.most_common():
            print(f"{name:10s}: {count}")

        print()
        print("[raw class counts]")
        for name, count in raw_counts.most_common():
            print(f"{name:10s}: {count}")

        print()
        print("[average probabilities over SIGNAL blocks]")
        if signal_rows:
            for name, prob in avg_probs.items():
                print(f"{name:10s}: {prob:.6f}")
        else:
            print("no signal blocks")

        return 0

    finally:
        _close_receiver(receiver)
