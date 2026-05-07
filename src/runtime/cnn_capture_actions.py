# src/runtime/cnn_capture_actions.py
from __future__ import annotations

import json
import math
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.calibration import load_calibration_params
from src.core import get_block_size, load_all_configs
from src.preprocess import get_cnn_input_iq, normalize_iq, remove_dc_offset
from src.receiver import build_receiver
from src.runtime.calibration_actions import (
    DEFAULT_NOISE_OUTPUT,
    DEFAULT_PHASE_GAIN_OUTPUT,
)


@dataclass
class CaptureConfig:
    label: str
    max_saved: int = 50
    rx_index: int = 0
    save_raw_iq: bool = False
    require_noise: bool = True
    require_phase_gain: bool = False
    stop_key: str = "q"
    verbose: bool = True


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _sanitize_label(label: str) -> str:
    label = label.strip().lower().replace("-", "_").replace(" ", "_")
    if not label:
        raise ValueError("label must not be empty")

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    cleaned = "".join(ch for ch in label if ch in allowed)

    if not cleaned:
        raise ValueError(f"invalid label: {label!r}")

    return cleaned


def _now_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)

    if iq.ndim == 1:
        return iq.reshape(1, -1)

    if iq.ndim == 2:
        return iq

    raise ValueError(f"iq must be 1D or 2D, got shape={iq.shape}")


def _build_scan_freqs_from_config(scan_cfg: dict[str, Any]) -> list[int]:
    band_start = int(scan_cfg.get("band_start", 2_400_000_000))
    band_end = int(scan_cfg.get("band_end", 2_485_000_000))
    channel_bw = int(scan_cfg.get("channel_bw", 5_000_000))

    if band_end < band_start:
        raise ValueError(f"band_end must be >= band_start: {band_start}, {band_end}")

    if channel_bw <= 0:
        raise ValueError(f"channel_bw must be positive: {channel_bw}")

    freqs = list(range(band_start, band_end + 1, channel_bw))

    if not freqs:
        raise ValueError("scan frequency list is empty")

    return freqs


def _set_receiver_center_freq(receiver: Any, center_freq: int) -> None:
    """
    PlutoReceiver / SimReceiver / RawFileReceiver 구조 차이를 최대한 흡수하기 위한 setter.

    PlutoReceiver는 내부에 self.sdr.rx_lo를 가질 가능성이 높고,
    SimReceiver / RawFileReceiver는 center_freq 속성만 바뀌어도 metadata 용도로 충분하다.
    """
    center_freq = int(center_freq)

    if hasattr(receiver, "center_freq"):
        try:
            receiver.center_freq = center_freq
        except Exception:
            pass

    if hasattr(receiver, "sdr"):
        sdr = getattr(receiver, "sdr")
        if sdr is not None:
            if hasattr(sdr, "rx_lo"):
                sdr.rx_lo = center_freq
                return

    if hasattr(receiver, "_set_center_freq"):
        receiver._set_center_freq(center_freq)
        return


def _read_block(receiver: Any, block_size: int) -> np.ndarray:
    if hasattr(receiver, "read_block"):
        return _ensure_2d_iq(receiver.read_block(block_size))

    if hasattr(receiver, "read_samples"):
        return _ensure_2d_iq(receiver.read_samples(block_size))

    raise AttributeError("receiver has neither read_block nor read_samples")


def _frame_signal_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}")

    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}")

    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    if x.size < frame_size:
        return x.reshape(1, -1)

    starts = range(0, x.size - frame_size + 1, hop_size)
    frames = [x[start : start + frame_size] for start in starts]
    return np.stack(frames, axis=0)


def _compute_frame_energies(
    iq_dc: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: str = "hann",
) -> np.ndarray:
    """
    후보 판단용 frame energy 계산.

    주의:
    - 입력 iq_dc는 이미 block 단위 DC offset 제거가 끝난 데이터여야 한다.
    - 여러 RX 채널이 있으면 채널별 frame energy를 평균낸다.
    """
    iq_dc = _ensure_2d_iq(iq_dc)

    all_ch_energies: list[np.ndarray] = []

    for ch in range(iq_dc.shape[0]):
        frames = _frame_signal_1d(iq_dc[ch], frame_size=frame_size, hop_size=hop_size)

        if window == "hann" and frames.shape[-1] == frame_size:
            win = np.hanning(frame_size).astype(np.float32)
            frames = frames * win
        elif window == "hamming" and frames.shape[-1] == frame_size:
            win = np.hamming(frame_size).astype(np.float32)
            frames = frames * win

        energies = np.mean(np.abs(frames) ** 2, axis=-1)
        all_ch_energies.append(energies.astype(np.float32))

    min_len = min(e.size for e in all_ch_energies)
    stacked = np.stack([e[:min_len] for e in all_ch_energies], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _compute_detection_ratio(frame_energies: np.ndarray, threshold: float) -> float:
    frame_energies = np.asarray(frame_energies)

    if frame_energies.size == 0:
        return 0.0

    detections = frame_energies >= float(threshold)
    return float(np.mean(detections))


def _compute_fft_score_db(iq_dc: np.ndarray, eps: float = 1e-12) -> float:
    """
    사람이 보기 위한 scan score.
    후보 판단 자체는 calibration threshold 기반 detection_ratio를 우선 사용한다.
    """
    iq_dc = _ensure_2d_iq(iq_dc)

    scores: list[float] = []

    for ch in range(iq_dc.shape[0]):
        x = iq_dc[ch]
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft(x)))
        power = float(np.max(fft_mag ** 2))
        score_db = 10.0 * math.log10(power + eps)
        scores.append(score_db)

    return float(np.max(scores))


def _compute_cnn_spectrogram_numpy(
    iq_1d: np.ndarray,
    nperseg: int = 512,
    hop_size: int = 128,
    nfft: int = 512,
    window: str = "hann",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    CNN 입력 저장용 spectrogram 생성.

    출력:
    - shape: (nfft, num_frames)
    - 값: log magnitude 기반 normalize된 float32
    """
    x = np.asarray(iq_1d).reshape(-1)

    if x.size < nperseg:
        raise ValueError(f"iq length {x.size} is shorter than nperseg {nperseg}")

    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    if window == "hann":
        win = np.hanning(nperseg).astype(np.float32)
    elif window == "hamming":
        win = np.hamming(nperseg).astype(np.float32)
    else:
        win = np.ones(nperseg, dtype=np.float32)

    starts = range(0, x.size - nperseg + 1, hop_size)
    cols: list[np.ndarray] = []

    for start in starts:
        frame = x[start : start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        mag = np.abs(spec)
        cols.append(mag.astype(np.float32))

    spectrogram = np.stack(cols, axis=1)

    log_spec = 20.0 * np.log10(spectrogram + eps)
    spec_min = float(np.min(log_spec))
    spec_max = float(np.max(log_spec))

    if spec_max - spec_min < eps:
        norm_spec = np.zeros_like(log_spec, dtype=np.float32)
    else:
        norm_spec = (log_spec - spec_min) / (spec_max - spec_min)

    return norm_spec.astype(np.float32)


def _stop_requested(stop_key: str) -> bool:
    """
    scan cycle 사이에 q + Enter 입력을 받기 위한 non-blocking check.
    WSL/Linux 터미널 기준으로 동작한다.
    """
    if not stop_key:
        return False

    try:
        if not sys.stdin.isatty():
            return False

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)

        if not ready:
            return False

        line = sys.stdin.readline().strip().lower()
        return line == stop_key.lower()

    except Exception:
        return False


def _append_metadata(metadata_path: Path, item: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _save_cnn_sample(
    output_dir: Path,
    sample_index: int,
    spectrogram: np.ndarray,
    cnn_input: np.ndarray,
    metadata: dict[str, Any],
    raw_iq: np.ndarray | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"sample_{sample_index:06d}.npz"
    sample_path = output_dir / filename

    save_kwargs: dict[str, Any] = {
        "spectrogram": spectrogram.astype(np.float32),
        "cnn_input": cnn_input.astype(np.float32),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }

    if raw_iq is not None:
        save_kwargs["raw_iq"] = raw_iq.astype(np.complex64)

    np.savez_compressed(sample_path, **save_kwargs)

    metadata_item = dict(metadata)
    metadata_item["file"] = filename
    _append_metadata(output_dir / "metadata.jsonl", metadata_item)

    return sample_path


def run_cnn_capture_action(
    label: str,
    max_saved: int = 50,
    rx_index: int = 0,
    save_raw_iq: bool = False,
    require_noise: bool = True,
    require_phase_gain: bool = False,
    stop_key: str = "q",
    cycle_delay_sec: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    CNN 학습 데이터 수집 모드.

    흐름:
    1. configs 로드
    2. noise calibration 로드
    3. receiver 생성
    4. 2.4 GHz 대역 scan
    5. 후보 주파수 발견
    6. 해당 주파수에서 precision block 수집
    7. 매 block마다 DC offset 제거
    8. RX 채널 선택 / IQ normalize / STFT spectrogram 생성
    9. label별 npz 저장
    """
    capture_cfg = CaptureConfig(
        label=_sanitize_label(label),
        max_saved=int(max_saved),
        rx_index=int(rx_index),
        save_raw_iq=bool(save_raw_iq),
        require_noise=bool(require_noise),
        require_phase_gain=bool(require_phase_gain),
        stop_key=stop_key,
        verbose=bool(verbose),
    )

    if capture_cfg.max_saved <= 0:
        raise ValueError(f"max_saved must be positive, got {capture_cfg.max_saved}")

    configs = load_all_configs("configs")

    receiver_cfg = configs.get("receiver", {})
    detect_cfg = configs.get("detect", {})
    scan_cfg = configs.get("scan", {})
    ml_cfg = configs.get("ml", {})

    block_size = int(get_block_size(configs))
    sample_rate = int(receiver_cfg.get("sample_rate", 5_000_000))

    energy_cfg = detect_cfg.get("energy_detector", {})
    frame_size = int(energy_cfg.get("frame_size", 1024))
    hop_size_energy = int(energy_cfg.get("hop_size", 512))
    energy_window = str(energy_cfg.get("window", "hann"))
    min_detection_ratio = float(energy_cfg.get("min_detection_ratio", 0.05))

    stft_cfg = ml_cfg.get("stft", {})
    nperseg = int(stft_cfg.get("nperseg", ml_cfg.get("nperseg", 512)))
    nfft = int(stft_cfg.get("nfft", ml_cfg.get("nfft", 512)))

    if "hop_size" in stft_cfg:
        hop_size_stft = int(stft_cfg["hop_size"])
    elif "hop_size" in ml_cfg:
        hop_size_stft = int(ml_cfg["hop_size"])
    elif "noverlap" in stft_cfg:
        hop_size_stft = nperseg - int(stft_cfg["noverlap"])
    elif "noverlap" in ml_cfg:
        hop_size_stft = nperseg - int(ml_cfg["noverlap"])
    else:
        hop_size_stft = 128

    stft_window = str(stft_cfg.get("window", ml_cfg.get("window", "hann")))

    scan_freqs = _build_scan_freqs_from_config(scan_cfg)
    scan_blocks = int(scan_cfg.get("scan_blocks", 3))
    min_pass_blocks = int(scan_cfg.get("min_pass_blocks", 1))
    precision_blocks_per_candidate = int(scan_cfg.get("precision_blocks_per_candidate", 1))
    settle_sec = float(scan_cfg.get("settle_sec", 0.02))

    calib = load_calibration_params(
        noise_path=Path(DEFAULT_NOISE_OUTPUT),
        phase_gain_path=Path(DEFAULT_PHASE_GAIN_OUTPUT),
        require_noise=capture_cfg.require_noise,
        require_phase_gain=capture_cfg.require_phase_gain,
    )

    if calib.noise is None:
        raise FileNotFoundError(
            "noise calibration result not found. Run noise calibration first."
        )

    noise_threshold = float(calib.noise.threshold)

    session_id = _now_session_id()
    output_dir = (
        Path("data")
        / "processed"
        / "cnn_capture"
        / capture_cfg.label
        / session_id
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    session_info = {
        "session_id": session_id,
        "label": capture_cfg.label,
        "max_saved": capture_cfg.max_saved,
        "rx_index": capture_cfg.rx_index,
        "save_raw_iq": capture_cfg.save_raw_iq,
        "sample_rate": sample_rate,
        "block_size": block_size,
        "frame_size": frame_size,
        "energy_hop_size": hop_size_energy,
        "energy_window": energy_window,
        "noise_threshold": noise_threshold,
        "min_detection_ratio": min_detection_ratio,
        "nperseg": nperseg,
        "stft_hop_size": hop_size_stft,
        "nfft": nfft,
        "stft_window": stft_window,
        "scan_freqs": scan_freqs,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with (output_dir / "session.json").open("w", encoding="utf-8") as f:
        json.dump(session_info, f, ensure_ascii=False, indent=2)

    if verbose:
        print()
        print("=== CNN Dataset Capture ===")
        print(f"label        : {capture_cfg.label}")
        print(f"output_dir   : {output_dir}")
        print(f"max_saved    : {capture_cfg.max_saved}")
        print(f"rx_index     : {capture_cfg.rx_index}")
        print(f"save_raw_iq  : {capture_cfg.save_raw_iq}")
        print(f"sample_rate  : {sample_rate}")
        print(f"block_size   : {block_size}")
        print(f"threshold    : {noise_threshold:.10g}")
        print(f"scan freqs   : {len(scan_freqs)} freqs")
        print()
        print(f"scan cycle 사이에 {stop_key} 입력 후 Enter를 누르면 중단한다.")
        print()

    receiver = build_receiver(receiver_cfg)

    saved_count = 0
    cycle_index = 0
    stop_requested = False

    try:
        while saved_count < capture_cfg.max_saved and not stop_requested:
            if _stop_requested(capture_cfg.stop_key):
                if verbose:
                    print("[STOP] user requested stop")
                stop_requested = True
                break

            cycle_index += 1

            if verbose:
                print(f"--- scan cycle {cycle_index} ---")

            for center_freq in scan_freqs:
                if saved_count >= capture_cfg.max_saved or stop_requested:
                    break

                if _stop_requested(capture_cfg.stop_key):
                    if verbose:
                        print("[STOP] user requested stop")
                    stop_requested = True
                    break

                _set_receiver_center_freq(receiver, center_freq)

                if settle_sec > 0:
                    time.sleep(settle_sec)

                pass_count = 0
                last_scan_score_db = float("nan")
                last_detection_ratio = 0.0

                for scan_block_idx in range(scan_blocks):
                    if _stop_requested(capture_cfg.stop_key):
                        if verbose:
                            print("[STOP] user requested stop")
                        stop_requested = True
                        break

                    iq_raw = _read_block(receiver, block_size)

                    # 중요:
                    # scan 후보 판단에도 block마다 DC offset 제거를 먼저 적용한다.
                    iq_dc = remove_dc_offset(iq_raw, axis=-1)

                    frame_energies = _compute_frame_energies(
                        iq_dc,
                        frame_size=frame_size,
                        hop_size=hop_size_energy,
                        window=energy_window,
                    )

                    detection_ratio = _compute_detection_ratio(
                        frame_energies,
                        threshold=noise_threshold,
                    )

                    scan_score_db = _compute_fft_score_db(iq_dc)

                    last_scan_score_db = scan_score_db
                    last_detection_ratio = detection_ratio

                    if detection_ratio >= min_detection_ratio:
                        pass_count += 1

                    if verbose:
                        print(
                            f"[scan] f={center_freq / 1e9:.4f} GHz "
                            f"block={scan_block_idx + 1}/{scan_blocks} "
                            f"ratio={detection_ratio:.3f} "
                            f"score={scan_score_db:.2f} dB "
                            f"pass={pass_count}/{min_pass_blocks}"
                        )
                if stop_requested:
                    break

                if pass_count < min_pass_blocks:
                    continue

                if verbose:
                    print(
                        f"[candidate] f={center_freq / 1e9:.4f} GHz "
                        f"pass_blocks={pass_count}/{scan_blocks}"
                    )

                for precision_idx in range(precision_blocks_per_candidate):
                    if saved_count >= capture_cfg.max_saved or stop_requested:
                        break

                    if _stop_requested(capture_cfg.stop_key):
                        if verbose:
                            print("[STOP] user requested stop")
                        stop_requested = True
                        break

                    iq_raw = _read_block(receiver, block_size)

                    # 중요:
                    # 저장되는 precision block마다 DC offset 제거를 다시 수행한다.
                    iq_dc = remove_dc_offset(iq_raw, axis=-1)

                    frame_energies = _compute_frame_energies(
                        iq_dc,
                        frame_size=frame_size,
                        hop_size=hop_size_energy,
                        window=energy_window,
                    )

                    detection_ratio = _compute_detection_ratio(
                        frame_energies,
                        threshold=noise_threshold,
                    )

                    if detection_ratio < min_detection_ratio:
                        if verbose:
                            print(
                                f"[skip precision] f={center_freq / 1e9:.4f} GHz "
                                f"ratio={detection_ratio:.3f}"
                            )
                        continue

                    cnn_iq = get_cnn_input_iq(iq_dc, rx_index=capture_cfg.rx_index)
                    cnn_iq = normalize_iq(cnn_iq, axis=-1, method="peak")

                    spectrogram = _compute_cnn_spectrogram_numpy(
                        cnn_iq,
                        nperseg=nperseg,
                        hop_size=hop_size_stft,
                        nfft=nfft,
                        window=stft_window,
                    )

                    cnn_input = spectrogram[..., np.newaxis].astype(np.float32)

                    saved_count += 1

                    metadata = {
                        "sample_index": saved_count,
                        "session_id": session_id,
                        "label": capture_cfg.label,
                        "center_freq": int(center_freq),
                        "center_freq_ghz": float(center_freq / 1e9),
                        "sample_rate": sample_rate,
                        "block_size": block_size,
                        "rx_index": capture_cfg.rx_index,
                        "spectrogram_shape": list(spectrogram.shape),
                        "cnn_input_shape": list(cnn_input.shape),
                        "noise_threshold": noise_threshold,
                        "detection_ratio": float(detection_ratio),
                        "scan_score_db": float(last_scan_score_db),
                        "last_scan_detection_ratio": float(last_detection_ratio),
                        "cycle_index": cycle_index,
                        "precision_index": precision_idx,
                        "dc_offset_before": [
                            [float(np.real(v)), float(np.imag(v))]
                            for v in np.mean(_ensure_2d_iq(iq_raw), axis=-1)
                        ],
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }

                    sample_path = _save_cnn_sample(
                        output_dir=output_dir,
                        sample_index=saved_count,
                        spectrogram=spectrogram,
                        cnn_input=cnn_input,
                        metadata=metadata,
                        raw_iq=iq_raw if capture_cfg.save_raw_iq else None,
                    )

                    if verbose:
                        print(
                            f"[saved] {saved_count}/{capture_cfg.max_saved} "
                            f"{sample_path}"
                        )

            if stop_requested:
                break

            if cycle_delay_sec > 0:
                time.sleep(cycle_delay_sec)

        if verbose:
            print()
            print("=== CNN Dataset Capture Finished ===")
            print(f"saved_count : {saved_count}")
            print(f"output_dir  : {output_dir}")
            print()

        return 0

    finally:
        if hasattr(receiver, "close"):
            receiver.close()