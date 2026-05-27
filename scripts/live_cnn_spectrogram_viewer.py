#!/usr/bin/env python3
"""
Live CNN Spectrogram Viewer v1.

Purpose:
- Read IQ blocks from the configured receiver.
- Compute raw IQ features before normalization.
- Select one representative block per update.
- Convert only the selected block to the same CNN-style spectrogram image.
- Display/save the spectrogram and append feature logs to CSV.

Notes:
- This script does not run CNN inference.
- This script does not compute AoA.
- This script does not scan frequencies.
- YAML is loaded only once at startup. Restart the script to apply config changes.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core import get_project_root, load_all_configs
from src.preprocess import get_cnn_input_iq, normalize_iq, remove_dc_offset
from src.receiver import build_receiver


EPS = 1e-12


@dataclass(frozen=True)
class RawFeatures:
    raw_rms: float
    raw_peak: float
    raw_abs_mean: float
    raw_abs_median: float
    raw_abs_p95: float
    raw_abs_p99: float
    clip_ratio: float
    frame_power_median: float
    frame_power_p95: float
    frame_power_p99: float
    noise_floor: float
    signal_ratio: float


@dataclass(frozen=True)
class CnnSpecFeatures:
    cnn_spec_mean: float
    cnn_spec_std: float
    cnn_spec_min: float
    cnn_spec_p50: float
    cnn_spec_p95: float
    cnn_spec_p99: float
    cnn_spec_max: float


@dataclass(frozen=True)
class ViewerThresholds:
    no_signal_ratio: float
    valid_signal_ratio: float
    overload_peak: float
    overload_clip_ratio: float


CSV_COLUMNS = [
    "timestamp",
    "session_id",
    "update_index",
    "center_freq",
    "sample_rate",
    "rf_bandwidth",
    "block_size",
    "rx_index",
    "gain",
    "distance_m",
    "memo",
    "blocks_per_update",
    "selected_block_index",
    "select_policy",
    "status",
    "suggestion",
    "raw_rms",
    "raw_peak",
    "raw_abs_mean",
    "raw_abs_median",
    "raw_abs_p95",
    "raw_abs_p99",
    "clip_ratio",
    "frame_power_median",
    "frame_power_p95",
    "frame_power_p99",
    "noise_floor",
    "signal_ratio",
    "cnn_spec_mean",
    "cnn_spec_std",
    "cnn_spec_min",
    "cnn_spec_p50",
    "cnn_spec_p95",
    "cnn_spec_p99",
    "cnn_spec_max",
    "latency_sec",
    "processing_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live viewer for CNN-style RF spectrogram input quality."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--center-freq", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--rf-bandwidth", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--distance-m", type=float, default=math.nan)
    parser.add_argument("--memo", default="")

    parser.add_argument("--update-interval-sec", type=float, default=1.0)
    parser.add_argument("--blocks-per-update", type=int, default=20)
    parser.add_argument("--select-policy", default="max_signal_ratio")
    parser.add_argument("--max-updates", type=int, default=None)

    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--no-signal-ratio", type=float, default=2.0)
    parser.add_argument("--valid-signal-ratio", type=float, default=5.0)
    parser.add_argument("--overload-peak", type=float, default=0.95)
    parser.add_argument("--overload-clip-ratio", type=float, default=0.001)

    parser.add_argument("--stft-nperseg", type=int, default=None)
    parser.add_argument("--stft-noverlap", type=int, default=None)
    parser.add_argument("--stft-nfft", type=int, default=None)

    parser.add_argument("--log-dir", default="outputs/live_viewer/logs")
    parser.add_argument("--latest-image-dir", default="outputs/live_viewer/latest")
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def now_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_project_root() / path


def nested_get(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_receiver_value(receiver_cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read common receiver values from either top-level or nested config blocks."""
    if key in receiver_cfg:
        return receiver_cfg[key]
    for section in ("sdr", "pluto", "pluto_plus", "file", "sim"):
        value = nested_get(receiver_cfg, [section, key], None)
        if value is not None:
            return value
    return default


def set_receiver_value(receiver_cfg: dict[str, Any], key: str, value: Any) -> None:
    """Set a receiver value in top-level and known nested blocks when present."""
    receiver_cfg[key] = value
    for section in ("sdr", "pluto", "pluto_plus", "file", "sim"):
        if isinstance(receiver_cfg.get(section), dict):
            receiver_cfg[section][key] = value


def load_viewer_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = load_all_configs(args.config_dir)
    receiver_cfg = dict(configs.get("receiver", {}))
    ml_cfg = dict(configs.get("ml", {}))

    if args.gain is not None:
        set_receiver_value(receiver_cfg, "gain", args.gain)
    if args.center_freq is not None:
        set_receiver_value(receiver_cfg, "center_freq", int(args.center_freq))
    if args.sample_rate is not None:
        set_receiver_value(receiver_cfg, "sample_rate", int(args.sample_rate))
    if args.rf_bandwidth is not None:
        set_receiver_value(receiver_cfg, "rf_bandwidth", int(args.rf_bandwidth))
    if args.block_size is not None:
        set_receiver_value(receiver_cfg, "block_size", args.block_size)

    return receiver_cfg, ml_cfg


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"IQ block must be 1D or 2D, got shape={arr.shape}")


def select_rx_1d(iq: np.ndarray, rx_index: int) -> np.ndarray:
    arr = ensure_2d_iq(iq)
    if rx_index < 0 or rx_index >= arr.shape[0]:
        raise IndexError(
            f"rx_index={rx_index} is out of range for IQ shape={arr.shape}"
        )
    return np.asarray(arr[rx_index], dtype=np.complex64)


def frame_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if x.size < frame_size:
        return x.reshape(1, -1)

    num_frames = 1 + (x.size - frame_size) // hop_size
    shape = (num_frames, frame_size)
    strides = (x.strides[0] * hop_size, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def compute_raw_features(
    iq_block: np.ndarray,
    rx_index: int,
    frame_size: int,
    hop_size: int,
    overload_peak: float,
) -> RawFeatures:
    raw = select_rx_1d(iq_block, rx_index)
    abs_raw = np.abs(raw).astype(np.float64)

    raw_rms = float(np.sqrt(np.mean(abs_raw ** 2)))
    raw_peak = float(np.max(abs_raw))
    raw_abs_mean = float(np.mean(abs_raw))
    raw_abs_median = float(np.median(abs_raw))
    raw_abs_p95 = float(np.percentile(abs_raw, 95))
    raw_abs_p99 = float(np.percentile(abs_raw, 99))
    clip_ratio = float(np.mean(abs_raw >= overload_peak))

    frames = frame_1d(raw, frame_size=frame_size, hop_size=hop_size)
    frame_power = np.mean(np.abs(frames).astype(np.float64) ** 2, axis=1)
    frame_power_median = float(np.median(frame_power))
    frame_power_p95 = float(np.percentile(frame_power, 95))
    frame_power_p99 = float(np.percentile(frame_power, 99))
    noise_floor = max(frame_power_median, EPS)
    signal_ratio = float(frame_power_p99 / noise_floor)

    return RawFeatures(
        raw_rms=raw_rms,
        raw_peak=raw_peak,
        raw_abs_mean=raw_abs_mean,
        raw_abs_median=raw_abs_median,
        raw_abs_p95=raw_abs_p95,
        raw_abs_p99=raw_abs_p99,
        clip_ratio=clip_ratio,
        frame_power_median=frame_power_median,
        frame_power_p95=frame_power_p95,
        frame_power_p99=frame_power_p99,
        noise_floor=noise_floor,
        signal_ratio=signal_ratio,
    )


def classify_signal_status(
    raw_features: RawFeatures,
    thresholds: ViewerThresholds,
) -> tuple[str, str]:
    if (
        raw_features.raw_peak >= thresholds.overload_peak
        or raw_features.clip_ratio >= thresholds.overload_clip_ratio
    ):
        return "OVERLOAD", "TRY_LOWER_GAIN"

    if raw_features.signal_ratio < thresholds.no_signal_ratio:
        return "NO_SIGNAL", "KEEP_GAIN"

    if raw_features.signal_ratio < thresholds.valid_signal_ratio:
        return "WEAK_SIGNAL", "TRY_HIGHER_GAIN_IF_REPEATED"

    return "VALID_SIGNAL", "KEEP_GAIN"


def select_representative_block(
    blocks: list[np.ndarray],
    raw_features_list: list[RawFeatures],
    select_policy: str,
) -> int:
    if not blocks:
        raise ValueError("No IQ blocks were collected")
    if select_policy != "max_signal_ratio":
        raise ValueError(f"Unsupported select_policy={select_policy!r}")
    scores = [features.signal_ratio for features in raw_features_list]
    return int(np.argmax(scores))


def infer_stft_params(
    ml_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    stft_cfg = ml_cfg.get("stft", {}) if isinstance(ml_cfg.get("stft", {}), dict) else {}

    nperseg = args.stft_nperseg or int(
        stft_cfg.get("nperseg", stft_cfg.get("frame_size", 128))
    )
    noverlap = args.stft_noverlap or int(
        stft_cfg.get("noverlap", nperseg - stft_cfg.get("hop_length", 32))
    )
    nfft = args.stft_nfft or int(stft_cfg.get("nfft", nperseg))

    if nperseg <= 0:
        raise ValueError("stft nperseg must be positive")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("stft noverlap must satisfy 0 <= noverlap < nperseg")
    if nfft < nperseg:
        raise ValueError("stft nfft must be >= nperseg")
    return nperseg, noverlap, nfft


def compute_cnn_input_spectrogram(
    iq_block: np.ndarray,
    rx_index: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> np.ndarray:
    """Create normalized log-magnitude STFT spectrogram for CNN input viewing."""
    iq_no_dc = remove_dc_offset(iq_block)
    cnn_iq = get_cnn_input_iq(iq_no_dc, rx_index=rx_index)
    cnn_iq = normalize_iq(cnn_iq, method="peak")
    cnn_iq = np.asarray(cnn_iq).reshape(-1).astype(np.complex64)

    hop = nperseg - noverlap
    frames = frame_1d(cnn_iq, frame_size=nperseg, hop_size=hop)
    window = np.hanning(nperseg).astype(np.float32)
    windowed = frames * window.reshape(1, -1)

    stft = np.fft.fft(windowed, n=nfft, axis=1)
    stft = np.fft.fftshift(stft, axes=1)
    mag = np.abs(stft).astype(np.float32)
    spec = np.log1p(mag)

    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec = (spec - spec_min) / max(spec_max - spec_min, EPS)

    # Image convention: frequency bins x time frames.
    return spec.T.astype(np.float32)


def compute_cnn_spec_features(spec: np.ndarray) -> CnnSpecFeatures:
    x = np.asarray(spec, dtype=np.float64)
    return CnnSpecFeatures(
        cnn_spec_mean=float(np.mean(x)),
        cnn_spec_std=float(np.std(x)),
        cnn_spec_min=float(np.min(x)),
        cnn_spec_p50=float(np.percentile(x, 50)),
        cnn_spec_p95=float(np.percentile(x, 95)),
        cnn_spec_p99=float(np.percentile(x, 99)),
        cnn_spec_max=float(np.max(x)),
    )


def empty_cnn_features() -> dict[str, float]:
    return {
        "cnn_spec_mean": math.nan,
        "cnn_spec_std": math.nan,
        "cnn_spec_min": math.nan,
        "cnn_spec_p50": math.nan,
        "cnn_spec_p95": math.nan,
        "cnn_spec_p99": math.nan,
        "cnn_spec_max": math.nan,
    }


def prepare_matplotlib(no_display: bool):
    if no_display:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_latest_image(
    plt: Any,
    latest_image_dir: Path,
    spec: np.ndarray | None,
    title: str,
    side_text: str,
) -> Path | None:
    if spec is None:
        return None
    latest_image_dir.mkdir(parents=True, exist_ok=True)
    path = latest_image_dir / "live_cnn_spectrogram_latest.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(spec, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.text(
        1.02,
        0.5,
        side_text,
        transform=ax.transAxes,
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def update_display(
    plt: Any,
    image_handle: Any,
    text_handle: Any,
    spec: np.ndarray | None,
    title: str,
    side_text: str,
) -> tuple[Any, Any]:
    if spec is None:
        plt.gcf().suptitle(title)
        text_handle.set_text(side_text)
        plt.pause(0.001)
        return image_handle, text_handle

    if image_handle is None:
        image_handle = plt.imshow(spec, aspect="auto", origin="lower")
        plt.xlabel("Time frame")
        plt.ylabel("Frequency bin")
    else:
        image_handle.set_data(spec)
        image_handle.set_clim(float(np.min(spec)), float(np.max(spec)))

    plt.title(title)
    text_handle.set_text(side_text)
    plt.pause(0.001)
    return image_handle, text_handle


def append_csv_log(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def build_side_text(row: dict[str, Any]) -> str:
    lines = [
        f"center_freq: {row['center_freq']}",
        f"sample_rate: {row['sample_rate']}",
        f"rf_bandwidth: {row['rf_bandwidth']}",
        f"gain: {row['gain']}",
        f"rx_index: {row['rx_index']}",
        f"distance_m: {row['distance_m']}",
        f"memo: {row['memo']}",
        f"status: {row['status']}",
        f"suggestion: {row['suggestion']}",
        "",
        f"raw_rms: {row['raw_rms']:.6g}",
        f"raw_peak: {row['raw_peak']:.6g}",
        f"raw_p99: {row['raw_abs_p99']:.6g}",
        f"clip_ratio: {row['clip_ratio']:.6g}",
        f"noise_floor: {row['noise_floor']:.6g}",
        f"signal_ratio: {row['signal_ratio']:.3f}",
        f"frame_p99: {row['frame_power_p99']:.6g}",
        "",
        f"cnn_mean: {row['cnn_spec_mean']:.6g}",
        f"cnn_std: {row['cnn_spec_std']:.6g}",
        f"cnn_p95: {row['cnn_spec_p95']:.6g}",
        f"cnn_p99: {row['cnn_spec_p99']:.6g}",
        f"cnn_max: {row['cnn_spec_max']:.6g}",
    ]
    return "\n".join(lines)


def print_update(row: dict[str, Any]) -> None:
    print(
        "[update={update_index:04d}] status={status:<12} "
        "signal_ratio={signal_ratio:.3f} raw_peak={raw_peak:.6g} "
        "clip_ratio={clip_ratio:.6g} suggestion={suggestion}".format(**row),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    receiver_cfg, ml_cfg = load_viewer_configs(args)

    session_id = now_session_id()
    log_dir = to_project_path(args.log_dir)
    latest_image_dir = to_project_path(args.latest_image_dir)
    csv_path = log_dir / f"{session_id}_live_cnn_viewer_log.csv"

    thresholds = ViewerThresholds(
        no_signal_ratio=args.no_signal_ratio,
        valid_signal_ratio=args.valid_signal_ratio,
        overload_peak=args.overload_peak,
        overload_clip_ratio=args.overload_clip_ratio,
    )

    block_size = int(infer_receiver_value(receiver_cfg, "block_size", args.block_size or 16384))
    center_freq = infer_receiver_value(receiver_cfg, "center_freq", args.center_freq)
    sample_rate = infer_receiver_value(receiver_cfg, "sample_rate", args.sample_rate)
    rf_bandwidth = infer_receiver_value(receiver_cfg, "rf_bandwidth", args.rf_bandwidth)
    gain = infer_receiver_value(receiver_cfg, "gain", args.gain)
    nperseg, noverlap, nfft = infer_stft_params(ml_cfg, args)

    print("=== Live CNN Spectrogram Viewer v1 ===")
    print(f"session_id         : {session_id}")
    print(f"csv_log            : {csv_path}")
    print(f"center_freq        : {center_freq}")
    print(f"sample_rate        : {sample_rate}")
    print(f"rf_bandwidth       : {rf_bandwidth}")
    print(f"gain               : {gain}")
    print(f"block_size         : {block_size}")
    print(f"rx_index           : {args.rx_index}")
    print(f"distance_m         : {args.distance_m}")
    print(f"memo               : {args.memo}")
    print(f"blocks_per_update  : {args.blocks_per_update}")
    print(f"stft               : nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}")
    print("note               : YAML is loaded once at startup. Restart to apply changes.")
    print("Press Ctrl+C to stop.")

    plt = prepare_matplotlib(args.no_display)
    image_handle = None
    text_handle = None
    if not args.no_display:
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 6))
        text_handle = ax.text(
            1.02,
            0.5,
            "starting...",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )
    else:
        # Dummy text handle for no-display mode.
        class _TextHandle:
            def set_text(self, _text: str) -> None:
                return None

        text_handle = _TextHandle()

    receiver = build_receiver(receiver_cfg)
    update_index = 0

    try:
        while args.max_updates is None or update_index < args.max_updates:
            loop_start = time.perf_counter()
            blocks: list[np.ndarray] = []
            raw_features_list: list[RawFeatures] = []

            for _ in range(args.blocks_per_update):
                block = receiver.read_block(block_size)
                features = compute_raw_features(
                    block,
                    rx_index=args.rx_index,
                    frame_size=args.frame_size,
                    hop_size=args.hop_size,
                    overload_peak=args.overload_peak,
                )
                blocks.append(block)
                raw_features_list.append(features)

            selected_idx = select_representative_block(
                blocks,
                raw_features_list,
                select_policy=args.select_policy,
            )
            selected_block = blocks[selected_idx]
            raw_features = raw_features_list[selected_idx]
            status, suggestion = classify_signal_status(raw_features, thresholds)

            spec: np.ndarray | None = None
            cnn_features: dict[str, float]
            if status in {"WEAK_SIGNAL", "VALID_SIGNAL", "OVERLOAD"}:
                spec = compute_cnn_input_spectrogram(
                    selected_block,
                    rx_index=args.rx_index,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                )
                cnn_features = asdict(compute_cnn_spec_features(spec))
            else:
                cnn_features = empty_cnn_features()

            processing_time_sec = time.perf_counter() - loop_start
            latency_sec = processing_time_sec

            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "session_id": session_id,
                "update_index": update_index,
                "center_freq": center_freq,
                "sample_rate": sample_rate,
                "rf_bandwidth": rf_bandwidth,
                "block_size": block_size,
                "rx_index": args.rx_index,
                "gain": gain,
                "distance_m": args.distance_m,
                "memo": args.memo,
                "blocks_per_update": args.blocks_per_update,
                "selected_block_index": selected_idx,
                "select_policy": args.select_policy,
                "status": status,
                "suggestion": suggestion,
                **asdict(raw_features),
                **cnn_features,
                "latency_sec": latency_sec,
                "processing_time_sec": processing_time_sec,
            }

            append_csv_log(csv_path, row)
            print_update(row)

            title = (
                f"Live CNN Spectrogram | {status} | "
                f"gain={gain} | d={args.distance_m}m | update={update_index}"
            )
            side_text = build_side_text(row)

            if not args.no_display:
                image_handle, text_handle = update_display(
                    plt, image_handle, text_handle, spec, title, side_text
                )
            if args.save_latest:
                save_latest_image(plt, latest_image_dir, spec, title, side_text)

            update_index += 1
            elapsed = time.perf_counter() - loop_start
            sleep_sec = max(0.0, args.update_interval_sec - elapsed)
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        close = getattr(receiver, "close", None)
        if callable(close):
            close()
        if not args.no_display:
            plt.ioff()
        print(f"CSV log saved to: {csv_path}")
        if args.save_latest:
            print(f"Latest image dir: {latest_image_dir}")


if __name__ == "__main__":
    main()
