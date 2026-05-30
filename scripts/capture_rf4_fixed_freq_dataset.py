#!/usr/bin/env python3
"""
RF4 fixed-frequency dataset capture script.

Purpose:
- Capture SDR IQ blocks at a fixed center frequency.
- Convert each block to the same canonical 0~1 spectrogram used by RF4 CNN.
- Optionally run RF4 CNN inference.
- Save only blocks that match a selected save policy:
    - all
    - weak_or_valid
    - valid_only
    - no_signal_only
    - final_drone_only

Main use for today's drone dataset:
- Drone-like:
    --save-policy weak_or_valid
    --min-signal-ratio 2.0
    --valid-signal-ratio 5.0

- Background:
    --save-policy no_signal_only
    --max-background-signal-ratio 1.5

- Controller-only:
    --save-policy all
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core import get_block_size, load_all_configs
from src.ml import RF4Classifier
from src.preprocess.dc_blocker import remove_dc_offset
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-frequency RF4 dataset capture with signal-ratio save policy."
    )

    # Session / IO
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-root", default="outputs/datasets/rf4_fixed_capture")
    parser.add_argument("--config-dir", default="configs")

    # SDR / capture
    parser.add_argument("--center-freq", type=int, default=2_450_000_000)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=100)
    parser.add_argument("--max-saved", type=int, default=None)
    parser.add_argument(
        "--until-max-saved",
        action="store_true",
        help="Keep capturing until --max-saved files are saved.",
    )
    parser.add_argument(
        "--max-total-blocks",
        type=int,
        default=5000,
        help="Safety limit when --until-max-saved is used.",
    )
    parser.add_argument("--warmup-reads", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument(
        "--start-discard-sec",
        type=float,
        default=0.0,
        help="Discard initial capture seconds before saving. Useful for stick/control preparation.",
    )

    # Raw signal detector
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--clip-peak", type=float, default=1000.0)
    parser.add_argument("--max-clip-ratio", type=float, default=0.001)
    parser.add_argument("--min-signal-ratio", type=float, default=2.0)
    parser.add_argument("--valid-signal-ratio", type=float, default=5.0)
    parser.add_argument("--max-background-signal-ratio", type=float, default=1.5)

    # Spectrogram
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=96)
    parser.add_argument("--stft-nfft", type=int, default=128)
    parser.add_argument("--window", default="hann", choices=["hann", "hamming", "rect"])
    parser.add_argument("--vmin", type=float, default=-40.0)
    parser.add_argument("--vmax", type=float, default=40.0)

    # CNN
    parser.add_argument("--model", default="outputs/ml/rf4_cnn_live2450_v2/best_model.pt")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--general-threshold", type=float, default=0.50)
    parser.add_argument("--drone-threshold", type=float, default=0.70)
    parser.add_argument("--strong-drone-p99", type=float, default=0.65)
    parser.add_argument("--strong-drone-max", type=float, default=0.80)

    # Save policy
    parser.add_argument(
        "--save-policy",
        default="weak_or_valid",
        choices=[
            "all",
            "weak_or_valid",
            "valid_only",
            "no_signal_only",
            "final_drone_only",
        ],
    )

    return parser.parse_args()


def safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.ndim == 1:
        return iq.reshape(1, -1)
    if iq.ndim == 2:
        return iq
    raise ValueError(f"iq must be 1D or 2D, got shape={iq.shape}")


def frame_1d(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)

    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")

    if x.size < frame_size:
        return x.reshape(1, -1)

    num_frames = 1 + (x.size - frame_size) // hop_size
    shape = (num_frames, frame_size)
    strides = (x.strides[0] * hop_size, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def compute_raw_features(
    iq_1d: np.ndarray,
    frame_size: int,
    hop_size: int,
    clip_peak: float,
) -> RawFeatures:
    raw = np.asarray(iq_1d).reshape(-1)
    abs_raw = np.abs(raw).astype(np.float64)

    raw_rms = float(np.sqrt(np.mean(abs_raw ** 2)))
    raw_peak = float(np.max(abs_raw))
    raw_abs_mean = float(np.mean(abs_raw))
    raw_abs_median = float(np.median(abs_raw))
    raw_abs_p95 = float(np.percentile(abs_raw, 95))
    raw_abs_p99 = float(np.percentile(abs_raw, 99))

    # clip_peak default is intentionally high because Pluto/receiver output scale
    # may be integer-like rather than normalized to [-1, 1].
    clip_ratio = float(np.mean(abs_raw >= clip_peak))

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
    features: RawFeatures,
    min_signal_ratio: float,
    valid_signal_ratio: float,
    max_background_signal_ratio: float,
    max_clip_ratio: float,
) -> str:
    if features.clip_ratio > max_clip_ratio:
        return "CLIPPED"
    if features.signal_ratio >= valid_signal_ratio:
        return "VALID_SIGNAL"
    if features.signal_ratio >= min_signal_ratio:
        return "WEAK_SIGNAL"
    if features.signal_ratio <= max_background_signal_ratio:
        return "NO_SIGNAL"
    return "AMBIGUOUS"


def should_save_block(
    save_policy: str,
    status: str,
    raw_features: RawFeatures,
    final_class: str,
    max_clip_ratio: float,
) -> tuple[bool, str]:
    if raw_features.clip_ratio > max_clip_ratio:
        return False, "skip_clipped"

    if save_policy == "all":
        return True, "save_all"

    if save_policy == "weak_or_valid":
        if status in {"WEAK_SIGNAL", "VALID_SIGNAL"}:
            return True, "save_weak_or_valid"
        return False, f"skip_status_{status}"

    if save_policy == "valid_only":
        if status == "VALID_SIGNAL":
            return True, "save_valid_only"
        return False, f"skip_status_{status}"

    if save_policy == "no_signal_only":
        if status == "NO_SIGNAL":
            return True, "save_no_signal_only"
        return False, f"skip_status_{status}"

    if save_policy == "final_drone_only":
        if final_class == "Drone-like":
            return True, "save_final_drone_only"
        return False, f"skip_final_{final_class}"

    raise ValueError(f"unsupported save_policy={save_policy!r}")


def set_config_value(configs: dict[str, Any], key: str, value: Any) -> None:
    """Set common receiver values in both top-level and nested receiver sections."""
    receiver_cfg = configs.setdefault("receiver", {})
    if isinstance(receiver_cfg, dict):
        receiver_cfg[key] = value
        for section in ("sdr", "pluto", "pluto_plus", "file", "sim"):
            if isinstance(receiver_cfg.get(section), dict):
                receiver_cfg[section][key] = value

    # Some older configs may have an sdr section at root.
    if isinstance(configs.get("sdr"), dict):
        configs["sdr"][key] = value


def read_block(receiver: Any, block_size: int) -> np.ndarray:
    if hasattr(receiver, "read_block"):
        return ensure_2d_iq(receiver.read_block(block_size))
    if hasattr(receiver, "read_samples"):
        return ensure_2d_iq(receiver.read_samples(block_size))
    raise AttributeError("receiver has neither read_block nor read_samples")


def close_receiver(receiver: Any) -> None:
    if hasattr(receiver, "close"):
        try:
            receiver.close()
        except Exception:
            pass


def set_center_freq(receiver: Any, center_freq: int) -> None:
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


def set_runtime_gain(receiver: Any, gain: float | None) -> None:
    if gain is None:
        return

    gain = float(gain)

    if hasattr(receiver, "gain"):
        try:
            receiver.gain = gain
        except Exception:
            pass

    # Project PlutoReceiver may support this helper.
    if hasattr(receiver, "_set_channel_gain"):
        try:
            # Try both common RX channels. If one fails, ignore.
            receiver._set_channel_gain(0)
            receiver._set_channel_gain(1)
        except Exception:
            pass

    # pyadi-iio AD936x style attributes.
    if hasattr(receiver, "sdr"):
        sdr = getattr(receiver, "sdr")
        if sdr is None:
            return

        for attr in (
            "rx_hardwaregain_chan0",
            "rx_hardwaregain_chan1",
            "rx_hardwaregain",
        ):
            if hasattr(sdr, attr):
                try:
                    setattr(sdr, attr, gain)
                except Exception:
                    pass


def get_block_size_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        safe_get(configs, "receiver", "block_size"),
        safe_get(configs, "receiver", "num_samples"),
        safe_get(configs, "sdr", "block_size"),
        safe_get(configs, "sdr", "num_samples"),
    ]:
        if value is not None:
            return int(value)

    try:
        return int(get_block_size(configs))
    except Exception:
        return 16384


def get_sample_rate_from_configs(configs: dict[str, Any]) -> int:
    for value in [
        safe_get(configs, "receiver", "sample_rate"),
        safe_get(configs, "sdr", "sample_rate"),
    ]:
        if value is not None:
            return int(value)
    return 5_000_000


def build_runtime_receiver(configs: dict[str, Any]) -> Any:
    receiver_cfg = configs.get("receiver", configs)
    try:
        return build_receiver(receiver_cfg)
    except TypeError:
        return build_receiver(configs)


def compute_canonical01_spectrogram(
    iq_1d: np.ndarray,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
    window: str = "hann",
    vmin: float = -40.0,
    vmax: float = 40.0,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(iq_1d).reshape(-1)

    if x.size < nperseg:
        raise ValueError(f"iq length {x.size} is shorter than nperseg {nperseg}")

    hop_size = nperseg - noverlap
    if hop_size <= 0:
        raise ValueError(f"invalid hop_size={hop_size}")

    if window == "hann":
        win = np.hanning(nperseg).astype(np.float32)
    elif window == "hamming":
        win = np.hamming(nperseg).astype(np.float32)
    else:
        win = np.ones(nperseg, dtype=np.float32)

    cols: list[np.ndarray] = []
    for start in range(0, x.size - nperseg + 1, hop_size):
        frame = x[start:start + nperseg] * win
        fft_result = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        mag = np.abs(fft_result) / (float(np.sum(win)) + eps)
        cols.append(mag.astype(np.float32))

    mag_spec = np.stack(cols, axis=1)
    db_spec = 20.0 * np.log10(mag_spec + eps)
    db_spec = np.clip(db_spec, vmin, vmax)

    norm_spec = (db_spec - vmin) / (vmax - vmin)
    return np.clip(norm_spec, 0.0, 1.0).astype(np.float32)


def spec_stats(spec: np.ndarray) -> dict[str, float]:
    x = np.asarray(spec, dtype=np.float32)
    return {
        "spec_mean": float(np.mean(x)),
        "spec_std": float(np.std(x)),
        "spec_median": float(np.median(x)),
        "spec_p95": float(np.percentile(x, 95)),
        "spec_p99": float(np.percentile(x, 99)),
        "spec_min": float(np.min(x)),
        "spec_max": float(np.max(x)),
    }


def make_filename(
    idx: int,
    label: str,
    status: str,
    raw_class: str,
    final_class: str,
    confidence: float,
    signal_ratio: float,
    saved_index: int,
) -> str:
    def _safe_name(text: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
            for ch in str(text)
        )

    safe_status = _safe_name(status)
    safe_raw = _safe_name(raw_class)
    safe_final = _safe_name(final_class)
    safe_label = _safe_name(label)

    return (
        f"{saved_index:04d}__srcidx_{idx:04d}"
        f"__{safe_label}"
        f"__status_{safe_status}"
        f"__sr_{signal_ratio:.3f}"
        f"__raw_{safe_raw}"
        f"__final_{safe_final}"
        f"__conf_{confidence:.4f}.npy"
    )


def main() -> None:
    args = parse_args()

    if args.vmax <= args.vmin:
        raise ValueError("--vmax must be greater than --vmin")

    configs = load_all_configs(args.config_dir)

    if args.center_freq is not None:
        set_config_value(configs, "center_freq", int(args.center_freq))
    if args.gain is not None:
        set_config_value(configs, "gain", float(args.gain))

    block_size = get_block_size_from_configs(configs)
    sample_rate = get_sample_rate_from_configs(configs)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{session}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier: RF4Classifier | None = None
    if not args.skip_inference:
        classifier = RF4Classifier(
            checkpoint_path=args.model,
            general_threshold=args.general_threshold,
            drone_threshold=args.drone_threshold,
        )

    receiver = build_runtime_receiver(configs)

    rows: list[dict[str, Any]] = []
    saved_count = 0
    skipped_count = 0
    status_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}

    try:
        set_center_freq(receiver, args.center_freq)
        set_runtime_gain(receiver, args.gain)

        for _ in range(args.warmup_reads):
            read_block(receiver, block_size)

        print("=== RF4 Fixed-Frequency Dataset Capture ===")
        print(f"label              : {args.label}")
        print(f"out_dir            : {out_dir}")
        print(f"center_freq        : {args.center_freq}")
        print(f"sample_rate        : {sample_rate}")
        print(f"gain               : {args.gain if args.gain is not None else 'config'}")
        print(f"block_size         : {block_size}")
        print(f"rx_index           : {args.rx_index}")
        print(f"num_blocks         : {args.num_blocks}")
        print(f"max_saved          : {args.max_saved}")
        print(f"until_max_saved    : {args.until_max_saved}")
        print(f"max_total_blocks   : {args.max_total_blocks}")
        print(f"save_policy        : {args.save_policy}")
        print(f"min_signal_ratio   : {args.min_signal_ratio}")
        print(f"valid_signal_ratio : {args.valid_signal_ratio}")
        print(f"max_bg_signal_ratio: {args.max_background_signal_ratio}")
        print(f"max_clip_ratio     : {args.max_clip_ratio}")
        print(f"clip_peak          : {args.clip_peak}")
        print(f"inference          : {'off' if args.skip_inference else 'on'}")
        print()

        if args.start_discard_sec > 0:
            print(f"[START DISCARD] discarding first {args.start_discard_sec:.2f} sec...")
            discard_start = time.time()
            discard_count = 0

            while time.time() - discard_start < args.start_discard_sec:
                read_block(receiver, block_size)
                discard_count += 1
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            print(f"[START DISCARD DONE] discarded_blocks={discard_count}")
            print()

        idx = 0
        max_attempts = args.max_total_blocks if args.until_max_saved else args.num_blocks

        while idx < max_attempts:
            if args.max_saved is not None and saved_count >= args.max_saved:
                print(f"[STOP] reached max_saved={args.max_saved}")
                break

            iq = read_block(receiver, block_size)
            iq = remove_dc_offset(iq, axis=-1)
            iq = ensure_2d_iq(iq)

            if args.rx_index < 0 or args.rx_index >= iq.shape[0]:
                raise ValueError(
                    f"rx_index out of range: {args.rx_index}, num_channels={iq.shape[0]}"
                )

            iq_1d = np.asarray(iq[args.rx_index]).reshape(-1)

            raw_features = compute_raw_features(
                iq_1d,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                clip_peak=args.clip_peak,
            )

            status = classify_signal_status(
                raw_features,
                min_signal_ratio=args.min_signal_ratio,
                valid_signal_ratio=args.valid_signal_ratio,
                max_background_signal_ratio=args.max_background_signal_ratio,
                max_clip_ratio=args.max_clip_ratio,
            )

            spec = compute_canonical01_spectrogram(
                iq_1d,
                nperseg=args.stft_nperseg,
                noverlap=args.stft_noverlap,
                nfft=args.stft_nfft,
                window=args.window,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            stats = spec_stats(spec)

            raw_class = "NA"
            final_class = "NA"
            confidence = float("nan")

            if classifier is not None:
                result = classifier.predict_array(spec)
                raw_class = result.class_name
                confidence = float(result.confidence)

                is_strong_drone = (
                    result.class_name == "Drone-like"
                    and stats["spec_p99"] >= args.strong_drone_p99
                    and stats["spec_max"] >= args.strong_drone_max
                )

                if result.class_name == "Drone-like":
                    final_class = "Drone-like" if is_strong_drone else "Background"
                else:
                    final_class = result.final_class

            should_save, save_reason = should_save_block(
                save_policy=args.save_policy,
                status=status,
                raw_features=raw_features,
                final_class=final_class,
                max_clip_ratio=args.max_clip_ratio,
            )

            save_path = ""
            if should_save:
                filename = make_filename(
                    idx=idx,
                    label=args.label,
                    status=status,
                    raw_class=raw_class,
                    final_class=final_class,
                    confidence=confidence if np.isfinite(confidence) else -1.0,
                    signal_ratio=raw_features.signal_ratio,
                    saved_index=saved_count,
                )
                save_path = str(out_dir / filename)
                np.save(save_path, spec)
                saved_count += 1
            else:
                skipped_count += 1

            status_counts[status] = status_counts.get(status, 0) + 1
            final_counts[final_class] = final_counts.get(final_class, 0) + 1

            row = {
                "index": idx,
                "saved": should_save,
                "save_reason": save_reason,
                "file": save_path,
                "label": args.label,
                "center_freq": args.center_freq,
                "sample_rate": sample_rate,
                "gain": args.gain if args.gain is not None else "",
                "rx_index": args.rx_index,
                "block_size": block_size,
                "status": status,
                "raw_class": raw_class,
                "final_class": final_class,
                "confidence": confidence,
                **asdict(raw_features),
                **stats,
            }
            rows.append(row)

            save_mark = "SAVE" if should_save else "SKIP"
            print(
                f"[{idx:04d}] {save_mark:<4s} "
                f"status={status:<12s} "
                f"sr={raw_features.signal_ratio:8.3f} "
                f"clip={raw_features.clip_ratio:.6f} "
                f"raw={raw_class:<10s} "
                f"final={final_class:<10s} "
                f"conf={confidence if np.isfinite(confidence) else -1.0:.4f} "
                f"p99={stats['spec_p99']:.4f} "
                f"max={stats['spec_max']:.4f} "
                f"reason={save_reason}",
                flush=True,
            )

            idx += 1
            time.sleep(args.sleep_sec)

    finally:
        close_receiver(receiver)

    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    meta_path = out_dir / "session_meta.txt"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write("RF4 Fixed-Frequency Dataset Capture\n")
        f.write(f"session={session}\n")
        f.write(f"label={args.label}\n")
        f.write(f"center_freq={args.center_freq}\n")
        f.write(f"sample_rate={sample_rate}\n")
        f.write(f"gain={args.gain if args.gain is not None else 'config'}\n")
        f.write(f"block_size={block_size}\n")
        f.write(f"rx_index={args.rx_index}\n")
        f.write(f"num_blocks={args.num_blocks}\n")
        f.write(f"max_saved={args.max_saved}\n")
        f.write(f"until_max_saved={args.until_max_saved}\n")
        f.write(f"max_total_blocks={args.max_total_blocks}\n")
        f.write(f"start_discard_sec={args.start_discard_sec}\n")
        f.write(f"save_policy={args.save_policy}\n")
        f.write(f"min_signal_ratio={args.min_signal_ratio}\n")
        f.write(f"valid_signal_ratio={args.valid_signal_ratio}\n")
        f.write(f"max_background_signal_ratio={args.max_background_signal_ratio}\n")
        f.write(f"max_clip_ratio={args.max_clip_ratio}\n")
        f.write(f"clip_peak={args.clip_peak}\n")
        f.write(f"saved_count={saved_count}\n")
        f.write(f"skipped_count={skipped_count}\n")
        f.write(f"status_counts={status_counts}\n")
        f.write(f"final_counts={final_counts}\n")

    print()
    print("=== Capture Summary ===")
    print(f"saved_count  : {saved_count}")
    print(f"skipped_count: {skipped_count}")
    print(f"status_counts: {status_counts}")
    print(f"final_counts : {final_counts}")
    print(f"[OK] saved spectrograms to: {out_dir}")
    print(f"[OK] saved summary: {csv_path}")
    print(f"[OK] saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
