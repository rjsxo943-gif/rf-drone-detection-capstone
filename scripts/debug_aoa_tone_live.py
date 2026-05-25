from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core import load_yaml
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver import build_receiver


C = 299_792_458.0


def wrap_phase_rad(x: float | np.ndarray) -> float | np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def ensure_2d_iq(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)

    if iq.ndim == 1:
        return iq.reshape(1, -1)

    if iq.ndim == 2:
        return iq

    raise ValueError(f"IQ must be 1D or 2D, got shape={iq.shape}")


def safe_get(configs: dict[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        cur: Any = configs

        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]

        if cur is not None:
            return cur

    return default


def build_runtime_receiver(configs: dict[str, Any]) -> Any:
    receiver_cfg = configs.get("receiver", configs)

    try:
        return build_receiver(receiver_cfg)
    except TypeError:
        return build_receiver(configs)


def close_receiver(receiver: Any) -> None:
    if hasattr(receiver, "close"):
        try:
            receiver.close()
        except Exception:
            pass


def read_block(receiver: Any, block_size: int) -> np.ndarray:
    if hasattr(receiver, "read_block"):
        return ensure_2d_iq(receiver.read_block(block_size))

    if hasattr(receiver, "read_samples"):
        return ensure_2d_iq(receiver.read_samples(block_size))

    raise AttributeError("receiver has neither read_block nor read_samples")


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


def get_block_size(configs: dict[str, Any], fallback: int = 8192) -> int:
    return int(
        safe_get(
            configs,
            ("receiver", "sdr", "block_size"),
            ("receiver", "sdr", "num_samples"),
            ("sdr", "block_size"),
            ("sdr", "num_samples"),
            ("receiver", "block_size"),
            ("receiver", "num_samples"),
            default=fallback,
        )
    )


def get_sample_rate(configs: dict[str, Any], fallback: int = 5_000_000) -> int:
    return int(
        safe_get(
            configs,
            ("receiver", "sdr", "sample_rate"),
            ("sdr", "sample_rate"),
            ("receiver", "sample_rate"),
            default=fallback,
        )
    )


def load_phase_gain(path: Path) -> tuple[float, float]:
    if not path.exists():
        print(f"[WARN] calibration file not found: {path}")
        print("[WARN] using phase_offset=0.0, gain_correction=1.0")
        return 0.0, 1.0

    data = json.loads(path.read_text(encoding="utf-8"))

    # calibration_actions.py 결과는 *_mean 이름으로 저장될 수 있다.
    # tone debug에서 직접 저장한 파일은 phase_offset / gain_correction 이름일 수 있다.
    phase_offset = float(
        data.get(
            "phase_offset",
            data.get(
                "phase_offset_rad",
                data.get("phase_offset_rad_mean", 0.0),
            ),
        )
    )

    gain_correction = float(
        data.get(
            "gain_correction",
            data.get("gain_correction_mean", 1.0),
        )
    )

    return phase_offset, gain_correction


def estimate_phase_from_tone_fft(
    ref: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    expected_offset_hz: float,
    search_bw_hz: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    n = int(ref.size)

    win = np.hanning(n).astype(np.float32)

    ref_fft = np.fft.fftshift(np.fft.fft(ref * win))
    target_fft = np.fft.fftshift(np.fft.fft(target * win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))

    mask = np.abs(freqs - expected_offset_hz) <= search_bw_hz

    if not np.any(mask):
        raise RuntimeError("No FFT bins found in tone search range")

    idx_candidates = np.where(mask)[0]
    score = np.abs(ref_fft[idx_candidates]) + np.abs(target_fft[idx_candidates])
    peak_idx = int(idx_candidates[int(np.argmax(score))])

    lo = max(0, peak_idx - 2)
    hi = min(n, peak_idx + 3)

    ref_bins = ref_fft[lo:hi]
    target_bins = target_fft[lo:hi]

    cross = np.sum(target_bins * np.conj(ref_bins))
    denom = np.sqrt(np.sum(np.abs(ref_bins) ** 2) * np.sum(np.abs(target_bins) ** 2)) + eps

    raw_phase = float(np.angle(cross))
    coherence = float(np.abs(cross) / denom)
    peak_freq = float(freqs[peak_idx])
    peak_db = float(20.0 * np.log10(np.max(score) + eps))

    return {
        "raw_phase_rad": raw_phase,
        "coherence": coherence,
        "peak_freq_hz": peak_freq,
        "peak_db": peak_db,
    }


def phase_to_angle_deg(
    phase_rad: float,
    carrier_freq_hz: float,
    antenna_spacing_m: float,
    invert: bool = False,
) -> tuple[float, float, bool]:
    if invert:
        phase_rad = -phase_rad

    wavelength = C / carrier_freq_hz

    arg = phase_rad * wavelength / (2.0 * np.pi * antenna_spacing_m)
    clipped = bool(arg < -1.0 or arg > 1.0)

    arg = float(np.clip(arg, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arcsin(arg)))

    return angle_deg, arg, clipped


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--center-freq", type=int, default=2449000000)
    parser.add_argument("--signal-freq", type=int, default=2450000000)

    parser.add_argument("--antenna-spacing", type=float, default=0.060)
    parser.add_argument("--num-blocks", type=int, default=30)
    parser.add_argument("--rx0", type=int, default=0)
    parser.add_argument("--rx1", type=int, default=1)

    parser.add_argument("--search-bw", type=float, default=200000.0)
    parser.add_argument("--warmup-reads", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.02)

    parser.add_argument("--calib", default="outputs/calibration/phase_gain_latest.json")
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--out-dir", default="outputs/debug/aoa_tone")

    args = parser.parse_args()

    # AoA tone debug는 receiver 설정만 필요하다.
    # load_all_configs()는 detect/ml/aoa block_size까지 검사하므로,
    # receiver.yaml만 직접 읽어 block_size mismatch를 피한다.
    configs = load_yaml("configs/receiver.yaml")
    receiver = build_runtime_receiver(configs)

    block_size = get_block_size(configs)
    sample_rate = get_sample_rate(configs)

    phase_offset, gain_correction = load_phase_gain(Path(args.calib))

    expected_offset_hz = float(args.signal_freq - args.center_freq)
    wavelength = C / float(args.signal_freq)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{session}_sig{int(args.signal_freq/1e6)}_cf{int(args.center_freq/1e6)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | bool]] = []

    print("=== AoA Tone Live Debug ===")
    print(f"signal_freq       : {args.signal_freq} Hz")
    print(f"center_freq       : {args.center_freq} Hz")
    print(f"expected_offset   : {expected_offset_hz:+.0f} Hz")
    print(f"sample_rate       : {sample_rate}")
    print(f"block_size        : {block_size}")
    print(f"antenna_spacing   : {args.antenna_spacing:.4f} m")
    print(f"wavelength        : {wavelength:.6f} m")
    print(f"phase_offset      : {phase_offset:+.6f} rad")
    print(f"gain_correction   : {gain_correction:.6f}")
    print(f"invert            : {args.invert}")
    print(f"out_dir           : {out_dir}")
    print()

    try:
        set_center_freq(receiver, args.center_freq)

        for _ in range(args.warmup_reads):
            read_block(receiver, block_size)

        for i in range(args.num_blocks):
            iq = read_block(receiver, block_size)
            iq = remove_dc_offset(iq, axis=-1)
            iq = ensure_2d_iq(iq)

            if iq.shape[0] < 2:
                raise RuntimeError(f"2 channels required, got shape={iq.shape}")

            ref = iq[args.rx0].astype(np.complex64)
            target = iq[args.rx1].astype(np.complex64) * gain_correction

            est = estimate_phase_from_tone_fft(
                ref=ref,
                target=target,
                sample_rate=sample_rate,
                expected_offset_hz=expected_offset_hz,
                search_bw_hz=args.search_bw,
            )

            raw_phase = float(est["raw_phase_rad"])
            corrected_phase = float(wrap_phase_rad(raw_phase - phase_offset))

            angle_deg, arcsin_arg, clipped = phase_to_angle_deg(
                corrected_phase,
                carrier_freq_hz=float(args.signal_freq),
                antenna_spacing_m=args.antenna_spacing,
                invert=args.invert,
            )

            row = {
                "block": i,
                "raw_phase_rad": raw_phase,
                "corrected_phase_rad": corrected_phase,
                "angle_deg": angle_deg,
                "arcsin_arg": arcsin_arg,
                "clipped": clipped,
                "coherence": float(est["coherence"]),
                "peak_freq_hz": float(est["peak_freq_hz"]),
                "peak_db": float(est["peak_db"]),
            }
            rows.append(row)

            print(
                f"[{i:04d}] "
                f"peak={row['peak_freq_hz']:+.0f} Hz "
                f"raw={row['raw_phase_rad']:+.4f} rad "
                f"corr={row['corrected_phase_rad']:+.4f} rad "
                f"coh={row['coherence']:.4f} "
                f"angle={row['angle_deg']:+.2f} deg "
                f"clip={row['clipped']}"
            )

            time.sleep(args.sleep_sec)

    finally:
        close_receiver(receiver)

    angle_values = np.array([float(r["angle_deg"]) for r in rows], dtype=np.float32)
    coherence_values = np.array([float(r["coherence"]) for r in rows], dtype=np.float32)

    csv_path = out_dir / "aoa_summary.csv"
    json_path = out_dir / "final_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "block",
            "raw_phase_rad",
            "corrected_phase_rad",
            "angle_deg",
            "arcsin_arg",
            "clipped",
            "coherence",
            "peak_freq_hz",
            "peak_db",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "signal_freq": args.signal_freq,
        "center_freq": args.center_freq,
        "expected_offset_hz": expected_offset_hz,
        "sample_rate": sample_rate,
        "block_size": block_size,
        "antenna_spacing_m": args.antenna_spacing,
        "phase_offset_rad": phase_offset,
        "gain_correction": gain_correction,
        "invert": args.invert,
        "num_blocks": len(rows),
        "angle_mean_deg": float(np.mean(angle_values)) if len(rows) else None,
        "angle_median_deg": float(np.median(angle_values)) if len(rows) else None,
        "angle_std_deg": float(np.std(angle_values)) if len(rows) else None,
        "coherence_mean": float(np.mean(coherence_values)) if len(rows) else None,
        "coherence_median": float(np.median(coherence_values)) if len(rows) else None,
        "csv_path": str(csv_path),
    }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"angle mean      : {summary['angle_mean_deg']:+.2f} deg")
    print(f"angle median    : {summary['angle_median_deg']:+.2f} deg")
    print(f"angle std       : {summary['angle_std_deg']:.2f} deg")
    print(f"coherence mean  : {summary['coherence_mean']:.4f}")
    print(f"saved csv       : {csv_path}")
    print(f"saved summary   : {json_path}")


if __name__ == "__main__":
    main()
