from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import stft

from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver.pluto_receiver import PlutoReceiver


# ============================================================
# Basic utilities
# ============================================================

def now_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    ensure_dir(path.parent)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# STFT / spectrogram
# ============================================================

def compute_noverlap(nperseg: int, hop_size: int) -> int:
    nperseg = int(nperseg)
    hop_size = int(hop_size)

    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}")

    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    if hop_size > nperseg:
        raise ValueError(
            f"hop_size must be <= nperseg. got hop_size={hop_size}, nperseg={nperseg}"
        )

    return nperseg - hop_size


def expected_stft_frames(block_size: int, nperseg: int, hop_size: int) -> int:
    if block_size < nperseg:
        return 0

    return ((block_size - nperseg) // hop_size) + 1


def compute_spectrogram_db(
    iq_1d: np.ndarray,
    sample_rate: int,
    nperseg: int,
    hop_size: int,
    nfft: int,
    window: str,
    vmin: float,
    vmax: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    1D complex IQ block을 dB spectrogram으로 변환한다.

    기본값:
    - block_size = 16384
    - nperseg = 128
    - hop_size = 32
    - noverlap = 96
    - nfft = 128
    - output shape ~= (128, 509)
    """
    iq_1d = np.asarray(iq_1d, dtype=np.complex64)

    if iq_1d.ndim != 1:
        raise ValueError(f"iq_1d must be 1-D, got shape={iq_1d.shape}")

    # 프로젝트 공통 DC offset 제거
    iq_1d = remove_dc_offset(iq_1d, axis=-1)

    noverlap = compute_noverlap(nperseg=nperseg, hop_size=hop_size)

    _, _, zxx = stft(
        iq_1d,
        fs=float(sample_rate),
        window=window,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        nfft=int(nfft),
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    zxx = np.fft.fftshift(zxx, axes=0)
    magnitude = np.abs(zxx).astype(np.float32)

    spec_db = 20.0 * np.log10(magnitude + eps)
    spec_db = np.clip(spec_db, float(vmin), float(vmax))

    return spec_db.astype(np.float32)


def compute_stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_value = float(np.max(arr))
    burst_score = p99 - median

    return {
        "median": median,
        "p95": p95,
        "p99": p99,
        "max": max_value,
        "burst_score": burst_score,
    }


# ============================================================
# Selection rules
# ============================================================

def is_bluetooth_meaningful(
    stats: dict[str, float],
    burst_threshold: float,
    p99_threshold: float,
    max_allowed: float,
    keep_weak: bool,
    weak_burst_threshold: float,
) -> tuple[bool, str]:
    """
    Bluetooth meaningful block 판정.

    clean 기준:
    - max <= max_allowed
    - burst_score >= burst_threshold
    - p99 >= p99_threshold

    keep_weak 사용 시:
    - clean 기준을 못 넘더라도 burst_score >= weak_burst_threshold면 저장
    """
    if stats["max"] > max_allowed:
        return False, "too_high_max"

    if stats["burst_score"] >= burst_threshold and stats["p99"] >= p99_threshold:
        return True, "clean_bt_burst"

    if keep_weak and stats["burst_score"] >= weak_burst_threshold:
        return True, "weak_bt_burst"

    return False, "weak_or_background"


def judge_selected(
    stats: dict[str, float],
    accept_mode: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """
    클래스별 selected 여부를 판단한다.

    accept_mode:
    - all
    - energy_like
    - bluetooth_meaningful
    - background_fail
    """
    if accept_mode == "all":
        return True, "all"

    if accept_mode == "bluetooth_meaningful":
        return is_bluetooth_meaningful(
            stats=stats,
            burst_threshold=args.burst_threshold,
            p99_threshold=args.p99_threshold,
            max_allowed=args.max_allowed,
            keep_weak=args.keep_weak,
            weak_burst_threshold=args.weak_burst_threshold,
        )

    if accept_mode == "energy_like":
        if stats["max"] > args.max_allowed:
            return False, "too_high_max"

        if (
            stats["burst_score"] >= args.burst_threshold
            and stats["p99"] >= args.p99_threshold
        ):
            return True, "energy_like_burst"

        if args.keep_weak and stats["burst_score"] >= args.weak_burst_threshold:
            return True, "weak_energy_like_burst"

        return False, "weak_or_background"

    if accept_mode == "background_fail":
        """
        Background용.
        강한 burst가 없는 block을 저장한다.
        """
        if stats["max"] > args.max_allowed:
            return False, "too_high_max"

        if (
            stats["burst_score"] < args.burst_threshold
            and stats["p99"] < args.p99_threshold
        ):
            return True, "background_like"

        return False, "signal_like"

    raise ValueError(f"unknown accept_mode: {accept_mode}")


# ============================================================
# Naming / printing
# ============================================================

def build_selected_filename(
    selected_index: int,
    label: str,
    center_freq: int,
    gain: float,
    attempt_index: int,
) -> str:
    center_freq_mhz = int(round(center_freq / 1_000_000))
    gain_str = f"{gain:g}".replace(".", "p")

    return (
        f"{selected_index:04d}"
        f"__{label}"
        f"__cf{center_freq_mhz}"
        f"__g{gain_str}"
        f"__attempt{attempt_index:04d}"
        f".npy"
    )


def should_print_attempt(
    attempt_index: int,
    selected: bool,
    print_every: int,
) -> bool:
    if selected:
        return True

    if print_every <= 0:
        return False

    return (attempt_index + 1) % print_every == 0


def print_attempt_line(
    attempt_index: int,
    max_attempts: int,
    selected_count: int,
    target_selected: int,
    selected: bool,
    reason: str,
    stats: dict[str, float],
) -> None:
    status = "SELECT" if selected else "DROP"

    print(
        f"[attempt {attempt_index + 1:04d}/{max_attempts:04d}] "
        f"{status:6s} | "
        f"selected={selected_count:04d}/{target_selected:04d} | "
        f"reason={reason:24s} | "
        f"median={stats['median']:+7.2f}, "
        f"p99={stats['p99']:+7.2f}, "
        f"max={stats['max']:+7.2f}, "
        f"burst={stats['burst_score']:+7.2f}"
    )


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Target-selected CNN spectrogram capture script for Pluto+/AD9361. "
            "It keeps capturing until selected spectrogram count reaches target-selected."
        )
    )

    # Output/session
    parser.add_argument("--label", required=True, help="수집 라벨: Background, WiFi, Bluetooth, Drone_like")
    parser.add_argument(
        "--base-dir",
        default="data/processed/cnn_capture",
        help="저장 기준 폴더",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="세션 ID. 예: 20260521_BT_music_cf2450_g10_d1m_s01. 생략하면 YYYYMMDD_HHMMSS 자동 생성",
    )

    # Target-selected capture
    parser.add_argument(
        "--target-selected",
        type=int,
        default=100,
        help="최종적으로 저장하고 싶은 selected spectrogram 개수",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="무한 수집 방지를 위한 최대 block 수신 횟수",
    )
    parser.add_argument(
        "--accept-mode",
        choices=["all", "energy_like", "bluetooth_meaningful", "background_fail"],
        default="energy_like",
        help="selected 판정 방식",
    )

    # SDR
    parser.add_argument("--uri", default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=5_000_000)
    parser.add_argument("--block-size", type=int, default=16_384)
    parser.add_argument("--gain", type=float, default=15.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--warmup-reads", type=int, default=3)

    # STFT
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--hop-size", type=int, default=32)
    parser.add_argument("--nfft", type=int, default=128)
    parser.add_argument("--window", default="hann")
    parser.add_argument("--vmin", type=float, default=-40.0)
    parser.add_argument("--vmax", type=float, default=40.0)

    # Selection thresholds
    parser.add_argument("--burst-threshold", type=float, default=24.0)
    parser.add_argument("--p99-threshold", type=float, default=0.0)
    parser.add_argument("--max-allowed", type=float, default=80.0)
    parser.add_argument("--keep-weak", action="store_true")
    parser.add_argument("--weak-burst-threshold", type=float, default=18.0)

    # Save/log options
    parser.add_argument(
        "--save-raw-selected",
        action="store_true",
        help="선택된 block의 raw IQ도 같이 저장",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=50,
        help="몇 attempt마다 CSV 중간 저장할지. 0이면 마지막에만 저장",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="몇 attempt마다 진행 상황을 출력할지. selected block은 항상 출력. 0이면 selected block만 출력",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.target_selected <= 0:
        raise ValueError(
            f"target-selected must be positive, got {args.target_selected}"
        )

    if args.max_attempts <= 0:
        raise ValueError(
            f"max-attempts must be positive, got {args.max_attempts}"
        )

    if args.max_attempts < args.target_selected:
        raise ValueError(
            "max-attempts must be >= target-selected. "
            f"got max_attempts={args.max_attempts}, "
            f"target_selected={args.target_selected}"
        )

    noverlap = compute_noverlap(nperseg=args.nperseg, hop_size=args.hop_size)
    expected_frames = expected_stft_frames(
        block_size=args.block_size,
        nperseg=args.nperseg,
        hop_size=args.hop_size,
    )

    session_id = args.session_id or now_session_id()

    # 저장 구조:
    # data/processed/cnn_capture/{session_id}/{label}/*.npy
    session_dir = ensure_dir(Path(args.base_dir) / session_id)
    selected_dir = ensure_dir(session_dir / args.label)

    raw_selected_dir = (
        ensure_dir(session_dir / f"{args.label}_raw_iq")
        if args.save_raw_selected
        else None
    )

    metadata = {
        "script": "capture_cnn_spectrogram_target_128hop32.py",
        "label": args.label,
        "session_id": session_id,
        "base_dir": args.base_dir,
        "session_dir": str(session_dir),
        "selected_dir": str(selected_dir),
        "target_selected": args.target_selected,
        "max_attempts": args.max_attempts,
        "accept_mode": args.accept_mode,
        "uri": args.uri,
        "center_freq": args.center_freq,
        "center_freq_mhz": args.center_freq / 1_000_000,
        "sample_rate": args.sample_rate,
        "rf_bandwidth": args.rf_bandwidth,
        "block_size": args.block_size,
        "gain": args.gain,
        "channel": args.channel,
        "warmup_reads": args.warmup_reads,
        "stft": {
            "nperseg": args.nperseg,
            "hop_size": args.hop_size,
            "noverlap": noverlap,
            "nfft": args.nfft,
            "window": args.window,
            "vmin": args.vmin,
            "vmax": args.vmax,
            "expected_shape": [args.nfft, expected_frames],
        },
        "selection": {
            "burst_threshold": args.burst_threshold,
            "p99_threshold": args.p99_threshold,
            "max_allowed": args.max_allowed,
            "keep_weak": args.keep_weak,
            "weak_burst_threshold": args.weak_burst_threshold,
        },
    }

    save_json(session_dir / "metadata.json", metadata)

    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    print("=== Target-Selected CNN Spectrogram Capture / STFT 128 Hop 32 ===")
    print(f"label           : {args.label}")
    print(f"session_id      : {session_id}")
    print(f"session_dir     : {session_dir}")
    print(f"selected_dir    : {selected_dir}")
    print(f"accept_mode     : {args.accept_mode}")
    print(f"target_selected : {args.target_selected}")
    print(f"max_attempts    : {args.max_attempts}")
    print(f"uri             : {args.uri}")
    print(f"center_freq     : {args.center_freq} Hz")
    print(f"sample_rate     : {args.sample_rate} Hz")
    print(f"gain            : {args.gain}")
    print(f"channel         : {args.channel}")
    print(f"block_size      : {args.block_size}")
    print(
        "STFT            : "
        f"nperseg={args.nperseg}, "
        f"hop={args.hop_size}, "
        f"noverlap={noverlap}, "
        f"nfft={args.nfft}"
    )
    print(f"expected shape  : ({args.nfft}, {expected_frames})")
    print(
        "threshold       : "
        f"burst>={args.burst_threshold}, "
        f"p99>={args.p99_threshold}, "
        f"max<={args.max_allowed}"
    )
    print(f"keep_weak       : {args.keep_weak}")
    print()

    receiver = PlutoReceiver(
        uri=args.uri,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        num_channels=1,
        channels=[args.channel],
        gain_control_mode="manual",
        gain=args.gain,
        block_size=args.block_size,
        num_samples=args.block_size,
        rf_bandwidth=args.rf_bandwidth,
        warmup_reads=args.warmup_reads,
    )

    try:
        attempt_count = 0
        selected_count = 0

        while (
            selected_count < args.target_selected
            and attempt_count < args.max_attempts
        ):
            attempt_index = attempt_count
            iq_block = receiver.read_block(args.block_size)
            attempt_count += 1

            if iq_block.ndim != 2:
                raise ValueError(
                    f"receiver output must be 2-D, got shape={iq_block.shape}"
                )

            # channels=[args.channel]로 열었으므로 row 0이 실제 선택 채널이다.
            iq_1d = iq_block[0].astype(np.complex64, copy=False)

            spec_db = compute_spectrogram_db(
                iq_1d=iq_1d,
                sample_rate=args.sample_rate,
                nperseg=args.nperseg,
                hop_size=args.hop_size,
                nfft=args.nfft,
                window=args.window,
                vmin=args.vmin,
                vmax=args.vmax,
            )

            stats = compute_stats(spec_db)

            selected, reason = judge_selected(
                stats=stats,
                accept_mode=args.accept_mode,
                args=args,
            )

            selected_file = ""
            raw_selected_file = ""
            selected_index_for_row: int | str = ""

            if selected:
                selected_index_for_row = selected_count

                selected_name = build_selected_filename(
                    selected_index=selected_count,
                    label=args.label,
                    center_freq=args.center_freq,
                    gain=args.gain,
                    attempt_index=attempt_index,
                )
                selected_path = selected_dir / selected_name
                np.save(selected_path, spec_db)
                selected_file = str(selected_path)

                if raw_selected_dir is not None:
                    raw_name = selected_name.replace(".npy", "__raw_iq.npy")
                    raw_path = raw_selected_dir / raw_name
                    np.save(raw_path, iq_1d)
                    raw_selected_file = str(raw_path)

                selected_count += 1

            row = {
                "attempt_index": attempt_index,
                "selected_index": selected_index_for_row,
                "selected": selected,
                "reason": reason,
                "label": args.label,
                "session_id": session_id,
                "target_selected": args.target_selected,
                "max_attempts": args.max_attempts,
                "accept_mode": args.accept_mode,
                "center_freq": args.center_freq,
                "center_freq_mhz": args.center_freq / 1_000_000,
                "sample_rate": args.sample_rate,
                "rf_bandwidth": args.rf_bandwidth,
                "gain": args.gain,
                "channel": args.channel,
                "block_size": args.block_size,
                "nperseg": args.nperseg,
                "hop_size": args.hop_size,
                "noverlap": noverlap,
                "nfft": args.nfft,
                "spectrogram_shape": str(tuple(spec_db.shape)),
                "vmin": args.vmin,
                "vmax": args.vmax,
                "selected_file": selected_file,
                "raw_selected_file": raw_selected_file,
                **stats,
            }

            all_rows.append(row)

            if selected:
                selected_rows.append(row)

            if should_print_attempt(
                attempt_index=attempt_index,
                selected=selected,
                print_every=args.print_every,
            ):
                print_attempt_line(
                    attempt_index=attempt_index,
                    max_attempts=args.max_attempts,
                    selected_count=selected_count,
                    target_selected=args.target_selected,
                    selected=selected,
                    reason=reason,
                    stats=stats,
                )

            if (
                args.summary_every > 0
                and attempt_count % args.summary_every == 0
            ):
                save_csv(session_dir / "capture_summary.csv", all_rows)
                save_csv(selected_dir / "selected_summary.csv", selected_rows)

    finally:
        receiver.close()

    save_csv(session_dir / "capture_summary.csv", all_rows)
    save_csv(selected_dir / "selected_summary.csv", selected_rows)

    reason_counts = Counter(row["reason"] for row in all_rows)

    completed = len(selected_rows) >= args.target_selected
    accept_rate = len(selected_rows) / len(all_rows) if all_rows else 0.0

    final_summary = {
        "session_id": session_id,
        "label": args.label,
        "completed": completed,
        "target_selected": args.target_selected,
        "selected_count": len(selected_rows),
        "attempt_count": len(all_rows),
        "max_attempts": args.max_attempts,
        "accept_rate": accept_rate,
        "accept_mode": args.accept_mode,
        "center_freq": args.center_freq,
        "center_freq_mhz": args.center_freq / 1_000_000,
        "sample_rate": args.sample_rate,
        "rf_bandwidth": args.rf_bandwidth,
        "gain": args.gain,
        "channel": args.channel,
        "block_size": args.block_size,
        "reason_counts": dict(reason_counts),
        "session_dir": str(session_dir),
        "selected_dir": str(selected_dir),
    }

    if not completed:
        final_summary["stop_reason"] = "max_attempts_reached"
    else:
        final_summary["stop_reason"] = "target_selected_reached"

    save_json(session_dir / "final_summary.json", final_summary)

    print()
    print("=== Final Summary ===")
    print(f"session_dir      : {session_dir}")
    print(f"selected_dir     : {selected_dir}")
    print(f"completed        : {completed}")
    print(f"stop_reason      : {final_summary['stop_reason']}")
    print(f"attempt blocks   : {len(all_rows)}")
    print(f"selected blocks  : {len(selected_rows)}")
    print(f"target selected  : {args.target_selected}")
    print(f"accept rate      : {accept_rate * 100:.2f}%")

    print("reason counts:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")

    print()
    print("saved:")
    print(f"  {session_dir / 'metadata.json'}")
    print(f"  {session_dir / 'capture_summary.csv'}")
    print(f"  {selected_dir / 'selected_summary.csv'}")
    print(f"  {session_dir / 'final_summary.json'}")

    if not completed:
        print()
        print("WARNING:")
        print(
            f"목표 개수 {args.target_selected}장에 도달하지 못했습니다. "
            f"현재 selected={len(selected_rows)}, attempts={len(all_rows)}입니다."
        )
        print("중심주파수, 거리, gain, 송신 상태, threshold를 확인하세요.")


if __name__ == "__main__":
    main()