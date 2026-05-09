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

    이번 Bluetooth selected-only 스크립트 기본값:
    - block_size = 16384
    - nperseg = 128
    - hop_size = 32
    - noverlap = 96
    - nfft = 128
    - output shape = 약 (128, 509)
    """
    iq_1d = np.asarray(iq_1d, dtype=np.complex64)

    if iq_1d.ndim != 1:
        raise ValueError(f"iq_1d must be 1-D, got shape={iq_1d.shape}")

    # 프로젝트 공통 DC offset 제거 함수 사용
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
    - clean 기준을 못 넘더라도 burst_score >= weak_burst_threshold면 저장한다.
    """
    if stats["max"] > max_allowed:
        return False, "too_high_max"

    if stats["burst_score"] >= burst_threshold and stats["p99"] >= p99_threshold:
        return True, "clean_bt_burst"

    if keep_weak and stats["burst_score"] >= weak_burst_threshold:
        return True, "weak_bt_burst"

    return False, "weak_or_background"


def should_print_block(
    block_idx: int,
    selected: bool,
    print_every: int,
) -> bool:
    if selected:
        return True

    if print_every <= 0:
        return False

    return (block_idx + 1) % print_every == 0


def print_block_line(
    block_idx: int,
    total_blocks: int,
    selected_count: int,
    selected: bool,
    reason: str,
    stats: dict[str, float],
) -> None:
    status = "SELECT" if selected else "DROP"

    print(
        f"[{block_idx + 1:04d}/{total_blocks:04d}] {status:6s} | "
        f"selected={selected_count:4d} | "
        f"reason={reason:18s} | "
        f"median={stats['median']:+7.2f}, "
        f"p99={stats['p99']:+7.2f}, "
        f"max={stats['max']:+7.2f}, "
        f"burst={stats['burst_score']:+7.2f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bluetooth selected-only capture script for Pluto+/AD9361. "
            "Default STFT is nperseg=128, hop_size=32."
        )
    )

    # Output/session
    parser.add_argument("--label", required=True, help="수집 라벨")
    parser.add_argument(
        "--base-dir",
        default="data/processed/cnn_capture",
        help="저장 기준 폴더",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="세션 ID. 생략하면 YYYYMMDD_HHMMSS 자동 생성",
    )

    # SDR
    parser.add_argument("--uri", default="ip:192.168.2.1")
    parser.add_argument("--blocks", type=int, default=400)
    parser.add_argument("--center-freq", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=5_000_000)
    parser.add_argument("--block-size", type=int, default=16_384)
    parser.add_argument("--gain", type=float, default=10.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--warmup-reads", type=int, default=3)

    # STFT
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--hop-size", type=int, default=32)
    parser.add_argument("--nfft", type=int, default=128)
    parser.add_argument("--window", default="hann")
    parser.add_argument("--vmin", type=float, default=-40.0)
    parser.add_argument("--vmax", type=float, default=40.0)

    # Bluetooth selection
    parser.add_argument("--burst-threshold", type=float, default=30.0)
    parser.add_argument("--p99-threshold", type=float, default=5.0)
    parser.add_argument("--max-allowed", type=float, default=80.0)
    parser.add_argument("--keep-weak", action="store_true")
    parser.add_argument("--weak-burst-threshold", type=float, default=25.0)

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
        help="몇 block마다 CSV 중간 저장할지. 0이면 마지막에만 저장",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="몇 block마다 진행 상황을 출력할지. 선택된 block은 항상 출력. 0이면 선택된 block만 출력",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.blocks <= 0:
        raise ValueError(f"blocks must be positive, got {args.blocks}")

    noverlap = compute_noverlap(nperseg=args.nperseg, hop_size=args.hop_size)
    expected_frames = expected_stft_frames(
        block_size=args.block_size,
        nperseg=args.nperseg,
        hop_size=args.hop_size,
    )

    session_id = args.session_id or now_session_id()
    session_dir = ensure_dir(Path(args.base_dir) / args.label / session_id)
    selected_dir = ensure_dir(session_dir / "selected_bluetooth_meaningful")
    raw_selected_dir = (
        ensure_dir(session_dir / "selected_raw_iq")
        if args.save_raw_selected
        else None
    )

    metadata = {
        "script": "capture_bluetooth_selected_only_128hop32.py",
        "label": args.label,
        "session_id": session_id,
        "uri": args.uri,
        "blocks": args.blocks,
        "center_freq": args.center_freq,
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

    print("=== Bluetooth Selected-Only Capture / STFT 128 Hop 32 ===")
    print(f"label           : {args.label}")
    print(f"session_id      : {session_id}")
    print(f"session_dir     : {session_dir}")
    print(f"selected_dir    : {selected_dir}")
    print(f"uri             : {args.uri}")
    print(f"center_freq     : {args.center_freq} Hz")
    print(f"sample_rate     : {args.sample_rate} Hz")
    print(f"gain            : {args.gain}")
    print(f"channel         : {args.channel}")
    print(f"block_size      : {args.block_size}")
    print(f"STFT            : nperseg={args.nperseg}, hop={args.hop_size}, noverlap={noverlap}, nfft={args.nfft}")
    print(f"expected shape  : ({args.nfft}, {expected_frames})")
    print(f"threshold       : burst>={args.burst_threshold}, p99>={args.p99_threshold}, max<={args.max_allowed}")
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
        for block_idx in range(args.blocks):
            iq_block = receiver.read_block(args.block_size)

            if iq_block.ndim != 2:
                raise ValueError(f"receiver output must be 2-D, got shape={iq_block.shape}")

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

            selected, reason = is_bluetooth_meaningful(
                stats=stats,
                burst_threshold=args.burst_threshold,
                p99_threshold=args.p99_threshold,
                max_allowed=args.max_allowed,
                keep_weak=args.keep_weak,
                weak_burst_threshold=args.weak_burst_threshold,
            )

            selected_file = ""
            raw_selected_file = ""

            if selected:
                selected_name = f"{args.label}_block_{block_idx:04d}.npy"
                selected_path = selected_dir / selected_name
                np.save(selected_path, spec_db)
                selected_file = str(selected_path)

                if raw_selected_dir is not None:
                    raw_name = f"{args.label}_block_{block_idx:04d}_raw_iq.npy"
                    raw_path = raw_selected_dir / raw_name
                    np.save(raw_path, iq_1d)
                    raw_selected_file = str(raw_path)

            row = {
                "block_index": block_idx,
                "selected": selected,
                "reason": reason,
                "label": args.label,
                "session_id": session_id,
                "center_freq": args.center_freq,
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

            if should_print_block(
                block_idx=block_idx,
                selected=selected,
                print_every=args.print_every,
            ):
                print_block_line(
                    block_idx=block_idx,
                    total_blocks=args.blocks,
                    selected_count=len(selected_rows),
                    selected=selected,
                    reason=reason,
                    stats=stats,
                )

            if args.summary_every > 0 and (block_idx + 1) % args.summary_every == 0:
                save_csv(session_dir / "capture_summary.csv", all_rows)
                save_csv(selected_dir / "selected_summary.csv", selected_rows)

    finally:
        receiver.close()

    save_csv(session_dir / "capture_summary.csv", all_rows)
    save_csv(selected_dir / "selected_summary.csv", selected_rows)

    reason_counts = Counter(row["reason"] for row in all_rows)

    print()
    print("=== Final Summary ===")
    print(f"session_dir     : {session_dir}")
    print(f"selected_dir    : {selected_dir}")
    print(f"total blocks    : {len(all_rows)}")
    print(f"selected blocks : {len(selected_rows)}")
    if all_rows:
        print(f"yield           : {len(selected_rows) / len(all_rows) * 100:.2f}%")

    print("reason counts:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")

    print()
    print("saved:")
    print(f"  {session_dir / 'metadata.json'}")
    print(f"  {session_dir / 'capture_summary.csv'}")
    print(f"  {selected_dir / 'selected_summary.csv'}")


if __name__ == "__main__":
    main()
