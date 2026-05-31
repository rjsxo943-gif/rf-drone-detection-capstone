from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.receiver.pluto_receiver import PlutoReceiver
from src.calibration import dominant_cluster_phase, wrap_phase_rad


def quality_from_meta(meta: dict) -> str:
    std_deg = float(meta["phase_std_deg"])
    valid = int(meta["valid_blocks"])
    cluster = int(meta["cluster_blocks"])
    ratio = float(meta["cluster_ratio"])
    coh_med = float(meta["coherence_median"])

    if std_deg < 3.0 and valid >= 100 and ratio >= 0.80 and coh_med >= 0.70:
        return "OK"

    if std_deg < 7.0 and valid >= 50 and ratio >= 0.60 and coh_med >= 0.55:
        return "WARNING"

    return "FAIL"


def print_quality_hint(quality: str) -> None:
    if quality == "OK":
        print("[QUALITY] OK - 현재 세션 phase calibration 값 사용 가능")
        return

    if quality == "WARNING":
        print("[QUALITY] WARNING - 사용은 가능하지만 재측정 권장")
        print("  - 안테나 정렬 확인")
        print("  - 사람/폰/노트북을 RX 주변에서 멀리 두기")
        print("  - 신호발생기 출력 또는 거리 확인")
        print("  - 주변 Wi-Fi/블루투스 간섭 확인")
        return

    print("[QUALITY] FAIL - 이 calibration 값은 사용 비추천")
    print("  - 장소 변경 또는 반사체 제거 권장")
    print("  - coherence threshold 낮추기보다 신호원 조건 개선 우선")
    print("  - 신호발생기 주파수를 center에서 ±1~2 MHz offset으로 설정")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust outdoor 0-degree phase-offset calibration for RX0/RX1."
    )

    parser.add_argument("--uri", type=str, default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, default=2_450_000_000)
    parser.add_argument("--signal-freq", type=int, default=2_452_000_000)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=None)
    parser.add_argument("--gain", type=float, default=30.0)
    parser.add_argument("--block-size", type=int, default=16_384)

    # 새 기본값: 총 200블럭, 앞 30블럭 discard
    parser.add_argument("--num-blocks", type=int, default=200)
    parser.add_argument("--discard-blocks", type=int, default=30)
    parser.add_argument("--warmup-reads", type=int, default=20)

    parser.add_argument("--coherence-threshold", type=float, default=0.50)
    parser.add_argument("--cluster-window-deg", type=float, default=5.0)

    # 기존 호환용: 예전 명령어의 --min-coherence도 받되 coherence_threshold로 사용
    parser.add_argument("--min-coherence", type=float, default=None)

    parser.add_argument("--memo", type=str, default="outdoor_0deg_phase_calibration")
    parser.add_argument("--out-dir", type=str, default="outputs/calibration")
    parser.add_argument(
        "--current-config",
        type=str,
        default="configs/calibration/current_phase_offset.json",
    )

    args = parser.parse_args()

    coherence_threshold = (
        float(args.min_coherence)
        if args.min_coherence is not None
        else float(args.coherence_threshold)
    )

    if args.num_blocks <= args.discard_blocks:
        raise ValueError(
            f"num_blocks must be greater than discard_blocks. "
            f"got num_blocks={args.num_blocks}, discard_blocks={args.discard_blocks}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        / f"{timestamp}_phase_offset_cf{args.center_freq}_g{int(args.gain)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Robust Outdoor Phase Offset Calibration ===")
    print(f"uri                 : {args.uri}")
    print(f"center_freq          : {args.center_freq}")
    print(f"signal_freq          : {args.signal_freq}")
    print(f"signal_offset        : {args.signal_freq - args.center_freq} Hz")
    print(f"sample_rate          : {args.sample_rate}")
    print(f"gain                 : {args.gain}")
    print(f"block_size           : {args.block_size}")
    print(f"num_blocks           : {args.num_blocks}")
    print(f"discard_blocks       : {args.discard_blocks}")
    print(f"used_blocks_target   : {args.num_blocks - args.discard_blocks}")
    print(f"warmup_reads         : {args.warmup_reads}")
    print(f"coherence_threshold  : {coherence_threshold}")
    print(f"cluster_window_deg   : {args.cluster_window_deg}")
    print(f"out_dir              : {out_dir}")
    print()
    print("[SETUP CHECK]")
    print("- 신호발생기 주파수는 --signal-freq 값과 맞추기")
    print("- 송신원은 RX0/RX1 배열 정면 0도")
    print("- 송신원-RX 거리 1.5~2.5m 권장")
    print("- 안테나/케이블/RX 포트/간격은 측정 중 변경 금지")
    print("- 핫스팟/블루투스/불필요한 2.4GHz 장치는 가능하면 OFF")
    print("- 사람/폰/노트북은 RX 주변에서 멀리 두기")
    print()

    receiver = PlutoReceiver(
        uri=args.uri,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        num_channels=2,
        channels=[0, 1],
        gain_control_mode="manual",
        gain=args.gain,
        block_size=args.block_size,
        rf_bandwidth=args.rf_bandwidth,
        warmup_reads=args.warmup_reads,
    )

    blocks_ch0: list[np.ndarray] = []
    blocks_ch1: list[np.ndarray] = []

    try:
        for block_idx in range(args.num_blocks):
            iq = receiver.read_block(args.block_size)

            if iq.shape[0] < 2:
                raise RuntimeError(f"Need 2 RX channels, got shape={iq.shape}")

            blocks_ch0.append(iq[0].copy())
            blocks_ch1.append(iq[1].copy())

            if (block_idx + 1) % 50 == 0:
                print(f"collected {block_idx + 1}/{args.num_blocks} blocks")

    finally:
        receiver.close()

    # 앞 discard block 제거
    used_ch0 = blocks_ch0[args.discard_blocks:]
    used_ch1 = blocks_ch1[args.discard_blocks:]

    meta = dominant_cluster_phase(
        blocks_ch0=used_ch0,
        blocks_ch1=used_ch1,
        coherence_threshold=coherence_threshold,
        cluster_window_deg=args.cluster_window_deg,
    )

    phase_offset_rad = float(wrap_phase_rad(meta["phase"]))
    phase_offset_deg = float(np.rad2deg(phase_offset_rad))

    quality = quality_from_meta(meta)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "robust_outdoor_0deg_over_air",
        "memo": args.memo,
        "quality": quality,

        "phase_offset_rad": phase_offset_rad,
        "phase_offset_deg": phase_offset_deg,

        "ref_channel": 0,
        "target_channel": 1,
        "center_freq": int(args.center_freq),
        "signal_freq": int(args.signal_freq),
        "signal_offset_hz": int(args.signal_freq - args.center_freq),
        "sample_rate": int(args.sample_rate),
        "rf_bandwidth": args.rf_bandwidth,
        "gain": float(args.gain),
        "block_size": int(args.block_size),

        "num_blocks_total": int(args.num_blocks),
        "discard_blocks": int(args.discard_blocks),
        "num_blocks_used_target": int(args.num_blocks - args.discard_blocks),
        "warmup_reads": int(args.warmup_reads),

        "coherence_threshold": float(coherence_threshold),
        "cluster_window_deg": float(args.cluster_window_deg),

        # dominant_cluster_phase 결과
        "phase_std_rad": float(meta["phase_std"]),
        "phase_std_deg": float(meta["phase_std_deg"]),
        "coherence_mean": float(meta["coherence_mean"]),
        "coherence_median": float(meta["coherence_median"]),
        "coherence_min": float(meta["coherence_min"]),
        "coherence_max": float(meta["coherence_max"]),
        "valid_blocks": int(meta["valid_blocks"]),
        "cluster_blocks": int(meta["cluster_blocks"]),
        "cluster_ratio": float(meta["cluster_ratio"]),

        "notes": [
            "phase_offset = RX1 phase - RX0 phase",
            "Correction should multiply RX1 by exp(-j * phase_offset_rad)",
            "This calibration is valid for the same SDR session, gain, center frequency, antennas, cables, and layout.",
            "If gain changes during AoA, use gain phase delta table if available.",
        ],
    }

    json_path = out_dir / "phase_offset_calibration.json"
    current_config_path = Path(args.current_config)
    current_config_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    current_config_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 원본 block 자체는 용량이 커질 수 있으니 기본 저장은 하지 않고 요약만 저장
    summary_npz_path = out_dir / "phase_offset_summary.npz"
    np.savez(
        summary_npz_path,
        phase_offset_rad=np.array([phase_offset_rad], dtype=np.float64),
        phase_offset_deg=np.array([phase_offset_deg], dtype=np.float64),
        phase_std_rad=np.array([meta["phase_std"]], dtype=np.float64),
        phase_std_deg=np.array([meta["phase_std_deg"]], dtype=np.float64),
        coherence_mean=np.array([meta["coherence_mean"]], dtype=np.float64),
        coherence_median=np.array([meta["coherence_median"]], dtype=np.float64),
        cluster_ratio=np.array([meta["cluster_ratio"]], dtype=np.float64),
        valid_blocks=np.array([meta["valid_blocks"]], dtype=np.int32),
        cluster_blocks=np.array([meta["cluster_blocks"]], dtype=np.int32),
    )

    print()
    print("=== Calibration Result ===")
    print(f"quality           : {quality}")
    print(f"phase_offset      : {phase_offset_deg:+.3f} deg")
    print(f"phase_offset      : {phase_offset_rad:+.6f} rad")
    print(f"phase_std         : {meta['phase_std_deg']:.3f} deg")
    print(f"coherence mean    : {meta['coherence_mean']:.3f}")
    print(f"coherence median  : {meta['coherence_median']:.3f}")
    print(
        f"cluster blocks    : {meta['cluster_blocks']} / {meta['valid_blocks']} "
        f"= {meta['cluster_ratio'] * 100:.1f}%"
    )
    print()
    print_quality_hint(quality)
    print()
    print(f"saved json        : {json_path}")
    print(f"saved npz         : {summary_npz_path}")
    print(f"current config    : {current_config_path}")
    print()
    print("[APPLY]")
    print("AoA 계산 전에 RX1에 exp(-j * phase_offset_rad)를 곱해서 보정하면 됨.")


if __name__ == "__main__":
    main()
