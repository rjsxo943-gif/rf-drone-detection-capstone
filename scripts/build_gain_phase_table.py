from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from src.receiver.pluto_receiver import PlutoReceiver
from src.calibration import build_gain_phase_table


def parse_gain_list(text: str) -> list[int]:
    """
    "20,25,30,35" -> [20, 25, 30, 35]
    """
    gains = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        gains.append(int(item))

    if not gains:
        raise ValueError("gain list is empty.")

    return gains


class PlutoGainBlockCollector:
    """
    gain별로 PlutoReceiver를 재생성해서 n_blocks 수집.

    주의:
    - gain을 바꿀 때마다 receiver를 새로 만들면 SDR 내부 상태가 바뀔 수 있다.
    - 하지만 gain별 phase table 제작 목적에서는 gain 변경 후 안정화까지 포함해 보는 것이 현실적이다.
    - 각 gain마다 warmup_reads를 충분히 준다.
    """

    def __init__(
        self,
        uri: str,
        center_freq: int,
        sample_rate: int,
        block_size: int,
        warmup_reads: int,
        rf_bandwidth: int | None = None,
    ) -> None:
        self.uri = uri
        self.center_freq = int(center_freq)
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.warmup_reads = int(warmup_reads)
        self.rf_bandwidth = rf_bandwidth

    def __call__(self, gain: int, n_blocks: int):
        receiver = PlutoReceiver(
            uri=self.uri,
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            num_channels=2,
            channels=[0, 1],
            gain_control_mode="manual",
            gain=float(gain),
            block_size=self.block_size,
            rf_bandwidth=self.rf_bandwidth,
            warmup_reads=self.warmup_reads,
        )

        blocks_ch0: list[np.ndarray] = []
        blocks_ch1: list[np.ndarray] = []

        try:
            for block_idx in range(int(n_blocks)):
                iq = receiver.read_block(self.block_size)

                if iq.shape[0] < 2:
                    raise RuntimeError(f"Need 2 RX channels, got shape={iq.shape}")

                blocks_ch0.append(iq[0].copy())
                blocks_ch1.append(iq[1].copy())

                if (block_idx + 1) % 50 == 0:
                    print(f"    collected {block_idx + 1}/{n_blocks} blocks")

        finally:
            receiver.close()

        return blocks_ch0, blocks_ch1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gain-dependent phase delta table for RX0/RX1 AoA calibration."
    )

    parser.add_argument("--uri", type=str, default="ip:192.168.2.1")
    parser.add_argument("--center-freq", type=int, default=2_450_000_000)
    parser.add_argument("--signal-freq", type=int, default=2_452_000_000)
    parser.add_argument("--sample-rate", type=int, default=5_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16_384)

    parser.add_argument(
        "--gains",
        type=str,
        default="20,25,30,35",
        help="Comma-separated gain list. Example: 20,25,30,35",
    )
    parser.add_argument("--reference-gain", type=int, default=30)

    parser.add_argument("--total-blocks", type=int, default=200)
    parser.add_argument("--discard-blocks", type=int, default=30)
    parser.add_argument("--warmup-reads", type=int, default=20)

    parser.add_argument("--coherence-threshold", type=float, default=0.50)
    parser.add_argument("--cluster-window-deg", type=float, default=5.0)

    parser.add_argument(
        "--output",
        type=str,
        default="configs/calibration/gain_phase_table_2450.json",
    )
    parser.add_argument(
        "--memo",
        type=str,
        default="outdoor gain-dependent phase delta table",
    )

    args = parser.parse_args()

    gain_list = parse_gain_list(args.gains)

    if args.reference_gain not in gain_list:
        raise ValueError(
            f"reference_gain={args.reference_gain} must be included in gains={gain_list}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== Build Gain Phase Table ===")
    print(f"uri                 : {args.uri}")
    print(f"center_freq          : {args.center_freq}")
    print(f"signal_freq          : {args.signal_freq}")
    print(f"signal_offset        : {args.signal_freq - args.center_freq} Hz")
    print(f"sample_rate          : {args.sample_rate}")
    print(f"block_size           : {args.block_size}")
    print(f"gains                : {gain_list}")
    print(f"reference_gain       : {args.reference_gain}")
    print(f"total_blocks         : {args.total_blocks}")
    print(f"discard_blocks       : {args.discard_blocks}")
    print(f"used_blocks          : {args.total_blocks - args.discard_blocks}")
    print(f"warmup_reads/gain    : {args.warmup_reads}")
    print(f"coherence_threshold  : {args.coherence_threshold}")
    print(f"cluster_window_deg   : {args.cluster_window_deg}")
    print(f"output               : {args.output}")
    print()
    print("[SETUP CHECK]")
    print("- 신호발생기 주파수는 --signal-freq 값과 맞추기")
    print("- 송신원은 RX0/RX1 배열 정면 0도")
    print("- 안테나/케이블/RX 포트/간격은 테이블 제작 중 절대 변경 금지")
    print("- 핫스팟/블루투스/불필요한 2.4GHz 장치는 가능하면 OFF")
    print("- 사람이 안테나 주변에 가까이 있지 않기")
    print()

    collector = PlutoGainBlockCollector(
        uri=args.uri,
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        warmup_reads=args.warmup_reads,
        rf_bandwidth=args.rf_bandwidth,
    )

    metadata = {
        "created_at": timestamp,
        "memo": args.memo,
        "uri": args.uri,
        "center_freq": args.center_freq,
        "signal_freq": args.signal_freq,
        "signal_offset_hz": args.signal_freq - args.center_freq,
        "sample_rate": args.sample_rate,
        "rf_bandwidth": args.rf_bandwidth,
        "block_size": args.block_size,
        "rx_channels": [0, 1],
        "method": "outdoor_0deg_gain_phase_delta_table",
        "notes": [
            "phase = RX1 phase - RX0 phase",
            "phase_delta is relative to reference_gain",
            "apply phase offset by multiplying RX1 with exp(-j * phase_offset_to_apply)",
        ],
    }

    table = build_gain_phase_table(
        gain_list=gain_list,
        collect_fn=collector,
        output_path=args.output,
        reference_gain=args.reference_gain,
        total_blocks=args.total_blocks,
        discard_blocks=args.discard_blocks,
        coherence_threshold=args.coherence_threshold,
        cluster_window_deg=args.cluster_window_deg,
        metadata=metadata,
    )

    print()
    print("=== Gain Phase Delta Summary ===")
    for gain in sorted(table.keys()):
        entry = table[gain]
        print(
            f"gain={gain:3d} | "
            f"phase={entry['phase_deg']:+8.3f} deg | "
            f"delta={entry['phase_delta_deg']:+8.3f} deg | "
            f"std={entry['phase_std_deg']:.3f} deg | "
            f"coh_med={entry['coherence_median']:.3f} | "
            f"cluster={entry['cluster_blocks']}/{entry['valid_blocks']} "
            f"({entry['cluster_ratio'] * 100:.1f}%) | "
            f"{entry['quality']}"
        )

    print()
    print(f"Saved: {Path(args.output)}")
    print()
    print("[NEXT]")
    print("현장 시작 시 reference_gain에서 current_ref_phase_offset을 한 번 측정한 뒤,")
    print("실시간 gain 변경 시 gain_phase_table의 delta를 더해서 보정하면 됨.")


if __name__ == "__main__":
    main()
