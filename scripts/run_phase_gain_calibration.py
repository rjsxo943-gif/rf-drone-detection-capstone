from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import calibrate_phase_gain_from_blocks  # noqa: E402
from src.core import load_yaml  # noqa: E402
from src.receiver import build_receiver  # noqa: E402


def _unwrap_section(cfg: dict[str, Any], section_name: str) -> dict[str, Any]:
    """
    YAML 구조가 아래 둘 중 어느 형태여도 대응한다.

    1)
    source_type: sim
    sample_rate: 5000000

    2)
    receiver:
      source_type: sim
      sample_rate: 5000000
    """
    if section_name in cfg and isinstance(cfg[section_name], dict):
        return cfg[section_name]

    return cfg


def _get_cfg_value(
    cfg: dict[str, Any],
    keys: list[str],
    default: Any,
) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]

    return default


def _close_receiver(receiver: Any) -> None:
    close_fn = getattr(receiver, "close", None)

    if callable(close_fn):
        close_fn()


def collect_phase_gain_blocks(
    receiver: Any,
    *,
    num_blocks: int,
    block_size: int,
) -> list[np.ndarray]:
    """
    receiver에서 IQ block을 num_blocks개 수집한다.

    현재 프로젝트 receiver 기준:
    - read_block(block_size) 사용
    - 반환 shape: (num_channels, num_samples)
    - phase/gain calibration은 최소 2채널 필요
    """
    blocks: list[np.ndarray] = []

    for block_index in range(num_blocks):
        block = receiver.read_block(block_size)
        block = np.asarray(block)

        if block.ndim != 2:
            raise ValueError(
                f"Expected 2D IQ block, got shape={block.shape} at block {block_index}"
            )

        # 정상: (2, N)
        # 예외: (N, 2)
        if block.shape[0] < 2 and block.shape[1] >= 2:
            raise ValueError(
                f"Phase/gain calibration requires 2 channels. "
                f"Got shape={block.shape} at block {block_index}"
            )

        blocks.append(block)

        print(
            f"[collect] "
            f"{block_index + 1:03d}/{num_blocks:03d} "
            f"shape={block.shape} "
            f"dtype={block.dtype}"
        )

    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase/gain calibration and save gain correction / phase offset."
    )

    parser.add_argument(
        "--receiver-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "receiver.yaml",
        help="Path to receiver.yaml",
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=50,
        help="Number of IQ blocks for phase/gain calibration.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="IQ samples per block. Default: receiver config block_size/num_samples or 16384.",
    )

    parser.add_argument(
        "--ref-channel",
        type=int,
        default=0,
        help="Reference channel index. Default: 0",
    )

    parser.add_argument(
        "--target-channel",
        type=int,
        default=1,
        help="Target channel index to correct. Default: 1",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibration" / "phase_gain_latest.json",
        help="Output calibration JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    receiver_cfg_raw = load_yaml(args.receiver_config)
    receiver_cfg = _unwrap_section(receiver_cfg_raw, "receiver")

    block_size = int(
        args.block_size
        or _get_cfg_value(receiver_cfg, ["block_size", "num_samples"], 16_384)
    )

    sample_rate = _get_cfg_value(receiver_cfg, ["sample_rate", "fs"], None)
    center_freq = _get_cfg_value(receiver_cfg, ["center_freq", "frequency"], None)

    # receiver에도 block_size/num_samples를 명시해 둔다.
    # factory가 어떤 키를 쓰더라도 최대한 맞게 들어가도록 둘 다 넣는다.
    receiver_cfg = dict(receiver_cfg)
    receiver_cfg["block_size"] = block_size
    receiver_cfg["num_samples"] = block_size

    print("=== Phase/Gain Calibration Runner ===")
    print(f"receiver_config : {args.receiver_config}")
    print(f"num_blocks      : {args.num_blocks}")
    print(f"block_size      : {block_size}")
    print(f"sample_rate     : {sample_rate}")
    print(f"center_freq     : {center_freq}")
    print(f"ref_channel     : {args.ref_channel}")
    print(f"target_channel  : {args.target_channel}")
    print(f"output          : {args.output}")
    print()
    print("[주의] 위상/게인 캘리브레이션은 2채널 RX0/RX1 동시 수집이 필요하다.")
    print("[주의] 신호원은 두 안테나의 정면 0도 방향에 두는 것이 기준이다.")
    print("[주의] 이 결과는 target channel에 적용할 gain_correction과 phase_offset이다.")
    print()

    receiver = build_receiver(receiver_cfg)

    try:
        blocks = collect_phase_gain_blocks(
            receiver,
            num_blocks=args.num_blocks,
            block_size=block_size,
        )
    finally:
        _close_receiver(receiver)

    result = calibrate_phase_gain_from_blocks(
        blocks,
        ref_channel=args.ref_channel,
        target_channel=args.target_channel,
        sample_rate=float(sample_rate) if sample_rate is not None else None,
        center_freq=float(center_freq) if center_freq is not None else None,
    )

    save_path = result.save_json(args.output)

    print()
    print("=== Phase/Gain Calibration Result ===")
    print(f"num_blocks              : {result.num_blocks}")
    print(f"block_size              : {result.block_size}")
    print(f"channels                : {result.num_channels}")
    print(f"gain_ref_rms_mean       : {result.gain_ref_rms_mean:.10g}")
    print(f"gain_target_rms_mean    : {result.gain_target_rms_mean:.10g}")
    print(f"gain_correction_mean    : {result.gain_correction_mean:.10g}")
    print(f"gain_correction_std     : {result.gain_correction_std:.10g}")
    print(f"phase_offset_rad_mean   : {result.phase_offset_rad_mean:.10g}")
    print(f"phase_offset_rad_std    : {result.phase_offset_rad_std:.10g}")
    print(f"phase_offset_deg_mean   : {result.phase_offset_deg_mean:.6f} deg")
    print(f"phase_offset_deg_std    : {result.phase_offset_deg_std:.6f} deg")
    print(f"coherence_like_mean     : {result.coherence_like_mean:.10g}")
    print(f"coherence_like_std      : {result.coherence_like_std:.10g}")
    print(f"saved to                : {save_path}")


if __name__ == "__main__":
    main()