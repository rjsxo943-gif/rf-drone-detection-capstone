from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import calibrate_noise_from_blocks  # noqa: E402
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


def _get_energy_cfg(detect_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    detect.yaml 구조가 아래 둘 중 어느 형태여도 대응한다.

    1)
    energy_detector:
      method: time_power

    2)
    method: time_power
    """
    if "energy_detector" in detect_cfg and isinstance(detect_cfg["energy_detector"], dict):
        return detect_cfg["energy_detector"]

    return detect_cfg


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


def collect_noise_blocks(
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
    """
    blocks: list[np.ndarray] = []

    for block_index in range(num_blocks):
        block = receiver.read_block(block_size)

        block = np.asarray(block)
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
        description="Run noise calibration and save noise floor / threshold."
    )

    parser.add_argument(
        "--receiver-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "receiver.yaml",
        help="Path to receiver.yaml",
    )

    parser.add_argument(
        "--detect-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "detect.yaml",
        help="Path to detect.yaml",
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Number of IQ blocks for noise calibration. Default: detect config calibration_blocks or 50.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="IQ samples per block. Default: receiver config block_size/num_samples or 16384.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["time_power", "fft_power"],
        help="Energy method. Default: detect config method or time_power.",
    )

    parser.add_argument(
        "--frame-size",
        type=int,
        default=None,
        help="Frame size for energy detector. Default: detect config frame_size or 1024.",
    )

    parser.add_argument(
        "--hop-size",
        type=int,
        default=None,
        help="Hop size for energy detector. Default: detect config hop_size or 512.",
    )

    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=None,
        help="Threshold multiplier. Default: detect config threshold_multiplier or 5.0.",
    )

    parser.add_argument(
        "--min-detection-ratio",
        type=float,
        default=None,
        help="Min detection ratio. Default: detect config min_detection_ratio or 0.05.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibration" / "noise_latest.json",
        help="Output calibration JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    receiver_cfg_raw = load_yaml(args.receiver_config)
    detect_cfg_raw = load_yaml(args.detect_config)

    receiver_cfg = _unwrap_section(receiver_cfg_raw, "receiver")
    energy_cfg = _get_energy_cfg(_unwrap_section(detect_cfg_raw, "detect"))

    block_size = int(
        args.block_size
        or _get_cfg_value(receiver_cfg, ["block_size", "num_samples"], 16_384)
    )

    num_blocks = int(
        args.num_blocks
        or _get_cfg_value(energy_cfg, ["calibration_blocks"], 50)
    )

    method = str(
        args.method
        or _get_cfg_value(energy_cfg, ["method"], "time_power")
    )

    frame_size = int(
        args.frame_size
        or _get_cfg_value(energy_cfg, ["frame_size"], 1024)
    )

    hop_size = int(
        args.hop_size
        or _get_cfg_value(energy_cfg, ["hop_size"], 512)
    )

    threshold_multiplier = float(
        args.threshold_multiplier
        or _get_cfg_value(energy_cfg, ["threshold_multiplier"], 5.0)
    )

    min_detection_ratio = float(
        args.min_detection_ratio
        or _get_cfg_value(energy_cfg, ["min_detection_ratio"], 0.05)
    )

    sample_rate = _get_cfg_value(receiver_cfg, ["sample_rate", "fs"], None)
    center_freq = _get_cfg_value(receiver_cfg, ["center_freq", "frequency"], None)

    # receiver에도 block_size/num_samples를 명시해 둔다.
    # receiver factory가 어떤 키를 쓰더라도 최대한 맞게 들어가도록 둘 다 넣는다.
    receiver_cfg = dict(receiver_cfg)
    receiver_cfg["block_size"] = block_size
    receiver_cfg["num_samples"] = block_size

    print("=== Noise Calibration Runner ===")
    print(f"receiver_config      : {args.receiver_config}")
    print(f"detect_config        : {args.detect_config}")
    print(f"num_blocks           : {num_blocks}")
    print(f"block_size           : {block_size}")
    print(f"sample_rate          : {sample_rate}")
    print(f"center_freq          : {center_freq}")
    print(f"method               : {method}")
    print(f"frame_size           : {frame_size}")
    print(f"hop_size             : {hop_size}")
    print(f"threshold_multiplier : {threshold_multiplier}")
    print(f"min_detection_ratio  : {min_detection_ratio}")
    print(f"output               : {args.output}")
    print()
    print("[주의] 노이즈 캘리브레이션 중에는 의도적인 신호원을 켜지 않는 것이 좋다.")
    print("[주의] WiFi/블루투스/드론 신호가 강하면 noise_floor가 과대평가될 수 있다.")
    print()

    receiver = build_receiver(receiver_cfg)

    try:
        blocks = collect_noise_blocks(
            receiver,
            num_blocks=num_blocks,
            block_size=block_size,
        )
    finally:
        _close_receiver(receiver)

    result = calibrate_noise_from_blocks(
        blocks,
        method=method,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold_multiplier=threshold_multiplier,
        calibration_blocks=num_blocks,
        min_detection_ratio=min_detection_ratio,
        sample_rate=float(sample_rate) if sample_rate is not None else None,
        center_freq=float(center_freq) if center_freq is not None else None,
    )

    save_path = result.save_json(args.output)

    print()
    print("=== Noise Calibration Result ===")
    print(f"num_blocks  : {result.num_blocks}")
    print(f"block_size  : {result.block_size}")
    print(f"channels    : {result.num_channels}")
    print(f"noise_floor : {result.noise_floor:.10g}")
    print(f"noise_mean  : {result.noise_mean:.10g}")
    print(f"noise_std   : {result.noise_std:.10g}")
    print(f"noise_min   : {result.noise_min:.10g}")
    print(f"noise_max   : {result.noise_max:.10g}")
    print(f"threshold   : {result.threshold:.10g}")
    print(f"saved to    : {save_path}")


if __name__ == "__main__":
    main()