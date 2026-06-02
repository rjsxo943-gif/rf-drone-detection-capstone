from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.gain_noise_calibration import calibrate_noise_by_gain  # noqa: E402
from src.core import load_yaml  # noqa: E402
from src.receiver import build_receiver  # noqa: E402


def _unwrap_section(cfg: dict[str, Any], section_name: str) -> dict[str, Any]:
    if section_name in cfg and isinstance(cfg[section_name], dict):
        return cfg[section_name]
    return cfg


def _get_energy_cfg(detect_cfg: dict[str, Any]) -> dict[str, Any]:
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


def _set_receiver_gain(
    receiver: Any,
    gain: float,
    *,
    warmup_reads: int,
) -> float:
    set_gain_fn = getattr(receiver, "set_gain", None)
    if callable(set_gain_fn):
        return float(set_gain_fn(gain, warmup_reads=warmup_reads))

    if hasattr(receiver, "gain"):
        receiver.gain = float(gain)
        return float(receiver.gain)

    raise AttributeError(
        "Receiver does not support gain update. "
        "Expected receiver.set_gain(gain, warmup_reads=...) or receiver.gain."
    )


def _collect_blocks_at_current_gain(
    receiver: Any,
    *,
    num_blocks: int,
    block_size: int,
) -> list[np.ndarray]:
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


def _parse_gain_list(values: list[str]) -> list[float]:
    gains: list[float] = []
    for value in values:
        parts = value.replace(",", " ").split()
        for part in parts:
            gains.append(float(part))
    if not gains:
        raise ValueError("At least one gain must be provided.")
    return gains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gain-wise noise calibration and save JSON profile set."
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
        "--gains",
        nargs="+",
        default=["20", "25", "30", "35", "40"],
        help="Gain list. Default: 20 25 30 35 40. Example: --gains 20 25 30 35 40",
    )

    parser.add_argument(
        "--num-blocks-per-gain",
        type=int,
        default=None,
        help="Number of IQ blocks per gain. Default: detect config calibration_blocks or 50.",
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
        "--gain-warmup-reads",
        type=int,
        default=2,
        help="Number of discarded reads after each gain change.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibration" / "noise_by_gain_latest.json",
        help="Output gain-wise calibration JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    receiver_cfg_raw = load_yaml(args.receiver_config)
    detect_cfg_raw = load_yaml(args.detect_config)

    receiver_cfg = _unwrap_section(receiver_cfg_raw, "receiver")
    energy_cfg = _get_energy_cfg(_unwrap_section(detect_cfg_raw, "detect"))

    gain_list = _parse_gain_list(args.gains)

    block_size = int(
        args.block_size
        or _get_cfg_value(receiver_cfg, ["block_size", "num_samples"], 16_384)
    )

    num_blocks_per_gain = int(
        args.num_blocks_per_gain
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

    receiver_cfg = dict(receiver_cfg)
    receiver_cfg["block_size"] = block_size
    receiver_cfg["num_samples"] = block_size

    # 최초 receiver 생성 gain은 gain sweep 첫 값으로 맞춘다.
    receiver_cfg["gain"] = gain_list[0]
    receiver_cfg.setdefault("gain_control_mode", "manual")

    print("=== Gain-wise Noise Calibration Runner ===")
    print(f"receiver_config       : {args.receiver_config}")
    print(f"detect_config         : {args.detect_config}")
    print(f"gain_list             : {gain_list}")
    print(f"num_blocks_per_gain   : {num_blocks_per_gain}")
    print(f"block_size            : {block_size}")
    print(f"sample_rate           : {sample_rate}")
    print(f"center_freq           : {center_freq}")
    print(f"method                : {method}")
    print(f"frame_size            : {frame_size}")
    print(f"hop_size              : {hop_size}")
    print(f"threshold_multiplier  : {threshold_multiplier}")
    print(f"min_detection_ratio   : {min_detection_ratio}")
    print(f"gain_warmup_reads     : {args.gain_warmup_reads}")
    print(f"output                : {args.output}")
    print()
    print("[주의] gain별 노이즈 캘리브레이션 중에는 의도적인 신호원을 켜지 않는 것이 좋다.")
    print("[주의] gain이 올라갈수록 주변 Wi-Fi/블루투스가 threshold에 더 크게 반영될 수 있다.")
    print()

    receiver = build_receiver(receiver_cfg)

    def collect_fn(gain: float, n_blocks: int) -> list[np.ndarray]:
        applied_gain = _set_receiver_gain(
            receiver,
            float(gain),
            warmup_reads=args.gain_warmup_reads,
        )
        print(f"[gain] requested={gain:g}, applied={applied_gain:g}")
        return _collect_blocks_at_current_gain(
            receiver,
            num_blocks=n_blocks,
            block_size=block_size,
        )

    try:
        result_set = calibrate_noise_by_gain(
            gain_list=gain_list,
            collect_fn=collect_fn,
            num_blocks_per_gain=num_blocks_per_gain,
            method=method,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_multiplier=threshold_multiplier,
            min_detection_ratio=min_detection_ratio,
            sample_rate=float(sample_rate) if sample_rate is not None else None,
            center_freq=float(center_freq) if center_freq is not None else None,
        )
    finally:
        _close_receiver(receiver)

    save_path = result_set.save_json(args.output)

    print()
    print("=== Gain-wise Noise Calibration Result ===")
    print(f"num_gains            : {result_set.num_gains}")
    print(f"num_blocks_per_gain  : {result_set.num_blocks_per_gain}")
    print(f"saved to             : {save_path}")
    print()

    for key in sorted(result_set.profiles.keys(), key=lambda x: float(x)):
        profile = result_set.profiles[key]
        print(
            f"gain={float(key):g} | "
            f"noise_floor={float(profile['noise_floor']):.10g} | "
            f"threshold={float(profile['threshold']):.10g} | "
            f"noise_std={float(profile['noise_std']):.10g}"
        )


if __name__ == "__main__":
    main()
