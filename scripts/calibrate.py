# calibrate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.aoa.phase_diff import estimate_phase_diff, wrap_phase_rad
from src.core.config import load_all_configs
from src.preprocess.dc_blocker import remove_dc_offset
from src.receiver.factory import build_receiver


def circular_mean_rad(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=np.float64)

    if angles_rad.size == 0:
        raise ValueError("angles_rad is empty.")

    mean_complex = np.mean(np.exp(1j * angles_rad))
    return float(np.angle(mean_complex))


def read_receiver_block(receiver: Any, block_size: int) -> np.ndarray:
    """
    receiver 구현마다 다른 block read 함수 이름을 흡수한다.
    """
    candidate_methods = [
        "read_block",
        "read_samples",
        "read",
        "receive",
        "get_block",
        "capture",
        "receive_samples",
        "get_samples",
        "sample",
        "recv",
    ]

    for method_name in candidate_methods:
        if hasattr(receiver, method_name):
            method = getattr(receiver, method_name)

            try:
                return method(block_size)
            except TypeError:
                return method()

    raise AttributeError(
        f"{type(receiver).__name__} has no supported block read method. "
        f"Available attributes: {sorted(dir(receiver))}"
    )

def close_receiver(receiver: Any) -> None:
    """
    close()가 있는 receiver만 닫는다.
    """
    if hasattr(receiver, "close"):
        receiver.close()


def ensure_channel_first_iq(block: np.ndarray) -> np.ndarray:
    """
    IQ block을 shape=(channels, samples) 형태로 맞춘다.

    허용:
    - (2, N)
    - (N, 2)
    """
    block = np.asarray(block)

    if block.ndim != 2:
        raise ValueError(
            f"IQ block must be 2-D for calibration. got shape={block.shape}"
        )

    if block.shape[0] >= 2:
        return block

    if block.shape[1] >= 2:
        return block.T

    raise ValueError(f"Calibration requires at least 2 channels. got shape={block.shape}")


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def save_yaml_file(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def update_aoa_yaml_phase_offset(
    aoa_yaml_path: str | Path,
    phase_offset_rad: float,
) -> None:
    aoa_yaml_path = Path(aoa_yaml_path)
    aoa_cfg = load_yaml_file(aoa_yaml_path)

    aoa_cfg["phase_offset_rad"] = float(phase_offset_rad)

    save_yaml_file(aoa_yaml_path, aoa_cfg)


def run_calibration(
    config_dir: str | Path = "configs",
    num_blocks: int = 20,
    min_coherence: float = 0.60,
    output_dir: str | Path = "outputs/runs/latest/calibration",
    write: bool = False,
) -> dict[str, Any]:
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = load_all_configs(config_dir)

    receiver_cfg = configs["receiver"]
    aoa_cfg = configs["aoa"]

    block_size = int(receiver_cfg.get("block_size", aoa_cfg.get("block_size", 16_384)))
    sample_rate = int(receiver_cfg.get("sample_rate", 5_000_000))
    center_freq = int(
        receiver_cfg.get(
            "center_freq",
            aoa_cfg.get("carrier_freq", 2_400_000_000),
        )
    )

    ref_channel = int(aoa_cfg.get("ref_channel", 0))
    target_channel = int(aoa_cfg.get("target_channel", 1))

    receiver = build_receiver(receiver_cfg)

    phase_offsets: list[float] = []
    coherences: list[float] = []

    try:
        for block_index in range(num_blocks):
            block = read_receiver_block(receiver, block_size)
            block = ensure_channel_first_iq(block)
            block_dc = remove_dc_offset(block)
            block_dc = ensure_channel_first_iq(block_dc)

            if block_dc.shape[0] < 2:
                raise ValueError(
                    f"Phase calibration requires 2 channels, got shape={block_dc.shape}. "
                    "For sim mode, set num_channels: 2 in receiver.yaml."
                )

            estimate = estimate_phase_diff(
                block_dc,
                ref_channel=ref_channel,
                target_channel=target_channel,
            )

            if estimate.coherence_like >= min_coherence:
                phase_offsets.append(estimate.phase_diff_rad)
                coherences.append(estimate.coherence_like)

            print(
                f"[block {block_index:04d}] "
                f"phase_offset={estimate.phase_diff_rad:+.6f} rad "
                f"({estimate.phase_diff_deg:+.2f} deg), "
                f"coherence={estimate.coherence_like:.4f}"
            )

    finally:
        close_receiver(receiver)

    if len(phase_offsets) == 0:
        raise RuntimeError(
            "No valid calibration blocks were collected. "
            f"Try lowering min_coherence or check RX0/RX1 signal. "
            f"min_coherence={min_coherence}"
        )

    phase_offsets_arr = np.asarray(phase_offsets, dtype=np.float64)
    coherences_arr = np.asarray(coherences, dtype=np.float64)

    final_offset_rad = circular_mean_rad(phase_offsets_arr)
    final_offset_rad = float(wrap_phase_rad(final_offset_rad))
    final_offset_deg = float(np.rad2deg(final_offset_rad))

    result = {
        "phase_offset_rad": final_offset_rad,
        "phase_offset_deg": final_offset_deg,
        "num_blocks_requested": int(num_blocks),
        "num_blocks_used": int(len(phase_offsets)),
        "min_coherence": float(min_coherence),
        "mean_coherence": float(np.mean(coherences_arr)),
        "median_coherence": float(np.median(coherences_arr)),
        "std_phase_offset_rad": float(np.std(phase_offsets_arr)),
        "std_phase_offset_deg": float(np.rad2deg(np.std(phase_offsets_arr))),
        "block_size": int(block_size),
        "sample_rate": int(sample_rate),
        "center_freq": int(center_freq),
        "ref_channel": int(ref_channel),
        "target_channel": int(target_channel),
    }

    result_path = output_dir / "phase_offset_calibration.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if write:
        aoa_yaml_path = config_dir / "aoa.yaml"
        update_aoa_yaml_phase_offset(
            aoa_yaml_path=aoa_yaml_path,
            phase_offset_rad=final_offset_rad,
        )
        result["updated_aoa_yaml"] = str(aoa_yaml_path)
    else:
        result["updated_aoa_yaml"] = None

    print()
    print("=== Phase Offset Calibration Result ===")
    print(f"phase_offset_rad: {final_offset_rad:+.8f}")
    print(f"phase_offset_deg: {final_offset_deg:+.4f}")
    print(f"used blocks     : {len(phase_offsets)} / {num_blocks}")
    print(f"mean coherence  : {np.mean(coherences_arr):.4f}")
    print(f"saved to        : {result_path}")

    if write:
        print(f"aoa.yaml updated: {config_dir / 'aoa.yaml'}")
    else:
        print()
        print("Copy this value into configs/aoa.yaml if it looks valid:")
        print(f"phase_offset_rad: {final_offset_rad:+.8f}")
        print()
        print("Or run again with --write to update configs/aoa.yaml automatically.")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate RX0/RX1 fixed phase offset for AoA."
    )

    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--min-coherence", type=float, default=0.60)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/runs/latest/calibration",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update configs/aoa.yaml phase_offset_rad automatically.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_calibration(
        config_dir=args.config_dir,
        num_blocks=args.num_blocks,
        min_coherence=args.min_coherence,
        output_dir=args.output_dir,
        write=args.write,
    )


if __name__ == "__main__":
    main()