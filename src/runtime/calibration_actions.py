from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.calibration import (
    NoiseCalibrationResult,
    PhaseGainCalibrationResult,
    calibrate_noise_from_blocks,
    calibrate_phase_gain_from_blocks,
    calibrate_noise_by_gain,
    calibrate_phase_gain_by_gain,
)
from src.core import load_yaml
from src.receiver import build_receiver


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RECEIVER_CONFIG = PROJECT_ROOT / "configs" / "receiver.yaml"
DEFAULT_DETECT_CONFIG = PROJECT_ROOT / "configs" / "detect.yaml"

DEFAULT_NOISE_OUTPUT = PROJECT_ROOT / "outputs" / "calibration" / "noise_latest.json"
DEFAULT_PHASE_GAIN_OUTPUT = (
    PROJECT_ROOT / "outputs" / "calibration" / "phase_gain_latest.json"
)

# Gain-wise calibration outputs used by live viewer / CLI / common runtime.
DEFAULT_GAIN_LIST = (20.0, 25.0, 30.0, 35.0, 40.0)
DEFAULT_GAIN_NOISE_OUTPUT = (
    PROJECT_ROOT / "outputs" / "calibration" / "noise_by_gain_latest.json"
)
DEFAULT_GAIN_PHASE_GAIN_OUTPUT = (
    PROJECT_ROOT / "outputs" / "calibration" / "phase_gain_by_gain_latest.json"
)


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


def _set_receiver_gain(receiver: Any, gain: float) -> None:
    """Best-effort gain update for PlutoReceiver while keeping SimReceiver safe."""
    gain_value = float(gain)

    if hasattr(receiver, "gain"):
        receiver.gain = gain_value

    set_channel_gain = getattr(receiver, "_set_channel_gain", None)
    channels = getattr(receiver, "channels", None)

    if callable(set_channel_gain) and channels is not None:
        for ch in channels:
            set_channel_gain(ch)


def _prepare_receiver_cfg(
    receiver_config: str | Path,
    *,
    block_size: int | None = None,
) -> tuple[dict[str, Any], int, float | None, float | None]:
    """
    receiver.yaml을 읽고 runtime action에서 쓰기 좋은 형태로 정리한다.
    """
    receiver_cfg_raw = load_yaml(receiver_config)
    receiver_cfg = _unwrap_section(receiver_cfg_raw, "receiver")

    final_block_size = int(
        block_size
        or _get_cfg_value(receiver_cfg, ["block_size", "num_samples"], 16_384)
    )

    sample_rate = _get_cfg_value(receiver_cfg, ["sample_rate", "fs"], None)
    center_freq = _get_cfg_value(receiver_cfg, ["center_freq", "frequency"], None)

    receiver_cfg = dict(receiver_cfg)
    receiver_cfg["block_size"] = final_block_size
    receiver_cfg["num_samples"] = final_block_size

    return receiver_cfg, final_block_size, sample_rate, center_freq


def collect_blocks(
    receiver: Any,
    *,
    num_blocks: int,
    block_size: int,
    label: str = "collect",
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    receiver에서 IQ block을 여러 개 수집한다.

    현재 프로젝트 기준:
    - receiver.read_block(block_size)
    - 반환 shape = (num_channels, num_samples)
    """
    blocks: list[np.ndarray] = []

    for block_index in range(num_blocks):
        block = receiver.read_block(block_size)
        block = np.asarray(block)

        blocks.append(block)

        if verbose:
            print(
                f"[{label}] "
                f"{block_index + 1:03d}/{num_blocks:03d} "
                f"shape={block.shape} "
                f"dtype={block.dtype}"
            )

    return blocks



def run_gain_wise_noise_calibration_action(
    *,
    receiver_config: str | Path = DEFAULT_RECEIVER_CONFIG,
    detect_config: str | Path = DEFAULT_DETECT_CONFIG,
    output: str | Path = DEFAULT_GAIN_NOISE_OUTPUT,
    gain_list: list[int | float] | tuple[int | float, ...] = DEFAULT_GAIN_LIST,
    num_blocks_per_gain: int | None = None,
    block_size: int | None = None,
    method: str | None = None,
    frame_size: int | None = None,
    hop_size: int | None = None,
    threshold_multiplier: float | None = None,
    min_detection_ratio: float | None = None,
    verbose: bool = True,
):
    """
    Gain-wise noise calibration action.

    Output:
        outputs/calibration/noise_by_gain_latest.json
    """
    receiver_cfg, final_block_size, sample_rate, center_freq = _prepare_receiver_cfg(
        receiver_config,
        block_size=block_size,
    )

    detect_cfg_raw = load_yaml(detect_config)
    detect_cfg = _unwrap_section(detect_cfg_raw, "detect")
    energy_cfg = _get_energy_cfg(detect_cfg)

    final_num_blocks = int(
        num_blocks_per_gain
        or _get_cfg_value(energy_cfg, ["calibration_blocks"], 50)
    )
    final_method = str(method or _get_cfg_value(energy_cfg, ["method"], "time_power"))
    final_frame_size = int(frame_size or _get_cfg_value(energy_cfg, ["frame_size"], 1024))
    final_hop_size = int(hop_size or _get_cfg_value(energy_cfg, ["hop_size"], 512))
    final_threshold_multiplier = float(
        threshold_multiplier
        or _get_cfg_value(energy_cfg, ["threshold_multiplier"], 5.0)
    )
    final_min_detection_ratio = float(
        min_detection_ratio
        or _get_cfg_value(energy_cfg, ["min_detection_ratio"], 0.05)
    )

    if verbose:
        print("=== Gain-wise Noise Calibration Action ===")
        print(f"receiver_config      : {receiver_config}")
        print(f"detect_config        : {detect_config}")
        print(f"gain_list            : {list(gain_list)}")
        print(f"num_blocks_per_gain  : {final_num_blocks}")
        print(f"block_size           : {final_block_size}")
        print(f"sample_rate          : {sample_rate}")
        print(f"center_freq          : {center_freq}")
        print(f"method               : {final_method}")
        print(f"frame_size           : {final_frame_size}")
        print(f"hop_size             : {final_hop_size}")
        print(f"threshold_multiplier : {final_threshold_multiplier}")
        print(f"min_detection_ratio  : {final_min_detection_ratio}")
        print(f"output               : {output}")
        print()

    receiver = build_receiver(receiver_cfg)

    def collect_for_gain(gain: float, num_blocks: int) -> list[np.ndarray]:
        _set_receiver_gain(receiver, gain)
        return collect_blocks(
            receiver,
            num_blocks=int(num_blocks),
            block_size=final_block_size,
            label=f"noise_g{float(gain):g}",
            verbose=verbose,
        )

    try:
        result = calibrate_noise_by_gain(
            list(gain_list),
            collect_for_gain,
            num_blocks_per_gain=final_num_blocks,
            method=final_method,
            frame_size=final_frame_size,
            hop_size=final_hop_size,
            threshold_multiplier=final_threshold_multiplier,
            min_detection_ratio=final_min_detection_ratio,
            sample_rate=float(sample_rate) if sample_rate is not None else None,
            center_freq=float(center_freq) if center_freq is not None else None,
        )
    finally:
        _close_receiver(receiver)

    result.save_json(output)

    if verbose:
        print()
        print("=== Gain-wise Noise Calibration Saved ===")
        print(f"saved to : {output}")

    return result


def run_gain_wise_phase_gain_calibration_action(
    *,
    receiver_config: str | Path = DEFAULT_RECEIVER_CONFIG,
    output: str | Path = DEFAULT_GAIN_PHASE_GAIN_OUTPUT,
    gain_list: list[int | float] | tuple[int | float, ...] = DEFAULT_GAIN_LIST,
    num_blocks_per_gain: int = 50,
    block_size: int | None = None,
    ref_channel: int = 0,
    target_channel: int = 1,
    verbose: bool = True,
):
    """
    Gain-wise phase/gain calibration action.

    Output:
        outputs/calibration/phase_gain_by_gain_latest.json
    """
    receiver_cfg, final_block_size, sample_rate, center_freq = _prepare_receiver_cfg(
        receiver_config,
        block_size=block_size,
    )

    if verbose:
        print("=== Gain-wise Phase/Gain Calibration Action ===")
        print(f"receiver_config      : {receiver_config}")
        print(f"gain_list            : {list(gain_list)}")
        print(f"num_blocks_per_gain  : {num_blocks_per_gain}")
        print(f"block_size           : {final_block_size}")
        print(f"sample_rate          : {sample_rate}")
        print(f"center_freq          : {center_freq}")
        print(f"ref_channel          : {ref_channel}")
        print(f"target_channel       : {target_channel}")
        print(f"output               : {output}")
        print()

    receiver = build_receiver(receiver_cfg)

    def collect_for_gain(gain: float, num_blocks: int) -> list[np.ndarray]:
        _set_receiver_gain(receiver, gain)
        return collect_blocks(
            receiver,
            num_blocks=int(num_blocks),
            block_size=final_block_size,
            label=f"phase_gain_g{float(gain):g}",
            verbose=verbose,
        )

    try:
        result = calibrate_phase_gain_by_gain(
            list(gain_list),
            collect_for_gain,
            num_blocks_per_gain=int(num_blocks_per_gain),
            ref_channel=int(ref_channel),
            target_channel=int(target_channel),
            sample_rate=float(sample_rate) if sample_rate is not None else None,
            center_freq=float(center_freq) if center_freq is not None else None,
        )
    finally:
        _close_receiver(receiver)

    result.save_json(output)

    if verbose:
        print()
        print("=== Gain-wise Phase/Gain Calibration Saved ===")
        print(f"saved to : {output}")

    return result


def run_noise_calibration_action(
    *,
    receiver_config: str | Path = DEFAULT_RECEIVER_CONFIG,
    detect_config: str | Path = DEFAULT_DETECT_CONFIG,
    output: str | Path = DEFAULT_NOISE_OUTPUT,
    num_blocks: int | None = None,
    block_size: int | None = None,
    method: str | None = None,
    frame_size: int | None = None,
    hop_size: int | None = None,
    threshold_multiplier: float | None = None,
    min_detection_ratio: float | None = None,
    verbose: bool = True,
) -> NoiseCalibrationResult:
    """
    노이즈 캘리브레이션 action.

    역할:
    1. receiver 생성
    2. IQ block 여러 개 수집
    3. calibrate_noise_from_blocks() 호출
    4. noise_latest.json 저장
    5. NoiseCalibrationResult 반환
    """
    receiver_cfg, final_block_size, sample_rate, center_freq = _prepare_receiver_cfg(
        receiver_config,
        block_size=block_size,
    )

    detect_cfg_raw = load_yaml(detect_config)
    detect_cfg = _unwrap_section(detect_cfg_raw, "detect")
    energy_cfg = _get_energy_cfg(detect_cfg)

    final_num_blocks = int(
        num_blocks
        or _get_cfg_value(energy_cfg, ["calibration_blocks"], 50)
    )

    final_method = str(
        method
        or _get_cfg_value(energy_cfg, ["method"], "time_power")
    )

    final_frame_size = int(
        frame_size
        or _get_cfg_value(energy_cfg, ["frame_size"], 1024)
    )

    final_hop_size = int(
        hop_size
        or _get_cfg_value(energy_cfg, ["hop_size"], 512)
    )

    final_threshold_multiplier = float(
        threshold_multiplier
        or _get_cfg_value(energy_cfg, ["threshold_multiplier"], 5.0)
    )

    final_min_detection_ratio = float(
        min_detection_ratio
        or _get_cfg_value(energy_cfg, ["min_detection_ratio"], 0.05)
    )

    if verbose:
        print("=== Noise Calibration Action ===")
        print(f"receiver_config      : {receiver_config}")
        print(f"detect_config        : {detect_config}")
        print(f"num_blocks           : {final_num_blocks}")
        print(f"block_size           : {final_block_size}")
        print(f"sample_rate          : {sample_rate}")
        print(f"center_freq          : {center_freq}")
        print(f"method               : {final_method}")
        print(f"frame_size           : {final_frame_size}")
        print(f"hop_size             : {final_hop_size}")
        print(f"threshold_multiplier : {final_threshold_multiplier}")
        print(f"min_detection_ratio  : {final_min_detection_ratio}")
        print(f"output               : {output}")
        print()

    receiver = build_receiver(receiver_cfg)

    try:
        blocks = collect_blocks(
            receiver,
            num_blocks=final_num_blocks,
            block_size=final_block_size,
            label="noise",
            verbose=verbose,
        )
    finally:
        _close_receiver(receiver)

    result = calibrate_noise_from_blocks(
        blocks,
        method=final_method,
        frame_size=final_frame_size,
        hop_size=final_hop_size,
        threshold_multiplier=final_threshold_multiplier,
        calibration_blocks=final_num_blocks,
        min_detection_ratio=final_min_detection_ratio,
        sample_rate=float(sample_rate) if sample_rate is not None else None,
        center_freq=float(center_freq) if center_freq is not None else None,
    )

    result.save_json(output)

    if verbose:
        print()
        print("=== Noise Calibration Result ===")
        print(f"noise_floor : {result.noise_floor:.10g}")
        print(f"noise_mean  : {result.noise_mean:.10g}")
        print(f"noise_std   : {result.noise_std:.10g}")
        print(f"threshold   : {result.threshold:.10g}")
        print(f"saved to    : {output}")

    return result


def run_phase_gain_calibration_action(
    *,
    receiver_config: str | Path = DEFAULT_RECEIVER_CONFIG,
    output: str | Path = DEFAULT_PHASE_GAIN_OUTPUT,
    num_blocks: int = 50,
    block_size: int | None = None,
    ref_channel: int = 0,
    target_channel: int = 1,
    verbose: bool = True,
) -> PhaseGainCalibrationResult:
    """
    위상/게인 캘리브레이션 action.

    역할:
    1. receiver 생성
    2. IQ block 여러 개 수집
    3. calibrate_phase_gain_from_blocks() 호출
    4. phase_gain_latest.json 저장
    5. PhaseGainCalibrationResult 반환
    """
    receiver_cfg, final_block_size, sample_rate, center_freq = _prepare_receiver_cfg(
        receiver_config,
        block_size=block_size,
    )

    if verbose:
        print("=== Phase/Gain Calibration Action ===")
        print(f"receiver_config : {receiver_config}")
        print(f"num_blocks      : {num_blocks}")
        print(f"block_size      : {final_block_size}")
        print(f"sample_rate     : {sample_rate}")
        print(f"center_freq     : {center_freq}")
        print(f"ref_channel     : {ref_channel}")
        print(f"target_channel  : {target_channel}")
        print(f"output          : {output}")
        print()

    receiver = build_receiver(receiver_cfg)

    try:
        blocks = collect_blocks(
            receiver,
            num_blocks=num_blocks,
            block_size=final_block_size,
            label="phase_gain",
            verbose=verbose,
        )
    finally:
        _close_receiver(receiver)

    result = calibrate_phase_gain_from_blocks(
        blocks,
        ref_channel=ref_channel,
        target_channel=target_channel,
        sample_rate=float(sample_rate) if sample_rate is not None else None,
        center_freq=float(center_freq) if center_freq is not None else None,
    )

    result.save_json(output)

    if verbose:
        print()
        print("=== Phase/Gain Calibration Result ===")
        print(f"gain_correction_mean  : {result.gain_correction_mean:.10g}")
        print(f"gain_correction_std   : {result.gain_correction_std:.10g}")
        print(f"phase_offset_rad_mean : {result.phase_offset_rad_mean:.10g}")
        print(f"phase_offset_deg_mean : {result.phase_offset_deg_mean:.6f} deg")
        print(f"coherence_like_mean   : {result.coherence_like_mean:.10g}")
        print(f"saved to              : {output}")

    return result