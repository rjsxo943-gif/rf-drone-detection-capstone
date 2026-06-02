from src.calibration.noise_calibration import (
    NoiseCalibrationResult,
    calibrate_noise_from_blocks,
)

from src.calibration.phase_gain_calibration import (
    PhaseGainCalibrationResult,
    calibrate_phase_gain_from_blocks,
)

from src.calibration.gain_phase_table import (
    build_gain_phase_table,
    circular_distance_rad,
    circular_weighted_mean_rad,
    compute_block_phase_and_coherence,
    dominant_cluster_phase,
    evaluate_table_entry,
    get_phase_offset_to_apply,
    interpolate_phase_delta,
    load_gain_phase_table,
    select_dominant_cluster,
    wrap_phase_rad,
)

from src.calibration.gain_noise_calibration import (
    GainNoiseCalibrationResult,
    GainNoiseCalibrationSet,
    calibrate_noise_by_gain,
    gain_to_key,
    get_noise_profile_for_gain,
    get_noise_threshold_for_gain,
    load_gain_noise_calibration,
)

from src.calibration.raw_iq_safety import (
    RawIQSafetyResult,
    check_raw_iq_safety,
    is_raw_iq_safe,
    summarize_raw_iq_safety,
)

from src.calibration.phase_gain_by_gain_calibration import (
    GainPhaseGainCalibrationResult,
    GainPhaseGainCalibrationSet,
    calibrate_phase_gain_by_gain,
    get_phase_gain_correction_for_gain,
    get_phase_gain_profile_for_gain,
    load_phase_gain_by_gain_calibration,
)

__all__ = [
    "NoiseCalibrationResult",
    "calibrate_noise_from_blocks",
    "PhaseGainCalibrationResult",
    "calibrate_phase_gain_from_blocks",
    "build_gain_phase_table",
    "circular_distance_rad",
    "circular_weighted_mean_rad",
    "compute_block_phase_and_coherence",
    "dominant_cluster_phase",
    "evaluate_table_entry",
    "get_phase_offset_to_apply",
    "interpolate_phase_delta",
    "load_gain_phase_table",
    "select_dominant_cluster",
    "wrap_phase_rad",
    "GainNoiseCalibrationResult",
    "GainNoiseCalibrationSet",
    "calibrate_noise_by_gain",
    "gain_to_key",
    "get_noise_profile_for_gain",
    "get_noise_threshold_for_gain",
    "load_gain_noise_calibration",
    "RawIQSafetyResult",
    "check_raw_iq_safety",
    "is_raw_iq_safe",
    "summarize_raw_iq_safety",
    "GainPhaseGainCalibrationResult",
    "GainPhaseGainCalibrationSet",
    "calibrate_phase_gain_by_gain",
    "get_phase_gain_correction_for_gain",
    "get_phase_gain_profile_for_gain",
    "load_phase_gain_by_gain_calibration",
]

# ============================================================
# Legacy single-calibration exports
# ============================================================
# 기존 runtime/cnn_capture_actions.py, scan_actions.py, cli.py 호환용.
# gain-wise calibration 구조를 추가해도 기존 단일 calibration import는 유지해야 한다.

from src.calibration.params import (
    load_calibration_params,
)

from src.calibration.noise_calibration import (
    NoiseCalibrationResult,
    calibrate_noise_from_blocks,
)

from src.calibration.phase_gain_calibration import (
    PhaseGainCalibrationResult,
    calibrate_phase_gain_from_blocks,
)
