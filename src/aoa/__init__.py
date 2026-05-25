from __future__ import annotations

# ============================================================
# Phase difference
# ============================================================

from .phase_diff import (
    PhaseDiffResult,
    compute_instant_phase_diff,
    estimate_phase_diff,
    estimate_phase_diff_deg,
    estimate_phase_diff_rad,
    wrap_phase_rad,
)

# ============================================================
# Angle estimation
# ============================================================

from .angle_estimator import (
    AngleEstimateResult,
    angle_to_phase_diff,
    estimate_angle_from_phase_result,
    phase_diff_to_angle,
)

# ============================================================
# Coherence
# ============================================================

from .coherence import (
    CoherenceResult,
    coherence_gate,
    compute_stft_coherence,
)

# ============================================================
# AoA compute gate
# ============================================================

from .aoa_gate import (
    AoAComputeGate,
    AoAComputeGateResult,
    normalize_class_name,
    should_compute_aoa,
)

# ============================================================
# Sector quantizer
# ============================================================

from .sector_quantizer import (
    SectorResult,
    SectorVoter,
    quantize_front_angle_to_sector,
    sector_index_to_label,
)


__all__ = [
    # phase_diff
    "PhaseDiffResult",
    "wrap_phase_rad",
    "estimate_phase_diff",
    "estimate_phase_diff_rad",
    "estimate_phase_diff_deg",
    "compute_instant_phase_diff",

    # angle_estimator
    "AngleEstimateResult",
    "phase_diff_to_angle",
    "estimate_angle_from_phase_result",
    "angle_to_phase_diff",

    # coherence
    "CoherenceResult",
    "compute_stft_coherence",
    "coherence_gate",

    # aoa_gate
    "AoAComputeGateResult",
    "AoAComputeGate",
    "normalize_class_name",
    "should_compute_aoa",

    # sector quantizer
    "SectorResult",
    "SectorVoter",
    "quantize_front_angle_to_sector",
    "sector_index_to_label",
]