#src/preprocess/__init__.py
"""
Preprocessing package for RF IQ blocks.

현재 프로젝트 기준:
- 입력 IQ block shape: (num_channels, 16384)
- 공통 전처리: DC offset 제거
- CNN branch 옵션: IQ amplitude normalization, RX 채널 선택
- AoA branch 전처리: phase offset correction
"""

from src.preprocess.dc_blocker import (
    remove_dc_offset,
    estimate_dc_offset,
    DCBlocker,
)

from src.preprocess.framing import (
    ensure_2d_iq,
    split_into_blocks,
    get_num_blocks,
    frame_signal,
)

from src.preprocess.iq_normalizer import (
    normalize_iq,
    estimate_iq_scale,
    IQNormalizer,
)

from src.preprocess.channel_filter import (
    select_rx,
    get_cnn_input_iq,
)

from src.preprocess.phaseoffset import (
    PhaseOffsetEstimate,
    PhaseOffsetCorrector,
    estimate_phase_offset,
    remove_phase_offset,
    estimate_and_remove_phase_offset,
    wrap_phase_rad,
)

__all__ = [
    # DC offset
    "remove_dc_offset",
    "estimate_dc_offset",
    "DCBlocker",

    # Framing / block
    "ensure_2d_iq",
    "split_into_blocks",
    "get_num_blocks",
    "frame_signal",

    # IQ normalization
    "normalize_iq",
    "estimate_iq_scale",
    "IQNormalizer",

    # Channel selection
    "select_rx",
    "get_cnn_input_iq",

    # Phase offset
    "PhaseOffsetEstimate",
    "PhaseOffsetCorrector",
    "estimate_phase_offset",
    "remove_phase_offset",
    "estimate_and_remove_phase_offset",
    "wrap_phase_rad",
]