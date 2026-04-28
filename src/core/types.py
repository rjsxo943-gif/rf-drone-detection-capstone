#src/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


# ============================================================
# Basic type aliases
# ============================================================

ComplexArray: TypeAlias = NDArray[np.complex64]
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int_]

ClassLabel: TypeAlias = Literal[
    "background",
    "wifi",
    "bluetooth",
    "drone_like",
    "unknown",
]

SourceType: TypeAlias = Literal[
    "sim",
    "file",
    "sdr",
    "pluto",
]


# ============================================================
# Raw IQ block
# ============================================================

@dataclass
class RawIQBlock:
    """
    Pluto+ 또는 파일/시뮬레이션에서 읽은 1개 block 단위의 raw IQ 데이터.

    기준:
    - rx0_iq, rx1_iq는 1-D complex array
    - 두 채널의 길이는 같아야 함
    - block_index 기준으로 sample_start/sample_end 계산 가능
    """

    rx0_iq: ComplexArray
    rx1_iq: ComplexArray
    block_index: int
    sample_rate: float
    center_freq: float
    label: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rx0_iq = np.asarray(self.rx0_iq).astype(np.complex64, copy=False)
        self.rx1_iq = np.asarray(self.rx1_iq).astype(np.complex64, copy=False)

        if self.rx0_iq.ndim != 1:
            raise ValueError(f"rx0_iq must be 1-D. got shape={self.rx0_iq.shape}")

        if self.rx1_iq.ndim != 1:
            raise ValueError(f"rx1_iq must be 1-D. got shape={self.rx1_iq.shape}")

        if self.rx0_iq.shape != self.rx1_iq.shape:
            raise ValueError(
                f"rx0_iq and rx1_iq must have same shape. "
                f"got rx0={self.rx0_iq.shape}, rx1={self.rx1_iq.shape}"
            )

        self.block_index = int(self.block_index)
        self.sample_rate = float(self.sample_rate)
        self.center_freq = float(self.center_freq)

    @property
    def block_size(self) -> int:
        return int(self.rx0_iq.shape[0])

    @property
    def sample_start(self) -> int:
        return self.block_index * self.block_size

    @property
    def sample_end(self) -> int:
        return self.sample_start + self.block_size


# ============================================================
# STFT / Stage 1 types
# ============================================================

@dataclass(frozen=True)
class STFTParams:
    """
    STFT 계산에 사용하는 공통 파라미터.

    현재 프로젝트 기본값:
    - nperseg = 512
    - noverlap = 384
    - hop_length = 128
    - window = hann
    """

    nperseg: int = 512
    noverlap: int = 384
    hop_length: int = 128
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.nperseg <= 0:
            raise ValueError(f"nperseg must be positive. got {self.nperseg}")

        if self.noverlap < 0:
            raise ValueError(f"noverlap must be non-negative. got {self.noverlap}")

        if self.noverlap >= self.nperseg:
            raise ValueError(
                f"noverlap must be smaller than nperseg. "
                f"got noverlap={self.noverlap}, nperseg={self.nperseg}"
            )

        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive. got {self.hop_length}")


@dataclass
class Stage1Artifacts:
    """
    Stage 1에서 생성된 중간 산출물.

    기본 핵심:
    - cnn_spectrogram

    선택 산출물:
    - complex STFT
    - phase
    - log magnitude
    """

    block_index: int
    sample_rate: float
    center_freq: float
    block_size: int
    stft_params: STFTParams
    cnn_spectrogram: FloatArray

    rx0_complex_stft: ComplexArray | None = None
    rx1_complex_stft: ComplexArray | None = None
    rx0_phase: FloatArray | None = None
    rx1_phase: FloatArray | None = None
    rx0_log_magnitude: FloatArray | None = None
    rx1_log_magnitude: FloatArray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block_index = int(self.block_index)
        self.sample_rate = float(self.sample_rate)
        self.center_freq = float(self.center_freq)
        self.block_size = int(self.block_size)

        self.cnn_spectrogram = np.asarray(self.cnn_spectrogram).astype(
            np.float32,
            copy=False,
        )

        if self.cnn_spectrogram.size == 0:
            raise ValueError("cnn_spectrogram is empty.")

    @property
    def sample_start(self) -> int:
        return self.block_index * self.block_size

    @property
    def sample_end(self) -> int:
        return self.sample_start + self.block_size


# ============================================================
# Detection / Classification / AoA result types
# ============================================================

@dataclass(frozen=True)
class ClassificationResult:
    """
    CNN 또는 임시 분류기 결과.

    label:
    - background
    - wifi
    - bluetooth
    - drone_like
    - unknown
    """

    label: ClassLabel
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def is_drone_like(self) -> bool:
        return self.label == "drone_like"


@dataclass(frozen=True)
class AOAResult:
    """
    AoA 추정 결과.

    valid=False이면 angle_deg가 None일 수 있다.
    """

    angle_deg: float | None
    phase_diff_rad: float | None
    coherence: float | None
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class BlockPipelineResult:
    """
    block 하나에 대한 최종 파이프라인 결과.

    구성:
    - block index
    - sample range
    - classification result
    - AoA result
    """

    block_index: int
    sample_start: int
    sample_end: int
    classification: ClassificationResult
    aoa: AOAResult | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detected_drone_like(self) -> bool:
        return self.classification.is_drone_like