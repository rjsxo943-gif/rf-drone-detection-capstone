from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.detect.energy_detector import EnergyDetector
from src.preprocess.dc_blocker import remove_dc_offset


@dataclass(frozen=True)
class RawNoiseGateResult:
    enabled: bool
    passed: bool
    label: str

    gain: float
    matched_gain: float | None
    matched_by: str

    detector_method: str
    frame_size: int
    hop_size: int

    noise_floor: float | None
    threshold_multiplier: float
    threshold: float | None

    detection_ratio: float
    min_detection_ratio: float
    score_max: float
    score_median: float

    reason: str


class RawNoiseGate:
    """
    Gain-wise noise calibration JSON + configs/detect.yaml 기반 raw IQ gate.

    목적:
    - CNN 입력 전에 정규화 전 raw IQ 신호 세기가 배경 noise floor보다 충분히 큰지 판단.
    - gate fail이면 CNN voting/AoA를 막아 background -> Drone 오탐 누적을 줄임.

    기준:
    - noise_by_gain_latest.json의 profile["noise_floor"] 사용
    - runtime threshold = noise_floor * YAML threshold_multiplier
    - JSON profile["threshold"]는 참고하지 않음
    """

    def __init__(
        self,
        *,
        detect_config_path: str | Path = "configs/detect.yaml",
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.detect_config_path = self._resolve_path(detect_config_path)

        self.detect_cfg = self._load_yaml(self.detect_config_path)
        self.gate_cfg = dict(self.detect_cfg.get("raw_noise_gate", {}) or {})

        self.enabled = bool(self.gate_cfg.get("enabled", False))
        self.pass_label = str(self.gate_cfg.get("pass_label", "RAW_GATE_PASS"))
        self.fail_label = str(self.gate_cfg.get("fail_label", "GATE_LOW_BACKGROUND"))

        self.noise_profile_path = self._resolve_path(
            self.gate_cfg.get(
                "noise_profile_path",
                "outputs/calibration/noise_by_gain_latest.json",
            )
        )

        self.allow_nearest_gain = bool(self.gate_cfg.get("allow_nearest_gain", True))

        self.detector_method = str(self.gate_cfg.get("detector_method", "time_power"))
        self.frame_size = int(self.gate_cfg.get("frame_size", 1024))
        self.hop_size = int(self.gate_cfg.get("hop_size", 512))
        self.min_detection_ratio = float(self.gate_cfg.get("min_detection_ratio", 0.05))

        self.default_threshold_multiplier = float(
            self.gate_cfg.get("threshold_multiplier", 5.0)
        )
        self.gain_threshold_multiplier = dict(
            self.gate_cfg.get("gain_threshold_multiplier", {}) or {}
        )

        self.noise_data = self._load_json(self.noise_profile_path)
        self.profiles = dict(self.noise_data.get("profiles", {}) or {})

    def evaluate(self, iq_block: np.ndarray, *, gain: float) -> RawNoiseGateResult:
        gain = float(gain)

        if not self.enabled:
            return RawNoiseGateResult(
                enabled=False,
                passed=True,
                label="RAW_GATE_DISABLED",
                gain=gain,
                matched_gain=None,
                matched_by="disabled",
                detector_method=self.detector_method,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                noise_floor=None,
                threshold_multiplier=1.0,
                threshold=None,
                detection_ratio=1.0,
                min_detection_ratio=self.min_detection_ratio,
                score_max=float("nan"),
                score_median=float("nan"),
                reason="gate_disabled",
            )

        profile, matched_gain, matched_by = self._select_profile(gain)
        if profile is None:
            return RawNoiseGateResult(
                enabled=True,
                passed=False,
                label=self.fail_label,
                gain=gain,
                matched_gain=None,
                matched_by="none",
                detector_method=self.detector_method,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                noise_floor=None,
                threshold_multiplier=self._multiplier_for_gain(gain),
                threshold=None,
                detection_ratio=0.0,
                min_detection_ratio=self.min_detection_ratio,
                score_max=float("nan"),
                score_median=float("nan"),
                reason="no_noise_profile_for_gain",
            )

        noise_floor = float(profile["noise_floor"])
        multiplier = self._multiplier_for_gain(gain)
        threshold = noise_floor * multiplier

        iq_dc = remove_dc_offset(np.asarray(iq_block), axis=-1)

        detector = EnergyDetector(
            mode="initial_calibration",
            threshold_multiplier=multiplier,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            method=self.detector_method,
            min_detection_ratio=self.min_detection_ratio,
            require_calibration=True,
        )
        detector.noise_floor = noise_floor
        detector.threshold = threshold

        result = detector.detect_block(iq_dc)

        frame_energies = np.asarray(result.frame_energies, dtype=np.float32)
        if frame_energies.size:
            score_max = float(np.max(frame_energies))
            score_median = float(np.median(frame_energies))
        else:
            score_max = 0.0
            score_median = 0.0

        passed = bool(result.detected)
        return RawNoiseGateResult(
            enabled=True,
            passed=passed,
            label=self.pass_label if passed else self.fail_label,
            gain=gain,
            matched_gain=matched_gain,
            matched_by=matched_by,
            detector_method=self.detector_method,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            noise_floor=noise_floor,
            threshold_multiplier=multiplier,
            threshold=threshold,
            detection_ratio=float(result.detection_ratio),
            min_detection_ratio=self.min_detection_ratio,
            score_max=score_max,
            score_median=score_median,
            reason="passed" if passed else "below_noise_gate",
        )

    def reset_cnn_history_on_fail(self) -> bool:
        return bool(self.gate_cfg.get("reset_cnn_history_on_fail", True))

    def block_cnn_on_fail(self) -> bool:
        return bool(self.gate_cfg.get("block_cnn_on_fail", True))

    def block_aoa_on_fail(self) -> bool:
        return bool(self.gate_cfg.get("block_aoa_on_fail", True))

    def status_text(self, result: RawNoiseGateResult) -> str:
        if not result.enabled:
            return "RAW_GATE disabled"

        thr = "NA" if result.threshold is None else f"{result.threshold:.4g}"
        nf = "NA" if result.noise_floor is None else f"{result.noise_floor:.4g}"
        return (
            f"RAW_GATE {result.label} "
            f"gain={result.gain:g} match={result.matched_gain}({result.matched_by}) "
            f"score_max={result.score_max:.4g} noise={nf} "
            f"thr={thr} x{result.threshold_multiplier:g} "
            f"ratio={result.detection_ratio:.3f}/{result.min_detection_ratio:.3f}"
        )

    def _multiplier_for_gain(self, gain: float) -> float:
        # exact string keys first: "40", "40.0"
        candidates = [
            str(int(gain)) if float(gain).is_integer() else str(gain),
            str(float(gain)),
        ]
        for key in candidates:
            if key in self.gain_threshold_multiplier:
                return float(self.gain_threshold_multiplier[key])

        return float(self.default_threshold_multiplier)

    def _select_profile(
        self,
        gain: float,
    ) -> tuple[dict[str, Any] | None, float | None, str]:
        if not self.profiles:
            return None, None, "none"

        exact_keys = [
            str(int(gain)) if float(gain).is_integer() else str(gain),
            str(float(gain)),
        ]

        for key in exact_keys:
            if key in self.profiles:
                profile = dict(self.profiles[key])
                return profile, float(profile.get("gain", gain)), "exact"

        if not self.allow_nearest_gain:
            return None, None, "not_found"

        best_key = min(
            self.profiles.keys(),
            key=lambda k: abs(float(k) - float(gain)),
        )
        profile = dict(self.profiles[best_key])
        return profile, float(profile.get("gain", best_key)), "nearest"

    def _resolve_path(self, path_like: str | Path) -> Path:
        p = Path(path_like)
        if p.is_absolute():
            return p
        return self.project_root / p

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"raw noise gate profile not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"detect config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(data or {})
