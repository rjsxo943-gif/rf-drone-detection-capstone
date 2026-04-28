from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AoAComputeGateResult:
    """
    AoA 계산 여부 판단 결과.
    """

    should_compute: bool
    reason: str
    class_name: str
    confidence: float


class AoAComputeGate:
    """
    CNN 분류 결과를 기준으로 AoA 계산 여부를 결정한다.

    현재 정책:
    - CNN class가 Drone-like
    - confidence가 threshold 이상
    - 위 조건을 만족할 때만 AoA 계산 진행
    """

    def __init__(
        self,
        target_class: str = "Drone-like",
        confidence_threshold: float = 0.85,
        enabled: bool = True,
    ) -> None:
        self.target_class = target_class
        self.confidence_threshold = float(confidence_threshold)
        self.enabled = bool(enabled)

    def apply(
        self,
        class_name: str,
        confidence: float,
    ) -> AoAComputeGateResult:
        confidence = float(confidence)

        if not self.enabled:
            return AoAComputeGateResult(
                should_compute=True,
                reason="gate_disabled",
                class_name=class_name,
                confidence=confidence,
            )

        if class_name != self.target_class:
            return AoAComputeGateResult(
                should_compute=False,
                reason=f"class_is_not_{self.target_class}",
                class_name=class_name,
                confidence=confidence,
            )

        if confidence < self.confidence_threshold:
            return AoAComputeGateResult(
                should_compute=False,
                reason="confidence_too_low",
                class_name=class_name,
                confidence=confidence,
            )

        return AoAComputeGateResult(
            should_compute=True,
            reason="passed",
            class_name=class_name,
            confidence=confidence,
        )


def should_compute_aoa(
    class_name: str,
    confidence: float,
    target_class: str = "Drone-like",
    confidence_threshold: float = 0.85,
) -> bool:
    """
    AoA 계산을 수행할지 True/False로 판단한다.
    """
    return (
        class_name == target_class
        and float(confidence) >= float(confidence_threshold)
    )