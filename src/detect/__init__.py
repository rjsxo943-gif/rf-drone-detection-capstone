"""
Detection package for RF signal presence detection.

현재 프로젝트 기준:
- block 단위 IQ 입력
- Energy Detector로 신호 있음/없음 1차 판단
- 신호가 없으면 CNN/AoA branch를 skip할 수 있게 함
"""

from src.detect.energy_detector import (
    EnergyDetector,
    EnergyDetectionResult,
)

__all__ = [
    "EnergyDetector",
    "EnergyDetectionResult",
]