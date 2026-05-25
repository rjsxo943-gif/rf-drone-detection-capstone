from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class SectorResult:
    sector_index: int | None
    sector_label: str
    angle_deg: float | None
    valid: bool


def quantize_front_angle_to_sector(
    angle_deg: float | None,
    *,
    num_sectors: int = 8,
    min_angle: float = -90.0,
    max_angle: float = 90.0,
) -> SectorResult:
    """
    전방 AoA 각도(-90~+90 deg)를 sector로 양자화한다.

    반환:
    - sector_index: 0 ~ num_sectors-1
    - sector_label: 사람이 읽기 쉬운 방향 라벨
    - valid: 범위 안이면 True
    """
    if angle_deg is None:
        return SectorResult(
            sector_index=None,
            sector_label="Unknown",
            angle_deg=None,
            valid=False,
        )

    angle = float(angle_deg)

    if angle < min_angle or angle > max_angle:
        return SectorResult(
            sector_index=None,
            sector_label="Out of Front Range",
            angle_deg=angle,
            valid=False,
        )

    sector_width = (max_angle - min_angle) / num_sectors
    sector = int((angle - min_angle) // sector_width)

    if sector >= num_sectors:
        sector = num_sectors - 1

    label = sector_index_to_label(sector, num_sectors=num_sectors)

    return SectorResult(
        sector_index=sector,
        sector_label=label,
        angle_deg=angle,
        valid=True,
    )


def sector_index_to_label(
    sector_index: int | None,
    *,
    num_sectors: int = 8,
) -> str:
    if sector_index is None:
        return "Unknown"

    if num_sectors != 8:
        return f"Sector {sector_index}"

    labels = [
        "Far Left",
        "Left 2",
        "Left 1",
        "Center Left",
        "Center Right",
        "Right 1",
        "Right 2",
        "Far Right",
    ]

    if 0 <= sector_index < len(labels):
        return labels[sector_index]

    return "Unknown"


class SectorVoter:
    """
    최근 N개 sector 결과를 모아서 다수결로 최종 방향을 정한다.

    예:
    - window_size=5
    - min_votes=3
    최근 5개 중 같은 sector가 3번 이상 나오면 인정.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        min_votes: int = 3,
        num_sectors: int = 8,
    ) -> None:
        self.window_size = int(window_size)
        self.min_votes = int(min_votes)
        self.num_sectors = int(num_sectors)
        self.history: Deque[int | None] = deque(maxlen=self.window_size)

    def update(self, sector_index: int | None) -> SectorResult:
        self.history.append(sector_index)

        valid = [s for s in self.history if s is not None]

        if not valid:
            return SectorResult(
                sector_index=None,
                sector_label="Unknown",
                angle_deg=None,
                valid=False,
            )

        sector, count = Counter(valid).most_common(1)[0]

        if count < self.min_votes:
            return SectorResult(
                sector_index=None,
                sector_label="Unstable",
                angle_deg=None,
                valid=False,
            )

        return SectorResult(
            sector_index=sector,
            sector_label=sector_index_to_label(sector, num_sectors=self.num_sectors),
            angle_deg=None,
            valid=True,
        )

    def reset(self) -> None:
        self.history.clear()
