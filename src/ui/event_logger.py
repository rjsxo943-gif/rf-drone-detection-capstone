from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.core.utils import now_iso


class EventLogger:
    """
    탐지/분류/AoA 이벤트를 CSV 파일로 기록한다.

    저장 예:
    outputs/runs/latest/events.csv
    """

    HEADER = [
        "timestamp",
        "block_index",
        "event_type",
        "class_name",
        "confidence",
        "angle_deg",
        "coherence",
        "message",
    ]

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)

    def log(
        self,
        event_type: str,
        message: str = "",
        block_index: int | None = None,
        class_name: str = "",
        confidence: float | None = None,
        angle_deg: float | None = None,
        coherence: float | None = None,
    ) -> None:
        """
        이벤트 한 줄을 CSV에 추가한다.
        """

        row = [
            now_iso(timespec="seconds"),
            "" if block_index is None else int(block_index),
            event_type,
            class_name,
            "" if confidence is None else float(confidence),
            "" if angle_deg is None else float(angle_deg),
            "" if coherence is None else float(coherence),
            message,
        ]

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_detection(
        self,
        block_index: int,
        class_name: str,
        confidence: float,
        message: str = "classification result",
    ) -> None:
        """
        CNN/분류 결과 이벤트를 기록한다.
        """
        self.log(
            event_type="detection",
            block_index=block_index,
            class_name=class_name,
            confidence=confidence,
            message=message,
        )

    def log_aoa(
        self,
        block_index: int,
        angle_deg: float,
        coherence: float | None = None,
        message: str = "aoa result",
    ) -> None:
        """
        AoA 결과 이벤트를 기록한다.
        """
        self.log(
            event_type="aoa",
            block_index=block_index,
            angle_deg=angle_deg,
            coherence=coherence,
            message=message,
        )

    def log_pipeline_result(
        self,
        block_index: int,
        class_name: str,
        confidence: float,
        angle_deg: float | None = None,
        coherence: float | None = None,
        message: str = "pipeline result",
    ) -> None:
        """
        block 하나에 대한 최종 pipeline 결과를 기록한다.
        """
        self.log(
            event_type="pipeline",
            block_index=block_index,
            class_name=class_name,
            confidence=confidence,
            angle_deg=angle_deg,
            coherence=coherence,
            message=message,
        )