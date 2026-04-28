from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.core.utils import now_iso


@dataclass
class DashboardSnapshot:
    """
    현재 block 하나에 대한 대시보드 표시용 요약 정보.
    """

    timestamp: str
    block_index: int
    class_name: str
    confidence: float
    detected: bool
    aoa_enabled: bool
    angle_deg: float | None
    coherence: float | None
    message: str


def make_dashboard_snapshot(
    block_index: int,
    class_name: str,
    confidence: float,
    detected: bool | None = None,
    aoa_enabled: bool = False,
    angle_deg: float | None = None,
    coherence: float | None = None,
    message: str = "",
) -> DashboardSnapshot:
    """
    block 하나에 대한 dashboard snapshot을 생성한다.

    detected를 직접 주지 않으면 class_name이 drone_like인지로 판단한다.
    """

    normalized_class = normalize_class_name(class_name)

    if detected is None:
        detected = normalized_class == "drone_like"

    return DashboardSnapshot(
        timestamp=now_iso(timespec="seconds"),
        block_index=int(block_index),
        class_name=normalized_class,
        confidence=float(confidence),
        detected=bool(detected),
        aoa_enabled=bool(aoa_enabled),
        angle_deg=None if angle_deg is None else float(angle_deg),
        coherence=None if coherence is None else float(coherence),
        message=message,
    )


def snapshot_to_dict(snapshot: DashboardSnapshot) -> dict[str, Any]:
    """
    DashboardSnapshot을 dict로 변환한다.

    나중에 JSON, 웹 대시보드, 로그 출력에 바로 사용 가능하다.
    """
    return asdict(snapshot)


def make_dashboard_dict(
    block_index: int,
    class_name: str,
    confidence: float,
    detected: bool | None = None,
    aoa_enabled: bool = False,
    angle_deg: float | None = None,
    coherence: float | None = None,
    message: str = "",
) -> dict[str, Any]:
    """
    dashboard snapshot을 바로 dict 형태로 생성한다.
    """

    snapshot = make_dashboard_snapshot(
        block_index=block_index,
        class_name=class_name,
        confidence=confidence,
        detected=detected,
        aoa_enabled=aoa_enabled,
        angle_deg=angle_deg,
        coherence=coherence,
        message=message,
    )

    return snapshot_to_dict(snapshot)


def print_dashboard(snapshot: DashboardSnapshot | dict[str, Any]) -> None:
    """
    터미널에 현재 block 결과를 보기 좋게 출력한다.
    """

    if isinstance(snapshot, DashboardSnapshot):
        data = snapshot_to_dict(snapshot)
    else:
        data = snapshot

    print("=== Dashboard Snapshot ===")
    print(f"timestamp   : {data.get('timestamp')}")
    print(f"block_index : {data.get('block_index')}")
    print(f"class       : {data.get('class_name')}")
    print(f"confidence  : {data.get('confidence'):.3f}")
    print(f"detected    : {data.get('detected')}")
    print(f"aoa_enabled : {data.get('aoa_enabled')}")

    angle_deg = data.get("angle_deg")
    coherence = data.get("coherence")

    if angle_deg is not None:
        print(f"angle_deg   : {float(angle_deg):.2f}")
    else:
        print("angle_deg   : None")

    if coherence is not None:
        print(f"coherence   : {float(coherence):.3f}")
    else:
        print("coherence   : None")

    message = data.get("message", "")
    if message:
        print(f"message     : {message}")


def normalize_class_name(class_name: str) -> str:
    """
    class 이름을 내부 표준 형식으로 정리한다.

    예:
    Drone-like -> drone_like
    drone like -> drone_like
    DRONE_LIKE -> drone_like
    """

    return (
        str(class_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )