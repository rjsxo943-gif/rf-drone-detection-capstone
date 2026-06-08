from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScanRailState:
    mode: str
    scan_freqs: list[float]
    current_freq: float | None = None
    locked_freq: float | None = None
    candidate_freq: float | None = None
    status: str = ""
    rail_width: int = 190


def _as_state(value: ScanRailState | dict[str, Any]) -> ScanRailState:
    if isinstance(value, ScanRailState):
        return value

    return ScanRailState(
        mode=str(value.get("mode", "SCAN")),
        scan_freqs=[float(x) for x in value.get("scan_freqs", [])],
        current_freq=value.get("current_freq", None),
        locked_freq=value.get("locked_freq", None),
        candidate_freq=value.get("candidate_freq", None),
        status=str(value.get("status", "")),
        rail_width=int(value.get("rail_width", 190)),
    )


def _fmt_freq(freq: float | None) -> str:
    if freq is None:
        return "--"
    return f"{float(freq) / 1e9:.3f}G"


def _nearest_freq(freqs: list[float], target: float | None) -> float | None:
    if target is None or not freqs:
        return None

    arr = np.asarray(freqs, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def draw_scan_rail(
    canvas: np.ndarray,
    scan_rail: ScanRailState | dict[str, Any],
    *,
    rail_width: int | None = None,
) -> None:
    """
    Draw a thin vertical scan-status rail on the left side of an OpenCV canvas.

    Visual policy:
    - SCAN mode      : green border, moving marker at current_freq
    - PRECISION mode : gray scan rail, marker locked at candidate/locked freq
    - Frequency list : generated outside from configs/scan.yaml, not hardcoded
    """
    import cv2

    state = _as_state(scan_rail)
    h, w = canvas.shape[:2]
    rw = int(rail_width or state.rail_width)
    rw = max(120, min(rw, max(120, w // 3)))

    mode = str(state.mode or "SCAN").strip().upper()
    is_scan = mode == "SCAN"

    # BGR colors
    bg = (16, 16, 18)
    panel_bg = (20, 22, 24)
    green = (80, 230, 90)
    green_dim = (50, 110, 55)
    yellow = (0, 220, 255)
    gray = (85, 88, 92)
    gray_text = (160, 165, 170)
    white = (235, 238, 240)

    border = green if is_scan else gray
    marker_color = green if is_scan else yellow
    title_color = green if is_scan else gray_text

    cv2.rectangle(canvas, (0, 0), (rw - 1, h - 1), panel_bg, -1)
    cv2.rectangle(canvas, (8, 10), (rw - 10, h - 12), border, 2)

    cv2.putText(
        canvas,
        "SCAN",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        title_color,
        2,
        cv2.LINE_AA,
    )

    status = state.status.strip()
    if not status:
        status = "SWEEPING" if is_scan else "HANDOFF"

    cv2.putText(
        canvas,
        status[:14],
        (24, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        gray_text if not is_scan else green,
        1,
        cv2.LINE_AA,
    )

    freqs = [float(x) for x in state.scan_freqs]
    if not freqs:
        cv2.putText(
            canvas,
            "NO FREQS",
            (24, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            gray_text,
            1,
            cv2.LINE_AA,
        )
        return

    # Display high frequency at top, low frequency at bottom.
    shown_freqs = list(reversed(freqs))

    top_y = 112
    bottom_y = h - 115
    span = max(1, bottom_y - top_y)

    if len(shown_freqs) == 1:
        y_positions = [top_y + span // 2]
    else:
        y_positions = [
            int(round(top_y + i * span / (len(shown_freqs) - 1)))
            for i in range(len(shown_freqs))
        ]

    line_x = rw // 2 + 15
    label_x = 24

    cv2.line(canvas, (line_x, top_y), (line_x, bottom_y), green_dim if is_scan else gray, 1)

    target = state.current_freq if is_scan else (
        state.locked_freq if state.locked_freq is not None else state.candidate_freq
    )

    # Display policy:
    # - SCAN mode: CUR follows the sweeping frequency.
    # - PRECISION mode: CUR freezes at the locked/candidate frequency.
    display_current_freq = state.current_freq if is_scan else target

    nearest = _nearest_freq(shown_freqs, target)

    for freq, y in zip(shown_freqs, y_positions):
        active = nearest is not None and abs(float(freq) - float(nearest)) < 1.0

        tick_color = marker_color if active else gray
        text_color = white if active else gray_text

        cv2.circle(canvas, (line_x, y), 3, tick_color, -1)

        cv2.putText(
            canvas,
            _fmt_freq(freq),
            (label_x, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            text_color,
            1,
            cv2.LINE_AA,
        )

        if active:
            # SCAN: moving diamond-like marker
            # PRECISION: locked circle marker
            if is_scan:
                pts = np.array(
                    [
                        [line_x + 22, y],
                        [line_x + 32, y - 9],
                        [line_x + 42, y],
                        [line_x + 32, y + 9],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(canvas, pts, marker_color)
                cv2.putText(
                    canvas,
                    "SWEEP",
                    (line_x + 48, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    marker_color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(canvas, (line_x + 32, y), 9, marker_color, -1)
                cv2.putText(
                    canvas,
                    "LOCK",
                    (line_x + 48, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    marker_color,
                    1,
                    cv2.LINE_AA,
                )

    footer_y = h - 72
    cv2.line(canvas, (20, footer_y - 22), (rw - 22, footer_y - 22), (60, 62, 66), 1)

    current_text = f"CUR  {_fmt_freq(display_current_freq)}"
    lock_text = f"LOCK {_fmt_freq(state.locked_freq or state.candidate_freq)}"

    cv2.putText(
        canvas,
        current_text,
        (22, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        gray_text,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        lock_text,
        (22, footer_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        yellow if not is_scan else gray_text,
        1,
        cv2.LINE_AA,
    )
