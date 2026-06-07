from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as:
#   PYTHONPATH=. python scripts/experimental/live_aoa_sector_dashboard.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experimental import live_aoa_sector_experiment_capture as base
from src.preprocess.dc_blocker import remove_dc_offset
from src.runtime.raw_noise_gate import RawNoiseGate
from src.viewer import ViewerState, compute_raw_features
from src.viewer.sector_range_estimator import SectorRangeEstimator


def load_dashboard_cfg(args: Any) -> dict[str, Any]:
    ui_path = Path(args.config_dir) / "ui.yaml"
    if ui_path.exists():
        ui = base.load_yaml(ui_path)
    else:
        ui = {}

    cfg = dict(ui.get("sector_dashboard", {}) or {})

    distance_cfg = dict(cfg.get("distance", {}) or {})
    default_distance = {
        "enabled": False,
        "show_rings": False,
        "profile_path": None,
    }
    default_distance.update(distance_cfg)

    default = {
        "enabled": True,
        "show_spectrogram": False,
        "show_sector": True,
        "show_angle_text": True,
        "show_coherence": True,
        "show_p99": True,
        "p99_source": "median_raw_p99",
        "sector_display_mode": "seven",
        "blink_on_hold": True,
        "fade_on_signal_lost": True,
        "canvas_width": 1120,
        "canvas_height": 720,
        "distance": default_distance,
    }
    default.update(cfg)
    default["distance"] = default_distance
    return default


def fmt_value(value: Any, digits: int = 3) -> str:
    try:
        if value is None or value == "":
            return "n/a"
        v = float(value)
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def get_p99_value(
    sector: dict[str, Any],
    selected_raw: dict[str, Any],
    p99_source: str,
) -> Any:
    src = str(p99_source or "median_raw_p99").strip()

    if src == "median_raw_p99":
        v = sector.get("median_raw_p99")
        if v not in (None, ""):
            return v

    if src in selected_raw:
        return selected_raw.get(src)

    if "raw_abs_p99" in selected_raw:
        return selected_raw.get("raw_abs_p99")

    return ""


class SectorDashboardRenderer:
    def __init__(
        self,
        *,
        window_name: str = "RF Sector Dashboard",
        target_fps: float = 5.0,
        width: int = 1120,
        height: int = 720,
        blink_on_hold: bool = True,
        fade_on_signal_lost: bool = True,
    ) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required. Install it with: pip install opencv-python"
            ) from exc

        self.window_name = str(window_name)
        self.target_fps = max(0.1, float(target_fps))
        self.width = max(800, int(width))
        self.height = max(520, int(height))
        self.blink_on_hold = bool(blink_on_hold)
        self.fade_on_signal_lost = bool(fade_on_signal_lost)
        self._last_render_time = 0.0

    def render(
        self,
        *,
        state: ViewerState,
        args: Any,
        dash_cfg: dict[str, Any],
        sector: dict[str, Any],
        selected_raw: dict[str, Any],
        cnn_result: dict[str, Any],
        raw_pass_count: int,
        cnn_drone_count: int,
        topk_count: int,
        paused: bool = False,
    ) -> str | None:
        import cv2

        self._sleep_to_limit_fps()

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (16, 16, 18)

        range_result = get_dashboard_range_result(dash_cfg, sector, selected_raw)

        demo_cycle_enabled = bool(dash_cfg.get("demo_cycle", False))
        demo_update_idx = int(time.monotonic() * 5.0)

        sector, range_result = apply_demo_range_cycle(
            sector=sector,
            range_result=range_result,
            update_idx=demo_update_idx,
            enabled=demo_cycle_enabled,
        )

        self._draw_title(canvas, paused=paused)
        self._draw_sector_fan_v2(
            canvas,
            args=args,
            sector=sector,
            range_result=range_result,
            paused=paused,
        )
        self._draw_text_panel(
            canvas,
            state=state,
            args=args,
            dash_cfg=dash_cfg,
            sector=sector,
            selected_raw=selected_raw,
            cnn_result=cnn_result,
            raw_pass_count=raw_pass_count,
            cnn_drone_count=cnn_drone_count,
            topk_count=topk_count,
            paused=paused,
        )
        self._draw_footer(canvas)

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        self._last_render_time = time.monotonic()

        if key == 255:
            return None
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("p"):
            return "pause"
        if key == ord("["):
            return "gain_down"
        if key == ord("]"):
            return "gain_up"
        if key == ord("s"):
            return "save_profile"
        if key == ord("x"):
            return "cancel_capture"

        return chr(key) if 0 <= key <= 255 else None

    def close(self) -> None:
        import cv2

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            cv2.destroyAllWindows()

    def _sleep_to_limit_fps(self) -> None:
        min_interval = 1.0 / self.target_fps
        elapsed = time.monotonic() - self._last_render_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _draw_title(self, canvas: np.ndarray, *, paused: bool) -> None:
        import cv2

        title = "RF Sector Dashboard"
        if paused:
            title += "  [PAUSED]"

        cv2.putText(
            canvas,
            title,
            (30, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        cv2.line(canvas, (30, 66), (self.width - 30, 66), (80, 80, 85), 1)

    @staticmethod
    def _angle_point(cx: int, cy: int, radius: float, angle_deg: float) -> tuple[int, int]:
        # Visual convention:
        #   0 deg  : straight ahead/up
        #   - deg  : left
        #   + deg  : right
        rad = math.radians(float(angle_deg))
        x = cx + radius * math.sin(rad)
        y = cy - radius * math.cos(rad)
        return int(round(x)), int(round(y))

    def _draw_sector_fan(
        self,
        canvas: np.ndarray,
        *,
        args: Any,
        sector: dict[str, Any],
        range_result: dict[str, Any] | None,
        paused: bool,
    ) -> None:
        import cv2

        bins = list(args.sector_preset.get("bins", []) or [])
        if not bins:
            return

        cx = int(self.width * 0.34)
        cy = int(self.height * 0.86)
        radius = min(int(self.width * 0.30), int(self.height * 0.63))

        status = str(sector.get("sector_status", "no_signal")).lower().strip()
        locked = str(sector.get("locked_sector_name", "") or "").strip()
        instant = str(sector.get("instant_sector_name", "") or "").strip()

        if status == "trusted" and locked:
            active_name = locked
            active_mode = "trusted"
        elif status.startswith("hold") and (locked or instant):
            active_name = locked or instant
            active_mode = "hold"
        elif status == "uncertain" and (locked or instant):
            active_name = locked or instant
            active_mode = "uncertain"
        else:
            active_name = ""
            active_mode = "none"

        blink_on = True
        if active_mode in {"hold", "uncertain"} and self.blink_on_hold:
            blink_on = int(time.monotonic() * 2.0) % 2 == 0

        for b in bins:
            name = str(b.get("name", ""))
            mn = float(b.get("min_deg", 0.0))
            mx = float(b.get("max_deg", 0.0))
            mid = (mn + mx) / 2.0

            is_active = bool(active_name and name == active_name and blink_on)

            if paused:
                color = (90, 90, 90)
            elif is_active and active_mode == "trusted":
                color = (70, 220, 110)
            elif is_active and active_mode == "hold":
                color = (80, 190, 255)
            elif is_active and active_mode == "uncertain":
                color = (80, 140, 255)
            elif self.fade_on_signal_lost and status in {"no_signal", "none"}:
                color = (42, 42, 45)
            else:
                color = (62, 62, 68)

            pts: list[tuple[int, int]] = [(cx, cy)]
            steps = max(4, int(abs(mx - mn) / 2))
            for a in np.linspace(mn, mx, steps):
                pts.append(self._angle_point(cx, cy, radius, float(a)))

            poly = np.asarray(pts, dtype=np.int32)
            cv2.fillPoly(canvas, [poly], color)
            cv2.polylines(canvas, [poly], isClosed=True, color=(30, 30, 35), thickness=2)

            # Sector label
            tx, ty = self._angle_point(cx, cy, radius * 0.62, mid)
            label = name.replace("LEFT_", "L").replace("RIGHT_", "R")
            font_scale = 0.46 if len(label) > 8 else 0.54
            cv2.putText(
                canvas,
                label,
                (tx - 42, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (245, 245, 245) if is_active else (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

        # Direction guide
        for angle, label in [(-60, "-60"), (-30, "-30"), (0, "0"), (30, "+30"), (60, "+60")]:
            x1, y1 = self._angle_point(cx, cy, radius * 0.90, angle)
            x2, y2 = self._angle_point(cx, cy, radius * 1.03, angle)
            cv2.line(canvas, (x1, y1), (x2, y2), (120, 120, 125), 1)
            lx, ly = self._angle_point(cx, cy, radius * 1.10, angle)
            cv2.putText(
                canvas,
                label,
                (lx - 18, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 185),
                1,
                cv2.LINE_AA,
            )

        # Receiver center
        cv2.circle(canvas, (cx, cy), 8, (235, 235, 235), -1)
        cv2.putText(
            canvas,
            "RX0/RX1",
            (cx - 38, cy + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


        self._draw_range_band_overlay(
            canvas,
            sector=sector,
            range_result=range_result,
        )

    def _sector5_angle_bounds(self, sector_name: str) -> tuple[float, float] | None:
        name = str(sector_name or "").strip().upper()

        seven_to_five = {
            "LEFT_60_45": "LEFT_OUTER",
            "LEFT_45_30": "LEFT_OUTER",
            "LEFT_30_15": "LEFT_INNER",
            "CENTER": "CENTER",
            "RIGHT_15_30": "RIGHT_INNER",
            "RIGHT_30_45": "RIGHT_OUTER",
            "RIGHT_45_60": "RIGHT_OUTER",
        }

        name = seven_to_five.get(name, name)

        bounds = {
            "LEFT_OUTER": (-60.0, -30.0),
            "LEFT_INNER": (-30.0, -15.0),
            "CENTER": (-15.0, 15.0),
            "RIGHT_INNER": (15.0, 30.0),
            "RIGHT_OUTER": (30.0, 60.0),
        }

        return bounds.get(name)

    def _range_band_polygon(
        self,
        *,
        cx: int,
        cy: int,
        r_inner: float,
        r_outer: float,
        angle_min: float,
        angle_max: float,
        num_points: int = 28,
    ) -> np.ndarray:
        angles = np.linspace(angle_min, angle_max, num_points)

        outer_points = [
            self._angle_point(cx, cy, r_outer, float(angle_deg))
            for angle_deg in angles
        ]

        inner_points = [
            self._angle_point(cx, cy, r_inner, float(angle_deg))
            for angle_deg in angles[::-1]
        ]

        return np.array(outer_points + inner_points, dtype=np.int32)

    def _draw_transparent_polygon(
        self,
        canvas: np.ndarray,
        *,
        polygon: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
        outline: bool = True,
    ) -> None:
        import cv2

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, float(alpha), canvas, 1.0 - float(alpha), 0.0, canvas)

        if outline:
            cv2.polylines(canvas, [polygon], True, (210, 210, 210), 1, cv2.LINE_AA)

    def _draw_range_band_overlay(
        self,
        canvas: np.ndarray,
        *,
        sector: dict[str, Any],
        range_result: dict[str, Any] | None,
    ) -> None:
        import cv2

        if not range_result:
            return

        locked = str(sector.get("locked_sector_name", "") or "").strip()
        if not locked:
            return

        range_class = str(range_result.get("range_class", "") or "").strip().upper()
        display_mode = str(range_result.get("display_mode", "") or "").strip().lower()

        if range_class not in {"WITHIN_9M", "RANGE_9_TO_15M", "SECTOR_ONLY"}:
            return

        bounds = self._sector5_angle_bounds(locked)
        if bounds is None:
            return

        angle_min, angle_max = bounds

        # Fan geometry. 기존 fan과 맞추기 위한 값.
        cx = int(self.width * 0.345)
        cy = int(self.height * 0.860)
        radius = int(min(self.width * 0.310, self.height * 0.570))

        # 기존 trusted sector full-fill을 깔끔하게 지우기 위한 clear 영역.
        clear_poly = self._range_band_polygon(
            cx=cx,
            cy=cy,
            r_inner=0.0,
            r_outer=radius + 5,
            angle_min=angle_min - 1.3,
            angle_max=angle_max + 1.3,
        )

        # range_bin이면 기존 초록색 전체 sector를 먼저 어둡게 정리한다.
        if display_mode == "range_bin":
            self._draw_transparent_polygon(
                canvas,
                polygon=clear_poly,
                color=(38, 35, 35),
                alpha=0.94,
                outline=False,
            )

            # 선택 sector의 기본 영역을 아주 약하게 다시 깔아준다.
            base_poly = self._range_band_polygon(
                cx=cx,
                cy=cy,
                r_inner=0.0,
                r_outer=radius,
                angle_min=angle_min + 0.6,
                angle_max=angle_max - 0.6,
            )
            self._draw_transparent_polygon(
                canvas,
                polygon=base_poly,
                color=(70, 67, 67),
                alpha=0.22,
                outline=True,
            )

        # band는 sector 전체를 꽉 채우지 않고, 안쪽/바깥쪽 ribbon으로만 표시한다.
        near_inner = radius * 0.14
        near_outer = radius * 0.48
        far_inner = radius * 0.68
        far_outer = radius * 0.97

        # 미선택 range guide를 아주 약하게 표시해서 2구간 구조를 보여준다.
        if display_mode == "range_bin":
            guide_near = self._range_band_polygon(
                cx=cx,
                cy=cy,
                r_inner=near_inner,
                r_outer=near_outer,
                angle_min=angle_min + 1.1,
                angle_max=angle_max - 1.1,
            )
            guide_far = self._range_band_polygon(
                cx=cx,
                cy=cy,
                r_inner=far_inner,
                r_outer=far_outer,
                angle_min=angle_min + 1.1,
                angle_max=angle_max - 1.1,
            )

            self._draw_transparent_polygon(
                canvas,
                polygon=guide_near,
                color=(75, 75, 78),
                alpha=0.16,
                outline=True,
            )
            self._draw_transparent_polygon(
                canvas,
                polygon=guide_far,
                color=(75, 75, 78),
                alpha=0.16,
                outline=True,
            )

        if range_class == "WITHIN_9M":
            r_inner = near_inner
            r_outer = near_outer
            # BGR: clean green
            main_color = (95, 235, 120)
            glow_color = (35, 130, 65)
            label = "<=9m"
            label_r = (r_inner + r_outer) * 0.5

        elif range_class == "RANGE_9_TO_15M":
            r_inner = far_inner
            r_outer = far_outer
            # BGR: clean sky blue
            main_color = (255, 185, 80)
            glow_color = (135, 85, 20)
            label = "9-15m"
            label_r = (r_inner + r_outer) * 0.5

        else:
            # 거리 구분이 불안정하면 전체 sector를 부드럽게 점등
            r_inner = 0.0
            r_outer = radius
            main_color = (95, 235, 120)
            glow_color = (35, 130, 65)
            label = "SECTOR"
            label_r = radius * 0.58

        # 선택 band 각도는 sector 경계보다 살짝 안쪽으로 넣어서 선이 덜 지저분하게 보이게 한다.
        band_angle_min = angle_min + 1.4
        band_angle_max = angle_max - 1.4

        # glow layer
        glow_poly = self._range_band_polygon(
            cx=cx,
            cy=cy,
            r_inner=max(0.0, r_inner - radius * 0.025),
            r_outer=min(radius + 5, r_outer + radius * 0.025),
            angle_min=band_angle_min - 0.5,
            angle_max=band_angle_max + 0.5,
        )
        self._draw_transparent_polygon(
            canvas,
            polygon=glow_poly,
            color=glow_color,
            alpha=0.34,
            outline=False,
        )

        # main ribbon
        band_poly = self._range_band_polygon(
            cx=cx,
            cy=cy,
            r_inner=r_inner,
            r_outer=r_outer,
            angle_min=band_angle_min,
            angle_max=band_angle_max,
        )
        self._draw_transparent_polygon(
            canvas,
            polygon=band_poly,
            color=main_color,
            alpha=0.82,
            outline=True,
        )

        # inner/outer arc 느낌을 주는 얇은 border
        angles = np.linspace(band_angle_min, band_angle_max, 42)
        outer_arc = np.array(
            [self._angle_point(cx, cy, r_outer, float(a)) for a in angles],
            dtype=np.int32,
        )
        inner_arc = np.array(
            [self._angle_point(cx, cy, r_inner, float(a)) for a in angles],
            dtype=np.int32,
        )

        cv2.polylines(canvas, [outer_arc], False, (245, 245, 245), 1, cv2.LINE_AA)
        if r_inner > 2:
            cv2.polylines(canvas, [inner_arc], False, (190, 190, 190), 1, cv2.LINE_AA)

        # 작은 pill label. 큰 글씨를 sector 안에 박지 않고 짧게 표시.
        angle_mid = (angle_min + angle_max) / 2.0
        tx, ty = self._angle_point(cx, cy, label_r, angle_mid)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pad_x = 8
        pad_y = 5
        x1 = int(tx - tw / 2 - pad_x)
        y1 = int(ty - th / 2 - pad_y)
        x2 = int(tx + tw / 2 + pad_x)
        y2 = int(ty + th / 2 + pad_y + baseline)

        # label background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (24, 24, 28), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), main_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (int(tx - tw / 2), int(ty + th / 2)),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )



    def _draw_sector_fan_v2(
        self,
        canvas: np.ndarray,
        *,
        args: Any,
        sector: dict[str, Any],
        range_result: dict[str, Any] | None,
        paused: bool,
    ) -> None:
        import cv2

        # ------------------------------------------------------------
        # Fan V2
        # - sector grid와 range band를 같은 geometry로 직접 그림
        # - overlay 덧칠 방식이 아니라 같은 polygon generator를 사용
        # ------------------------------------------------------------

        status = str(sector.get("sector_status", "no_signal") or "no_signal").lower().strip()
        locked = str(sector.get("locked_sector_name", "") or "").strip().upper()
        instant = str(sector.get("instant_sector_name", "") or "").strip().upper()

        # 화면 오른쪽 패널을 침범하지 않도록 왼쪽 영역 기준으로 geometry 고정
        left_area_w = int(self.width * 0.66)

        cx = int(left_area_w * 0.52)
        cy = int(self.height * 0.86)
        radius = int(min(left_area_w * 0.42, self.height * 0.57))

        # 7-sector grid
        sectors7 = [
            ("LEFT_60_45", -60.0, -45.0, "L60_45"),
            ("LEFT_45_30", -45.0, -30.0, "L45_30"),
            ("LEFT_30_15", -30.0, -15.0, "L30_15"),
            ("CENTER", -15.0, 15.0, "CENTER"),
            ("RIGHT_15_30", 15.0, 30.0, "R15_30"),
            ("RIGHT_30_45", 30.0, 45.0, "R30_45"),
            ("RIGHT_45_60", 45.0, 60.0, "R45_60"),
        ]

        # 5-sector mapping for range display
        seven_to_five = {
            "LEFT_60_45": "LEFT_OUTER",
            "LEFT_45_30": "LEFT_OUTER",
            "LEFT_30_15": "LEFT_INNER",
            "CENTER": "CENTER",
            "RIGHT_15_30": "RIGHT_INNER",
            "RIGHT_30_45": "RIGHT_OUTER",
            "RIGHT_45_60": "RIGHT_OUTER",
        }

        bounds5 = {
            "LEFT_OUTER": (-60.0, -30.0),
            "LEFT_INNER": (-30.0, -15.0),
            "CENTER": (-15.0, 15.0),
            "RIGHT_INNER": (15.0, 30.0),
            "RIGHT_OUTER": (30.0, 60.0),
        }

        locked5 = seven_to_five.get(locked, locked)

        # 색상: OpenCV BGR
        bg_sector = (58, 54, 54)
        bg_sector_alt = (64, 60, 60)
        grid_line = (34, 32, 32)
        grid_outer = (92, 88, 88)
        text_dim = (180, 180, 185)

        trusted_green = (95, 235, 120)
        hold_orange = (80, 190, 255)
        uncertain_yellow = (80, 220, 235)
        near_green = (90, 235, 120)
        far_blue = (255, 185, 85)
        sector_only_green = (90, 225, 115)

        # ------------------------------------------------------------
        # local polygon helpers
        # ------------------------------------------------------------
        def wedge_poly(angle_min: float, angle_max: float, r0: float, r1: float, n: int = 40) -> np.ndarray:
            angles = np.linspace(angle_min, angle_max, n)

            outer = [
                self._angle_point(cx, cy, r1, float(a))
                for a in angles
            ]

            if r0 <= 1:
                inner = [(cx, cy)]
            else:
                inner = [
                    self._angle_point(cx, cy, r0, float(a))
                    for a in angles[::-1]
                ]

            return np.array(outer + inner, dtype=np.int32)

        def fill_poly(poly: np.ndarray, color: tuple[int, int, int], alpha: float = 1.0) -> None:
            if alpha >= 0.999:
                cv2.fillPoly(canvas, [poly], color)
            else:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [poly], color)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        def draw_arc(angle_min: float, angle_max: float, r: float, color: tuple[int, int, int], thickness: int = 1) -> None:
            angles = np.linspace(angle_min, angle_max, 64)
            pts = np.array(
                [self._angle_point(cx, cy, r, float(a)) for a in angles],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

        def draw_radial(angle_deg: float, r0: float, r1: float, color: tuple[int, int, int], thickness: int = 1) -> None:
            p0 = self._angle_point(cx, cy, r0, angle_deg)
            p1 = self._angle_point(cx, cy, r1, angle_deg)
            cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)

        def draw_centered_text(text: str, x: int, y: int, scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(
                canvas,
                text,
                (int(x - tw / 2), int(y + th / 2)),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        def draw_pill(text: str, x: int, y: int, border_color: tuple[int, int, int]) -> None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.50
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

            pad_x = 9
            pad_y = 6
            x1 = int(x - tw / 2 - pad_x)
            y1 = int(y - th / 2 - pad_y)
            x2 = int(x + tw / 2 + pad_x)
            y2 = int(y + th / 2 + pad_y + baseline)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (22, 22, 26), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                text,
                (int(x - tw / 2), int(y + th / 2)),
                font,
                scale,
                (245, 245, 245),
                thickness,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # 1) base 7-sector grid
        # ------------------------------------------------------------
        for i, (name, a0, a1, label) in enumerate(sectors7):
            poly = wedge_poly(a0, a1, 0.0, radius)
            fill_poly(poly, bg_sector if i % 2 == 0 else bg_sector_alt, 1.0)

            # sector label
            mid = (a0 + a1) / 2.0
            lx, ly = self._angle_point(cx, cy, radius * 0.58, mid)
            draw_centered_text(label, lx, ly, 0.52, text_dim, 1)

        # grid boundaries
        for angle in [-60, -45, -30, -15, 15, 30, 45, 60]:
            draw_radial(float(angle), 0.0, radius, grid_line, 2 if angle in [-60, -30, 30, 60] else 1)

        draw_arc(-60.0, 60.0, radius, grid_outer, 1)

        # 5-sector soft separators
        for angle in [-60, -30, -15, 15, 30, 60]:
            draw_radial(float(angle), 0.0, radius, (78, 75, 75), 1)

        # angle marks
        for angle, label in [(-60, "-60"), (-30, "-30"), (0, "0"), (30, "+30"), (60, "+60")]:
            px, py = self._angle_point(cx, cy, radius + 34, float(angle))
            draw_centered_text(label, px, py, 0.48, (185, 185, 190), 1)
            p0 = self._angle_point(cx, cy, radius + 4, float(angle))
            p1 = self._angle_point(cx, cy, radius + 20, float(angle))
            cv2.line(canvas, p0, p1, (120, 120, 125), 1, cv2.LINE_AA)

        # center line
        draw_radial(0.0, radius * 0.84, radius + 10, (105, 105, 110), 1)

        # ------------------------------------------------------------
        # 2) locked sector / range cell
        # ------------------------------------------------------------
        highlight_name = locked5 if locked5 in bounds5 else seven_to_five.get(instant, instant)
        bounds = bounds5.get(highlight_name)

        if bounds is not None and status not in {"no_signal", "none"}:
            a0, a1 = bounds

            # selected 5-sector frame
            frame_poly = wedge_poly(a0, a1, 0.0, radius)
            if status == "trusted":
                frame_color = trusted_green
            elif status.startswith("hold"):
                frame_color = hold_orange
            else:
                frame_color = uncertain_yellow

            fill_poly(frame_poly, frame_color, 0.12)
            cv2.polylines(canvas, [frame_poly], True, frame_color, 2, cv2.LINE_AA)

            # range structure
            range_class = ""
            display_mode = ""
            if range_result:
                range_class = str(range_result.get("range_class", "") or "").strip().upper()
                display_mode = str(range_result.get("display_mode", "") or "").strip().lower()

            # range guide cells: exactly same sector bounds
            near_inner = 0.0
            near_outer = radius * 0.52
            far_inner = near_outer
            far_outer = radius

            # 내부 분할선
            draw_arc(a0, a1, near_outer, (128, 128, 132), 1)

            if display_mode == "range_bin" and range_class in {"WITHIN_9M", "RANGE_9_TO_15M"}:
                if range_class == "WITHIN_9M":
                    cell_poly = wedge_poly(a0, a1, near_inner, near_outer)
                    main_color = near_green
                    label = "<=9m"
                    label_r = near_outer * 0.56
                else:
                    cell_poly = wedge_poly(a0, a1, far_inner, far_outer)
                    main_color = far_blue
                    label = "9-15m"
                    label_r = far_inner + (far_outer - far_inner) * 0.58

                # 선택 cell만 정확히 채움
                fill_poly(cell_poly, main_color, 0.78)
                cv2.polylines(canvas, [cell_poly], True, (238, 238, 238), 2, cv2.LINE_AA)

                mid = (a0 + a1) / 2.0
                tx, ty = self._angle_point(cx, cy, label_r, mid)
                draw_pill(label, tx, ty, main_color)

            elif range_class == "SECTOR_ONLY":
                # 거리 구분 불안정: 해당 sector 전체만 부드럽게 표시
                fill_poly(frame_poly, sector_only_green, 0.38)
                cv2.polylines(canvas, [frame_poly], True, sector_only_green, 2, cv2.LINE_AA)

                mid = (a0 + a1) / 2.0
                tx, ty = self._angle_point(cx, cy, radius * 0.62, mid)
                draw_pill("SECTOR", tx, ty, sector_only_green)

            else:
                # distance OFF 또는 range 결과 없음: 기존 sector highlight 느낌만 유지
                if status == "trusted":
                    fill_poly(frame_poly, trusted_green, 0.45)
                elif status.startswith("hold"):
                    fill_poly(frame_poly, hold_orange, 0.35)
                else:
                    fill_poly(frame_poly, uncertain_yellow, 0.28)

                mid = (a0 + a1) / 2.0
                tx, ty = self._angle_point(cx, cy, radius * 0.60, mid)
                draw_centered_text(highlight_name, tx, ty, 0.55, (245, 245, 245), 1)

        # ------------------------------------------------------------
        # 3) receiver point
        # ------------------------------------------------------------
        cv2.circle(canvas, (cx, cy), 8, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 11, (36, 36, 40), 1, cv2.LINE_AA)

        draw_centered_text("RX0/RX1", cx, cy + 30, 0.48, (225, 225, 230), 1)


    def _draw_text_panel(
        self,
        canvas: np.ndarray,
        *,
        state: ViewerState,
        args: Any,
        dash_cfg: dict[str, Any],
        sector: dict[str, Any],
        selected_raw: dict[str, Any],
        cnn_result: dict[str, Any],
        raw_pass_count: int,
        cnn_drone_count: int,
        topk_count: int,
        paused: bool,
    ) -> None:
        import cv2

        panel_x = int(self.width * 0.66)
        panel_y = 92
        panel_w = self.width - panel_x - 30
        panel_h = self.height - 150

        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (25, 25, 28),
            -1,
        )
        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (85, 85, 90),
            1,
        )

        status = str(sector.get("sector_status", "no_signal") or "no_signal").upper()
        locked = str(sector.get("locked_sector_name", "") or "None")
        instant = str(sector.get("instant_sector_name", "") or "None")
        p99 = get_p99_value(sector, selected_raw, str(dash_cfg.get("p99_source", "median_raw_p99")))

        distance_cfg = dict(dash_cfg.get("distance", {}) or {})
        distance_enabled = bool(distance_cfg.get("enabled", False))
        range_result = get_dashboard_range_result(dash_cfg, sector, selected_raw)
        distance_text = format_range_display(range_result, distance_enabled=distance_enabled)

        lines = [
            ("Status", "PAUSED" if paused else status),
            ("Locked Sector", locked),
            ("Instant Sector", instant),
            ("Angle Median", f"{fmt_value(sector.get('angle_median'), 2)} deg"),
            ("Angle Spread", f"{fmt_value(sector.get('angle_spread'), 2)} deg"),
            ("Median Coherence", fmt_value(sector.get("median_coherence"), 3)),
            ("Raw P99", fmt_value(p99, 2)),
            ("Distance Bin", distance_text),
            ("Range Conf", format_range_field(range_result, "confidence")),
            ("Range Score", format_range_score(range_result)),
            ("Range Feature", format_range_features(range_result)),
            ("Capture", f"{'ACTIVE' if bool(getattr(state, 'capture_active', False)) else 'IDLE'} {int(getattr(state, 'capture_saved_count', 0))}/{int(getattr(state, 'capture_target', 0))}"),
            ("Save CSV", str(getattr(state, "profile_csv_name", "n/a"))),
            ("", ""),
            ("CNN Raw", str(cnn_result.get("cnn_raw_class_name", "n/a"))),
            ("CNN Conf", fmt_value(cnn_result.get("cnn_raw_confidence"), 3)),
            ("CNN Vote", f"{cnn_result.get('cnn_positive_votes', 0)}/{cnn_result.get('cnn_confirm_votes', 0)}"),
            ("Raw Pass", str(raw_pass_count)),
            ("Top-K", str(topk_count)),
            ("Drone Top-K", str(cnn_drone_count)),
            ("Valid AoA", str(sector.get("valid_aoa_count", 0))),
            ("Votes", str(sector.get("sector_votes", "") or "None")),
            ("", ""),
            ("Gain", f"{float(state.gain):.1f}"),
            ("Center Freq", f"{float(state.center_freq) / 1e6:.3f} MHz"),
            ("Phase Delta", f"{float(getattr(state, 'phase_offset_live_delta_deg', 0.0)):.1f} deg"),
            ("Reason", str(sector.get("reason", "") or "n/a")),
        ]

        x_key = panel_x + 18
        x_val = panel_x + 170
        y = panel_y + 34

        cv2.putText(
            canvas,
            "Runtime Result",
            (x_key, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += 30

        for key, val in lines:
            if y > panel_y + panel_h - 16:
                break

            if key == "":
                y += 12
                cv2.line(canvas, (x_key, y), (panel_x + panel_w - 18, y), (70, 70, 75), 1)
                y += 20
                continue

            cv2.putText(
                canvas,
                f"{key}:",
                (x_key, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (185, 185, 190),
                1,
                cv2.LINE_AA,
            )

            wrapped = self._wrap_text(str(val), max_chars=42)
            for j, chunk in enumerate(wrapped[:2]):
                cv2.putText(
                    canvas,
                    chunk,
                    (x_val, y + j * 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            y += 22 + max(0, len(wrapped[:2]) - 1) * 18

    @staticmethod
    def _wrap_text(text: str, *, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        words = text.split(" ")
        out: list[str] = []
        cur = ""

        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                out.append(cur)
                cur = w

        if cur:
            out.append(cur)

        final: list[str] = []
        for item in out:
            if len(item) <= max_chars:
                final.append(item)
            else:
                for i in range(0, len(item), max_chars):
                    final.append(item[i:i + max_chars])
        return final

    def _draw_footer(self, canvas: np.ndarray) -> None:
        import cv2

        text = "q quit | p pause | [/] gain | 1/2 dist | a/d angle | ,/. phase | m reset | s capture | x cancel"
        cv2.line(canvas, (30, self.height - 58), (self.width - 30, self.height - 58), (80, 80, 85), 1)
        cv2.putText(
            canvas,
            text,
            (30, self.height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )



_RANGE_ESTIMATOR_CACHE: dict[tuple[str, str, float], SectorRangeEstimator] = {}


def get_dashboard_range_result(
    dash_cfg: dict[str, Any],
    sector: dict[str, Any],
    selected_raw: dict[str, Any],
) -> dict[str, Any] | None:
    distance_cfg = dict(dash_cfg.get("distance", {}) or {})
    distance_enabled = bool(distance_cfg.get("enabled", False))

    if not distance_enabled:
        return None

    locked_sector = str(sector.get("locked_sector_name", "") or "").strip()
    if not locked_sector:
        return None

    profile_path = distance_cfg.get("profile_path")
    min_reliability = str(distance_cfg.get("min_reliability", "LOW"))
    min_margin = float(distance_cfg.get("min_margin_for_range", 0.25))

    if not profile_path:
        return {
            "range_class": "SECTOR_ONLY",
            "range_label_ko": "섹터 전체 표시",
            "display_mode": "sector_only",
            "sector_fill": True,
            "confidence": "LOW",
            "reliability": "LOW",
            "score": None,
            "threshold": None,
            "margin": None,
            "features_used": [],
            "enabled": True,
            "reason": "profile_path_empty",
        }

    key = (str(profile_path), min_reliability, min_margin)

    try:
        estimator = _RANGE_ESTIMATOR_CACHE.get(key)
        if estimator is None:
            estimator = SectorRangeEstimator(
                profile_path,
                min_reliability=min_reliability,
                min_margin_for_range=min_margin,
            )
            _RANGE_ESTIMATOR_CACHE[key] = estimator

        features = dict(selected_raw)
        features.update(dict(sector))

        return estimator.estimate(
            sector5=locked_sector,
            features=features,
        )

    except Exception as exc:
        return {
            "range_class": "SECTOR_ONLY",
            "range_label_ko": "섹터 전체 표시",
            "display_mode": "sector_only",
            "sector_fill": True,
            "confidence": "LOW",
            "reliability": "LOW",
            "score": None,
            "threshold": None,
            "margin": None,
            "features_used": [],
            "enabled": True,
            "reason": f"range_error:{type(exc).__name__}",
        }


def format_range_display(
    range_result: dict[str, Any] | None,
    *,
    distance_enabled: bool,
) -> str:
    if not distance_enabled:
        return "OFF"

    if not range_result:
        return "N/A"

    range_class = str(range_result.get("range_class", "N/A"))

    # cv2.putText 기본 폰트는 한글을 안정적으로 표시하지 못하므로
    # dashboard 화면에서는 ASCII only label을 사용한다.
    if range_class == "WITHIN_9M":
        return "WITHIN_9M / <=9m"

    if range_class == "RANGE_9_TO_15M":
        return "RANGE_9_TO_15M / 9-15m"

    if range_class == "SECTOR_ONLY":
        return "SECTOR_ONLY / full sector"

    return range_class


def format_range_field(
    range_result: dict[str, Any] | None,
    key: str,
) -> str:
    if not range_result:
        return "N/A"

    value = range_result.get(key)
    if value in (None, ""):
        return "N/A"

    return str(value)


def format_range_score(range_result: dict[str, Any] | None) -> str:
    if not range_result:
        return "N/A"

    score = range_result.get("score")
    threshold = range_result.get("threshold")
    margin = range_result.get("margin")

    if score is None:
        return "N/A"

    s = fmt_value(score, 3)

    if threshold is None:
        return s

    t = fmt_value(threshold, 3)

    if margin is None:
        return f"{s} / th {t}"

    m = fmt_value(margin, 3)
    return f"{s} / th {t} / m {m}"


def format_range_features(range_result: dict[str, Any] | None) -> str:
    if not range_result:
        return "N/A"

    features = range_result.get("features_used") or []
    if not features:
        reason = str(range_result.get("reason", ""))
        return reason or "N/A"

    aliases = {
        "median_raw_p99": "med_p99",
        "raw_abs_mean": "mean",
        "raw_abs_p50": "p50",
        "raw_abs_p95": "p95",
        "raw_abs_p99": "p99",
        "raw_abs_max": "max",
        "raw_rms": "rms",
        "frame_power_p99": "fp99",
        "ratio_p99_to_rms": "p99/rms",
        "ratio_p95_to_rms": "p95/rms",
        "ratio_p99_to_mean": "p99/mean",
        "ratio_framepower_to_rms2": "fp99/rms2",
        "angle_spread": "ang_spread",
        "median_coherence": "coh",
        "dominant_sector_ratio": "dom_ratio",
        "valid_aoa_count": "valid_aoa",
    }

    return " + ".join(aliases.get(str(x), str(x)) for x in features)



def apply_demo_range_cycle(
    *,
    sector: dict[str, Any],
    range_result: dict[str, Any] | None,
    update_idx: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    UI visual self-test mode.

    실제 CNN/AoA/sector 정책을 바꾸는 기능이 아니라,
    dashboard 그림이 sector/range cell을 제대로 그리는지 보기 위한 표시 전용 override다.
    """
    if not enabled:
        return sector, range_result

    sectors = [
        "LEFT_OUTER",
        "LEFT_INNER",
        "CENTER",
        "RIGHT_INNER",
        "RIGHT_OUTER",
    ]

    range_classes = [
        ("WITHIN_9M", "<=9m", "range_bin"),
        ("RANGE_9_TO_15M", "9-15m", "range_bin"),
        ("SECTOR_ONLY", "full sector", "sector_only"),
    ]

    sector_name = sectors[(update_idx // 10) % len(sectors)]
    range_class, label, display_mode = range_classes[(update_idx // 5) % len(range_classes)]

    demo_sector = dict(sector)
    demo_sector.update(
        {
            "sector_status": "trusted",
            "locked_sector_name": sector_name,
            "instant_sector_name": sector_name,
            "valid_aoa_count": 4,
            "reason": "demo_cycle",
        }
    )

    demo_range = {
        "range_class": range_class,
        "range_label_ko": label,
        "display_mode": display_mode,
        "sector_fill": display_mode == "sector_only",
        "confidence": "DEMO",
        "reliability": "DEMO",
        "score": 0.0,
        "threshold": 0.0,
        "margin": 1.0,
        "features_used": ["demo"],
        "enabled": True,
        "reason": "demo_cycle",
    }

    return demo_sector, demo_range


def handle_key(
    key: str | None,
    *,
    state: ViewerState,
    args: Any,
    receiver: Any,
    aoa_runtime: Any,
    cnn_runtime: Any,
) -> None:
    if key is None:
        return

    if key == "save_profile":
        state.capture_active = True
        state.capture_saved_count = 0
        state.capture_target = max(1, int(args.capture_trusted_n))
        print(
            f"[CAPTURE] start trusted-only capture target={state.capture_target} "
            f"d={state.distance_m:.1f}m "
            f"angle={float(getattr(state, 'true_angle_deg', 0.0)):.1f}deg "
            f"-> {getattr(state, 'profile_csv_path', 'n/a')}",
            flush=True,
        )
        return

    if key == "cancel_capture":
        state.capture_active = False
        print(
            f"[CAPTURE] canceled at {int(getattr(state, 'capture_saved_count', 0))}/"
            f"{int(getattr(state, 'capture_target', 0))}",
            flush=True,
        )
        return

    if key == "quit":
        state.running = False
    elif key == "pause":
        state.toggle_pause()
    elif key == "gain_down":
        state.step_gain(-args.gain_step, args.min_gain, args.max_gain)
        base.apply_receiver_gain(receiver, state.gain)
        aoa_runtime.update_gain(state.gain)
        cnn_runtime.reset_history()
        print(f"[GAIN] gain -> {state.gain:.1f}", flush=True)
    elif key == "gain_up":
        state.step_gain(args.gain_step, args.min_gain, args.max_gain)
        base.apply_receiver_gain(receiver, state.gain)
        aoa_runtime.update_gain(state.gain)
        cnn_runtime.reset_history()
        print(f"[GAIN] gain -> {state.gain:.1f}", flush=True)
    elif key == "1":
        state.distance_m = max(0.0, float(state.distance_m) - float(args.distance_step_m))
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "2":
        state.distance_m = float(state.distance_m) + float(args.distance_step_m)
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "0":
        state.distance_m = 0.0
        print(f"[LABEL] distance_m -> {state.distance_m:.1f}m", flush=True)
    elif key == "a":
        state.true_angle_deg = float(getattr(state, "true_angle_deg", 0.0)) - float(args.angle_step_deg)
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == "d":
        state.true_angle_deg = float(getattr(state, "true_angle_deg", 0.0)) + float(args.angle_step_deg)
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == "c":
        state.true_angle_deg = 0.0
        print(f"[LABEL] true_angle_deg -> {state.true_angle_deg:.1f}deg", flush=True)
    elif key == ",":
        delta_deg = -1.0
        state.phase_offset_live_delta_deg = float(getattr(state, "phase_offset_live_delta_deg", 0.0)) + delta_deg
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(delta_deg))
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta={state.phase_offset_live_delta_deg:.1f}deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )
    elif key == ".":
        delta_deg = 1.0
        state.phase_offset_live_delta_deg = float(getattr(state, "phase_offset_live_delta_deg", 0.0)) + delta_deg
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(delta_deg))
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta={state.phase_offset_live_delta_deg:.1f}deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )
    elif key == "m":
        reset_deg = -float(getattr(state, "phase_offset_live_delta_deg", 0.0))
        aoa_runtime.phase_offset_to_apply_rad += float(np.deg2rad(reset_deg))
        state.phase_offset_live_delta_deg = 0.0
        state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))
        print(
            f"[PHASE] live_delta reset -> 0.0deg "
            f"total={state.phase_offset_total_deg:.1f}deg",
            flush=True,
        )


def run() -> int:
    args = base.apply_defaults(base.parse_args())
    dash_cfg = load_dashboard_cfg(args)

    if not bool(dash_cfg.get("enabled", True)):
        print("[WARN] sector_dashboard.enabled=false, but script will still run.")

    state = ViewerState(
        mode="sector_dashboard",
        gain=args.gain,
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        distance_m=args.distance_m,
        memo=args.memo,
        target_fps=args.target_fps,
    )
    state.true_angle_deg = float(args.true_angle_deg)
    state.phase_offset_live_delta_deg = 0.0
    state.phase_offset_total_deg = 0.0

    raw_gate = RawNoiseGate(detect_config_path=Path(args.config_dir) / "detect.yaml")
    if bool(args.disable_raw_gate):
        raw_gate.enabled = False
        print("[DRY-RUN] raw noise gate disabled by --disable-raw-gate")

    receiver = base.build_receiver(args)
    cnn_runtime = base.build_cnn_runtime(args)
    aoa_runtime = base.build_aoa_runtime(args)
    state.phase_offset_total_deg = float(np.rad2deg(aoa_runtime.phase_offset_to_apply_rad))

    renderer = SectorDashboardRenderer(
        window_name="RF Sector Dashboard",
        target_fps=args.target_fps,
        width=int(dash_cfg.get("canvas_width", 1120)),
        height=int(dash_cfg.get("canvas_height", 720)),
        blink_on_hold=bool(dash_cfg.get("blink_on_hold", True)),
        fade_on_signal_lost=bool(dash_cfg.get("fade_on_signal_lost", True)),
    )

    lock = base.SectorLockState()

    out_dir = Path(args.sector_profile.get("output_dir", "outputs/aoa_sector_profiles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_csv = out_dir / f"{ts}_gain{args.gain:g}_cf{args.center_freq}_sector_profile.csv"

    state.capture_active = False
    state.capture_saved_count = 0
    state.capture_target = max(1, int(args.capture_trusted_n))
    state.profile_csv_name = profile_csv.name
    state.profile_csv_path = str(profile_csv)

    last_sector: dict[str, Any] = {
        "sector_status": "no_signal",
        "locked_sector_name": "",
        "instant_sector_name": "",
        "angle_median": "",
        "angle_spread": "",
        "median_coherence": "",
        "median_raw_p99": "",
        "valid_aoa_count": 0,
        "sector_votes": "",
        "reason": "initial",
    }
    last_selected_raw: dict[str, Any] = {}
    last_cnn_result: dict[str, Any] = {
        "cnn_raw_class_name": "n/a",
        "cnn_raw_confidence": 0.0,
        "cnn_positive_votes": 0,
        "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
    }
    last_raw_pass_count = 0
    last_cnn_drone_count = 0
    last_topk_count = 0

    try:
        while state.running:
            if state.paused:
                key = renderer.render(
                    state=state,
                    args=args,
                    dash_cfg=dash_cfg,
                    sector=last_sector,
                    selected_raw=last_selected_raw,
                    cnn_result=last_cnn_result,
                    raw_pass_count=last_raw_pass_count,
                    cnn_drone_count=last_cnn_drone_count,
                    topk_count=last_topk_count,
                    paused=True,
                )
                handle_key(
                    key,
                    state=state,
                    args=args,
                    receiver=receiver,
                    aoa_runtime=aoa_runtime,
                    cnn_runtime=cnn_runtime,
                )
                continue

            blocks: list[np.ndarray] = []
            raws: list[dict[str, Any]] = []
            gate_results: list[Any] = []

            for _ in range(max(1, int(args.blocks_per_update))):
                try:
                    iq = receiver.read_block(args.block_size)
                except EOFError:
                    if hasattr(receiver, "reset"):
                        receiver.reset()
                        iq = receiver.read_block(args.block_size)
                    else:
                        raise

                if not args.disable_dc_offset_removal:
                    iq = remove_dc_offset(iq, axis=-1)

                raw = compute_raw_features(iq)
                gate = raw_gate.evaluate(iq, gain=state.gain)

                blocks.append(iq)
                raws.append(raw)
                gate_results.append(gate)

            state.mark_update()

            raw_pass_indices = [
                i for i, r in enumerate(gate_results)
                if bool((not r.enabled) or r.passed)
            ]
            raw_pass_count = len(raw_pass_indices)

            topk_idx = base.select_topk_indices(gate_results, args.top_k)

            topk_items: list[dict[str, Any]] = []
            for idx in topk_idx:
                image, cnn_raw = base.run_cnn_raw_no_history(cnn_runtime, blocks[idx])
                topk_items.append(
                    {
                        "index": idx,
                        "iq": blocks[idx],
                        "raw": raws[idx],
                        "gate": gate_results[idx],
                        "image": image,
                        "cnn": cnn_raw,
                    }
                )

            if raw_pass_count <= 0:
                if raw_gate.reset_cnn_history_on_fail():
                    cnn_runtime.reset_history()
                cnn_result = {
                    "cnn_raw_class_name": "RAW_GATE_BLOCKED",
                    "cnn_raw_confidence": 0.0,
                    "cnn_candidate": False,
                    "cnn_confirmed": False,
                    "cnn_positive_votes": 0,
                    "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
                    "cnn_skipped": True,
                    "cnn_skipped_reason": "raw_noise_gate_failed",
                }
                aoa_candidates: list[dict[str, Any]] = []
            else:
                vote_raw = base.pick_vote_cnn_result(topk_items)
                cnn_result = (
                    base.update_cnn_history_once(cnn_runtime, vote_raw)
                    if vote_raw
                    else {
                        "cnn_raw_class_name": "NO_CNN_RESULT",
                        "cnn_raw_confidence": 0.0,
                        "cnn_candidate": False,
                        "cnn_confirmed": False,
                        "cnn_positive_votes": 0,
                        "cnn_confirm_votes": int(cnn_runtime.confirm_votes),
                    }
                )
                aoa_candidates = base.build_aoa_candidates(topk_items, aoa_runtime, args)

            cnn_drone_count = sum(
                1 for item in topk_items
                if bool(item["cnn"].get("cnn_candidate", False))
            )

            sector = base.update_sector_lock(
                lock=lock,
                candidates=aoa_candidates,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                args=args,
            )

            selected_raw = dict(raws[topk_idx[0]]) if topk_idx else {}
            selected_raw.update(
                {
                    "selected_block_index": int(topk_idx[0]) if topk_idx else -1,
                    "blocks_per_update": int(args.blocks_per_update),
                    "top_k": int(args.top_k),
                    "raw_gate_pass_count": int(raw_pass_count),
                    "cnn_topk_count": int(len(topk_items)),
                    "cnn_drone_count": int(cnn_drone_count),
                }
            )

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "update_idx": int(state.update_index),
                "gain": float(state.gain),
                "center_freq": int(state.center_freq),
                "sample_rate": int(state.sample_rate),
                "blocks_per_update": int(args.blocks_per_update),
                "top_k": int(args.top_k),
                "sector_preset": str(args.sector_preset_name),
                "distance_m": float(state.distance_m),
                "true_angle_deg": float(getattr(state, "true_angle_deg", 0.0)),
                "phase_offset_live_delta_deg": float(getattr(state, "phase_offset_live_delta_deg", 0.0)),
                "phase_offset_total_deg": float(getattr(state, "phase_offset_total_deg", 0.0)),
                "memo": str(state.memo),
                **selected_raw,
                **cnn_result,
                **sector,
            }

            if bool(getattr(state, "capture_active", False)):
                capture_ok = (
                    str(sector.get("sector_status", "")) == "trusted"
                    and int(sector.get("valid_aoa_count", 0) or 0) >= 2
                    and int(cnn_drone_count) >= 2
                    and str(sector.get("locked_sector_name", "")).strip() != ""
                )

                if capture_ok:
                    state.capture_saved_count = int(getattr(state, "capture_saved_count", 0)) + 1

                    capture_row = dict(row)
                    capture_row.update(
                        {
                            "capture_active": True,
                            "capture_saved_index": int(state.capture_saved_count),
                            "capture_target": int(state.capture_target),
                            "capture_done": bool(int(state.capture_saved_count) >= int(state.capture_target)),
                            "capture_source": "sector_dashboard",
                        }
                    )

                    base.append_sector_profile(profile_csv, capture_row)

                    print(
                        f"[CAPTURE] saved {int(state.capture_saved_count)}/{int(state.capture_target)} "
                        f"sector={sector.get('locked_sector_name')} "
                        f"d={state.distance_m:.1f}m "
                        f"angle={float(getattr(state, 'true_angle_deg', 0.0)):.1f}deg "
                        f"-> {profile_csv}",
                        flush=True,
                    )

                    if int(state.capture_saved_count) >= int(state.capture_target):
                        state.capture_active = False
                        print(f"[CAPTURE] done -> {profile_csv}", flush=True)

            last_sector = sector
            last_selected_raw = selected_raw
            last_cnn_result = cnn_result
            last_raw_pass_count = raw_pass_count
            last_cnn_drone_count = cnn_drone_count
            last_topk_count = len(topk_items)

            if not args.disable_cli_log and state.update_index % int(args.cli_log_every_n) == 0:
                print(
                    f"[DASH] idx={state.update_index} "
                    f"raw_pass={raw_pass_count} topk={len(topk_items)} drone={cnn_drone_count} "
                    f"status={sector['sector_status']} "
                    f"instant={sector['instant_sector_name'] or 'None'} "
                    f"locked={sector['locked_sector_name'] or 'None'} "
                    f"angle_med={base.fmt(sector.get('angle_median'), 3)} "
                    f"coh={base.fmt(sector.get('median_coherence'), 3)} "
                    f"p99={base.fmt(sector.get('median_raw_p99'), 4)} "
                    f"reason={sector['reason']}",
                    flush=True,
                )

            key = renderer.render(
                state=state,
                args=args,
                dash_cfg=dash_cfg,
                sector=sector,
                selected_raw=selected_raw,
                cnn_result=cnn_result,
                raw_pass_count=raw_pass_count,
                cnn_drone_count=cnn_drone_count,
                topk_count=len(topk_items),
                paused=False,
            )

            handle_key(
                key,
                state=state,
                args=args,
                receiver=receiver,
                aoa_runtime=aoa_runtime,
                cnn_runtime=cnn_runtime,
            )

    except KeyboardInterrupt:
        return 130
    finally:
        try:
            receiver.close()
        except Exception:
            pass
        renderer.close()

    return 0


if __name__ == "__main__":
    sys.exit(run())
