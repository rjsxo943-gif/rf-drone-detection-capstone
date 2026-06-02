from __future__ import annotations

import time
from typing import Iterable

import numpy as np


class OpenCVRenderer:
    """Small OpenCV wrapper for low-latency spectrogram display.

    overlay_mode:
        - "image": 기존 방식처럼 spectrogram 위에 글자 출력
        - "right": spectrogram 오른쪽에 별도 status panel 출력
    """

    def __init__(
        self,
        window_name: str = "RF Spectrogram",
        target_fps: float = 10.0,
        display_scale: float = 2.0,
        display_width: int = 0,
        display_height: int = 0,
        auto_orient: bool = False,
        overlay_mode: str = "right",
        overlay_width: int = 460,
    ) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc

        self.window_name = window_name
        self.target_fps = max(0.1, float(target_fps))
        self.display_scale = max(0.1, float(display_scale))
        self.display_width = max(0, int(display_width))
        self.display_height = max(0, int(display_height))
        self.auto_orient = bool(auto_orient)
        self.overlay_mode = str(overlay_mode).lower().strip()
        self.overlay_width = max(260, int(overlay_width))
        self._last_render_time = 0.0

    def render(self, image: np.ndarray, overlay: Iterable[str] | None = None) -> str | None:
        import cv2

        self._sleep_to_limit_fps()

        spectrogram_frame = self._to_bgr_image(image)
        spectrogram_frame = self._resize_for_display(spectrogram_frame)

        overlay_lines = list(overlay or [])

        if self.overlay_mode in ("right", "sidebar", "side") and overlay_lines:
            frame = self._compose_right_panel(spectrogram_frame, overlay_lines)
        else:
            frame = spectrogram_frame
            self._draw_overlay(frame, overlay_lines)

        cv2.imshow(self.window_name, frame)
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
        return chr(key) if 0 <= key <= 255 else None

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.window_name)

    def _sleep_to_limit_fps(self) -> None:
        min_interval = 1.0 / self.target_fps
        elapsed = time.monotonic() - self._last_render_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2:
            raise ValueError(f"image must be 2-D spectrogram. got shape={arr.shape}")

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - min_val) / (max_val - min_val)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    def _to_bgr_image(self, image: np.ndarray) -> np.ndarray:
        import cv2

        gray = self._to_uint8(image)
        gray = np.flipud(gray)
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        height, width = frame.shape[:2]

        if self.display_width > 0 and self.display_height > 0:
            target_size = (self.display_width, self.display_height)
        elif self.display_width > 0:
            scale = self.display_width / max(1, width)
            target_size = (self.display_width, max(1, int(round(height * scale))))
        elif self.display_height > 0:
            scale = self.display_height / max(1, height)
            target_size = (max(1, int(round(width * scale))), self.display_height)
        else:
            if abs(self.display_scale - 1.0) < 1e-6:
                return frame
            target_size = (
                max(1, int(round(width * self.display_scale))),
                max(1, int(round(height * self.display_scale))),
            )

        return cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)

    def _compose_right_panel(self, frame: np.ndarray, overlay: list[str]) -> np.ndarray:
        import cv2

        frame_h, frame_w = frame.shape[:2]
        panel_w = int(self.overlay_width)

        line_height = 22
        padding = 14
        required_h = padding * 2 + line_height * (len(overlay) + 3)
        canvas_h = max(frame_h, required_h)
        canvas_w = frame_w + panel_w

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Spectrogram 영역: 왼쪽
        canvas[:frame_h, :frame_w] = frame

        # Panel 영역: 오른쪽
        panel_x = frame_w
        cv2.rectangle(
            canvas,
            (panel_x, 0),
            (canvas_w - 1, canvas_h - 1),
            (18, 18, 18),
            thickness=-1,
        )

        # 구분선
        cv2.line(
            canvas,
            (panel_x, 0),
            (panel_x, canvas_h - 1),
            (90, 90, 90),
            thickness=1,
        )

        x = panel_x + padding
        y = padding + 20

        cv2.putText(
            canvas,
            "RF VIEWER STATUS",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += line_height

        cv2.line(
            canvas,
            (x, y - 8),
            (canvas_w - padding, y - 8),
            (90, 90, 90),
            thickness=1,
        )

        max_text_width = panel_w - padding * 2
        y = self._draw_panel_lines(
            canvas,
            overlay,
            x=x,
            y=y + 8,
            max_text_width=max_text_width,
            line_height=line_height,
        )

        return canvas

    @staticmethod
    def _draw_panel_lines(
        frame: np.ndarray,
        lines: list[str],
        *,
        x: int,
        y: int,
        max_text_width: int,
        line_height: int = 22,
    ) -> int:
        import cv2

        # OpenCV 기본 폰트 기준 대략적인 wrap.
        max_chars = max(24, int(max_text_width / 8.5))

        for line in lines:
            for wrapped in OpenCVRenderer._wrap_text(str(line), max_chars=max_chars):
                if y > frame.shape[0] - 8:
                    return y

                cv2.putText(
                    frame,
                    wrapped,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                y += line_height

        return y

    @staticmethod
    def _wrap_text(text: str, *, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current = ""

        for token in text.split(" "):
            if not current:
                current = token
                continue

            if len(current) + 1 + len(token) <= max_chars:
                current += " " + token
            else:
                chunks.append(current)
                current = token

        if current:
            chunks.append(current)

        # 공백 없는 긴 문자열 대응
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
            else:
                for i in range(0, len(chunk), max_chars):
                    final.append(chunk[i : i + max_chars])

        return final

    @staticmethod
    def _draw_overlay(frame: np.ndarray, overlay: list[str]) -> None:
        """Legacy mode: draw text directly on the spectrogram image."""
        import cv2

        x = 10
        y = 22
        line_height = 22
        for line in overlay:
            cv2.putText(
                frame,
                str(line),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += line_height
