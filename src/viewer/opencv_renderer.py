from __future__ import annotations

import time
from typing import Iterable

import numpy as np


class OpenCVRenderer:
    """Small OpenCV wrapper for low-latency spectrogram display."""

    def __init__(self, window_name: str = "RF Spectrogram", target_fps: float = 10.0) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc

        self.window_name = window_name
        self.target_fps = max(0.1, float(target_fps))
        self._last_render_time = 0.0

    def render(self, image: np.ndarray, overlay: Iterable[str] | None = None) -> str | None:
        import cv2

        self._sleep_to_limit_fps()
        frame = self._to_bgr_image(image)
        self._draw_overlay(frame, list(overlay or []))

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

    @classmethod
    def _to_bgr_image(cls, image: np.ndarray) -> np.ndarray:
        import cv2

        gray = cls._to_uint8(image)
        gray = np.flipud(gray)
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    @staticmethod
    def _draw_overlay(frame: np.ndarray, overlay: list[str]) -> None:
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
