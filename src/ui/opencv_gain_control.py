from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpenCVGainControl:
    current_gain: float
    min_gain: float = -3.0
    max_gain: float = 71.0
    step_db: float = 1.0
    settle_after_update: int = 3

    input_mode: bool = False
    input_buffer: str = ""
    message: str = ""
    settle_blocks: int = 0

    def handle_key(self, key: int, receiver) -> str | None:
        """
        반환:
            "quit"이면 viewer 종료
            None이면 계속 실행
        """
        if key < 0:
            return None

        key = key & 0xFF

        if self.input_mode:
            return self._handle_input_mode_key(key, receiver)

        return self._handle_normal_mode_key(key, receiver)

    def tick_settle(self) -> bool:
        """
        gain 변경 직후 안정화 block이면 True.
        """
        if self.settle_blocks > 0:
            self.settle_blocks -= 1
            return True
        return False

    def _handle_normal_mode_key(self, key: int, receiver) -> str | None:
        if key == ord("q"):
            return "quit"

        if key == ord("g"):
            self.input_mode = True
            self.input_buffer = ""
            self.message = "Gain edit mode"
            return None

        if key == ord("["):
            self._apply_gain(receiver, self.current_gain - self.step_db)
            return None

        if key == ord("]"):
            self._apply_gain(receiver, self.current_gain + self.step_db)
            return None

        return None

    def _handle_input_mode_key(self, key: int, receiver) -> str | None:
        # Enter
        if key in (10, 13):
            if not self.input_buffer:
                self.message = "Empty gain input"
                self.input_mode = False
                return None

            try:
                new_gain = float(self.input_buffer)
            except ValueError:
                self.message = f"Invalid gain: {self.input_buffer}"
                self.input_mode = False
                self.input_buffer = ""
                return None

            self._apply_gain(receiver, new_gain)
            self.input_mode = False
            self.input_buffer = ""
            return None

        # ESC
        if key == 27:
            self.input_mode = False
            self.input_buffer = ""
            self.message = "Gain input canceled"
            return None

        # Backspace
        if key in (8, 127):
            self.input_buffer = self.input_buffer[:-1]
            return None

        char = chr(key)

        if char.isdigit():
            self.input_buffer += char
            return None

        if char == "." and "." not in self.input_buffer:
            self.input_buffer += char
            return None

        if char == "-" and not self.input_buffer:
            self.input_buffer += char
            return None

        return None

    def _apply_gain(self, receiver, gain: float) -> None:
        gain = max(self.min_gain, min(self.max_gain, float(gain)))

        if hasattr(receiver, "set_gain"):
            applied_gain = receiver.set_gain(gain)
        else:
            raise AttributeError(
                "receiver.set_gain()이 없습니다. "
                "src/receiver/pluto_receiver.py 패치가 먼저 필요합니다."
            )

        self.current_gain = float(applied_gain)
        self.settle_blocks = int(self.settle_after_update)
        self.message = f"Gain updated: {self.current_gain:.1f} dB"


def draw_gain_overlay(frame, gain_control: OpenCVGainControl):
    """
    OpenCV BGR 이미지 위에 gain 상태창을 그린다.
    frame은 원본을 직접 수정한다.
    """
    import cv2

    x, y = 12, 12
    w, h = 520, 118

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    line1 = f"Current Gain : {gain_control.current_gain:.1f} dB"

    if gain_control.input_mode:
        line2 = f"Input Gain   : [{gain_control.input_buffer}]"
        line3 = "Enter: apply | Esc: cancel | Backspace: delete"
    else:
        line2 = "g: edit gain | [: -1 dB | ]: +1 dB | q: quit"
        line3 = gain_control.message or "Gain control ready"

    if gain_control.settle_blocks > 0:
        line3 = f"GAIN SETTLING... {gain_control.settle_blocks} blocks"

    cv2.putText(frame, line1, (x + 14, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, line2, (x + 14, y + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, line3, (x + 14, y + 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2)

    return frame
