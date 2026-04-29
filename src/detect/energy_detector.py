from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocess.framing import frame_signal
from src.features.fft import compute_fft_power


@dataclass
class EnergyDetectionResult:
    """
    block 하나에 대한 에너지 탐지 결과.
    """

    detected: bool
    detection_ratio: float
    frame_energies: np.ndarray
    frame_detections: np.ndarray
    noise_floor: float
    threshold: float
    calibrated: bool


class EnergyDetector:
    """
    에너지 기반 1차 탐지기.

    현재 프로젝트 기준:
    - 입력은 block 단위 IQ
    - block_size = 16,384 samples
    - block 내부를 작은 energy frame으로 나눠 에너지 계산
    - 초기 N개 block으로 noise floor를 calibration한 뒤 detection 수행

    권장 사용:
    1. 시작 직후 목표 신호가 없는 상태에서 calibration block 수집
    2. calibrate_from_blocks() 또는 calibrate_block()으로 noise floor 설정
    3. 이후 detect_block() 실행
    """

    def __init__(
        self,
        mode: str = "initial_calibration",
        threshold_multiplier: float = 5.0,
        frame_size: int = 1024,
        hop_size: int = 512,
        window: str = "hann",
        method: str = "time_power",
        min_detection_ratio: float = 0.05,
        calibration_num_blocks: int = 20,
        require_calibration: bool = True,
    ) -> None:
        self.mode = mode.lower().strip()

        self.threshold_multiplier = float(threshold_multiplier)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.window = window
        self.method = method.lower().strip()
        self.min_detection_ratio = float(min_detection_ratio)

        self.calibration_num_blocks = int(calibration_num_blocks)
        self.require_calibration = bool(require_calibration)

        self.noise_floor: float | None = None
        self.threshold: float | None = None

        self._calibration_energies: list[np.ndarray] = []

    @property
    def is_calibrated(self) -> bool:
        """
        noise floor와 threshold가 설정되었는지 여부.
        """
        return self.noise_floor is not None and self.threshold is not None

    def reset_calibration(self) -> None:
        """
        noise calibration 상태를 초기화한다.
        """
        self.noise_floor = None
        self.threshold = None
        self._calibration_energies.clear()

    def fit(self, frame_energies: np.ndarray) -> None:
        """
        frame energy 배열로 noise floor와 threshold를 추정한다.

        기본:
        - noise_floor = median(frame_energies)
        - threshold = noise_floor * threshold_multiplier
        """
        frame_energies = np.asarray(frame_energies, dtype=np.float32)

        if frame_energies.size == 0:
            raise ValueError("frame_energies is empty.")

        self.noise_floor = float(np.median(frame_energies))
        self.threshold = float(self.noise_floor * self.threshold_multiplier)

    def calibrate_from_blocks(self, iq_blocks: np.ndarray | list[np.ndarray]) -> None:
        """
        여러 block을 한 번에 받아 noise floor를 calibration한다.

        입력 예:
        - list of block: [block0, block1, ...]
        - ndarray shape = (num_blocks, num_channels, block_size)

        사용 목적:
        - 시작 직후 목표 신호가 없는 상태에서 N개 block을 모아 noise 기준 설정
        """
        energies_list: list[np.ndarray] = []

        for block in iq_blocks:
            energies = self.compute_frame_energies(block)

            if energies.size > 0:
                energies_list.append(energies)

        if not energies_list:
            raise ValueError("No valid frame energies collected for calibration.")

        all_energies = np.concatenate(energies_list, axis=0)
        self.fit(all_energies)

    def calibrate_block(self, iq_block: np.ndarray) -> bool:
        """
        calibration용 block 하나를 추가한다.

        Returns:
            True  -> calibration_num_blocks만큼 모여서 calibration 완료
            False -> 아직 calibration block이 더 필요함
        """
        energies = self.compute_frame_energies(iq_block)

        if energies.size > 0:
            self._calibration_energies.append(energies)

        if len(self._calibration_energies) >= self.calibration_num_blocks:
            all_energies = np.concatenate(self._calibration_energies, axis=0)
            self.fit(all_energies)
            return True

        return False

    def get_calibration_progress(self) -> tuple[int, int]:
        """
        현재 calibration 진행 상황을 반환한다.

        Returns:
            (현재 모은 block 수, 필요한 block 수)
        """
        return len(self._calibration_energies), self.calibration_num_blocks

    def detect(self, frame_energies: np.ndarray) -> np.ndarray:
        """
        frame energy 배열에서 threshold 초과 여부를 반환한다.

        Returns:
            bool ndarray, shape = (num_frames,)
        """
        frame_energies = np.asarray(frame_energies, dtype=np.float32)

        if frame_energies.size == 0:
            return np.zeros((0,), dtype=bool)

        if self.threshold is None:
            if self.require_calibration:
                raise RuntimeError(
                    "EnergyDetector is not calibrated. "
                    "Call calibrate_from_blocks() or calibrate_block() before detect()."
                )

            # backward compatibility:
            # require_calibration=False이면 기존처럼 첫 입력으로 threshold 설정
            self.fit(frame_energies)

        return frame_energies > float(self.threshold)


    def detect_frame_energies(self, frame_energies: np.ndarray) -> np.ndarray:
        frame_energies = np.asarray(frame_energies, dtype=np.float32)

        if frame_energies.size == 0:
            return np.zeros((0,), dtype=bool)

        if self.mode == "block_median":
            self.fit(frame_energies)
            return frame_energies > float(self.threshold)

        if self.mode == "initial_calibration":
            return self.detect(frame_energies)

        raise ValueError(
            f"Unsupported EnergyDetector mode: {self.mode}. "
            "Expected 'block_median' or 'initial_calibration'."
    )

    def detect_block(self, iq_block: np.ndarray) -> EnergyDetectionResult:
        """
        block 하나에 대해 에너지 탐지를 수행한다.

        입력:
        - 단일 채널: shape = (16384,)
        - 2채널: shape = (2, 16384)

        출력:
        - EnergyDetectionResult
        """
        frame_energies = self.compute_frame_energies(iq_block)
        frame_detections = self.detect(frame_energies)

        if frame_detections.size == 0:
            detection_ratio = 0.0
            detected = False
        else:
            detection_ratio = float(np.mean(frame_detections))
            detected = detection_ratio >= self.min_detection_ratio

        return EnergyDetectionResult(
            detected=detected,
            detection_ratio=detection_ratio,
            frame_energies=frame_energies.astype(np.float32),
            frame_detections=frame_detections.astype(bool),
            noise_floor=float(self.noise_floor),
            threshold=float(self.threshold),
            calibrated=self.is_calibrated,
        )

    def compute_frame_energies(self, iq_block: np.ndarray) -> np.ndarray:
        """
        block 내부 frame별 energy를 계산한다.

        여러 채널이 들어오면 채널별 frame energy를 계산한 뒤 평균한다.

        Returns:
            frame_energies, shape = (num_frames,)
        """
        iq_block = self._ensure_2d_iq(iq_block)

        channel_energies = []

        for ch in range(iq_block.shape[0]):
            frames = frame_signal(
                iq_block[ch],
                frame_size=self.frame_size,
                hop_size=self.hop_size,
            )

            if frames.size == 0:
                continue

            if self.method == "time_power":
                energies = self._compute_time_power(frames)

            elif self.method == "fft_power":
                energies = self._compute_fft_power(frames)

            else:
                raise ValueError(
                    f"Unsupported energy detection method: {self.method}. "
                    "Expected 'time_power' or 'fft_power'."
                )

            channel_energies.append(energies)

        if not channel_energies:
            return np.zeros((0,), dtype=np.float32)

        # shape: (num_channels, num_frames)
        stacked = np.stack(channel_energies, axis=0)

        # 채널 평균 energy
        return np.mean(stacked, axis=0).astype(np.float32)

    def _compute_time_power(self, frames: np.ndarray) -> np.ndarray:
        """
        시간영역 power 기반 energy 계산.

        energy = mean(|x[n]|^2)
        """
        return np.mean(np.abs(frames) ** 2, axis=1).astype(np.float32)

    def _compute_fft_power(self, frames: np.ndarray) -> np.ndarray:
        """
        FFT power 기반 energy 계산.

        energy = mean(|FFT(x)|^2)
        """
        power = compute_fft_power(
            frames,
            window=self.window,
            fftshift=True,
            log_scale=False,
        )

        return np.mean(power, axis=1).astype(np.float32)

    def _ensure_2d_iq(self, iq: np.ndarray) -> np.ndarray:
        """
        IQ block을 (num_channels, num_samples) 형태로 맞춘다.
        """
        iq = np.asarray(iq)

        if iq.size == 0:
            raise ValueError("Input IQ block is empty.")

        if not np.iscomplexobj(iq):
            raise TypeError(f"IQ block must be complex, got dtype={iq.dtype}")

        if iq.ndim == 1:
            iq = iq[np.newaxis, :]

        if iq.ndim != 2:
            raise ValueError(
                f"IQ block must be 1D or 2D. "
                f"Expected (N,) or (C, N), got shape {iq.shape}"
            )

        return iq.astype(np.complex64, copy=False)