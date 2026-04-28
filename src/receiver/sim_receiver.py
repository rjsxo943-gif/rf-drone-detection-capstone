from __future__ import annotations

import numpy as np

from src.receiver.base import BaseReceiver


class SimReceiver(BaseReceiver):
    """
    Synthetic IQ 신호를 생성하는 Receiver.

    출력 shape:
    - num_channels = 1 -> (1, num_samples)
    - num_channels = 2 -> (2, num_samples)
    """

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        num_channels: int = 1,
        block_size: int = 16384,
        tone_freq_norm: float = 0.08,
        noise_std: float = 0.08,
        burst_amplitude: float = 1.5,
        burst_period: int = 4096,
        burst_length: int = 768,
        seed: int | None = None,
        channel_phase_offset_rad: float = 0.0,
    ) -> None:
        super().__init__(
            sample_rate=int(sample_rate),
            center_freq=int(center_freq),
            num_channels=int(num_channels),
        )

        # BaseReceiver가 num_channels를 저장하지 않는 경우를 대비해 명시적으로 저장
        self.sample_rate = int(sample_rate)
        self.center_freq = int(center_freq)
        self.num_channels = int(num_channels)

        self.block_size = int(block_size)
        self.tone_freq_norm = float(tone_freq_norm)
        self.noise_std = float(noise_std)
        self.burst_amplitude = float(burst_amplitude)
        self.burst_period = int(burst_period)
        self.burst_length = int(burst_length)
        self.channel_phase_offset_rad = float(channel_phase_offset_rad)

        self.rng = np.random.default_rng(seed)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Synthetic IQ samples를 생성한다.

        반환:
            shape = (num_channels, num_samples)
            dtype = complex64
        """
        num_samples = int(num_samples)

        base_signal = self._generate_base_signal(num_samples)

        channels = []

        for ch in range(self.num_channels):
            phase_offset = ch * self.channel_phase_offset_rad
            channel_signal = base_signal * np.exp(1j * phase_offset)

            noise = self._generate_noise(num_samples)
            channels.append(channel_signal + noise)

        samples = np.stack(channels, axis=0).astype(np.complex64)

        # 여기서는 2D shape을 유지해야 하므로 validate_samples가 squeeze하면 안 됨
        if samples.shape != (self.num_channels, num_samples):
            raise ValueError(
                f"Invalid SimReceiver output shape: {samples.shape}. "
                f"Expected {(self.num_channels, num_samples)}"
            )

        return samples

    def read_block(self, block_size: int | None = None) -> np.ndarray:
        if block_size is None:
            block_size = self.block_size

        return self.read_samples(int(block_size))

    def _generate_base_signal(self, num_samples: int) -> np.ndarray:
        n = np.arange(num_samples, dtype=np.float32)

        phase = 2.0 * np.pi * self.tone_freq_norm * n
        tone = np.exp(1j * phase).astype(np.complex64)

        envelope = self._generate_burst_envelope(num_samples)
        signal = envelope * tone

        return signal.astype(np.complex64)

    def _generate_burst_envelope(self, num_samples: int) -> np.ndarray:
        envelope = np.zeros(num_samples, dtype=np.float32)

        start = 0
        while start < num_samples:
            end = min(start + self.burst_length, num_samples)
            envelope[start:end] = self.burst_amplitude
            start += self.burst_period

        return envelope

    def _generate_noise(self, num_samples: int) -> np.ndarray:
        noise = self.noise_std * (
            self.rng.standard_normal(num_samples)
            + 1j * self.rng.standard_normal(num_samples)
        )

        return noise.astype(np.complex64)