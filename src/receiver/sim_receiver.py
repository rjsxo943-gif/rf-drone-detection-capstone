import numpy as np


class SimReceiver:
    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        tone_freq_norm: float = 0.08,
        noise_std: float = 0.08,
        burst_amplitude: float = 2.5,
        burst_period: int = 4096,
        burst_length: int = 768,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.tone_freq_norm = tone_freq_norm
        self.noise_std = noise_std
        self.burst_amplitude = burst_amplitude
        self.burst_period = burst_period
        self.burst_length = burst_length
        self.rng = np.random.default_rng(seed)

    def read_samples(self, num_samples: int) -> np.ndarray:
        n = np.arange(num_samples, dtype=np.float32)

        phase = 2.0 * np.pi * self.tone_freq_norm * n
        tone = np.exp(1j * phase).astype(np.complex64)

        envelope = np.zeros(num_samples, dtype=np.float32)
        start = 0
        while start < num_samples:
            end = min(start + self.burst_length, num_samples)
            envelope[start:end] = self.burst_amplitude
            start += self.burst_period

        signal = envelope * tone

        noise = self.noise_std * (
            self.rng.standard_normal(num_samples)
            + 1j * self.rng.standard_normal(num_samples)
        )

        return (signal + noise).astype(np.complex64)
