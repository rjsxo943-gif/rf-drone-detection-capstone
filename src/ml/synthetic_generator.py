import numpy as np


def generate_complex_tone(
    num_samples: int,
    freq_norm: float = 0.1,
    amplitude: float = 1.0,
    noise_std: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = np.arange(num_samples, dtype=np.float32)
    phase = 2 * np.pi * freq_norm * n
    signal = amplitude * np.exp(1j * phase)
    noise = noise_std * (
        rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    )
    return (signal + noise).astype(np.complex64)
