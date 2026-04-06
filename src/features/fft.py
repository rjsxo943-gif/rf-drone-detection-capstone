import numpy as np


def _get_window(window: str, frame_size: int) -> np.ndarray:
    if window == "hann":
        return np.hanning(frame_size).astype(np.float32)
    if window == "hamming":
        return np.hamming(frame_size).astype(np.float32)
    return np.ones(frame_size, dtype=np.float32)


def compute_fft_magnitude(frames: np.ndarray, window: str = "hann") -> np.ndarray:
    if len(frames) == 0:
        return np.empty((0, 0), dtype=np.float32)

    win = _get_window(window, frames.shape[1])
    windowed = frames * win[None, :]
    spectrum = np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)
    return np.abs(spectrum).astype(np.float32)
