import numpy as np


def frame_signal(iq: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(iq) < frame_size:
        return np.empty((0, frame_size), dtype=np.complex64)

    frames = []
    for start in range(0, len(iq) - frame_size + 1, hop_size):
        frames.append(iq[start:start + frame_size])

    return np.asarray(frames, dtype=np.complex64)
