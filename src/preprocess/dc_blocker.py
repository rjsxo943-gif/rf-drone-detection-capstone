import numpy as np


def remove_dc(iq: np.ndarray) -> np.ndarray:
    # DC offset 제거: 중심값을 0 근처로 맞춤
    return (iq - np.mean(iq)).astype(np.complex64)
