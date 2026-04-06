import numpy as np


def normalize_iq(iq: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # 최대 진폭 기준 정규화
    peak = np.max(np.abs(iq))
    if peak < eps:
        return iq.astype(np.complex64)
    return (iq / peak).astype(np.complex64)
