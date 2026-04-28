from __future__ import annotations

from pathlib import Path

import numpy as np

from src.receiver.base import BaseReceiver


class RawFileReceiver(BaseReceiver):
    """
    저장된 IQ 파일(.npy)을 읽는 Receiver.

    현재 프로젝트 기준:
    - 처리 단위는 block
    - 1 block = 16,384 samples
    - 출력 shape은 항상 (num_channels, num_samples)
    - 파일 전체를 로드한 뒤 read_block()으로 block 단위 순차 읽기
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        filepath: str | Path | None = None,
        sample_rate: float = 5_000_000,
        center_freq: float = 2_400_000_000,
        num_channels: int = 1,
        block_size: int = 16_384,
    ) -> None:
        path = file_path if file_path is not None else filepath

        if path is None:
            raise ValueError("file_path is required for RawFileReceiver")

        self.file_path = Path(path)
        self.block_size = int(block_size)
        self.cursor = 0

        super().__init__(
            sample_rate=int(sample_rate),
            center_freq=int(center_freq),
            num_channels=int(num_channels),
        )

        self.samples = self._load_iq_file(self.file_path)
        self.samples = self.validate_samples(self.samples)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        파일에서 num_samples만큼 IQ sample을 읽는다.

        반환:
            shape = (num_channels, num_samples)
            dtype = complex64
        """
        num_samples = int(num_samples)

        start = self.cursor
        end = start + num_samples

        if end > self.samples.shape[1]:
            raise EOFError(
                f"Not enough samples in file. "
                f"Requested {num_samples}, cursor={self.cursor}, "
                f"available={self.samples.shape[1]}"
            )

        block = self.samples[:, start:end]
        self.cursor = end

        return self.validate_samples(block, expected_samples=num_samples)

    def read_block(self, block_size: int | None = None) -> np.ndarray:
        """
        block 단위로 IQ sample을 읽는다.
        """
        if block_size is None:
            block_size = self.block_size

        return self.read_samples(int(block_size))

    def reset(self) -> None:
        """
        파일 읽기 위치를 처음으로 되돌린다.
        """
        self.cursor = 0

    def num_available_samples(self) -> int:
        """
        현재 파일에 남아 있는 sample 수를 반환한다.
        """
        return int(self.samples.shape[1] - self.cursor)

    def num_total_samples(self) -> int:
        """
        파일 전체 sample 수를 반환한다.
        """
        return int(self.samples.shape[1])

    def num_blocks(self, drop_last: bool = True) -> int:
        """
        현재 파일에서 읽을 수 있는 block 개수를 반환한다.
        """
        total = self.samples.shape[1]

        if drop_last:
            return int(total // self.block_size)

        return int(np.ceil(total / self.block_size))

    def _load_iq_file(self, file_path: Path) -> np.ndarray:
        """
        .npy IQ 파일을 읽어서 (num_channels, num_samples) 형태로 정리한다.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"IQ file not found: {file_path}")

        raw = np.load(file_path)

        if raw.size == 0:
            raise ValueError(f"IQ file is empty: {file_path}")

        raw = self._convert_to_complex(raw)
        raw = self._normalize_shape(raw)

        return raw.astype(np.complex64, copy=False)

    def _convert_to_complex(self, raw: np.ndarray) -> np.ndarray:
        """
        입력 데이터를 complex IQ 형태로 변환한다.

        지원:
        - complex array: 그대로 사용
        - (..., 2) real array: 마지막 축을 I/Q로 보고 complex 변환
        """
        raw = np.asarray(raw)

        if np.iscomplexobj(raw):
            return raw

        if raw.ndim >= 2 and raw.shape[-1] == 2:
            return raw[..., 0] + 1j * raw[..., 1]

        raise TypeError(
            f"Unsupported IQ file dtype/shape. "
            f"Expected complex array or real array with last dimension=2, "
            f"got dtype={raw.dtype}, shape={raw.shape}"
        )

    def _normalize_shape(self, raw: np.ndarray) -> np.ndarray:
        """
        IQ 배열을 (num_channels, num_samples) 형태로 통일한다.
        """
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        elif raw.ndim == 2:
            if raw.shape[0] == self.num_channels:
                pass

            elif raw.shape[1] == self.num_channels:
                raw = raw.T

            else:
                raise ValueError(
                    f"Cannot interpret IQ shape {raw.shape} "
                    f"as {self.num_channels} channel(s)."
                )

        else:
            raise ValueError(
                f"IQ array must be 1D or 2D after complex conversion, "
                f"got shape {raw.shape}"
            )

        return raw