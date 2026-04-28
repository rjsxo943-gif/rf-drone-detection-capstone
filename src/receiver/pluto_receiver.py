from __future__ import annotations

from typing import Sequence

import numpy as np

from src.receiver.base import BaseReceiver


class PlutoReceiver(BaseReceiver):
    """
    Pluto+ SDR Receiver.

    현재 프로젝트 기준:
    - Pluto+ RX0/RX1 사용 가능
    - 처리 단위는 block
    - 1 block = 16,384 samples
    - 출력 shape은 항상 (num_channels, num_samples)
    - RX0/RX1 2채널이면 shape = (2, 16384)
    """

    def __init__(
        self,
        uri: str = "ip:192.168.2.1",
        sample_rate: int = 5_000_000,
        center_freq: int = 2_400_000_000,
        num_channels: int = 2,
        channels: Sequence[int] | None = None,
        gain_control_mode: str = "manual",
        gain: int | float = 20,
        block_size: int = 16_384,
        num_samples: int | None = None,
        rf_bandwidth: int | None = None,
        warmup_reads: int = 1,
    ) -> None:
        if channels is None:
            channels = [0, 1] if num_channels == 2 else [0]

        self.uri = uri
        self.channels = list(channels)
        self.block_size = int(block_size)
        self.rx_buffer_size = int(num_samples or block_size)
        self.gain_control_mode = gain_control_mode
        self.gain = float(gain)
        self.rf_bandwidth = int(rf_bandwidth or sample_rate)
        self.warmup_reads = int(warmup_reads)

        super().__init__(
            sample_rate=int(sample_rate),
            center_freq=int(center_freq),
            num_channels=len(self.channels),
        )

        self.sdr = self._create_sdr()
        self._configure_sdr()
        self._warmup()

    def _create_sdr(self):
        """
        pyadi-iio SDR 객체를 생성한다.

        Pluto+는 AD9361 기반 2채널 수신을 써야 하므로
        우선 adi.ad9361(uri=...)로 연결한다.
        """
        try:
            import adi
        except ImportError as exc:
            raise ImportError(
                "pyadi-iio가 설치되어 있지 않습니다. "
                "다음 명령어로 설치하세요: pip install pyadi-iio"
            ) from exc

        try:
            return adi.ad9361(uri=self.uri)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Pluto+/AD9361 at uri={self.uri}. "
                "Pluto+ 연결, IP 주소, iio_info 동작 여부를 확인하세요."
            ) from exc

    def _configure_sdr(self) -> None:
        """
        Pluto+ 수신 파라미터를 설정한다.
        """
        self.sdr.sample_rate = int(self.sample_rate)
        self.sdr.rx_lo = int(self.center_freq)
        self.sdr.rx_rf_bandwidth = int(self.rf_bandwidth)
        self.sdr.rx_buffer_size = int(self.rx_buffer_size)

        # 사용할 RX 채널 설정
        self.sdr.rx_enabled_channels = self.channels

        # Gain 설정
        for ch in self.channels:
            self._set_channel_gain(ch)

    def _set_channel_gain(self, ch: int) -> None:
        """
        채널별 gain mode / gain 값을 설정한다.

        pyadi-iio에서는 보통 아래 속성을 사용한다.
        - gain_control_mode_chan0
        - gain_control_mode_chan1
        - rx_hardwaregain_chan0
        - rx_hardwaregain_chan1
        """
        gain_mode_attr = f"gain_control_mode_chan{ch}"
        gain_attr = f"rx_hardwaregain_chan{ch}"

        if hasattr(self.sdr, gain_mode_attr):
            setattr(self.sdr, gain_mode_attr, self.gain_control_mode)

        if hasattr(self.sdr, gain_attr):
            setattr(self.sdr, gain_attr, self.gain)

    def _warmup(self) -> None:
        """
        초기 버퍼 안정화를 위해 몇 번 읽고 버린다.
        """
        for _ in range(max(0, self.warmup_reads)):
            try:
                _ = self.sdr.rx()
            except Exception:
                # warmup 실패는 실제 read에서 다시 잡는다.
                pass

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Pluto+에서 num_samples만큼 IQ sample을 읽는다.

        반환:
            np.ndarray
            shape = (num_channels, num_samples)
            dtype = complex64
        """
        num_samples = int(num_samples)

        if num_samples != self.rx_buffer_size:
            self.rx_buffer_size = num_samples
            self.sdr.rx_buffer_size = num_samples

        try:
            raw = self.sdr.rx()
        except Exception as exc:
            raise RuntimeError(
                "Failed to read IQ samples from Pluto+. "
                "Pluto+ 연결 상태, sample_rate, center_freq, gain 설정을 확인하세요."
            ) from exc

        samples = self._normalize_rx_output(raw)

        return self.validate_samples(samples, expected_samples=num_samples)

    def read_block(self, block_size: int | None = None) -> np.ndarray:
        """
        block 단위로 IQ sample을 읽는다.

        기본:
            block_size = 16,384
        """
        if block_size is None:
            block_size = self.block_size

        return self.read_samples(int(block_size))

    def _normalize_rx_output(self, raw) -> np.ndarray:
        """
        pyadi-iio의 rx() 출력을 (num_channels, num_samples)로 통일한다.

        pyadi-iio 동작:
        - 1채널 수신: np.ndarray
        - 2채널 이상 수신: list[np.ndarray]
        """
        if isinstance(raw, list):
            channels = [np.asarray(ch, dtype=np.complex64) for ch in raw]
            samples = np.stack(channels, axis=0)

        else:
            samples = np.asarray(raw, dtype=np.complex64)

            if samples.ndim == 1:
                samples = samples[np.newaxis, :]

        return samples.astype(np.complex64, copy=False)

    def close(self) -> None:
        """
        Pluto+ RX buffer 정리.
        """
        if hasattr(self, "sdr") and self.sdr is not None:
            if hasattr(self.sdr, "rx_destroy_buffer"):
                try:
                    self.sdr.rx_destroy_buffer()
                except Exception:
                    pass

            self.sdr = None