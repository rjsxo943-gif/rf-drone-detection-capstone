"""
Receiver package for RF IQ input sources.

현재 프로젝트 기준:
- 처리 단위: block
- 1 block = 16,384 IQ samples
- Receiver 출력 shape: (num_channels, num_samples)

지원 입력:
- SimReceiver: synthetic IQ 생성
- RawFileReceiver: 저장된 .npy IQ 파일 읽기
- PlutoReceiver: Pluto+ SDR 실제 수신
"""

from src.receiver.base import BaseReceiver
from src.receiver.factory import build_receiver
from src.receiver.sim_receiver import SimReceiver
from src.receiver.raw_file_receiver import RawFileReceiver
from src.receiver.pluto_receiver import PlutoReceiver

__all__ = [
    "BaseReceiver",
    "build_receiver",
    "SimReceiver",
    "RawFileReceiver",
    "PlutoReceiver",
]