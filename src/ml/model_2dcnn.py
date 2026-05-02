from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


DEFAULT_CLASS_NAMES = [
    "Background",
    "WiFi",
    "Bluetooth",
    "Drone-like",
]


@dataclass
class CNNOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_index: torch.Tensor


class ConvBNAct(nn.Module):
    """
    Conv2D → BatchNorm → ReLU 블록
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    연산량을 줄이기 위한 Depthwise Separable Convolution 블록.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNN(nn.Module):
    """
    원본 STFT spectrogram을 그대로 사용하는 2D CNN.

    입력:
        x.shape = (B, 1, 512, 125)

    출력:
        logits.shape = (B, 4)

    클래스:
        Background / WiFi / Bluetooth / Drone-like
    """

    def __init__(
        self,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)

        self.features = nn.Sequential(
            ConvBNAct(1, 16),

            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=2),
            DepthwiseSeparableConv(96, 128, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B, 1, H, W)
            현재 프로젝트 기본값: (B, 1, 512, 125)
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 1, H, W), got {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"Expected 1 input channel, got {x.shape[1]}"
            )

        x = self.features(x)
        logits = self.classifier(x)

        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> CNNOutput:
        self.eval()

        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1)

        return CNNOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_index=predicted_index,
        )


def build_model(
    num_classes: int = 4,
) -> SpectrogramCNN:
    return SpectrogramCNN(
        num_classes=num_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)