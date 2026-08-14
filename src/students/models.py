import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """~1-5M param CNN, trained from scratch, image -> R^k embedding regression."""

    def __init__(self, k: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.head(feat)


def build_student(arch: str, k: int) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN(k=k)
    if arch in ("mobilenetv3_small", "efficientnet_lite0"):
        raise NotImplementedError(
            f"student arch '{arch}' is deferred until pilot with small_cnn is confirmed"
        )
    raise ValueError(f"unknown student arch: {arch}")
