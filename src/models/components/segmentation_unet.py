"""Simple UNet architecture for segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (conv => BN => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """Simple UNet for binary segmentation.
    
    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        out_channels: Number of output channels (e.g., 1 for binary segmentation)
        features: List of feature channels at each level
        bilinear: Whether to use bilinear upsampling (True) or transposed convolutions (False)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = None,
        bilinear: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)

        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
