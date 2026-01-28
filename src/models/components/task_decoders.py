"""Decoder heads for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


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


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

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


class FlowMatchingDecoder(nn.Module):
    """Decoder for flow matching (Head A: Virtual staining).
    
    Takes bottleneck features F and time t, outputs velocity field for IHC generation.
    
    Args:
        bottleneck_channels: Number of channels in bottleneck features
        features: List of feature channels at each decoder level
        out_channels: Number of output channels (3 for RGB)
        time_emb_dim: Dimension of time embeddings
        bilinear: Whether to use bilinear upsampling
    """

    def __init__(
        self,
        bottleneck_channels: int = 1024,
        features: List[int] = None,
        out_channels: int = 3,
        time_emb_dim: int = 256,
        bilinear: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [512, 256, 128, 64]

        self.bottleneck_channels = bottleneck_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Project time embedding to bottleneck channels for conditioning
        self.time_proj = nn.Linear(time_emb_dim, bottleneck_channels)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        in_ch = bottleneck_channels
        for feat_ch in features:
            # Each Up block expects (in_channels + skip_channels) as input
            self.ups.append(Up(in_ch + feat_ch, feat_ch, bilinear))
            in_ch = feat_ch
        
        # Final convolution
        self.outc = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_connections: List[torch.Tensor],
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features from encoder (B, C, H, W)
            skip_connections: List of skip connection features
            t_emb: Time embeddings (B, time_emb_dim)
            
        Returns:
            Velocity field (B, 3, H_orig, W_orig)
        """
        # Process time embedding
        t = self.time_mlp(t_emb)  # (B, time_emb_dim)
        t = self.time_proj(t)  # (B, bottleneck_channels)
        
        # Add time conditioning to bottleneck
        # Reshape t for broadcasting: (B, C, 1, 1)
        t = t.view(t.shape[0], t.shape[1], 1, 1)
        x = bottleneck + t
        
        # Upsampling path with skip connections
        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)
        
        # Final output
        velocity = self.outc(x)
        
        return velocity


class SegmentationDecoder(nn.Module):
    """Decoder for segmentation (Head B: Amyloid mask prediction).
    
    Takes bottleneck features F, outputs segmentation mask.
    
    Args:
        bottleneck_channels: Number of channels in bottleneck features
        features: List of feature channels at each decoder level
        out_channels: Number of output channels (1 for binary segmentation)
        bilinear: Whether to use bilinear upsampling
    """

    def __init__(
        self,
        bottleneck_channels: int = 1024,
        features: List[int] = None,
        out_channels: int = 1,
        bilinear: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [512, 256, 128, 64]

        # Upsampling path
        self.ups = nn.ModuleList()
        in_ch = bottleneck_channels
        for feat_ch in features:
            # Each Up block expects (in_channels + skip_channels) as input
            self.ups.append(Up(in_ch + feat_ch, feat_ch, bilinear))
            in_ch = feat_ch
        
        # Final convolution
        self.outc = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_connections: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bottleneck: Bottleneck features from encoder (B, C, H, W)
            skip_connections: List of skip connection features
            
        Returns:
            Mask logits (B, 1, H_orig, W_orig)
        """
        x = bottleneck
        
        # Upsampling path with skip connections
        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)
        
        # Final output (logits)
        mask_logits = self.outc(x)
        
        return mask_logits
