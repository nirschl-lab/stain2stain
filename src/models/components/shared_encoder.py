"""Shared encoder backbone for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


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


class SharedEncoder(nn.Module):
    """Shared encoder backbone that extracts features from H&E images.
    
    This encoder is shared between the flow matching decoder (Head A) 
    and the segmentation decoder (Head B). It learns features that are 
    useful for both virtual staining and amyloid segmentation.
    
    Args:
        in_channels: Number of input channels (3 for RGB H&E)
        features: List of feature channels at each level
        return_skip_connections: If True, returns skip connections for UNet-style decoders
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: List[int] = None,
        return_skip_connections: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.in_channels = in_channels
        self.features = features
        self.return_skip_connections = return_skip_connections

        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Downsampling path
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through encoder.
        
        Args:
            x: Input H&E image (B, 3, H, W)
            
        Returns:
            - bottleneck: Bottleneck features (B, features[-1], H/2^n, W/2^n)
            - skip_connections: List of skip connection features for each level
        """
        skip_connections = []
        
        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # The last skip connection is the bottleneck
        bottleneck = skip_connections[-1]
        
        # Return skip connections in reverse order (excluding bottleneck)
        # for easier use in decoders
        if self.return_skip_connections:
            return bottleneck, skip_connections[:-1][::-1]
        else:
            return bottleneck, []


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal time embeddings.
        
        Args:
            t: Time values (B,) or (B, 1)
            
        Returns:
            Time embeddings (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Ensure t is 2D: (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings
