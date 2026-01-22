"""Custom UNet wrapper for 4-channel input to 3-channel output."""

import torch
import torch.nn as nn
from torchcfm.models.unet.unet import UNetModel


class UNet4to3(nn.Module):
    """UNet wrapper that accepts 4-channel input and produces 3-channel output.
    
    This is useful for conditional flow matching where you want to condition on
    an additional channel (e.g., a mask) concatenated with the RGB input, but
    output only RGB channels.
    """

    def __init__(
        self,
        image_size: int = 256,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 2, 4),
        use_scale_shift_norm: bool = True,
        num_heads: int = 4,
        num_head_channels: int = 32,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        """Initialize UNet4to3.
        
        Args:
            image_size: Size of input images (assumes square)
            model_channels: Base channel count for the model
            num_res_blocks: Number of residual blocks per downsample
            attention_resolutions: Tuple of downsample rates where attention is applied
            dropout: Dropout probability
            channel_mult: Channel multiplier for each level
            use_scale_shift_norm: Use FiLM-like conditioning
            num_heads: Number of attention heads
            num_head_channels: Channels per attention head
            use_checkpoint: Use gradient checkpointing
            use_fp16: Use FP16 precision
            resblock_updown: Use residual blocks for up/downsampling
            use_new_attention_order: Use different attention pattern
        """
        super().__init__()
        
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=4,  # 3 RGB + 1 condition channel
            model_channels=model_channels,
            out_channels=3,  # 3 RGB output
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_scale_shift_norm=use_scale_shift_norm,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, y=None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            t: Time tensor of shape (batch_size,) or (1,)
            x: Input tensor of shape (batch_size, 4, height, width)
            y: Optional class labels (not used in this version)
            
        Returns:
            Output tensor of shape (batch_size, 3, height, width)
        """
        # UNetModel expects (t, x, y)
        return self.unet(t, x, y=y)
