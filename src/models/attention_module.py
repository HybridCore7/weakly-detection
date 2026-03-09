"""
Attention Modules — Spatial, Channel, and CBAM attention for deepfake detection.

Attention mechanisms help the model focus on manipulated regions of the face,
such as blending boundaries, texture inconsistencies, and color artifacts.

Reference: Woo, S. et al. (2018). CBAM: Convolutional Block Attention Module. ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Learns inter-channel dependencies to emphasize informative feature channels.
    Uses both global average pooling and global max pooling for richer statistics.

    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio (default: 16)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)

        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        max_out = self.mlp(max_pool)

        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Learns WHERE to focus by generating a spatial attention map from
    channel-wise statistics. Particularly useful for localizing manipulated
    regions in deepfake images.

    Args:
        kernel_size (int): Convolution kernel size (default: 7)
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and generate spatial attention
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies channel attention followed by spatial attention.
    This dual attention mechanism helps the model focus on both WHAT (channels)
    and WHERE (spatial locations) are important for deepfake detection.

    Args:
        channels (int): Number of input channels
        reduction (int): Channel attention reduction ratio (default: 16)
        spatial_kernel (int): Spatial attention kernel size (default: 7)
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for feature maps.

    Captures long-range dependencies in the feature map, useful for detecting
    global inconsistencies in deepfake images.

    Args:
        channels (int): Number of input channels
        num_heads (int): Number of attention heads (default: 8)
        qkv_bias (bool): Whether QKV projection has bias (default: False)
    """

    def __init__(self, channels, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv.unbind(1)  # Each: (B, heads, head_dim, N)

        # Scaled dot-product attention
        attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = (v @ attn.transpose(-2, -1))  # (B, heads, head_dim, N)
        out = out.reshape(b, c, h, w)

        return self.proj(out) + x  # Residual connection


class FrequencyAttention(nn.Module):
    """
    Frequency-domain Attention Module.

    Analyzes frequency components of the feature map to detect GAN artifacts,
    which often leave characteristic patterns in the frequency domain.

    Args:
        channels (int): Number of input channels
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Compute 2D FFT
        freq = torch.fft.fft2(x.float())
        freq_magnitude = torch.abs(freq)

        # Generate attention from frequency magnitudes
        attention = self.conv(freq_magnitude)

        return x * attention


if __name__ == "__main__":
    # Test attention modules
    x = torch.randn(2, 64, 32, 32)

    cbam = CBAM(64)
    out = cbam(x)
    print(f"CBAM: {x.shape} -> {out.shape}")

    mhsa = MultiHeadSelfAttention(64, num_heads=8)
    out = mhsa(x)
    print(f"MHSA: {x.shape} -> {out.shape}")

    freq_attn = FrequencyAttention(64)
    out = freq_attn(x)
    print(f"FreqAttn: {x.shape} -> {out.shape}")
