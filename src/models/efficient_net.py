"""
EfficientNet-B4 Detector — Leverages compound scaling for deepfake detection.

EfficientNet uses a principled compound scaling method that uniformly scales
width, depth, and resolution. The B4 variant provides an excellent balance
of accuracy and computational efficiency for deepfake detection.

Reference: Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks. ICML 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .attention_module import CBAM, FrequencyAttention


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = int(in_channels * expand_ratio)
        padding = kernel_size // 2

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride,
                      padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-excitation
        self.se = SqueezeExcitation(mid_channels,
                                     reduction=int(1 / se_ratio))

        # Pointwise projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)

        if self.use_residual:
            out = out + identity

        return out


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 based deepfake detector.

    Uses timm's pretrained EfficientNet-B4 as backbone when available,
    with custom attention heads for deepfake-specific feature enhancement.

    Architecture:
        EfficientNet-B4 backbone → CBAM Attention → Frequency Attention →
        Global Pooling → Dropout → Classifier

    Args:
        num_classes (int): Number of output classes (default: 2)
        dropout (float): Dropout rate (default: 0.5)
        use_attention (bool): Whether to use attention modules (default: True)
        pretrained (bool): Use pretrained ImageNet weights (default: True)
    """

    def __init__(self, num_classes=2, dropout=0.5, use_attention=True,
                 pretrained=True):
        super().__init__()

        self.use_attention = use_attention

        if HAS_TIMM:
            # Use timm's pretrained EfficientNet-B4
            self.backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier head
                global_pool='',  # Remove global pooling
            )
            feature_dim = self.backbone.num_features  # 1792 for B4
        else:
            # Fallback: build a simplified EfficientNet-like architecture
            self.backbone = self._build_fallback_backbone()
            feature_dim = 1792

        # Attention modules
        if use_attention:
            self.cbam = CBAM(feature_dim, reduction=16)
            self.freq_attention = FrequencyAttention(feature_dim)
        else:
            self.cbam = None
            self.freq_attention = None

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        self.feature_dim = feature_dim

    def _build_fallback_backbone(self):
        """Build simplified EfficientNet-like backbone when timm unavailable."""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),

            # Stage 2
            MBConvBlock(48, 24, kernel_size=3, stride=1, expand_ratio=1),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=1),

            # Stage 3
            MBConvBlock(24, 32, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(32, 32, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(32, 32, kernel_size=3, stride=1, expand_ratio=6),

            # Stage 4
            MBConvBlock(32, 56, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(56, 56, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(56, 56, kernel_size=5, stride=1, expand_ratio=6),

            # Stage 5
            MBConvBlock(56, 112, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=3, stride=1, expand_ratio=6),

            # Stage 6
            MBConvBlock(112, 160, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(160, 160, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(160, 160, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(160, 160, kernel_size=5, stride=1, expand_ratio=6),

            # Stage 7
            MBConvBlock(160, 272, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(272, 272, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(272, 272, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(272, 272, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(272, 272, kernel_size=5, stride=1, expand_ratio=6),

            # Stage 8
            MBConvBlock(272, 448, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(448, 448, kernel_size=3, stride=1, expand_ratio=6),

            # Head
            nn.Conv2d(448, 1792, 1, bias=False),
            nn.BatchNorm2d(1792),
            nn.SiLU(inplace=True),
        )

    def extract_features(self, x):
        """Extract feature maps before classification."""
        features = self.backbone(x)

        if self.cbam is not None:
            features = self.cbam(features)

        if self.freq_attention is not None:
            features = self.freq_attention(features)

        return features

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        features = self.extract_features(x)
        pooled = self.global_pool(features).flatten(1)
        logits = self.classifier(pooled)
        return logits

    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        if HAS_TIMM:
            # Get the last conv layer from timm backbone
            return list(self.backbone.children())[-1]
        else:
            return list(self.backbone.children())[-2]


if __name__ == "__main__":
    model = EfficientNetDetector(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 380, 380)  # EfficientNet-B4 default size
    out = model(x)
    print(f"EfficientNet output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
