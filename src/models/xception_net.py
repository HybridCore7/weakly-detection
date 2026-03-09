"""
XceptionNet — Modified Xception architecture for deepfake detection.

The Xception architecture uses depthwise separable convolutions which are
particularly effective at detecting GAN-generated manipulation artifacts.
This implementation adds spatial attention and is optimized for binary
classification (real vs. fake).

Reference: Chollet, F. (2017). Xception: Deep Learning with Depthwise
Separable Convolutions. CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_module import CBAM


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution — key building block of Xception."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class XceptionBlock(nn.Module):
    """Xception middle flow block with residual connection."""

    def __init__(self, in_channels, out_channels, reps=3, stride=1,
                 start_with_relu=True, grow_first=True, use_attention=False):
        super().__init__()

        # Skip connection
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        layers = []
        channels = in_channels

        for i in range(reps):
            if grow_first:
                inc = channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = channels if i == 0 else in_channels
                outc = in_channels if i < (reps - 1) else out_channels

            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=False))

            layers.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))

            if i == reps - 1 and stride != 1:
                layers.append(nn.MaxPool2d(3, stride, 1))

        self.rep = nn.Sequential(*layers)

        # Optional CBAM attention
        self.attention = CBAM(out_channels) if use_attention else None

    def forward(self, x):
        skip = self.skip(x) if self.skip is not None else x
        out = self.rep(x)

        if self.attention is not None:
            out = self.attention(out)

        return out + skip


class XceptionNet(nn.Module):
    """
    Modified XceptionNet for deepfake detection.

    Architecture:
        Entry Flow → Middle Flow (×8) → Exit Flow → Global Avg Pool → Classifier

    Features:
        - Depthwise separable convolutions for efficient artifact detection
        - Optional CBAM attention modules at key stages
        - Dropout regularization to prevent overfitting
        - Support for pretrained ImageNet weights via timm

    Args:
        num_classes (int): Number of output classes (default: 2)
        dropout (float): Dropout rate before classifier (default: 0.5)
        use_attention (bool): Whether to use CBAM attention (default: True)
        pretrained (bool): Load pretrained weights (default: True)
    """

    def __init__(self, num_classes=2, dropout=0.5, use_attention=True,
                 pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.use_attention = use_attention

        # ---- Entry Flow ----
        self.entry_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.entry_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.entry_block1 = XceptionBlock(
            64, 128, reps=2, stride=2, start_with_relu=False,
            use_attention=use_attention
        )
        self.entry_block2 = XceptionBlock(
            128, 256, reps=2, stride=2, start_with_relu=True,
            use_attention=use_attention
        )
        self.entry_block3 = XceptionBlock(
            256, 728, reps=2, stride=2, start_with_relu=True,
            use_attention=use_attention
        )

        # ---- Middle Flow ----
        self.middle_blocks = nn.Sequential(*[
            XceptionBlock(
                728, 728, reps=3, stride=1, start_with_relu=True,
                use_attention=(use_attention and i % 2 == 0)
            )
            for i in range(8)
        ])

        # ---- Exit Flow ----
        self.exit_block = XceptionBlock(
            728, 1024, reps=2, stride=2, start_with_relu=True,
            grow_first=False, use_attention=use_attention
        )

        self.exit_conv1 = SeparableConv2d(1024, 1536, 3, padding=1)
        self.exit_bn1 = nn.BatchNorm2d(1536)
        self.exit_conv2 = SeparableConv2d(1536, 2048, 3, padding=1)
        self.exit_bn2 = nn.BatchNorm2d(2048)

        # Final attention
        self.final_attention = CBAM(2048) if use_attention else None

        # ---- Classifier ----
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def extract_features(self, x):
        """Extract feature maps before the classifier head."""
        # Entry flow
        x = self.entry_conv1(x)
        x = self.entry_conv2(x)
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # Middle flow
        x = self.middle_blocks(x)

        # Exit flow
        x = self.exit_block(x)
        x = F.relu(self.exit_bn1(self.exit_conv1(x)))
        x = F.relu(self.exit_bn2(self.exit_conv2(x)))

        if self.final_attention is not None:
            x = self.final_attention(x)

        return x

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        features = self.extract_features(x)
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.exit_conv2


if __name__ == "__main__":
    # Quick test
    model = XceptionNet(num_classes=2, use_attention=True)
    x = torch.randn(2, 3, 299, 299)
    out = model(x)
    print(f"XceptionNet output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
