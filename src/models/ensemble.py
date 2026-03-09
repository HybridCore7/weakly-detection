"""
Ensemble Detector — Multi-model fusion for robust deepfake detection.

Combines predictions from XceptionNet, EfficientNet, Autoencoder, and
Contrastive Learning models using learned attention-based fusion.

The ensemble consistently outperforms individual models by leveraging
complementary strengths of different architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xception_net import XceptionNet
from .efficient_net import EfficientNetDetector
from .autoencoder import DeepfakeAutoencoder
from .contrastive import ContrastiveDetector


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module.

    Learns to dynamically weight predictions from different models
    based on the input, allowing the ensemble to adaptively rely
    on the most confident model for each sample.
    """

    def __init__(self, num_models, num_classes, hidden_dim=64):
        super().__init__()

        # Each model's logits are projected
        self.model_projections = nn.ModuleList([
            nn.Linear(num_classes, hidden_dim) for _ in range(num_models)
        ])

        # Attention weights generation
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1),
        )

        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(num_classes * num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.num_models = num_models
        self.num_classes = num_classes

    def forward(self, logits_list):
        """
        Fuse predictions from multiple models.

        Args:
            logits_list: List of tensors, each (B, num_classes)

        Returns:
            fused_logits: Fused prediction of shape (B, num_classes)
            attention_weights: Per-model weights of shape (B, num_models)
        """
        batch_size = logits_list[0].shape[0]

        # Project each model's logits
        projected = [
            proj(logits) for proj, logits in
            zip(self.model_projections, logits_list)
        ]

        # Concatenate projections for attention
        concat_proj = torch.cat(projected, dim=-1)
        attention_weights = self.attention_net(concat_proj)

        # Weighted sum of logits
        stacked = torch.stack(logits_list, dim=1)  # (B, M, C)
        weights = attention_weights.unsqueeze(-1)   # (B, M, 1)
        weighted = (stacked * weights).sum(dim=1)   # (B, C)

        # Also pass through final classifier for refinement
        all_logits = torch.cat(logits_list, dim=-1)
        refined = self.final_classifier(all_logits)

        # Combine weighted sum and refined
        fused = 0.6 * weighted + 0.4 * refined

        return fused, attention_weights


class EnsembleDetector(nn.Module):
    """
    Multi-Model Ensemble Deepfake Detector.

    Combines four complementary architectures:
    1. XceptionNet — Excellent at detecting GAN artifacts via separable convolutions
    2. EfficientNet-B4 — Efficient feature extraction with compound scaling
    3. Autoencoder — Reconstruction-based anomaly detection
    4. Contrastive — Discriminative embeddings via contrastive learning

    Fusion strategies:
    - 'attention': Learned attention-based fusion (recommended)
    - 'weighted_average': Fixed weighted average
    - 'mlp': MLP-based fusion

    Args:
        num_classes (int): Number of output classes (default: 2)
        dropout (float): Dropout rate for individual models (default: 0.5)
        pretrained (bool): Use pretrained weights for backbones (default: True)
        use_attention (bool): Use attention in individual models (default: True)
        fusion (str): Fusion strategy (default: 'attention')
        model_weights (list): Weights for weighted_average fusion
        models_to_use (list): Which models to include in ensemble
    """

    def __init__(self, num_classes=2, dropout=0.5, pretrained=True,
                 use_attention=True, fusion='attention',
                 model_weights=None,
                 models_to_use=None):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_type = fusion

        # Default models and weights
        if models_to_use is None:
            models_to_use = ['xception', 'efficientnet', 'autoencoder',
                             'contrastive']
        if model_weights is None:
            model_weights = [0.35, 0.30, 0.15, 0.20]

        self.model_names = models_to_use
        self.model_weights = model_weights
        self.models = nn.ModuleDict()

        # Initialize selected models
        if 'xception' in models_to_use:
            self.models['xception'] = XceptionNet(
                num_classes=num_classes, dropout=dropout,
                use_attention=use_attention, pretrained=pretrained
            )

        if 'efficientnet' in models_to_use:
            self.models['efficientnet'] = EfficientNetDetector(
                num_classes=num_classes, dropout=dropout,
                use_attention=use_attention, pretrained=pretrained
            )

        if 'autoencoder' in models_to_use:
            self.models['autoencoder'] = DeepfakeAutoencoder(
                latent_dim=512, num_classes=num_classes
            )

        if 'contrastive' in models_to_use:
            self.models['contrastive'] = ContrastiveDetector(
                embedding_dim=512, projection_dim=128,
                num_classes=num_classes
            )

        num_models = len(self.models)

        # Fusion module
        if fusion == 'attention':
            self.fusion = AttentionFusion(num_models, num_classes)
        elif fusion == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * num_models, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes),
            )
        else:
            self.fusion = None  # weighted_average

        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, return_individual=False):
        """
        Forward pass through all models and fuse predictions.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_individual: Whether to return individual model predictions

        Returns:
            logits: Fused classification logits (B, num_classes)
            info: Dict with individual predictions and attention weights
        """
        individual_logits = {}

        for name, model in self.models.items():
            if name == 'contrastive':
                model.set_mode('classify')

            # Handle different input sizes
            if name == 'autoencoder':
                # Autoencoder expects 256x256
                x_resized = F.interpolate(x, size=(256, 256),
                                          mode='bilinear',
                                          align_corners=False)
                logits = model(x_resized)
            elif name == 'efficientnet':
                # EfficientNet-B4 prefers 380x380
                x_resized = F.interpolate(x, size=(380, 380),
                                          mode='bilinear',
                                          align_corners=False)
                logits = model(x_resized)
            elif name == 'contrastive':
                # Contrastive uses 256x256
                x_resized = F.interpolate(x, size=(256, 256),
                                          mode='bilinear',
                                          align_corners=False)
                logits = model(x_resized)
            else:
                logits = model(x)

            individual_logits[name] = logits

        # Fuse predictions
        logits_list = [individual_logits[name]
                       for name in self.model_names
                       if name in individual_logits]

        if self.fusion_type == 'attention':
            fused_logits, attention_weights = self.fusion(logits_list)
        elif self.fusion_type == 'mlp':
            concatenated = torch.cat(logits_list, dim=-1)
            fused_logits = self.fusion(concatenated)
            attention_weights = None
        else:
            # Weighted average
            stacked = torch.stack(logits_list, dim=0)
            weights = torch.tensor(
                self.model_weights[:len(logits_list)],
                device=x.device
            ).view(-1, 1, 1)
            fused_logits = (stacked * weights).sum(dim=0)
            attention_weights = None

        # Temperature scaling
        fused_logits = fused_logits / self.temperature

        info = {
            'individual_logits': individual_logits,
            'attention_weights': attention_weights,
        }

        if return_individual:
            return fused_logits, info

        return fused_logits

    def get_model_contributions(self, x):
        """
        Get each model's contribution to the final prediction.
        Useful for interpretability and debugging.
        """
        _, info = self.forward(x, return_individual=True)
        contributions = {}

        for name, logits in info['individual_logits'].items():
            probs = F.softmax(logits, dim=-1)
            contributions[name] = {
                'logits': logits,
                'probabilities': probs,
                'prediction': probs.argmax(dim=-1),
                'confidence': probs.max(dim=-1).values,
            }

        return contributions


if __name__ == "__main__":
    # Test with subset to avoid memory issues
    model = EnsembleDetector(
        num_classes=2,
        pretrained=False,
        models_to_use=['xception', 'contrastive'],
        model_weights=[0.5, 0.5],
        fusion='attention'
    )

    x = torch.randn(2, 3, 299, 299)
    logits, info = model(x, return_individual=True)
    print(f"Ensemble output shape: {logits.shape}")
    print(f"Individual models: {list(info['individual_logits'].keys())}")

    if info['attention_weights'] is not None:
        print(f"Attention weights: {info['attention_weights']}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total ensemble parameters: {total_params:,}")
