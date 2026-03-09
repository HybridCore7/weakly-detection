"""
Unit tests for model architectures.

Tests model initialization, forward pass shapes, and key features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn


class TestXceptionNet:
    """Tests for XceptionNet model."""

    def test_initialization(self):
        from src.models.xception_net import XceptionNet
        model = XceptionNet(num_classes=2, pretrained=False, use_attention=True)
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from src.models.xception_net import XceptionNet
        model = XceptionNet(num_classes=2, pretrained=False, use_attention=False)
        model.eval()
        x = torch.randn(2, 3, 299, 299)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_feature_extraction(self):
        from src.models.xception_net import XceptionNet
        model = XceptionNet(num_classes=2, pretrained=False, use_attention=False)
        model.eval()
        x = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            features = model.extract_features(x)
        assert features.dim() == 4
        assert features.shape[1] == 2048

    def test_get_last_conv_layer(self):
        from src.models.xception_net import XceptionNet
        model = XceptionNet(num_classes=2, pretrained=False)
        layer = model.get_last_conv_layer()
        assert layer is not None


class TestEfficientNet:
    """Tests for EfficientNet detector."""

    def test_initialization(self):
        from src.models.efficient_net import EfficientNetDetector
        model = EfficientNetDetector(
            num_classes=2, pretrained=False, use_attention=False
        )
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from src.models.efficient_net import EfficientNetDetector
        model = EfficientNetDetector(
            num_classes=2, pretrained=False, use_attention=False
        )
        model.eval()
        x = torch.randn(2, 3, 380, 380)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)


class TestAutoencoder:
    """Tests for Autoencoder detector."""

    def test_initialization(self):
        from src.models.autoencoder import DeepfakeAutoencoder
        model = DeepfakeAutoencoder(latent_dim=256, num_classes=2)
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from src.models.autoencoder import DeepfakeAutoencoder
        model = DeepfakeAutoencoder(latent_dim=256, num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_reconstruction(self):
        from src.models.autoencoder import DeepfakeAutoencoder
        model = DeepfakeAutoencoder(latent_dim=256, num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out, extras = model(x, return_reconstruction=True)
        assert 'reconstruction' in extras
        assert 'mu' in extras
        assert 'logvar' in extras
        assert extras['mu'].shape == (2, 256)

    def test_reparameterize(self):
        from src.models.autoencoder import DeepfakeAutoencoder
        model = DeepfakeAutoencoder(latent_dim=128)
        mu = torch.zeros(4, 128)
        logvar = torch.zeros(4, 128)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (4, 128)


class TestContrastive:
    """Tests for Contrastive Learning detector."""

    def test_classification_mode(self):
        from src.models.contrastive import ContrastiveDetector
        model = ContrastiveDetector(
            embedding_dim=256, projection_dim=64, num_classes=2
        )
        model.eval()
        model.set_mode('classify')
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_contrastive_mode(self):
        from src.models.contrastive import ContrastiveDetector
        model = ContrastiveDetector(
            embedding_dim=256, projection_dim=64, num_classes=2
        )
        model.train()
        model.set_mode('contrastive')
        x = torch.randn(4, 3, 256, 256)
        x_aug = torch.randn(4, 3, 256, 256)
        labels = torch.tensor([0, 1, 0, 1])
        loss = model(x, x_aug, labels)
        assert loss.dim() == 0  # Scalar loss

    def test_embeddings(self):
        from src.models.contrastive import ContrastiveDetector
        model = ContrastiveDetector(embedding_dim=256)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            emb = model.get_embeddings(x)
        assert emb.shape == (2, 256)

    def test_freeze_unfreeze(self):
        from src.models.contrastive import ContrastiveDetector
        model = ContrastiveDetector(embedding_dim=256)
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad


class TestAttention:
    """Tests for attention modules."""

    def test_cbam(self):
        from src.models.attention_module import CBAM
        cbam = CBAM(64)
        x = torch.randn(2, 64, 32, 32)
        out = cbam(x)
        assert out.shape == x.shape

    def test_channel_attention(self):
        from src.models.attention_module import ChannelAttention
        ca = ChannelAttention(64)
        x = torch.randn(2, 64, 16, 16)
        out = ca(x)
        assert out.shape == x.shape

    def test_spatial_attention(self):
        from src.models.attention_module import SpatialAttention
        sa = SpatialAttention()
        x = torch.randn(2, 64, 16, 16)
        out = sa(x)
        assert out.shape == x.shape

    def test_multi_head_attention(self):
        from src.models.attention_module import MultiHeadSelfAttention
        mhsa = MultiHeadSelfAttention(64, num_heads=8)
        x = torch.randn(2, 64, 8, 8)
        out = mhsa(x)
        assert out.shape == x.shape


class TestLosses:
    """Tests for loss functions."""

    def test_focal_loss(self):
        from src.training.losses import FocalLoss
        loss_fn = FocalLoss()
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_combined_loss(self):
        from src.training.losses import CombinedLoss
        loss_fn = CombinedLoss()
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_metrics_calculator(self):
        from src.training.metrics import MetricsCalculator
        calc = MetricsCalculator()

        preds = torch.tensor([0, 1, 1, 0, 1])
        labels = torch.tensor([0, 1, 0, 0, 1])
        probs = torch.tensor([
            [0.9, 0.1], [0.2, 0.8], [0.4, 0.6],
            [0.8, 0.2], [0.1, 0.9]
        ])

        calc.update(preds, labels, probs)
        results = calc.compute_all()

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 0 <= results['accuracy'] <= 1

    def test_accuracy(self):
        from src.training.metrics import MetricsCalculator
        calc = MetricsCalculator()
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 1, 0])
        calc.update(preds, labels)
        assert calc.accuracy() == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
