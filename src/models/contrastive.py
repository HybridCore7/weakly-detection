"""
Contrastive Learning Detector — SimCLR-style contrastive representation
learning for deepfake detection.

Learns discriminative embeddings where real images cluster together and
fake images cluster separately. Uses a projection head during training
and a classification head for inference.

Reference: Chen, T. et al. (2020). A Simple Framework for Contrastive
Learning of Visual Representations. ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveBackbone(nn.Module):
    """
    ResNet-style backbone for contrastive feature extraction.

    Uses residual connections and progressively increasing channel depth
    to extract robust visual features.
    """

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (3, 256, 256) -> (64, 128, 128)
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 2: (64, 64, 64) -> (128, 32, 32)
            self._make_layer(64, 128, num_blocks=2, stride=2),

            # Block 3: (128, 32, 32) -> (256, 16, 16)
            self._make_layer(128, 256, num_blocks=2, stride=2),

            # Block 4: (256, 16, 16) -> (512, 8, 8)
            self._make_layer(256, 512, num_blocks=2, stride=2),

            # Block 5: (512, 8, 8) -> (embedding_dim, 4, 4)
            self._make_layer(512, embedding_dim, num_blocks=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a residual layer with the given number of blocks."""
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return pooled, features


class ResBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Maps embeddings to a lower-dimensional space where the contrastive
    loss is applied. Discarded after training.
    """

    def __init__(self, embedding_dim=512, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    The core contrastive loss function from SimCLR. Encourages positive pairs
    (same image, different augmentations) to be close in embedding space while
    pushing negative pairs apart.

    Args:
        temperature (float): Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss.

        Args:
            z_i: Embeddings from augmentation 1, shape (B, D)
            z_j: Embeddings from augmentation 2, shape (B, D)

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = z_i.shape[0]

        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Cosine similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        sim = sim / self.temperature

        # Create mask for positive pairs
        # Positive pairs: (i, i+B) and (i+B, i)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)  # Mask self-similarity

        # Positive pair indices
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size,
                               device=z.device, dtype=torch.bool)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True

        # Compute log-softmax loss for positive pairs
        log_prob = F.log_softmax(sim, dim=-1)
        loss = -(log_prob * pos_mask.float()).sum(dim=-1)
        loss = loss[pos_mask.any(dim=-1)].mean()

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Extends NT-Xent by using label information: all samples with the same
    label are treated as positive pairs. This is more effective than
    self-supervised contrastive learning when labels are available.

    Args:
        temperature (float): Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Normalized embeddings, shape (B, D)
            labels: Class labels, shape (B,)

        Returns:
            loss: Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask self-similarity
        mask_self = torch.eye(batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask_self, -1e9)

        # Create positive pair mask (same label, different sample)
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        # Count positives per sample
        num_positives = pos_mask.sum(dim=-1)

        # Compute loss
        log_prob = F.log_softmax(sim, dim=-1)
        loss = -(log_prob * pos_mask).sum(dim=-1)

        # Avoid division by zero
        valid = num_positives > 0
        if valid.any():
            loss = loss[valid] / num_positives[valid]
            return loss.mean()
        return torch.tensor(0.0, device=device, requires_grad=True)


class ContrastiveDetector(nn.Module):
    """
    Contrastive Learning-based Deepfake Detector.

    Architecture:
        Backbone → Embedding → (Projection Head during training)
                              → Classification Head during inference

    Training modes:
    1. Contrastive pretraining: Learn discriminative embeddings
    2. Fine-tuning: Train classification head on frozen/unfrozen embeddings

    Args:
        embedding_dim (int): Backbone embedding dimension (default: 512)
        projection_dim (int): Contrastive projection dimension (default: 128)
        num_classes (int): Number of output classes (default: 2)
        temperature (float): Contrastive loss temperature (default: 0.07)
        dropout (float): Dropout rate (default: 0.3)
    """

    def __init__(self, embedding_dim=512, projection_dim=128,
                 num_classes=2, temperature=0.07, dropout=0.3):
        super().__init__()

        self.backbone = ContrastiveBackbone(embedding_dim)

        # Contrastive projection head (used during pretraining)
        self.projection_head = ProjectionHead(embedding_dim, projection_dim)

        # Classification head (used during fine-tuning/inference)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # Loss functions
        self.ntxent_loss = NTXentLoss(temperature)
        self.supervised_contrastive_loss = SupervisedContrastiveLoss(temperature)

        self.mode = 'classify'  # 'contrastive' or 'classify'

    def set_mode(self, mode):
        """Set model mode: 'contrastive' for pretraining, 'classify' for inference."""
        assert mode in ('contrastive', 'classify')
        self.mode = mode

    def forward(self, x, x_aug=None, labels=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            x_aug: Augmented view (only for contrastive mode)
            labels: Labels for supervised contrastive loss

        Returns:
            In contrastive mode: contrastive loss
            In classify mode: classification logits (B, num_classes)
        """
        embedding, features = self.backbone(x)

        if self.mode == 'contrastive' and x_aug is not None:
            # Contrastive pretraining
            z_i = self.projection_head(embedding)
            embedding_aug, _ = self.backbone(x_aug)
            z_j = self.projection_head(embedding_aug)

            if labels is not None:
                # Supervised contrastive
                z_combined = torch.cat([z_i, z_j], dim=0)
                labels_combined = torch.cat([labels, labels], dim=0)
                return self.supervised_contrastive_loss(z_combined,
                                                        labels_combined)
            else:
                # Self-supervised contrastive
                return self.ntxent_loss(z_i, z_j)

        # Classification mode
        logits = self.classifier(embedding)
        return logits

    def get_embeddings(self, x):
        """Extract embeddings for visualization (t-SNE, etc.)."""
        embedding, _ = self.backbone(x)
        return embedding

    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return list(self.backbone.features.children())[-1]

    def freeze_backbone(self):
        """Freeze backbone for linear evaluation."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = ContrastiveDetector(embedding_dim=512, projection_dim=128)

    # Test classification mode
    x = torch.randn(4, 3, 256, 256)
    model.set_mode('classify')
    logits = model(x)
    print(f"Classification logits: {logits.shape}")

    # Test contrastive mode
    x_aug = torch.randn(4, 3, 256, 256)
    labels = torch.tensor([0, 1, 0, 1])
    model.set_mode('contrastive')
    loss = model(x, x_aug, labels)
    print(f"Contrastive loss: {loss.item():.4f}")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
