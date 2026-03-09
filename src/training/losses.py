"""
Custom Loss Functions for Deepfake Detection.

Includes:
- Focal Loss: Handles class imbalance by focusing on hard examples
- Combined Loss: Weighted combination of multiple losses
- Label smoothing variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Reduces the relative loss for well-classified examples, putting more
    focus on hard, misclassified examples. Particularly effective for
    deepfake detection where there may be class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Balancing factor for positive/negative classes
        gamma (float): Focusing parameter (higher = more focus on hard examples)
        label_smoothing (float): Label smoothing factor
        reduction (str): Reduction method ('mean', 'sum', 'none')

    Reference: Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model logits of shape (B, C)
            targets: Ground truth labels of shape (B,)
        """
        num_classes = inputs.shape[1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()

        # Compute probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Compute focal weight
        focal_weight = (1 - probs) ** self.gamma

        # Apply alpha balancing
        if self.alpha is not None:
            alpha_weight = torch.where(
                targets.unsqueeze(1).expand_as(targets_smooth) == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1 - self.alpha, device=inputs.device),
            )
            # Simpler approach: use alpha for class 1 (fake)
            alpha_t = torch.ones_like(targets_smooth)
            alpha_t[:, 1] = self.alpha
            alpha_t[:, 0] = 1 - self.alpha
        else:
            alpha_t = torch.ones_like(targets_smooth)

        # Compute focal loss
        loss = -alpha_t * focal_weight * targets_smooth * log_probs

        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for single-output models."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Weighted sum of Focal Loss and Cross-Entropy.

    Provides complementary gradient signals — focal loss for hard examples
    and CE for stable baseline gradients.

    Args:
        focal_weight (float): Weight for focal loss
        ce_weight (float): Weight for cross-entropy loss
        label_smoothing (float): Label smoothing factor
    """

    def __init__(self, focal_weight: float = 0.7, ce_weight: float = 0.3,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

        self.focal_loss = FocalLoss(
            alpha=0.25, gamma=2.0, label_smoothing=label_smoothing
        )
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.focal_weight * fl + self.ce_weight * ce


class ContrastiveClassificationLoss(nn.Module):
    """
    Combined contrastive + classification loss.

    Used when training the contrastive model with both losses simultaneously.

    Args:
        contrastive_weight (float): Weight for contrastive loss
        classification_weight (float): Weight for classification loss
        temperature (float): Contrastive loss temperature
    """

    def __init__(self, contrastive_weight: float = 0.5,
                 classification_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, classification_logits: torch.Tensor,
                targets: torch.Tensor,
                contrastive_loss: torch.Tensor = None) -> torch.Tensor:
        cls_loss = self.ce_loss(classification_logits, targets)

        if contrastive_loss is not None:
            return (self.classification_weight * cls_loss +
                    self.contrastive_weight * contrastive_loss)
        return cls_loss


class VAELoss(nn.Module):
    """
    VAE loss for the autoencoder model.

    Combines reconstruction loss, KL divergence, and classification loss.

    Args:
        recon_weight (float): Reconstruction loss weight
        kl_weight (float): KL divergence weight
        cls_weight (float): Classification loss weight
    """

    def __init__(self, recon_weight: float = 1.0,
                 kl_weight: float = 0.0005,
                 cls_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.cls_weight = cls_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, reconstruction=None,
                original=None, mu=None, logvar=None):
        total_loss = self.cls_weight * self.ce_loss(logits, targets)

        if reconstruction is not None and original is not None:
            # Resize if needed
            if reconstruction.shape != original.shape:
                reconstruction = F.interpolate(
                    reconstruction, size=original.shape[2:],
                    mode='bilinear', align_corners=False
                )
            recon_loss = F.mse_loss(reconstruction, original)
            total_loss += self.recon_weight * recon_loss

        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            total_loss += self.kl_weight * kl_loss

        return total_loss


if __name__ == "__main__":
    # Quick test
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))

    focal = FocalLoss()
    combined = CombinedLoss()

    print(f"Focal Loss: {focal(logits, targets):.4f}")
    print(f"Combined Loss: {combined(logits, targets):.4f}")
