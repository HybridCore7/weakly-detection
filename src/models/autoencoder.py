"""
Autoencoder-based Deepfake Detector.

Reconstruction-based anomaly detection: real faces are learned during training,
and deepfakes produce higher reconstruction errors because the model has not
seen their specific manipulation patterns.

The autoencoder also includes a classification branch for direct binary
prediction, making it a dual-task model.

Reference: Inspired by approaches from:
- Khalid, H. & Woo, S. S. (2020). OC-FakeDect: Classifying Deepfakes
  Using One-class Variational Autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for encoder/decoder."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.block(x) + x)


class Encoder(nn.Module):
    """
    Convolutional encoder that maps images to a latent representation.

    Progressively downsamples the input while increasing channel depth:
    3 → 64 → 128 → 256 → 512 → latent_dim
    """

    def __init__(self, latent_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            # (B, 3, 256, 256) -> (B, 64, 128, 128)
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            # (B, 64, 128, 128) -> (B, 128, 64, 64)
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            # (B, 128, 64, 64) -> (B, 256, 32, 32)
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),

            # (B, 256, 32, 32) -> (B, 512, 16, 16)
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),

            # (B, 512, 16, 16) -> (B, 512, 8, 8)
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        flat = features.view(features.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, features


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs images from latent representation.

    Mirrors the encoder architecture with transposed convolutions:
    latent_dim → 512 → 256 → 128 → 64 → 3
    """

    def __init__(self, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            # (B, 512, 8, 8) -> (B, 512, 16, 16)
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),

            # (B, 512, 16, 16) -> (B, 256, 32, 32)
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            # (B, 256, 32, 32) -> (B, 128, 64, 64)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            # (B, 128, 64, 64) -> (B, 64, 128, 128)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            # (B, 64, 128, 128) -> (B, 3, 256, 256)
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        return self.decoder(x)


class DeepfakeAutoencoder(nn.Module):
    """
    Variational Autoencoder for deepfake detection.

    Dual-task architecture:
    1. Reconstruction task: Learn to reconstruct real faces
    2. Classification task: Directly classify real vs. fake

    The reconstruction error serves as an additional signal — deepfakes
    produce higher reconstruction error since the model was trained
    primarily on real face distributions.

    Args:
        latent_dim (int): Dimensionality of latent space (default: 512)
        num_classes (int): Number of output classes (default: 2)
        reconstruction_weight (float): Weight for reconstruction loss (default: 1.0)
        kl_weight (float): Weight for KL divergence loss (default: 0.0005)
    """

    def __init__(self, latent_dim=512, num_classes=2,
                 reconstruction_weight=1.0, kl_weight=0.0005):
        super().__init__()

        self.latent_dim = latent_dim
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Classification head from latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Reconstruction error head
        self.recon_classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, return_reconstruction=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 256, 256)
            return_reconstruction: Whether to return reconstructed image

        Returns:
            logits: Classification logits of shape (B, num_classes)
            extras: Dict with reconstruction, mu, logvar if requested
        """
        # Encode
        mu, logvar, features = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Decode (reconstruct)
        reconstruction = self.decoder(z)

        # Compute reconstruction error
        with torch.no_grad():
            # Resize reconstruction to match input if needed
            if reconstruction.shape != x.shape:
                reconstruction_resized = F.interpolate(
                    reconstruction, size=x.shape[2:], mode='bilinear',
                    align_corners=False
                )
            else:
                reconstruction_resized = reconstruction

            recon_error = F.mse_loss(
                reconstruction_resized, x, reduction='none'
            ).mean(dim=[1, 2, 3]).unsqueeze(1)

        # Classification from latent space
        latent_logits = self.classifier(z)

        # Classification from reconstruction error
        recon_logits = self.recon_classifier(recon_error)

        # Fuse both classification signals
        logits = latent_logits + 0.3 * recon_logits

        if return_reconstruction:
            return logits, {
                'reconstruction': reconstruction,
                'mu': mu,
                'logvar': logvar,
                'recon_error': recon_error,
            }

        return logits

    def compute_vae_loss(self, reconstruction, target, mu, logvar):
        """Compute VAE loss = Reconstruction Loss + KL Divergence."""
        # Resize if needed
        if reconstruction.shape != target.shape:
            reconstruction = F.interpolate(
                reconstruction, size=target.shape[2:], mode='bilinear',
                align_corners=False
            )

        # Reconstruction loss (MSE + Perceptual-like L1)
        recon_loss = F.mse_loss(reconstruction, target) + \
                     0.5 * F.l1_loss(reconstruction, target)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return (self.reconstruction_weight * recon_loss +
                self.kl_weight * kl_loss)

    def get_last_conv_layer(self):
        """Return the last conv layer of encoder for Grad-CAM."""
        return list(self.encoder.encoder.children())[-2]


if __name__ == "__main__":
    model = DeepfakeAutoencoder(latent_dim=512, num_classes=2)
    x = torch.randn(2, 3, 256, 256)
    logits, extras = model(x, return_reconstruction=True)
    print(f"Autoencoder logits shape: {logits.shape}")
    print(f"Reconstruction shape: {extras['reconstruction'].shape}")
    print(f"Latent mu shape: {extras['mu'].shape}")
    print(f"Recon error: {extras['recon_error'].squeeze()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
