"""
Grad-CAM — Gradient-weighted Class Activation Mapping for explainability.

Generates visual heatmaps showing which regions of the image contributed
most to the model's prediction. Critical for interpretability in deepfake
detection — shows WHERE manipulation was detected.

Reference: Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization. ICCV 2017.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.

    Hooks into a target convolutional layer to capture activations and
    gradients, then computes a weighted combination to produce heatmaps.

    Args:
        model: PyTorch CNN model
        target_layer: Target convolutional layer for visualization
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 normalize: bool = True) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Target class index (None = predicted class)
            normalize: Normalize heatmap to [0, 1]

        Returns:
            heatmap: Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.eval()

        # Enable gradients temporarily
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Global average pooling of gradients -> weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )

        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()

        if normalize and heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def generate_multiple(self, input_tensor: torch.Tensor,
                          classes: list = None) -> dict:
        """
        Generate Grad-CAM heatmaps for multiple classes.

        Args:
            input_tensor: Input tensor
            classes: List of class indices

        Returns:
            heatmaps: Dict mapping class index to heatmap
        """
        if classes is None:
            classes = [0, 1]  # Real, Fake

        heatmaps = {}
        for cls in classes:
            heatmaps[cls] = self.generate(input_tensor, target_class=cls)

        return heatmaps

    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray,
                        alpha: float = 0.5,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            heatmap: Grad-CAM heatmap (H, W), values in [0, 1]
            image: Original image (H, W, 3), RGB
            alpha: Blend factor (0 = only image, 1 = only heatmap)
            colormap: OpenCV colormap

        Returns:
            overlay: Blended image (H, W, 3), RGB
        """
        # Resize heatmap to match image
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert to uint8 for colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

        return overlay

    @staticmethod
    def create_comparison(original: np.ndarray, heatmap: np.ndarray,
                          prediction: str, confidence: float) -> np.ndarray:
        """
        Create a side-by-side comparison image with original and heatmap.

        Args:
            original: Original image
            heatmap: Grad-CAM heatmap
            prediction: Prediction label
            confidence: Prediction confidence

        Returns:
            comparison: Side-by-side comparison image
        """
        overlay = GradCAM.overlay_heatmap(heatmap, original, alpha=0.4)

        # Ensure same size
        h = max(original.shape[0], overlay.shape[0])
        w = original.shape[1] + overlay.shape[1] + 10  # 10px gap

        comparison = np.ones((h + 40, w, 3), dtype=np.uint8) * 255

        # Place images
        comparison[:original.shape[0], :original.shape[1]] = original
        x_offset = original.shape[1] + 10
        comparison[:overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

        # Add text
        color = (0, 180, 0) if prediction == 'Real' else (220, 0, 0)
        text = f"{prediction} ({confidence:.1%})"
        cv2.putText(
            comparison, text,
            (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        return comparison


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — Improved Grad-CAM with better localization.

    Uses second and third order gradients for more precise heatmaps,
    especially for multiple instances of the same class.

    Reference: Chattopadhay, A. et al. (2018). Grad-CAM++: Generalized
    Gradient-based Visual Explanations for Deep Convolutional Networks.
    """

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 normalize: bool = True) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        grads = self.gradients
        acts = self.activations

        # Grad-CAM++ weights
        # alpha = grad^2 / (2 * grad^2 + sum(A * grad^3) + eps)
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_acts = acts.sum(dim=[2, 3], keepdim=True)
        denominator = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8

        alpha = grads_power_2 / denominator
        alpha = alpha * torch.relu(grads)

        # Weights
        weights = alpha.sum(dim=[2, 3], keepdim=True)

        # Weighted combination
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )

        heatmap = cam.squeeze().cpu().numpy()

        if normalize and heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully.")
    print("Available: GradCAM, GradCAMPlusPlus")
