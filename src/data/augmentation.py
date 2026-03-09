"""
Augmentation Pipeline — Deepfake-specific data augmentations.

Includes standard augmentations plus domain-specific transforms that
simulate common deepfake artifacts and compression patterns, making
the model more robust to real-world conditions.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torchvision import transforms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class JPEGCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, quality_range=(30, 100)):
        self.quality_range = quality_range

    def __call__(self, image):
        import io
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianNoise:
    """Add Gaussian noise to simulate camera sensor noise."""

    def __init__(self, mean=0, std_range=(0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, image):
        img_array = np.array(image, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.normal(self.mean, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))


class RandomGaussianBlur:
    """Random Gaussian blur to simulate focus/motion blur."""

    def __init__(self, radius_range=(0.5, 2.0)):
        self.radius_range = radius_range

    def __call__(self, image):
        radius = random.uniform(*self.radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomDownscale:
    """Random downscale + upscale to simulate low resolution."""

    def __init__(self, scale_range=(0.5, 0.9)):
        self.scale_range = scale_range

    def __call__(self, image):
        w, h = image.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        downscaled = image.resize((new_w, new_h), Image.BILINEAR)
        return downscaled.resize((w, h), Image.BILINEAR)


class ColorJitterAugment:
    """Enhanced color jitter with saturation and hue control."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2,
                 hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image):
        # Brightness
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Saturation
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        return image


class CutoutAugment:
    """Random rectangular cutout (erasing) augmentation."""

    def __init__(self, num_holes=1, max_h_size=40, max_w_size=40):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, image):
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        for _ in range(self.num_holes):
            hole_h = random.randint(10, self.max_h_size)
            hole_w = random.randint(10, self.max_w_size)

            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)

            img_array[y:y + hole_h, x:x + hole_w] = 128  # Gray fill

        return Image.fromarray(img_array)


def get_train_transforms(image_size=299, use_albumentations=True):
    """
    Get training augmentation transforms.

    Includes both standard augmentations and deepfake-specific transforms
    to improve model robustness.

    Args:
        image_size (int): Target image size
        use_albumentations (bool): Use albumentations library if available

    Returns:
        transform: Composed transform pipeline
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),

            # Color augmentations
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),

            # Deepfake-specific augmentations
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),

            # Geometric augmentations
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=10, p=0.3
            ),
            A.CoarseDropout(
                max_holes=1, max_height=40, max_width=40,
                min_holes=1, min_height=10, min_width=10, p=0.2
            ),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    # Fallback: torchvision transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        ColorJitterAugment(0.2, 0.2, 0.2, 0.1),
        transforms.RandomApply([JPEGCompression((30, 100))], p=0.5),
        transforms.RandomApply([RandomGaussianBlur((0.5, 2.0))], p=0.3),
        transforms.RandomApply([GaussianNoise(0, (0.01, 0.05))], p=0.3),
        transforms.RandomApply([RandomDownscale((0.5, 0.9))], p=0.2),
        transforms.RandomApply([CutoutAugment()], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size=299, use_albumentations=True):
    """
    Get validation/test transforms (only resize + normalize).

    Args:
        image_size (int): Target image size
        use_albumentations (bool): Use albumentations if available

    Returns:
        transform: Composed transform pipeline
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_contrastive_transforms(image_size=256):
    """
    Get augmentation transforms for contrastive learning.

    Creates two strongly-augmented views of the same image.
    Stronger augmentation than standard training transforms.

    Returns:
        (transform1, transform2): Two augmentation pipelines
    """
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([RandomGaussianBlur((0.1, 2.0))], p=0.5),
        transforms.RandomApply([JPEGCompression((20, 80))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return base_transform, base_transform


if __name__ == "__main__":
    print("Augmentation pipeline loaded successfully.")
    print(f"Albumentations available: {HAS_ALBUMENTATIONS}")

    # Test with dummy image
    dummy = Image.fromarray(np.random.randint(0, 255, (299, 299, 3),
                                               dtype=np.uint8))
    train_tf = get_train_transforms(299, use_albumentations=False)
    val_tf = get_val_transforms(299, use_albumentations=False)

    train_out = train_tf(dummy)
    val_out = val_tf(dummy)
    print(f"Train transform output: {train_out.shape}")
    print(f"Val transform output: {val_out.shape}")
