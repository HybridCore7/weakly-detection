"""
Unit tests for data pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from PIL import Image


class TestAugmentation:
    """Tests for augmentation pipeline."""

    def test_train_transforms(self):
        from src.data.augmentation import get_train_transforms
        transform = get_train_transforms(299, use_albumentations=False)
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 299, 299)

    def test_val_transforms(self):
        from src.data.augmentation import get_val_transforms
        transform = get_val_transforms(299, use_albumentations=False)
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 299, 299)

    def test_contrastive_transforms(self):
        from src.data.augmentation import get_contrastive_transforms
        t1, t2 = get_contrastive_transforms(256)
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        out1 = t1(img)
        out2 = t2(img)
        assert out1.shape == (3, 256, 256)
        assert out2.shape == (3, 256, 256)

    def test_jpeg_compression(self):
        from src.data.augmentation import JPEGCompression
        aug = JPEGCompression(quality_range=(30, 100))
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        out = aug(img)
        assert isinstance(out, Image.Image)

    def test_gaussian_noise(self):
        from src.data.augmentation import GaussianNoise
        aug = GaussianNoise()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        out = aug(img)
        assert isinstance(out, Image.Image)


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_preprocess_image_array(self):
        from src.data.preprocessing import preprocess_image_array
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        tensor = preprocess_image_array(img, target_size=(299, 299))
        assert tensor.shape == (1, 3, 299, 299)

    def test_denormalize(self):
        from src.data.preprocessing import denormalize
        tensor = torch.randn(1, 3, 64, 64)
        out = denormalize(tensor)
        assert out.shape == tensor.shape


class TestFaceExtractor:
    """Tests for face extraction."""

    def test_initialization(self):
        from src.data.face_extractor import FaceExtractor
        extractor = FaceExtractor(device='cpu')
        assert extractor is not None

    def test_crop_face(self):
        from src.data.face_extractor import FaceExtractor
        extractor = FaceExtractor(device='cpu', margin=20)
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        box = np.array([100, 100, 300, 300])
        face = extractor._crop_face(img, box, (299, 299))
        assert face is not None
        assert face.shape == (299, 299, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
