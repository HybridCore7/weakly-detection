"""
Preprocessing utilities for images and videos.

Handles loading, normalization, and preparation of media files
for deepfake detection inference.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (299, 299),
    normalize: bool = True,
    return_original: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
    """
    Preprocess a single image for model inference.

    Args:
        image_path: Path to image file
        target_size: Target (H, W) size
        normalize: Apply ImageNet normalization
        return_original: Also return original image

    Returns:
        tensor: Preprocessed image tensor (1, 3, H, W)
        original: Original image (only if return_original=True)
    """
    image = Image.open(image_path).convert('RGB')
    original = np.array(image)

    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    transform = transforms.Compose(transform_list)
    tensor = transform(image).unsqueeze(0)  # Add batch dim

    if return_original:
        return tensor, original
    return tensor


def preprocess_image_array(
    image: np.ndarray,
    target_size: Tuple[int, int] = (299, 299),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess a numpy array image for model inference.

    Args:
        image: RGB image array (H, W, 3)
        target_size: Target (H, W) size
        normalize: Apply ImageNet normalization

    Returns:
        tensor: Preprocessed image tensor (1, 3, H, W)
    """
    pil_image = Image.fromarray(image)

    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    transform = transforms.Compose(transform_list)
    return transform(pil_image).unsqueeze(0)


def preprocess_video(
    video_path: str,
    target_size: Tuple[int, int] = (299, 299),
    max_frames: int = 32,
    sample_strategy: str = 'uniform',
    normalize: bool = True,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Preprocess a video for frame-by-frame analysis.

    Args:
        video_path: Path to video file
        target_size: Target frame size (H, W)
        max_frames: Maximum number of frames to extract
        sample_strategy: 'uniform', 'first', or 'random'
        normalize: Apply ImageNet normalization

    Returns:
        tensors: Batch of frame tensors (N, 3, H, W)
        originals: List of original frame arrays
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine which frames to sample
    if sample_strategy == 'uniform':
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(
                0, total_frames - 1, max_frames, dtype=int
            ).tolist()
    elif sample_strategy == 'first':
        frame_indices = list(range(min(total_frames, max_frames)))
    elif sample_strategy == 'random':
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(
                np.random.choice(total_frames, max_frames, replace=False)
            )
    else:
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")

    frames = []
    originals = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        originals.append(frame_rgb.copy())

        # Preprocess
        pil_frame = Image.fromarray(frame_rgb)
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )
        transform = transforms.Compose(transform_list)
        frames.append(transform(pil_frame))

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    tensors = torch.stack(frames)  # (N, 3, H, W)
    return tensors, originals


def denormalize(tensor: torch.Tensor,
                mean: List[float] = None,
                std: List[float] = None) -> torch.Tensor:
    """
    Denormalize a tensor back to [0, 1] range.

    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    mean = torch.tensor(mean, device=tensor.device)
    std = torch.tensor(std, device=tensor.device)

    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.dim() == 3:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    return tensor * std + mean


def get_video_metadata(video_path: str) -> dict:
    """
    Extract metadata from a video file.

    Returns:
        metadata: Dict with fps, frame_count, width, height, duration
    """
    cap = cv2.VideoCapture(video_path)

    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0.0,
    }

    if metadata['fps'] > 0:
        metadata['duration'] = metadata['frame_count'] / metadata['fps']

    cap.release()
    return metadata


if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")
    print(f"ImageNet Mean: {IMAGENET_MEAN}")
    print(f"ImageNet Std: {IMAGENET_STD}")
