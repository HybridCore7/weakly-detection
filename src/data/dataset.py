"""
Dataset classes for deepfake detection.

Supports multiple dataset formats:
1. FaceForensics++ — Video-based with manipulation types
2. DFDC — Kaggle competition format
3. CelebDF — Celebrity deepfake dataset
4. Custom — Simple directory structure (real/ and fake/ folders)
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DeepfakeDataset(Dataset):
    """
    Generic deepfake detection dataset.

    Expects a directory structure like:
        root/
        ├── real/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        └── fake/
            ├── img001.jpg
            ├── img002.jpg
            └── ...

    Args:
        root_dir (str): Path to dataset root
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Image transforms
        max_samples (int, optional): Maximum samples per class
        balance_classes (bool): Whether to balance class distribution
    """

    LABEL_MAP = {'real': 0, 'fake': 1}

    def __init__(self, root_dir: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 max_samples: Optional[int] = None,
                 balance_classes: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []

        # Load samples
        self._load_samples(max_samples, balance_classes)

    def _load_samples(self, max_samples, balance_classes):
        """Scan directory structure and load image paths with labels."""
        real_dir = self.root_dir / self.split / 'real'
        fake_dir = self.root_dir / self.split / 'fake'

        # Also check flat structure (for custom datasets)
        if not real_dir.exists():
            real_dir = self.root_dir / 'real'
            fake_dir = self.root_dir / 'fake'

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        real_samples = []
        fake_samples = []

        if real_dir.exists():
            real_samples = [
                (str(p), 0)
                for p in sorted(real_dir.iterdir())
                if p.suffix.lower() in image_extensions
            ]

        if fake_dir.exists():
            fake_samples = [
                (str(p), 1)
                for p in sorted(fake_dir.iterdir())
                if p.suffix.lower() in image_extensions
            ]

        # Balance classes if requested
        if balance_classes and real_samples and fake_samples:
            min_count = min(len(real_samples), len(fake_samples))
            random.shuffle(real_samples)
            random.shuffle(fake_samples)
            real_samples = real_samples[:min_count]
            fake_samples = fake_samples[:min_count]

        all_samples = real_samples + fake_samples

        # Limit total samples
        if max_samples and len(all_samples) > max_samples:
            random.shuffle(all_samples)
            all_samples = all_samples[:max_samples]

        for path, label in all_samples:
            self.samples.append(path)
            self.labels.append(label)

        # If no real samples found, create synthetic data info
        if len(self.samples) == 0:
            print(f"⚠ Warning: No samples found in {self.root_dir}")
            print(f"  Expected structure: {self.root_dir}/[split]/real/ and "
                  f"{self.root_dir}/[split]/fake/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize and normalize
            image = image.resize((299, 299))
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label

    def get_class_weights(self):
        """Compute class weights for imbalanced datasets."""
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

    @staticmethod
    def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4,
                       pin_memory=True):
        """Create a DataLoader from the dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )


class VideoFrameDataset(Dataset):
    """
    Dataset that extracts frames from videos for deepfake detection.

    Processes videos frame-by-frame, optionally extracting faces.

    Args:
        video_dir (str): Directory containing video files
        labels_file (str): CSV/JSON file mapping video names to labels
        frames_per_video (int): Number of frames to extract per video
        transform (callable): Image transforms
        face_extractor: Optional FaceExtractor instance
    """

    def __init__(self, video_dir: str, labels_file: Optional[str] = None,
                 frames_per_video: int = 16,
                 transform: Optional[Callable] = None,
                 face_extractor=None):
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.face_extractor = face_extractor

        self.videos = []
        self.labels = []

        self._load_video_list(labels_file)

    def _load_video_list(self, labels_file):
        """Load video paths and labels."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

        if labels_file and Path(labels_file).exists():
            # Load from labels file
            import json
            with open(labels_file, 'r') as f:
                if labels_file.endswith('.json'):
                    label_data = json.load(f)
                else:
                    # CSV format: filename,label
                    import csv
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    label_data = {row[0]: int(row[1]) for row in reader}

            for video_name, label in label_data.items():
                video_path = self.video_dir / video_name
                if video_path.exists():
                    self.videos.append(str(video_path))
                    self.labels.append(label)
        else:
            # Infer from directory structure
            for label_name, label_idx in [('real', 0), ('fake', 1)]:
                label_dir = self.video_dir / label_name
                if label_dir.exists():
                    for f in sorted(label_dir.iterdir()):
                        if f.suffix.lower() in video_extensions:
                            self.videos.append(str(f))
                            self.labels.append(label_idx)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract evenly-spaced frames from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        # Calculate frame indices (evenly-spaced)
        if total_frames <= self.frames_per_video:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(
                0, total_frames - 1, self.frames_per_video, dtype=int
            ).tolist()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract face if extractor is provided
                if self.face_extractor:
                    face = self.face_extractor.extract_face(frame)
                    if face is not None:
                        frame = face

                frames.append(frame)

        cap.release()
        return frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)

        if not frames:
            # Return dummy data if video couldn't be read
            dummy = torch.zeros(self.frames_per_video, 3, 299, 299)
            return dummy, label

        # Apply transforms to each frame
        processed_frames = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            if self.transform:
                frame_tensor = self.transform(frame_pil)
            else:
                frame_pil = frame_pil.resize((299, 299))
                frame_arr = np.array(frame_pil, dtype=np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_arr).permute(2, 0, 1)
            processed_frames.append(frame_tensor)

        # Stack into temporal tensor (T, C, H, W)
        frames_tensor = torch.stack(processed_frames)

        # Pad if we got fewer frames than expected
        if frames_tensor.shape[0] < self.frames_per_video:
            padding = torch.zeros(
                self.frames_per_video - frames_tensor.shape[0],
                *frames_tensor.shape[1:]
            )
            frames_tensor = torch.cat([frames_tensor, padding], dim=0)

        return frames_tensor, label


class ContrastiveDataset(Dataset):
    """
    Dataset wrapper that returns two augmented views for contrastive learning.

    Wraps an existing dataset and applies two different random augmentations
    to create positive pairs.

    Args:
        base_dataset: Base DeepfakeDataset
        transform1: First augmentation transform
        transform2: Second augmentation transform
    """

    def __init__(self, base_dataset: DeepfakeDataset,
                 transform1: Callable, transform2: Callable):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_path = self.base_dataset.samples[idx]
        label = self.base_dataset.labels[idx]

        image = Image.open(img_path).convert('RGB')

        view1 = self.transform1(image)
        view2 = self.transform2(image)

        return view1, view2, label


if __name__ == "__main__":
    print("Dataset module loaded successfully.")
    print("Supported formats: FaceForensics++, DFDC, CelebDF, Custom")
    print("\nExpected directory structure:")
    print("  dataset_root/")
    print("  ├── train/")
    print("  │   ├── real/")
    print("  │   └── fake/")
    print("  ├── val/")
    print("  │   ├── real/")
    print("  │   └── fake/")
    print("  └── test/")
    print("      ├── real/")
    print("      └── fake/")
