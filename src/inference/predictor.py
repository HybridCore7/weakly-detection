"""
Inference Pipeline — Production-ready prediction for deepfake detection.

Handles single image, batch, and video predictions with optional
test-time augmentation (TTA) and Grad-CAM explainability.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..data.preprocessing import (
    preprocess_image, preprocess_image_array,
    preprocess_video, denormalize
)
from ..data.face_extractor import FaceExtractor
from .grad_cam import GradCAM


class DeepfakePredictor:
    """
    Production inference pipeline for deepfake detection.

    Features:
    - Single image and video prediction
    - Automatic face extraction
    - Test-time augmentation (TTA)
    - Grad-CAM heatmap generation
    - Confidence calibration
    - Batch processing

    Args:
        model: Trained PyTorch model
        device (str): Computing device
        threshold (float): Classification threshold (default: 0.5)
        use_tta (bool): Enable test-time augmentation
        tta_iterations (int): Number of TTA iterations
        generate_heatmap (bool): Generate Grad-CAM heatmaps
        extract_faces (bool): Auto-extract faces before prediction
    """

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 threshold: float = 0.5, use_tta: bool = False,
                 tta_iterations: int = 5, generate_heatmap: bool = True,
                 extract_faces: bool = True):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.use_tta = use_tta
        self.tta_iterations = tta_iterations
        self.generate_heatmap = generate_heatmap

        # Face extractor
        if extract_faces:
            self.face_extractor = FaceExtractor(device=device)
        else:
            self.face_extractor = None

        # Grad-CAM
        if generate_heatmap:
            try:
                target_layer = model.get_last_conv_layer()
                self.grad_cam = GradCAM(model, target_layer)
            except (AttributeError, Exception):
                self.grad_cam = None
                print("⚠ Grad-CAM not available for this model")
        else:
            self.grad_cam = None

        self.class_names = ['Real', 'Fake']

    @torch.no_grad()
    def predict_image(self, image_input: Union[str, np.ndarray, Image.Image],
                      return_heatmap: bool = True) -> Dict:
        """
        Predict whether a single image is real or fake.

        Args:
            image_input: Image path, numpy array, or PIL Image
            return_heatmap: Whether to generate Grad-CAM heatmap

        Returns:
            result: Dict with keys:
                - prediction: 'Real' or 'Fake'
                - confidence: Prediction confidence (0-1)
                - probabilities: {'Real': float, 'Fake': float}
                - heatmap: Grad-CAM heatmap (if requested)
                - face_detected: Whether a face was found
        """
        # Load image
        if isinstance(image_input, str):
            original_image = np.array(Image.open(image_input).convert('RGB'))
        elif isinstance(image_input, Image.Image):
            original_image = np.array(image_input.convert('RGB'))
        else:
            original_image = image_input.copy()

        # Extract face
        face_image = original_image
        face_detected = True
        if self.face_extractor:
            face = self.face_extractor.extract_face(original_image)
            if face is not None:
                face_image = face
            else:
                face_detected = False

        # Preprocess
        tensor = preprocess_image_array(face_image)
        tensor = tensor.to(self.device)

        # Predict (with optional TTA)
        if self.use_tta:
            probabilities = self._predict_with_tta(tensor)
        else:
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # Classification
        fake_prob = float(probabilities[1])
        prediction = 'Fake' if fake_prob >= self.threshold else 'Real'
        confidence = fake_prob if prediction == 'Fake' else (1 - fake_prob)

        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'Real': float(probabilities[0]),
                'Fake': float(probabilities[1]),
            },
            'face_detected': face_detected,
        }

        # Generate Grad-CAM heatmap
        if return_heatmap and self.grad_cam is not None:
            try:
                heatmap = self.grad_cam.generate(
                    tensor, target_class=1  # Focus on "fake" class
                )
                result['heatmap'] = heatmap

                # Overlay heatmap on original image
                overlay = self.grad_cam.overlay_heatmap(
                    heatmap, face_image
                )
                result['heatmap_overlay'] = overlay
            except Exception as e:
                print(f"⚠ Grad-CAM generation failed: {e}")

        return result

    def _predict_with_tta(self, tensor: torch.Tensor) -> np.ndarray:
        """Apply test-time augmentation for more robust predictions."""
        from torchvision import transforms

        all_probs = []

        # Original prediction
        outputs = self.model(tensor)
        all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

        # TTA augmentations
        tta_transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]

        denorm_tensor = denormalize(tensor.cpu())

        for i in range(min(self.tta_iterations - 1, len(tta_transforms))):
            aug_tensor = tta_transforms[i % len(tta_transforms)](
                denorm_tensor.clone()
            )
            # Re-normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            aug_tensor = (aug_tensor - mean) / std
            aug_tensor = aug_tensor.to(self.device)

            outputs = self.model(aug_tensor)
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs[0]

    def predict_video(self, video_path: str,
                      max_frames: int = 32,
                      return_timeline: bool = True) -> Dict:
        """
        Analyze a video for deepfakes, frame by frame.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            return_timeline: Include per-frame predictions

        Returns:
            result: Dict with overall and per-frame predictions
        """
        tensors, originals = preprocess_video(
            video_path, max_frames=max_frames
        )

        # Process faces
        face_tensors = []
        for frame in originals:
            if self.face_extractor:
                face = self.face_extractor.extract_face(frame)
                if face is not None:
                    face_tensor = preprocess_image_array(face)
                    face_tensors.append(face_tensor.squeeze(0))
                    continue

            # Use full frame if face not detected
            face_tensor = preprocess_image_array(frame)
            face_tensors.append(face_tensor.squeeze(0))

        if not face_tensors:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'error': 'No frames could be processed',
            }

        batch = torch.stack(face_tensors).to(self.device)

        # Batch prediction
        with torch.no_grad():
            outputs = self.model(batch)
            all_probs = F.softmax(outputs, dim=1).cpu().numpy()

        # Per-frame results
        frame_predictions = []
        for i, probs in enumerate(all_probs):
            fake_prob = float(probs[1])
            pred = 'Fake' if fake_prob >= self.threshold else 'Real'
            frame_predictions.append({
                'frame_index': i,
                'prediction': pred,
                'fake_probability': fake_prob,
                'real_probability': float(probs[0]),
            })

        # Overall prediction (majority vote + average confidence)
        fake_count = sum(1 for fp in frame_predictions
                         if fp['prediction'] == 'Fake')
        avg_fake_prob = np.mean([fp['fake_probability']
                                 for fp in frame_predictions])

        overall_pred = 'Fake' if avg_fake_prob >= self.threshold else 'Real'
        overall_confidence = avg_fake_prob if overall_pred == 'Fake' \
            else (1 - avg_fake_prob)

        result = {
            'prediction': overall_pred,
            'confidence': float(overall_confidence),
            'probabilities': {
                'Real': float(1 - avg_fake_prob),
                'Fake': float(avg_fake_prob),
            },
            'frames_analyzed': len(frame_predictions),
            'fake_frame_ratio': fake_count / len(frame_predictions),
        }

        if return_timeline:
            result['timeline'] = frame_predictions

        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict on a batch of images."""
        results = []
        for path in image_paths:
            try:
                result = self.predict_image(path, return_heatmap=False)
                result['file'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': path,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'error': str(e),
                })
        return results

    @staticmethod
    def load_model(model_class, checkpoint_path: str,
                   device: str = 'cpu', **model_kwargs):
        """
        Load a trained model from checkpoint.

        Args:
            model_class: Model class to instantiate
            checkpoint_path: Path to checkpoint file
            device: Computing device
            **model_kwargs: Arguments for model constructor

        Returns:
            model: Loaded model in eval mode
        """
        model = model_class(**model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model


if __name__ == "__main__":
    print("Predictor module loaded successfully.")
    print("Supported inputs: image files, numpy arrays, PIL Images, videos")
