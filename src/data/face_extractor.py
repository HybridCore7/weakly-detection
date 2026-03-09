"""
Face Extractor — MTCNN-based face detection and extraction.

Extracts face regions from images and videos as a preprocessing step.
Uses Multi-Task Cascaded Convolutional Networks (MTCNN) for robust
face detection with facial landmark alignment.

Reference: Zhang, K. et al. (2016). Joint Face Detection and Alignment
Using Multi-task Cascaded Convolutional Networks. SPL 2016.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

import torch


class FaceExtractor:
    """
    Face detection and extraction using MTCNN.

    Features:
    - Multi-face detection with confidence scoring
    - Facial landmark-based alignment
    - Configurable margin around detected faces
    - Batch processing support
    - Fallback to Haar cascades if MTCNN unavailable

    Args:
        device (str): Computing device ('cuda' or 'cpu')
        margin (int): Pixel margin around detected face (default: 40)
        min_face_size (int): Minimum face size to detect (default: 60)
        thresholds (list): Detection thresholds for 3 MTCNN stages
        keep_all (bool): Keep all detected faces or just the largest
        post_process (bool): Apply alignment post-processing
    """

    def __init__(self, device: str = 'cpu', margin: int = 40,
                 min_face_size: int = 60,
                 thresholds: Optional[List[float]] = None,
                 keep_all: bool = False, post_process: bool = True):

        self.device = device
        self.margin = margin
        self.keep_all = keep_all

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        if HAS_MTCNN:
            self.detector = MTCNN(
                image_size=299,
                margin=margin,
                min_face_size=min_face_size,
                thresholds=thresholds,
                factor=0.709,
                post_process=post_process,
                keep_all=keep_all,
                device=device,
            )
        else:
            self.detector = None
            # Fallback: OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + \
                           'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            print("⚠ MTCNN not available, using Haar cascade fallback")

    def extract_face(self, image: Union[np.ndarray, Image.Image],
                     target_size: Tuple[int, int] = (299, 299),
                     min_confidence: float = 0.95) -> Optional[np.ndarray]:
        """
        Extract the primary face from an image.

        Args:
            image: Input image (numpy array or PIL Image)
            target_size: Output face size (H, W)
            min_confidence: Minimum detection confidence

        Returns:
            face: Extracted face as numpy array, or None if no face found
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        if self.detector is not None:
            # MTCNN detection
            try:
                boxes, probs, landmarks = self.detector.detect(
                    pil_image, landmarks=True
                )

                if boxes is None or len(boxes) == 0:
                    return None

                # Find face with highest confidence
                best_idx = np.argmax(probs)
                if probs[best_idx] < min_confidence:
                    return None

                box = boxes[best_idx].astype(int)
                face = self._crop_face(np.array(pil_image), box, target_size)

                # Optional: align face using landmarks
                if landmarks is not None and landmarks[best_idx] is not None:
                    face = self._align_face(
                        np.array(pil_image), landmarks[best_idx],
                        box, target_size
                    )

                return face

            except Exception as e:
                print(f"⚠ MTCNN detection failed: {e}")
                return self._haar_fallback(np.array(pil_image), target_size)
        else:
            return self._haar_fallback(np.array(pil_image), target_size)

    def extract_all_faces(self, image: Union[np.ndarray, Image.Image],
                          target_size: Tuple[int, int] = (299, 299),
                          min_confidence: float = 0.9
                          ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract all faces from an image with their confidence scores.

        Returns:
            faces: List of (face_array, confidence) tuples
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        faces = []

        if self.detector is not None:
            try:
                boxes, probs, _ = self.detector.detect(pil_image,
                                                        landmarks=True)
                if boxes is None:
                    return faces

                img_array = np.array(pil_image)
                for box, prob in zip(boxes, probs):
                    if prob >= min_confidence:
                        box = box.astype(int)
                        face = self._crop_face(img_array, box, target_size)
                        if face is not None:
                            faces.append((face, float(prob)))

            except Exception as e:
                print(f"⚠ Multi-face detection failed: {e}")

        return faces

    def _crop_face(self, image: np.ndarray, box: np.ndarray,
                   target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Crop face region with margin from the image."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        # Add margin
        margin_x = int((x2 - x1) * self.margin / 100)
        margin_y = int((y2 - y1) * self.margin / 100)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, target_size, interpolation=cv2.INTER_LANCZOS4)

        return face

    def _align_face(self, image: np.ndarray, landmarks: np.ndarray,
                    box: np.ndarray,
                    target_size: Tuple[int, int]) -> np.ndarray:
        """Align face using eye landmarks for consistent orientation."""
        if landmarks is None or len(landmarks) < 2:
            return self._crop_face(image, box, target_size)

        # Eye centers
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Compute angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Eye center
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )

        # Rotation matrix
        M = cv2.getRotationMatrix2D(
            (float(eye_center[0]), float(eye_center[1])), angle, 1.0
        )

        # Rotate image
        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LANCZOS4)

        # Crop from rotated image
        return self._crop_face(rotated, box, target_size)

    def _haar_fallback(self, image: np.ndarray,
                       target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Fallback face detection using Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # Take the largest face
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        x, y, w, h = faces[largest_idx]

        box = np.array([x, y, x + w, y + h])
        return self._crop_face(image, box, target_size)

    def process_video(self, video_path: str, output_dir: str,
                      frames_per_second: float = 1.0,
                      target_size: Tuple[int, int] = (299, 299)
                      ) -> List[str]:
        """
        Extract faces from a video, saving them as individual images.

        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted faces
            frames_per_second: Frames to sample per second
            target_size: Output face size

        Returns:
            saved_paths: List of paths to saved face images
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return saved_paths

        frame_interval = max(1, int(fps / frames_per_second))
        video_name = Path(video_path).stem

        frame_idx = 0
        save_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.extract_face(frame_rgb, target_size)

                if face is not None:
                    save_path = os.path.join(
                        output_dir,
                        f"{video_name}_frame{save_idx:04d}.jpg"
                    )
                    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, face_bgr)
                    saved_paths.append(save_path)
                    save_idx += 1

            frame_idx += 1

        cap.release()
        return saved_paths


if __name__ == "__main__":
    extractor = FaceExtractor(device='cpu', margin=40)
    print("Face extractor initialized successfully.")
    print(f"Using: {'MTCNN' if HAS_MTCNN else 'Haar Cascade (fallback)'}")
