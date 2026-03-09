"""
Prediction Script — CLI for running deepfake detection inference.

Usage:
    python scripts/predict.py --input image.jpg --model-path checkpoints/best_model.pth
    python scripts/predict.py --input video.mp4 --model-path checkpoints/best_model.pth --output results/
    python scripts/predict.py --input-dir images/ --model-path checkpoints/best_model.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models import XceptionNet, EfficientNetDetector
from src.inference.predictor import DeepfakePredictor


MODEL_MAP = {
    'xception': XceptionNet,
    'efficientnet': EfficientNetDetector,
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}


def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection — Predict'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input image/video file or directory')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='xception',
                        help='Model architecture')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Disable Grad-CAM heatmaps')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation')
    parser.add_argument('--max-frames', type=int, default=32,
                        help='Max frames for video analysis')

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_class = MODEL_MAP.get(args.model, XceptionNet)
    model = DeepfakePredictor.load_model(
        model_class, args.model_path, device=device,
        num_classes=2, pretrained=False
    )

    # Create predictor
    predictor = DeepfakePredictor(
        model=model,
        device=device,
        threshold=args.threshold,
        use_tta=args.tta,
        generate_heatmap=not args.no_heatmap,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file():
        ext = input_path.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            # Single image prediction
            print(f"\n🔍 Analyzing image: {input_path.name}")
            result = predictor.predict_image(str(input_path))
            _print_result(result, str(input_path))

            # Save heatmap if available
            if 'heatmap_overlay' in result:
                import cv2
                overlay_path = output_dir / f"{input_path.stem}_heatmap.jpg"
                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(result['heatmap_overlay'], cv2.COLOR_RGB2BGR)
                )
                print(f"  🗺️  Heatmap saved: {overlay_path}")

        elif ext in VIDEO_EXTENSIONS:
            # Video prediction
            print(f"\n🎬 Analyzing video: {input_path.name}")
            result = predictor.predict_video(
                str(input_path), max_frames=args.max_frames
            )
            _print_video_result(result, str(input_path))

        else:
            print(f"❌ Unsupported file format: {ext}")
            return

    elif input_path.is_dir():
        # Batch prediction
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"\n📁 Analyzing {len(image_files)} images from: {input_path}")

        results = predictor.predict_batch(
            [str(f) for f in image_files]
        )

        for result in results:
            _print_result(result, result.get('file', ''))

        # Save summary
        summary_path = output_dir / 'prediction_summary.json'
        # Clean results for JSON serialization
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items()
                      if k not in ('heatmap', 'heatmap_overlay')}
            clean_results.append(clean_r)

        with open(summary_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\n📄 Summary saved: {summary_path}")

    else:
        print(f"❌ Input not found: {args.input}")


def _print_result(result, filename):
    """Print prediction result."""
    pred = result['prediction']
    conf = result['confidence']

    icon = '✅' if pred == 'Real' else '⚠️'
    print(f"\n  {icon} {Path(filename).name}")
    print(f"     Prediction: {pred}")
    print(f"     Confidence: {conf:.1%}")
    print(f"     Probabilities: Real={result['probabilities']['Real']:.3f} "
          f"| Fake={result['probabilities']['Fake']:.3f}")

    if 'face_detected' in result:
        print(f"     Face detected: {'Yes' if result['face_detected'] else 'No'}")


def _print_video_result(result, filename):
    """Print video prediction result."""
    pred = result['prediction']
    conf = result['confidence']

    icon = '✅' if pred == 'Real' else '⚠️'
    print(f"\n  {icon} {Path(filename).name}")
    print(f"     Overall: {pred} ({conf:.1%})")
    print(f"     Frames analyzed: {result['frames_analyzed']}")
    print(f"     Fake frame ratio: {result['fake_frame_ratio']:.1%}")

    if 'timeline' in result:
        print(f"     Frame timeline:")
        for fp in result['timeline'][:10]:
            frame_icon = '🔴' if fp['prediction'] == 'Fake' else '🟢'
            print(f"       {frame_icon} Frame {fp['frame_index']:3d}: "
                  f"{fp['prediction']} ({fp['fake_probability']:.3f})")
        if len(result['timeline']) > 10:
            print(f"       ... and {len(result['timeline']) - 10} more frames")


if __name__ == '__main__':
    main()
