"""
Evaluation Script — Evaluate trained models on test data.

Usage:
    python scripts/evaluate.py --model-path checkpoints/best_model.pth --data-dir ./data/test
    python scripts/evaluate.py --model-path checkpoints/best_model.pth --model xception --detailed
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np

from src.models import (
    XceptionNet, EfficientNetDetector, DeepfakeAutoencoder,
    ContrastiveDetector, EnsembleDetector
)
from src.data.dataset import DeepfakeDataset
from src.data.augmentation import get_val_transforms
from src.training.metrics import MetricsCalculator
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_model_comparison
)


MODEL_MAP = {
    'xception': XceptionNet,
    'efficientnet': EfficientNetDetector,
    'autoencoder': DeepfakeAutoencoder,
    'contrastive': ContrastiveDetector,
    'ensemble': EnsembleDetector,
}


def evaluate_model(model, dataloader, device, detailed=False):
    """Run evaluation on a dataset."""
    model.eval()
    metrics = MetricsCalculator(num_classes=2)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)

            metrics.update(predictions, labels, probabilities)

    # Print results
    print(metrics.summary())

    results = metrics.compute_all()

    if detailed:
        # ROC curve
        fpr, tpr, _ = metrics.get_roc_curve()
        if fpr is not None:
            plot_roc_curve(fpr, tpr, results['auc'],
                          save_path='evaluation_roc.png')

        # Confusion matrix
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            plot_confusion_matrix(cm, save_path='evaluation_confusion.png')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Deepfake Detection Model'
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='xception',
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed plots')
    parser.add_argument('--image-size', type=int, default=299,
                        help='Image size')

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Build model
    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model: {args.model}. "
                        f"Available: {list(MODEL_MAP.keys())}")

    model_class = MODEL_MAP[args.model]
    model = model_class(num_classes=2, pretrained=False)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"✅ Model loaded from {args.model_path}")

    # Setup data
    val_transforms = get_val_transforms(args.image_size)
    test_dataset = DeepfakeDataset(
        args.data_dir, split='test', transform=val_transforms
    )

    test_loader = DeepfakeDataset.get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"📊 Test samples: {len(test_dataset)}")

    # Evaluate
    results = evaluate_model(model, test_loader, device, args.detailed)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
