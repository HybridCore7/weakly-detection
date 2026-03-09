"""
Evaluation Metrics for Deepfake Detection.

Comprehensive metrics including accuracy, precision, recall, F1, AUC-ROC,
confusion matrix, and per-class statistics.
"""

from typing import Dict, Optional
import numpy as np
import torch

try:
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, average_precision_score, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MetricsCalculator:
    """
    Accumulates predictions and computes comprehensive evaluation metrics.

    Usage:
        metrics = MetricsCalculator(num_classes=2)
        for batch in dataloader:
            predictions, labels, probs = model(batch)
            metrics.update(predictions, labels, probs)
        results = metrics.compute_all()

    Args:
        num_classes (int): Number of classes (default: 2)
        class_names (list): Optional class names (default: ['Real', 'Fake'])
    """

    def __init__(self, num_classes: int = 2,
                 class_names: Optional[list] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ['Real', 'Fake']

        self.reset()

    def reset(self):
        """Reset all accumulated data."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None):
        """
        Update with a batch of predictions.

        Args:
            predictions: Predicted class indices (B,)
            labels: Ground truth labels (B,)
            probabilities: Class probabilities (B, C)
        """
        self.all_predictions.extend(
            predictions.cpu().detach().numpy().tolist()
        )
        self.all_labels.extend(
            labels.cpu().detach().numpy().tolist()
        )

        if probabilities is not None:
            self.all_probabilities.extend(
                probabilities.cpu().detach().numpy().tolist()
            )

    def accuracy(self) -> float:
        """Compute overall accuracy."""
        if not self.all_predictions:
            return 0.0
        correct = sum(
            p == l for p, l in zip(self.all_predictions, self.all_labels)
        )
        return correct / len(self.all_predictions)

    def compute_all(self) -> Dict:
        """
        Compute all metrics.

        Returns:
            metrics: Dict containing:
                - accuracy: Overall accuracy
                - precision: Weighted precision
                - recall: Weighted recall
                - f1: Weighted F1 score
                - auc: AUC-ROC score
                - confusion_matrix: Confusion matrix
                - per_class: Per-class metrics
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        if len(predictions) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'auc': 0.0,
            }

        # Basic accuracy
        accuracy = (predictions == labels).mean()

        # Per-class metrics
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            tp[c] = ((predictions == c) & (labels == c)).sum()
            fp[c] = ((predictions == c) & (labels != c)).sum()
            fn[c] = ((predictions != c) & (labels == c)).sum()

        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall), 0
        )

        # Weighted averages
        class_counts = np.bincount(labels.astype(int),
                                    minlength=self.num_classes)
        total = class_counts.sum()
        weights = class_counts / total if total > 0 else np.ones(
            self.num_classes) / self.num_classes

        weighted_precision = (precision * weights).sum()
        weighted_recall = (recall * weights).sum()
        weighted_f1 = (f1 * weights).sum()

        results = {
            'accuracy': float(accuracy),
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1),
            'auc': 0.0,
        }

        # AUC-ROC
        if self.all_probabilities and HAS_SKLEARN:
            try:
                probs = np.array(self.all_probabilities)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    results['auc'] = float(
                        roc_auc_score(labels, probs[:, 1])
                    )
                elif probs.ndim == 1:
                    results['auc'] = float(
                        roc_auc_score(labels, probs)
                    )
            except (ValueError, IndexError):
                pass

        # Confusion matrix
        if HAS_SKLEARN:
            results['confusion_matrix'] = confusion_matrix(
                labels, predictions
            ).tolist()

        # Per-class metrics
        results['per_class'] = {}
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f'Class_{c}'
            results['per_class'][name] = {
                'precision': float(precision[c]),
                'recall': float(recall[c]),
                'f1': float(f1[c]),
                'support': int(class_counts[c]),
            }

        return results

    def get_roc_curve(self):
        """Get ROC curve data for plotting."""
        if not self.all_probabilities or not HAS_SKLEARN:
            return None, None, None

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)

        if probs.ndim == 2:
            probs = probs[:, 1]

        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            return fpr, tpr, thresholds
        except ValueError:
            return None, None, None

    def get_precision_recall_curve(self):
        """Get precision-recall curve data for plotting."""
        if not self.all_probabilities or not HAS_SKLEARN:
            return None, None, None

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)

        if probs.ndim == 2:
            probs = probs[:, 1]

        try:
            precision, recall, thresholds = precision_recall_curve(
                labels, probs
            )
            ap = average_precision_score(labels, probs)
            return precision, recall, ap
        except ValueError:
            return None, None, None

    def summary(self) -> str:
        """Get a formatted summary string."""
        metrics = self.compute_all()

        lines = [
            f"\n{'='*50}",
            f"📊 Evaluation Results",
            f"{'='*50}",
            f"  Accuracy:  {metrics['accuracy']:.4f}",
            f"  Precision: {metrics['precision']:.4f}",
            f"  Recall:    {metrics['recall']:.4f}",
            f"  F1 Score:  {metrics['f1']:.4f}",
            f"  AUC-ROC:   {metrics['auc']:.4f}",
        ]

        if 'per_class' in metrics:
            lines.append(f"\n  Per-Class Metrics:")
            for name, class_metrics in metrics['per_class'].items():
                lines.append(
                    f"    {name}: P={class_metrics['precision']:.3f} "
                    f"R={class_metrics['recall']:.3f} "
                    f"F1={class_metrics['f1']:.3f} "
                    f"(n={class_metrics['support']})"
                )

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            lines.append(f"\n  Confusion Matrix:")
            lines.append(f"                Predicted")
            lines.append(f"                Real   Fake")
            lines.append(f"    Actual Real  {cm[0][0]:5d}  {cm[0][1]:5d}")
            lines.append(f"    Actual Fake  {cm[1][0]:5d}  {cm[1][1]:5d}")

        lines.append(f"{'='*50}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    calc = MetricsCalculator(num_classes=2)

    # Simulate some predictions
    preds = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 1, 1])
    probs = torch.tensor([
        [0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.8, 0.2],
        [0.1, 0.9], [0.6, 0.4], [0.3, 0.7], [0.15, 0.85]
    ])

    calc.update(preds, labels, probs)
    print(calc.summary())
