"""
Visualization utilities for training results and model analysis.
"""

from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_training_history(history: Dict[str, list],
                          save_path: str = 'training_history.png',
                          figsize: tuple = (16, 10)):
    """
    Plot training history (loss, accuracy, AUC, LR).

    Args:
        history: Dict with train_loss, val_loss, train_acc, val_acc, val_auc, lr
        save_path: Path to save the figure
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_title('Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax.set_title('Accuracy', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC-ROC
    ax = axes[1, 0]
    if 'val_auc' in history:
        ax.plot(epochs, history['val_auc'], 'g-', label='Validation AUC',
                linewidth=2)
    ax.set_title('AUC-ROC', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'm-', label='Learning Rate',
                linewidth=2)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training history saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str] = None,
                          save_path: str = 'confusion_matrix.png',
                          normalize: bool = True,
                          figsize: tuple = (8, 6)):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (2D numpy array)
        class_names: Class labels
        save_path: Path to save the figure
        normalize: Normalize values
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib not available")
        return

    if class_names is None:
        class_names = ['Real', 'Fake']

    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_normalized = cm.astype(float)

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%' if normalize else 'd',
            cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=ax,
            square=True, linewidths=0.5,
            annot_kws={'size': 14}
        )
    else:
        im = ax.imshow(cm_normalized, cmap='Blues')
        plt.colorbar(im)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_normalized[i, j]
                text = f'{val:.2%}' if normalize else f'{int(val)}'
                ax.text(j, i, text, ha='center', va='center', fontsize=14)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    if not HAS_SEABORN:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")


def plot_roc_curve(fpr, tpr, auc_score: float,
                   save_path: str = 'roc_curve.png',
                   figsize: tuple = (8, 6)):
    """Plot ROC curve."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 ROC curve saved to {save_path}")


def plot_model_comparison(model_results: Dict[str, Dict],
                          save_path: str = 'model_comparison.png',
                          figsize: tuple = (12, 6)):
    """
    Plot comparison of different model architectures.

    Args:
        model_results: Dict mapping model name to metrics dict
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    models = list(model_results.keys())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model_name, results) in enumerate(model_results.items()):
        values = [results.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=colors[i], edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Model comparison saved to {save_path}")
