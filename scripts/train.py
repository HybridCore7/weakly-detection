"""
Training Script — CLI for training deepfake detection models.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --config config/config.yaml --model xception --epochs 50
    python scripts/train.py --config config/config.yaml --model ensemble --batch-size 8
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch

from src.models import (
    XceptionNet, EfficientNetDetector, DeepfakeAutoencoder,
    ContrastiveDetector, EnsembleDetector
)
from src.data.dataset import DeepfakeDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_training_history


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config: dict, model_name: str = None) -> torch.nn.Module:
    """Build model from configuration."""
    model_cfg = config.get('model', {})
    arch = model_name or model_cfg.get('architecture', 'xception')
    num_classes = model_cfg.get('num_classes', 2)
    dropout = model_cfg.get('dropout', 0.5)
    pretrained = model_cfg.get('pretrained', True)

    print(f"\n🏗️  Building model: {arch}")

    if arch == 'xception':
        xcfg = model_cfg.get('xception', {})
        model = XceptionNet(
            num_classes=num_classes,
            dropout=dropout,
            use_attention=xcfg.get('use_attention', True),
            pretrained=pretrained,
        )
    elif arch == 'efficientnet':
        model = EfficientNetDetector(
            num_classes=num_classes,
            dropout=dropout,
            use_attention=model_cfg.get('efficientnet', {}).get('use_attention', True),
            pretrained=pretrained,
        )
    elif arch == 'autoencoder':
        ae_cfg = model_cfg.get('autoencoder', {})
        model = DeepfakeAutoencoder(
            latent_dim=ae_cfg.get('latent_dim', 512),
            num_classes=num_classes,
        )
    elif arch == 'contrastive':
        ct_cfg = model_cfg.get('contrastive', {})
        model = ContrastiveDetector(
            embedding_dim=ct_cfg.get('embedding_dim', 512),
            projection_dim=ct_cfg.get('projection_dim', 128),
            num_classes=num_classes,
            temperature=ct_cfg.get('temperature', 0.07),
        )
    elif arch == 'ensemble':
        ens_cfg = model_cfg.get('ensemble', {})
        model = EnsembleDetector(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            fusion=ens_cfg.get('fusion', 'attention'),
            models_to_use=ens_cfg.get('models', ['xception', 'contrastive']),
            model_weights=ens_cfg.get('weights', [0.5, 0.5]),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection Model'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model architecture override')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs override')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset directory override')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--experiment', type=str, default='deepfake',
                        help='Experiment name')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['learning_rate'] = args.lr
    if args.data_dir:
        config['data']['dataset_root'] = args.data_dir

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"\n🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Build model
    model = build_model(config, args.model)

    # Setup data
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('image_size', [299, 299])[0]
    batch_size = config['training'].get('batch_size', 16)

    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    data_root = data_cfg.get('dataset_root', './data')

    print(f"\n📂 Loading data from: {data_root}")

    train_dataset = DeepfakeDataset(
        data_root, split='train', transform=train_transforms
    )
    val_dataset = DeepfakeDataset(
        data_root, split='val', transform=val_transforms
    )

    train_loader = DeepfakeDataset.get_dataloader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=data_cfg.get('num_workers', 4)
    )
    val_loader = DeepfakeDataset.get_dataloader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=data_cfg.get('num_workers', 4)
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Setup trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        experiment_name=args.experiment,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Plot training history
    plot_training_history(
        history,
        save_path=str(Path(config['training']['checkpoint']['save_dir']) /
                       'training_history.png')
    )

    print("\n🎉 Training complete!")


if __name__ == '__main__':
    main()
