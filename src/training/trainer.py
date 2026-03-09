"""
Training Loop — Full-featured trainer for deepfake detection models.

Features:
- Mixed precision training (AMP)
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warmup
- Early stopping with patience
- Best model checkpointing
- TensorBoard logging
- Progress bars with rich formatting
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from .losses import FocalLoss, CombinedLoss
from .metrics import MetricsCalculator


class Trainer:
    """
    Comprehensive trainer for deepfake detection models.

    Handles the full training loop including validation, checkpointing,
    logging, and early stopping.

    Args:
        model: PyTorch model to train
        config (dict): Training configuration
        device (str): Computing device
        experiment_name (str): Name for logging
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: str = 'cuda', experiment_name: str = 'deepfake'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_name = experiment_name

        # Training settings
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 50)
        self.grad_accumulation = train_cfg.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.use_amp = train_cfg.get('mixed_precision', True) and \
                       device != 'cpu'

        # Setup optimizer
        self.optimizer = self._build_optimizer(train_cfg.get('optimizer', {}))

        # Setup scheduler
        self.scheduler = self._build_scheduler(train_cfg.get('scheduler', {}))

        # Setup loss
        self.criterion = self._build_loss(train_cfg.get('loss', {}))

        # Setup AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Metrics calculator
        self.metrics = MetricsCalculator(num_classes=2)

        # Early stopping
        es_cfg = train_cfg.get('early_stopping', {})
        self.early_stopping_enabled = es_cfg.get('enabled', True)
        self.patience = es_cfg.get('patience', 10)
        self.min_delta = es_cfg.get('min_delta', 0.001)
        self.monitor_metric = es_cfg.get('monitor', 'val_auc')
        self.best_metric = 0.0
        self.patience_counter = 0

        # Checkpointing
        ckpt_cfg = train_cfg.get('checkpoint', {})
        self.save_dir = Path(ckpt_cfg.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = ckpt_cfg.get('save_best_only', True)
        self.save_frequency = ckpt_cfg.get('save_frequency', 5)

        # Logging
        log_cfg = config.get('logging', {})
        self.log_interval = log_cfg.get('log_interval', 10)
        self.tensorboard = None
        if log_cfg.get('tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(log_cfg.get('log_dir', './logs'))
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard = SummaryWriter(
                    log_dir / experiment_name
                )
            except ImportError:
                print("⚠ TensorBoard not available")

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': [], 'lr': [],
        }

    def _build_optimizer(self, opt_cfg: dict) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_type = opt_cfg.get('type', 'adamw').lower()
        lr = opt_cfg.get('learning_rate', 1e-4)
        weight_decay = opt_cfg.get('weight_decay', 0.01)
        betas = tuple(opt_cfg.get('betas', [0.9, 0.999]))

        if opt_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr,
                weight_decay=weight_decay, betas=betas
            )
        elif opt_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, betas=betas
            )
        elif opt_type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(), lr=lr,
                weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _build_scheduler(self, sched_cfg: dict):
        """Build learning rate scheduler from config."""
        sched_type = sched_cfg.get('type', 'cosine_annealing_warm_restarts')

        if sched_type == 'cosine_annealing_warm_restarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_cfg.get('T_0', 10),
                T_mult=sched_cfg.get('T_mult', 2),
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
        elif sched_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get('step_size', 10),
                gamma=sched_cfg.get('gamma', 0.1),
            )
        elif sched_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max', factor=0.5, patience=5,
            )
        else:
            return None

    def _build_loss(self, loss_cfg: dict) -> nn.Module:
        """Build loss function from config."""
        loss_type = loss_cfg.get('type', 'focal').lower()

        if loss_type == 'focal':
            return FocalLoss(
                alpha=loss_cfg.get('focal_alpha', 0.25),
                gamma=loss_cfg.get('focal_gamma', 2.0),
                label_smoothing=loss_cfg.get('label_smoothing', 0.1),
            )
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(
                label_smoothing=loss_cfg.get('label_smoothing', 0.1)
            )
        elif loss_type == 'combined':
            return CombinedLoss(
                focal_weight=0.7,
                ce_weight=0.3,
                label_smoothing=loss_cfg.get('label_smoothing', 0.1),
            )
        else:
            return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """
        Run one training epoch.

        Returns:
            metrics: Dict with loss, accuracy, etc.
        """
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                    leave=True, ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accumulation

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accumulation == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item() * self.grad_accumulation
            num_batches += 1

            predictions = torch.argmax(outputs, dim=1)
            self.metrics.update(predictions, labels,
                                torch.softmax(outputs, dim=1))

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}',
                'acc': f'{self.metrics.accuracy():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })

            # TensorBoard logging
            if self.tensorboard and batch_idx % self.log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.tensorboard.add_scalar(
                    'train/batch_loss', loss.item(), global_step
                )

        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_metrics = self.metrics.compute_all()
        epoch_metrics['loss'] = epoch_loss

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Run validation.

        Returns:
            metrics: Dict with loss, accuracy, AUC, etc.
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc="Validating", leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(outputs, dim=1)
            self.metrics.update(predictions, labels,
                                torch.softmax(outputs, dim=1))

        epoch_loss = total_loss / max(num_batches, 1)
        epoch_metrics = self.metrics.compute_all()
        epoch_metrics['loss'] = epoch_loss

        return epoch_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            history: Training history dict
        """
        print(f"\n{'='*60}")
        print(f"🛡️  Deepfake Detection — Training Started")
        print(f"{'='*60}")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        total_params = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        print(f"  Trainable Parameters: {total_params:,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self.validate(val_loader)

            # Update scheduler
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('auc', 0))
            elif self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            self.history['lr'].append(
                self.optimizer.param_groups[0]['lr']
            )

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{self.epochs} "
                  f"({epoch_time:.1f}s)")
            print(f"  Train — Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   — Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"AUC: {val_metrics.get('auc', 0):.4f} | "
                  f"F1: {val_metrics.get('f1', 0):.4f}")

            # TensorBoard
            if self.tensorboard:
                self.tensorboard.add_scalars('loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss'],
                }, epoch)
                self.tensorboard.add_scalars('accuracy', {
                    'train': train_metrics['accuracy'],
                    'val': val_metrics['accuracy'],
                }, epoch)
                self.tensorboard.add_scalar(
                    'val/auc', val_metrics.get('auc', 0), epoch
                )

            # Checkpointing
            current_metric = val_metrics.get(
                self.monitor_metric.replace('val_', ''), 0
            )

            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✅ Best model saved! "
                      f"{self.monitor_metric}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                if not self.save_best_only and \
                        (epoch + 1) % self.save_frequency == 0:
                    self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if self.early_stopping_enabled and \
                    self.patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} "
                      f"epochs (patience: {self.patience})")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete — {total_time / 60:.1f} minutes")
        print(f"   Best {self.monitor_metric}: {self.best_metric:.4f}")
        print(f"{'='*60}")

        if self.tensorboard:
            self.tensorboard.close()

        return self.history

    def _save_checkpoint(self, epoch: int, metrics: Dict,
                         is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch{epoch + 1}.pth'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', 0)
        print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch'] + 1}")
        print(f"   Best metric: {self.best_metric:.4f}")

        return checkpoint.get('epoch', 0)


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
    print("Features: AMP, gradient accumulation, early stopping, "
          "TensorBoard, cosine annealing")
