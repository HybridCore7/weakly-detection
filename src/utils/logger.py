"""
Logging configuration for the deepfake detection system.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = 'deepfake', log_dir: str = './logs',
                 level: str = 'INFO', console: bool = True) -> logging.Logger:
    """
    Setup a configured logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to also log to console

    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_path / f'{name}_{timestamp}.log', encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Colored console output
        try:
            from rich.logging import RichHandler
            rich_handler = RichHandler(
                rich_tracebacks=True, markup=True,
                show_time=True, show_path=False
            )
            logger.addHandler(rich_handler)
        except ImportError:
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


class TrainingLogger:
    """
    Structured training logger that records metrics in a tabular format.

    Args:
        log_dir (str): Directory for log files
        experiment_name (str): Name of the experiment
    """

    def __init__(self, log_dir: str = './logs',
                 experiment_name: str = 'deepfake'):
        self.logger = setup_logger(experiment_name, log_dir)
        self.epoch_metrics = []

    def log_epoch(self, epoch: int, train_metrics: dict,
                  val_metrics: dict, lr: float):
        """Log metrics for one epoch."""
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
            f"Val AUC: {val_metrics.get('auc', 0):.4f} | "
            f"LR: {lr:.6f}"
        )

        self.epoch_metrics.append({
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0),
            'train_acc': train_metrics.get('accuracy', 0),
            'val_loss': val_metrics.get('loss', 0),
            'val_acc': val_metrics.get('accuracy', 0),
            'val_auc': val_metrics.get('auc', 0),
            'lr': lr,
        })

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
