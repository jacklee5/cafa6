"""
Weights & Biases integration for training monitoring.

Provides WandbCallback for logging metrics during model training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from models.Model import TrainingCallback


class WandbCallback:
    """
    Training callback that logs metrics to Weights & Biases.

    Implements the TrainingCallback protocol for use with Model.train().

    Example usage:
        callback = WandbCallback(
            project="cafa6",
            config=model_config,
            name="train-run-1",
        )
        model.train(train_dataset, val_dataset, callback=callback)
        callback.finish()
    """

    def __init__(
        self,
        project: str = "cafa6",
        config: dict | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ):
        """
        Initialize the callback and start a W&B run.

        Args:
            project: W&B project name
            config: Hyperparameters and configuration to log
            name: Run name (optional, W&B will generate one if not provided)
            group: Group name for organizing related runs (e.g., Optuna study)
            tags: List of tags for the run
        """
        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            group=group,
            tags=tags,
            reinit=True,
        )
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """
        Log metrics to W&B after each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dict of metrics from this epoch

        Returns:
            True to continue training
        """
        # Log all metrics
        wandb.log(metrics, step=epoch)

        # Track best F1
        f1 = metrics.get("f1", 0.0)
        if f1 > self.best_f1:
            self.best_f1 = f1
            wandb.run.summary["best_f1"] = self.best_f1

        return True

    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
