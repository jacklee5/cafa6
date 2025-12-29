from __future__ import annotations

from typing import Any, Protocol


class TrainingCallback(Protocol):
    """Protocol for training callbacks (e.g., Optuna pruning)."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """
        Called after each training epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dict of metrics from this epoch (e.g., train_loss, val_loss, f1)

        Returns:
            True to continue training, False to stop early
        """
        ...


class Model:
    """Base class for all models."""

    def __init__(self, config: dict):
        self._config = config

    def train(
        self,
        train_dataset: Any,
        val_dataset: Any | None = None,
        callback: TrainingCallback | None = None,
    ) -> "Model":
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            callback: Optional callback for epoch-level hooks (e.g., Optuna pruning)

        Returns:
            self for method chaining
        """
        raise NotImplementedError

    def predict(self, inputs: Any) -> Any:
        """Generate predictions."""
        raise NotImplementedError

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> Model:
        """Load a model from a checkpoint file."""
        raise NotImplementedError
