"""
ESMModel: Multi-label protein function prediction using MLP on ESM embeddings.

Uses PyTorch with GPU acceleration and BCEWithLogitsLoss for multi-label classification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .Model import Model, TrainingCallback

if TYPE_CHECKING:
    import optuna


# =============================================================================
# Search Space for Hyperparameter Optimization
# =============================================================================

@dataclass
class ESMSearchSpace:
    """Search space for ESMModel hyperparameters."""

    hidden_layer_choices: list[list[int]] = field(default_factory=lambda: [
        [],  # logistic regression
        [512],
        [1024],
        [512, 256],
        [1024, 512],
        [1024, 512, 256],
    ])
    dropout_range: tuple[float, float] = (0.1, 0.5)
    activation_choices: list[str] = field(default_factory=lambda: ["relu", "gelu", "silu"])
    batch_size_choices: list[int] = field(default_factory=lambda: [128, 256, 512])
    learning_rate_range: tuple[float, float] = (1e-5, 1e-2)
    weight_decay_range: tuple[float, float] = (1e-6, 1e-3)
    batch_norm: bool = True


# =============================================================================
# Optuna Mixin
# =============================================================================

class OptunaMixin:
    """Mixin class providing Optuna hyperparameter optimization support."""

    @classmethod
    def from_trial(
        cls,
        trial: "optuna.Trial",
        search_space: ESMSearchSpace,
        epochs: int = 20,
        seed: int = 42,
        num_workers: int = 4,
    ) -> "ESMModel":
        """
        Create an ESMModel with hyperparameters sampled from an Optuna trial.

        Args:
            trial: Optuna trial object for sampling hyperparameters
            search_space: ESMSearchSpace defining the search ranges
            epochs: Number of training epochs
            seed: Random seed
            num_workers: Number of data loader workers

        Returns:
            ESMModel instance with sampled hyperparameters
        """
        # Sample hyperparameters
        hidden_layers = trial.suggest_categorical(
            "hidden_layers",
            [str(h) for h in search_space.hidden_layer_choices]
        )
        hidden_layers = eval(hidden_layers)  # Convert string back to list

        dropout = trial.suggest_float(
            "dropout",
            search_space.dropout_range[0],
            search_space.dropout_range[1],
        )
        activation = trial.suggest_categorical(
            "activation",
            search_space.activation_choices,
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            search_space.batch_size_choices,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            search_space.learning_rate_range[0],
            search_space.learning_rate_range[1],
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            search_space.weight_decay_range[0],
            search_space.weight_decay_range[1],
            log=True,
        )

        config = {
            "model": {
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "activation": activation,
                "batch_norm": search_space.batch_norm,
            },
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "num_workers": num_workers,
                "seed": seed,
                "use_checkpoint": False,  # Don't checkpoint during HPO
            },
        }

        return cls(config)


# =============================================================================
# Neural Network
# =============================================================================

class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for multi-label classification.

    With empty hidden_layers, this is equivalent to logistic regression.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = True,
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_fn = activations.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim

        layers.append(nn.Dropout(dropout))
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# ESMModel
# =============================================================================

class ESMModel(Model, OptunaMixin):
    """
    Protein function prediction model using ESM embeddings.

    Config structure (nested dict):
    {
        "model": {
            "hidden_layers": [1024],      # List of hidden layer sizes
            "dropout": 0.1,               # Dropout rate
            "activation": "gelu",         # "relu", "gelu", or "silu"
            "batch_norm": True,           # Whether to use batch normalization
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "epochs": 20,
            "num_workers": 4,
            "seed": 42,
            "use_checkpoint": False,     # Save best model checkpoint
            "checkpoint_path": "checkpoints",  # Folder for checkpoints
        },
    }
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Extract config sections with defaults
        model_config = config.get("model", {})
        training_config = config.get("training", {})

        # Model architecture config
        self.hidden_layers = model_config.get("hidden_layers", [1024])
        self.dropout = model_config.get("dropout", 0.1)
        self.activation = model_config.get("activation", "gelu")
        self.batch_norm = model_config.get("batch_norm", True)

        # Training config
        self.batch_size = training_config.get("batch_size", 128)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 1e-5)
        self.epochs = training_config.get("epochs", 20)
        self.num_workers = training_config.get("num_workers", 4)
        self.seed = training_config.get("seed", 42)
        self.use_checkpoint = training_config.get("use_checkpoint", False)
        self.checkpoint_path = training_config.get("checkpoint_path", "checkpoints")

        # Store full config for checkpointing
        self._config = config

        # Will be set during train() or load_from_checkpoint()
        self._model: MLPClassifier | None = None
        self._device: torch.device | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None

        # Training history
        self.history: list[dict[str, Any]] = []

    def _build_model(self, input_dim: int, output_dim: int) -> MLPClassifier:
        """Build the MLP classifier with current config."""
        self._input_dim = input_dim
        self._output_dim = output_dim
        return MLPClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

    def _setup_device(self) -> torch.device:
        """Setup and return the compute device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        return device

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        callback: TrainingCallback | None = None,
    ) -> "ESMModel":
        """
        Train the model on the provided dataset.

        Args:
            train_dataset: Training dataset with (embeddings, labels) items
            val_dataset: Optional validation dataset
            callback: Optional callback for epoch-level hooks (e.g., Optuna pruning)

        Returns:
            self for method chaining
        """
        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = self._setup_device()
        print(f"Using device: {device}")

        # Get dimensions from first sample
        sample_embedding, sample_label = train_dataset[0]
        input_dim = sample_embedding.shape[0]
        output_dim = sample_label.shape[0]

        # Build model
        self._model = self._build_model(input_dim, output_dim)
        self._model = self._model.to(device)

        print(f"Model architecture: {self.hidden_layers}")
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"Total parameters: {sum(p.numel() for p in self._model.parameters()):,}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
        )

        # Training loop
        self.history = []
        best_f1 = 0.0

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train epoch
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            scheduler.step()
            print(f"  Train Loss: {train_loss:.4f}")

            epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss}

            # Validation
            if val_loader is not None:
                val_loss, metrics = self._evaluate(val_loader, criterion)
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

                epoch_metrics.update({
                    "val_loss": val_loss,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                })

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    epoch_metrics["is_best"] = True
                    print("  -> New best F1")
                    if self.use_checkpoint:
                        self.save_checkpoint(self.checkpoint_path)

            self.history.append(epoch_metrics)

            # Callback for external control (e.g., Optuna pruning)
            if callback is not None:
                callback.on_epoch_end(epoch, epoch_metrics)

        if val_loader is not None:
            print(f"\nTraining complete. Best F1: {best_f1:.4f}")
        else:
            print("\nTraining complete (no validation set)")

        return self

    def _train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train for one epoch and return average loss."""
        self._model.train()
        total_loss = 0.0

        for embeddings, labels in tqdm(dataloader, desc="Training", leave=False):
            embeddings = embeddings.to(self._device)
            labels = labels.to(self._device)

            optimizer.zero_grad()
            outputs = self._model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embeddings.size(0)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def _evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate on a dataloader and return loss and metrics."""
        self._model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for embeddings, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            embeddings = embeddings.to(self._device)
            labels = labels.to(self._device)

            outputs = self._model(embeddings)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * embeddings.size(0)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate metrics
        binary_preds = (all_preds > 0.5).astype(int)

        tp = (binary_preds * all_labels).sum()
        fp = (binary_preds * (1 - all_labels)).sum()
        fn = ((1 - binary_preds) * all_labels).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return avg_loss, {"precision": precision, "recall": recall, "f1": f1}

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Generate predictions for embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Array of shape (n_samples, n_labels) with probabilities
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first or load from checkpoint.")

        self._model.eval()
        dataset = torch.from_numpy(embeddings).float()
        all_preds = []

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Predicting"):
            batch = dataset[i : i + self.batch_size].to(self._device)
            outputs = self._model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)

        return np.concatenate(all_preds, axis=0)

    def save_checkpoint(self, folder: str, extra_data: dict | None = None) -> str:
        """
        Save model checkpoint to a folder with timestamped filename.

        Args:
            folder: Directory to save the checkpoint
            extra_data: Optional extra data to include (e.g., go_terms, mlb)

        Returns:
            Path to the saved checkpoint file
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Nothing to save.")

        # Create folder if it doesn't exist
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"esm_{timestamp}.pt"
        filepath = folder_path / filename

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self._config,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "history": self.history,
        }

        if extra_data:
            checkpoint.update(extra_data)

        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")

        return str(filepath)

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> "ESMModel":
        """
        Load a model from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            ESMModel instance with loaded weights
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create model instance from saved config
        config = checkpoint["config"]
        model = ESMModel(config)

        # Build and load the network
        input_dim = checkpoint["input_dim"]
        output_dim = checkpoint["output_dim"]

        # Set dropout to 0 for inference
        model._model = MLPClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=model.hidden_layers,
            dropout=0.0,  # No dropout during inference
            activation=model.activation,
            batch_norm=model.batch_norm,
        )
        model._model.load_state_dict(checkpoint["model_state_dict"])
        model._model = model._model.to(device)
        model._model.eval()

        model._device = device
        model._input_dim = input_dim
        model._output_dim = output_dim
        model.history = checkpoint.get("history", [])

        print(f"Loaded model from {checkpoint_path}")
        print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"  Architecture: {model.hidden_layers}")

        return model
