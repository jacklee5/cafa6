"""
Multi-label protein function prediction using MLP on protein embeddings.
Uses PyTorch with GPU acceleration and BCEWithLogitsLoss for 1vsRest classification.
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from models import ESMModel
from models.ESMModel import ESMSearchSpace
from utils.optuna import OptunaOptimizer, OptunaStudyConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings dataset - easy to swap."""
    train_embeddings_path: str = "input/esm1b-embeddings/train_esm1b_embeddings.npy"
    train_ids_path: str = "input/esm1b-embeddings/train_ids.pkl"
    val_embeddings_path: str | None = None  # None = no validation (train on all data)
    val_ids_path: str | None = None
    # Embedding pooling selection: "all", "mean", "max", "min"
    # The embeddings are concatenated as [mean, max, min] poolings (each 1/3 of dims)
    pooling: str = "mean"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    top_k_labels: int = 1000  # Number of most frequent GO terms to predict
    batch_size: int = 128
    learning_rate: float = 0.002320547782435886
    weight_decay: float = 4.056106044748933e-05
    epochs: int = 20
    seed: int = 42
    num_workers: int = 4
    # Model architecture
    hidden_layers: list[int] = field(default_factory=lambda: [1024])
    dropout: float = 0.1006961784393873
    activation: str = "gelu"
    batch_norm: bool = True
    # Checkpointing
    use_checkpoint: bool = True
    checkpoint_path: str = "checkpoints"


# =============================================================================
# Dataset
# =============================================================================

class ProteinDataset(Dataset):
    """PyTorch Dataset for protein embeddings and GO term labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# =============================================================================
# Data Loading
# =============================================================================

def slice_embeddings_by_pooling(embeddings: np.ndarray, pooling: str) -> np.ndarray:
    """
    Slice embeddings to select specific pooling type.

    The embeddings are concatenated as [mean, max, min] poolings,
    each comprising 1/3 of the total dimensions.

    Args:
        embeddings: Full embeddings array of shape (n_samples, full_dim)
        pooling: One of "all", "mean", "max", "min"

    Returns:
        Sliced embeddings array
    """
    if pooling == "all":
        return embeddings

    full_dim = embeddings.shape[1]
    third = full_dim // 3

    pooling_slices = {
        "mean": (0, third),              # First third
        "max": (third, 2 * third),       # Second third
        "min": (2 * third, full_dim),    # Last third
    }

    if pooling not in pooling_slices:
        raise ValueError(f"Unknown pooling '{pooling}'. Must be one of: all, mean, max, min")

    start, end = pooling_slices[pooling]
    return embeddings[:, start:end]


def load_embeddings(config: EmbeddingConfig):
    """Load embeddings and IDs from files, with optional pooling selection."""
    train_embeddings = np.load(config.train_embeddings_path)
    with open(config.train_ids_path, "rb") as f:
        train_ids = pickle.load(f)

    # Apply pooling selection
    train_embeddings = slice_embeddings_by_pooling(train_embeddings, config.pooling)

    # Load validation embeddings if paths provided
    val_embeddings = None
    val_ids = None
    if config.val_embeddings_path and config.val_ids_path:
        val_embeddings = np.load(config.val_embeddings_path)
        with open(config.val_ids_path, "rb") as f:
            val_ids = pickle.load(f)
        val_embeddings = slice_embeddings_by_pooling(val_embeddings, config.pooling)

    print(f"Pooling mode: {config.pooling} -> embedding dim: {train_embeddings.shape[1]}")

    return train_embeddings, train_ids, val_embeddings, val_ids


def load_labels(
    train_ids: list[str],
    top_k: int,
    terms_path: str = "input/cafa-6-protein-function-prediction/Train/train_terms.tsv",
):
    """
    Load GO term labels and filter to top-k most frequent terms.

    Returns:
        labels: np.ndarray of shape (n_samples, top_k) with binary labels
        mlb: fitted MultiLabelBinarizer
        top_terms: list of top-k GO term IDs
    """
    # Load terms
    terms_df = pd.read_csv(terms_path, sep="\t")

    # Get top-k most frequent terms
    term_counts = terms_df["term"].value_counts()
    if not top_k:
        top_terms = term_counts.index.tolist()
    else:
        top_terms = term_counts.head(top_k).index.tolist()

    print(f"Total unique terms: {len(term_counts)}")
    print(f"Using top {top_k} terms (min count: {term_counts.iloc[top_k-1]})")

    # Filter to top terms only
    terms_df = terms_df[terms_df["term"].isin(top_terms)]

    # Group by protein ID
    protein_terms = terms_df.groupby("EntryID")["term"].apply(list).to_dict()

    # Create label matrix aligned with train_ids
    mlb = MultiLabelBinarizer(classes=top_terms)
    mlb.fit([top_terms])  # Ensure all classes are present

    # Build labels in order of train_ids
    labels_list = [protein_terms.get(pid, []) for pid in train_ids]
    labels = mlb.transform(labels_list)

    return labels, mlb, top_terms


# =============================================================================
# Training
# =============================================================================

def train(
    embedding_config: EmbeddingConfig,
    training_config: TrainingConfig,
):
    """Main training function using ESMModel."""
    # Load data
    print("Loading embeddings...")
    train_embeddings, train_ids, val_embeddings, val_ids = load_embeddings(
        embedding_config
    )
    print(f"Train embeddings shape: {train_embeddings.shape}")
    has_validation = val_embeddings is not None
    if has_validation:
        print(f"Val embeddings shape: {val_embeddings.shape}")
    else:
        print("No validation set - training on all data")

    print("Loading labels...")
    train_labels, mlb, top_terms = load_labels(
        train_ids,
        training_config.top_k_labels,
    )
    print(f"Train labels shape: {train_labels.shape}")

    # Create datasets
    train_dataset = ProteinDataset(train_embeddings, train_labels)

    val_dataset = None
    if has_validation:
        val_labels, _, _ = load_labels(
            val_ids,
            training_config.top_k_labels,
        )
        print(f"Val labels shape: {val_labels.shape}")
        val_dataset = ProteinDataset(val_embeddings, val_labels)

    # Create ESMModel config
    config = {
        "model": {
            "hidden_layers": training_config.hidden_layers,
            "dropout": training_config.dropout,
            "activation": training_config.activation,
            "batch_norm": training_config.batch_norm,
        },
        "training": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "epochs": training_config.epochs,
            "num_workers": training_config.num_workers,
            "seed": training_config.seed,
            "use_checkpoint": training_config.use_checkpoint,
            "checkpoint_path": training_config.checkpoint_path,
        },
    }

    # Create and train model
    model = ESMModel(config)
    model.train(train_dataset, val_dataset)

    # Save final model if no validation (train on all data for submission)
    if not has_validation:
        model.save_checkpoint(
            training_config.checkpoint_path,
            extra_data={
                "top_terms": top_terms,
                "embedding_config": embedding_config,
            }
        )
        print("Saved final model (trained on all data)")

    return model, mlb, top_terms


# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================

def run_optuna_optimization(
    embedding_config: EmbeddingConfig,
    search_space: ESMSearchSpace,
    study_config: OptunaStudyConfig,
    top_k_labels: int = 1000,
    epochs: int = 20,
    seed: int = 42,
    num_workers: int = 4,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        embedding_config: Configuration for loading embeddings
        search_space: ESMSearchSpace defining hyperparameter search ranges
        study_config: OptunaStudyConfig for study settings
        top_k_labels: Number of GO terms to predict (fixed, not optimized)
        epochs: Number of training epochs per trial
        seed: Random seed
        num_workers: Number of data loader workers

    Returns:
        Completed Optuna study
    """
    # Pre-load embeddings and labels ONCE (top_k_labels is fixed)
    print("Loading embeddings...")
    train_embeddings, train_ids, val_embeddings, val_ids = load_embeddings(
        embedding_config
    )
    print(f"Train embeddings shape: {train_embeddings.shape}")

    if val_embeddings is None:
        raise ValueError("Validation set required for Optuna optimization")
    print(f"Val embeddings shape: {val_embeddings.shape}")

    print("Loading labels...")
    train_labels, _, top_terms = load_labels(train_ids, top_k_labels)
    val_labels, _, _ = load_labels(val_ids, top_k_labels)
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val labels shape: {val_labels.shape}")

    # Create datasets once
    train_dataset = ProteinDataset(train_embeddings, train_labels)
    val_dataset = ProteinDataset(val_embeddings, val_labels)

    # Create and run optimizer
    optimizer = OptunaOptimizer(ESMModel, search_space, study_config)
    study = optimizer.run_study(
        train_dataset,
        val_dataset,
        epochs=epochs,
        seed=seed,
        num_workers=num_workers,
    )

    return study


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP for protein function prediction")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    embedding_config = EmbeddingConfig(
        train_embeddings_path="data/train/embeddings_esm2.npy",
        train_ids_path="data/train/proteins.pkl",
        val_embeddings_path="data/val/embeddings_esm2.npy",
        val_ids_path="data/val/proteins.pkl",
        pooling="all"
    )

    if args.optuna:
        # Run Optuna hyperparameter optimization
        search_space = ESMSearchSpace(
            hidden_layer_choices=[[], [512], [1024], [512, 256], [1024, 512]],
            dropout_range=(0.1, 0.5),
            activation_choices=["relu", "gelu", "silu"],
            batch_size_choices=[128, 256, 512],
            learning_rate_range=(1e-5, 1e-2),
            weight_decay_range=(1e-6, 1e-3),
        )
        study_config = OptunaStudyConfig(
            n_trials=args.n_trials,
            study_name="protein_mlp_optimization",
            pruning=True,
            metric="f1",
        )
        study = run_optuna_optimization(
            embedding_config,
            search_space,
            study_config,
            top_k_labels=1000,
            epochs=args.epochs,
        )
    else:
        # Standard training with best hyperparameters from Optuna
        training_config = TrainingConfig(
            top_k_labels=1000,
            batch_size=128,
            learning_rate=0.002320547782435886,
            weight_decay=4.056106044748933e-05,
            epochs=args.epochs,
            seed=42,
            hidden_layers=[1024],
            dropout=0.1006961784393873,
            activation="gelu",
            batch_norm=False,
            use_checkpoint=True,
            checkpoint_path="checkpoints",
        )

        model, mlb, top_terms = train(
            embedding_config,
            training_config,
        )
