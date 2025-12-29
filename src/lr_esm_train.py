"""
Multi-label protein function prediction using MLP on protein embeddings.
Uses PyTorch with GPU acceleration and BCEWithLogitsLoss for 1vsRest classification.
"""

from dataclasses import dataclass, field

import optuna

from dataset import EmbeddingDataset
from models import ESMModel
from models.ESMModel import ESMSearchSpace
from utils.optuna import OptunaOptimizer, OptunaStudyConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data paths
    train_embeddings_path: str = "data/train/embeddings_esm2.npy"
    train_ids_path: str = "data/train/proteins.pkl"
    val_embeddings_path: str | None = "data/val/embeddings_esm2.npy"
    val_ids_path: str | None = "data/val/proteins.pkl"
    terms_path: str = "input/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    # Embedding pooling: "all", "mean", "max", "min"
    pooling: str = "all"
    # Labels
    top_k_labels: int = 1000
    # Training hyperparameters
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
# Training
# =============================================================================

def train(config: TrainingConfig):
    """Main training function using ESMModel."""
    # Load training data
    print("Loading training data...")
    train_dataset = EmbeddingDataset(
        embeddings_path=config.train_embeddings_path,
        ids_path=config.train_ids_path,
        pooling=config.pooling,
        top_k=config.top_k_labels,
        terms_path=config.terms_path,
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    # Load validation data if paths provided
    val_dataset = None
    if config.val_embeddings_path and config.val_ids_path:
        print("Loading validation data...")
        val_dataset = EmbeddingDataset(
            embeddings_path=config.val_embeddings_path,
            ids_path=config.val_ids_path,
            pooling=config.pooling,
            top_k=config.top_k_labels,
            terms_path=config.terms_path,
        )
        print(f"Val dataset: {len(val_dataset)} samples")
    else:
        print("No validation set - training on all data")

    # Create ESMModel config
    model_config = {
        "model": {
            "hidden_layers": config.hidden_layers,
            "dropout": config.dropout,
            "activation": config.activation,
            "batch_norm": config.batch_norm,
        },
        "training": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "use_checkpoint": config.use_checkpoint,
            "checkpoint_path": config.checkpoint_path,
        },
    }

    # Create and train model
    model = ESMModel(model_config)
    model.train(train_dataset, val_dataset)

    # Save final model if no validation (train on all data for submission)
    if val_dataset is None:
        model.save_checkpoint(
            config.checkpoint_path,
            extra_data={
                "top_terms": train_dataset.top_terms,
                "pooling": config.pooling,
            }
        )
        print("Saved final model (trained on all data)")

    return model, train_dataset.top_terms


# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================

def run_optuna_optimization(
    config: TrainingConfig,
    search_space: ESMSearchSpace,
    study_config: OptunaStudyConfig,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        config: Training configuration with data paths and fixed settings
        search_space: ESMSearchSpace defining hyperparameter search ranges
        study_config: OptunaStudyConfig for study settings

    Returns:
        Completed Optuna study
    """
    # Load training data
    print("Loading training data...")
    train_dataset = EmbeddingDataset(
        embeddings_path=config.train_embeddings_path,
        ids_path=config.train_ids_path,
        pooling=config.pooling,
        top_k=config.top_k_labels,
        terms_path=config.terms_path,
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    # Load validation data (required for Optuna)
    if not config.val_embeddings_path or not config.val_ids_path:
        raise ValueError("Validation set required for Optuna optimization")

    print("Loading validation data...")
    val_dataset = EmbeddingDataset(
        embeddings_path=config.val_embeddings_path,
        ids_path=config.val_ids_path,
        pooling=config.pooling,
        top_k=config.top_k_labels,
        terms_path=config.terms_path,
    )
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create and run optimizer
    optimizer = OptunaOptimizer(ESMModel, search_space, study_config)
    study = optimizer.run_study(
        train_dataset,
        val_dataset,
        epochs=config.epochs,
        seed=config.seed,
        num_workers=config.num_workers,
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

    config = TrainingConfig(
        train_embeddings_path="data/train/embeddings_esm2.npy",
        train_ids_path="data/train/proteins.pkl",
        val_embeddings_path="data/val/embeddings_esm2.npy",
        val_ids_path="data/val/proteins.pkl",
        pooling="all",
        epochs=args.epochs,
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
        study = run_optuna_optimization(config, search_space, study_config)
    else:
        # Standard training with best hyperparameters from Optuna
        model, top_terms = train(config)
