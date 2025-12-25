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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# =============================================================================
# Optuna Configuration
# =============================================================================

@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter optimization."""
    n_trials: int = 50
    study_name: str = "protein_mlp_optimization"
    storage: str | None = None  # e.g., "sqlite:///optuna_study.db"
    pruning: bool = True
    # Search spaces
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
    top_k_labels: int | None = None  # None = optimize, int = fixed value
    batch_size_choices: list[int] = field(default_factory=lambda: [128, 256, 512])
    learning_rate_range: tuple[float, float] = (1e-5, 1e-2)
    weight_decay_range: tuple[float, float] = (1e-6, 1e-3)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings dataset - easy to swap."""
    train_embeddings_path: str = "data/train/embeddings.npy"
    train_ids_path: str = "data/train/proteins.pkl"
    val_embeddings_path: str = "data/val/embeddings.npy"
    val_ids_path: str = "data/val/proteins.pkl"
    # Embedding pooling selection: "all", "mean", "max", "min"
    # The embeddings are concatenated as [mean, max, min] poolings (each 1/3 of dims)
    pooling: str = "all"


@dataclass
class ModelConfig:
    """Configuration for the MLP architecture."""
    # Hidden layer sizes (empty list = pure logistic regression)
    hidden_layers: list[int] = field(default_factory=lambda: [1024, 512])
    dropout: float = 0.3
    activation: str = "relu"  # "relu", "gelu", "silu"
    batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    top_k_labels: int = 1000  # Number of most frequent GO terms to predict
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    seed: int = 42
    num_workers: int = 4


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
# Model
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
        config: ModelConfig,
    ):
        super().__init__()

        # Build activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_fn = activations.get(config.activation, nn.ReLU)

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Output layer (logistic regression layer)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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

    val_embeddings = np.load(config.val_embeddings_path)
    with open(config.val_ids_path, "rb") as f:
        val_ids = pickle.load(f)

    # Apply pooling selection
    train_embeddings = slice_embeddings_by_pooling(train_embeddings, config.pooling)
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for embeddings, labels in tqdm(dataloader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for embeddings, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * embeddings.size(0)
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    # Threshold at 0.5 for binary predictions
    binary_preds = (all_preds > 0.5).astype(int)

    # Micro-averaged metrics (treat all predictions as one pool)
    tp = (binary_preds * all_labels).sum()
    fp = (binary_preds * (1 - all_labels)).sum()
    fn = ((1 - binary_preds) * all_labels).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, {"precision": precision, "recall": recall, "f1": f1}


def train(
    embedding_config: EmbeddingConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
):
    """Main training function."""
    # Set seeds
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading embeddings...")
    train_embeddings, train_ids, val_embeddings, val_ids = load_embeddings(
        embedding_config
    )
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Val embeddings shape: {val_embeddings.shape}")

    print("Loading labels...")
    train_labels, mlb, top_terms = load_labels(
        train_ids,
        training_config.top_k_labels,
    )
    val_labels, _, _ = load_labels(
        val_ids,
        training_config.top_k_labels,
    )
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val labels shape: {val_labels.shape}")

    train_dataset = ProteinDataset(train_embeddings, train_labels)
    val_dataset = ProteinDataset(val_embeddings, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
    )

    # Model
    input_dim = train_embeddings.shape[1]
    output_dim = len(top_terms)
    model = MLPClassifier(input_dim, output_dim, model_config).to(device)
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
    )

    # Training loop
    best_f1 = 0
    for epoch in range(training_config.epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "embedding_config": embedding_config,
                "top_terms": top_terms,
            }
            torch.save(checkpoint, "best_model.pt")
            print("  -> Saved best model")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")

    return model, mlb, top_terms


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(
    embedding_config: EmbeddingConfig,
    optuna_config: OptunaConfig,
    epochs: int = 20,
    seed: int = 42,
    num_workers: int = 4,
):
    """Create an Optuna objective function with pre-loaded embeddings."""
    # Pre-load embeddings once (they don't change between trials)
    print("Pre-loading embeddings for Optuna study...")
    train_embeddings_full = np.load(embedding_config.train_embeddings_path)
    with open(embedding_config.train_ids_path, "rb") as f:
        train_ids = pickle.load(f)
    val_embeddings_full = np.load(embedding_config.val_embeddings_path)
    with open(embedding_config.val_ids_path, "rb") as f:
        val_ids = pickle.load(f)

    # Apply pooling
    train_embeddings = slice_embeddings_by_pooling(train_embeddings_full, embedding_config.pooling)
    val_embeddings = slice_embeddings_by_pooling(val_embeddings_full, embedding_config.pooling)
    print(f"Embeddings loaded: train {train_embeddings.shape}, val {val_embeddings.shape}")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function - returns validation F1 score."""
        # Sample hyperparameters
        hidden_layers = trial.suggest_categorical(
            "hidden_layers",
            [str(h) for h in optuna_config.hidden_layer_choices]
        )
        hidden_layers = eval(hidden_layers)  # Convert string back to list

        dropout = trial.suggest_float(
            "dropout",
            optuna_config.dropout_range[0],
            optuna_config.dropout_range[1],
        )
        activation = trial.suggest_categorical(
            "activation",
            optuna_config.activation_choices,
        )
        if optuna_config.top_k_labels is not None:
            top_k_labels = optuna_config.top_k_labels
        else:
            top_k_labels = trial.suggest_int("top_k_labels", 500, 2000, step=100)
        batch_size = trial.suggest_categorical(
            "batch_size",
            optuna_config.batch_size_choices,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            optuna_config.learning_rate_range[0],
            optuna_config.learning_rate_range[1],
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            optuna_config.weight_decay_range[0],
            optuna_config.weight_decay_range[1],
            log=True,
        )

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load labels with sampled top_k
        train_labels, _, top_terms = load_labels(train_ids, top_k_labels)
        val_labels, _, _ = load_labels(val_ids, top_k_labels)

        # Create datasets and loaders
        train_dataset = ProteinDataset(train_embeddings, train_labels)
        val_dataset = ProteinDataset(val_embeddings, val_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Create model
        model_config = ModelConfig(
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            batch_norm=True,
        )
        input_dim = train_embeddings.shape[1]
        output_dim = len(top_terms)
        model = MLPClassifier(input_dim, output_dim, model_config).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop with pruning
        best_f1 = 0
        for epoch in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            f1 = metrics["f1"]
            if f1 > best_f1:
                best_f1 = f1

            # Report to Optuna for pruning
            trial.report(f1, epoch)

            if optuna_config.pruning and trial.should_prune():
                raise optuna.TrialPruned()

        return best_f1

    return objective


def run_optuna_study(
    embedding_config: EmbeddingConfig,
    optuna_config: OptunaConfig,
    epochs: int = 20,
    seed: int = 42,
):
    """Run Optuna hyperparameter optimization study."""
    objective = create_objective(
        embedding_config,
        optuna_config,
        epochs=epochs,
        seed=seed,
    )

    # Create or load study
    pruner = optuna.pruners.MedianPruner() if optuna_config.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=optuna_config.study_name,
        storage=optuna_config.storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=optuna_config.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 60)
    print("Optuna Study Complete")
    print("=" * 60)
    print(f"Best trial F1: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    return study


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate predictions for embeddings."""
    model.eval()
    dataset = torch.from_numpy(embeddings).float()
    all_preds = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size].to(device)
        outputs = model(batch)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)

    return np.concatenate(all_preds, axis=0)


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

    embedding_config = EmbeddingConfig()

    if args.optuna:
        # Run Optuna hyperparameter optimization
        optuna_config = OptunaConfig(
            n_trials=args.n_trials,
            pruning=True,
            top_k_labels=1000,  # Fixed; set to None to optimize
        )
        study = run_optuna_study(
            embedding_config,
            optuna_config,
            epochs=args.epochs,
        )
    else:
        # Standard training with default hyperparameters
        model_config = ModelConfig(
            hidden_layers=[1024, 512],
            dropout=0.3,
            activation="relu",
            batch_norm=True,
        )

        training_config = TrainingConfig(
            top_k_labels=1000,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-5,
            epochs=args.epochs,
            seed=42,
        )

        model, mlb, top_terms = train(
            embedding_config,
            model_config,
            training_config,
        )
