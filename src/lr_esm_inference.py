"""
Run inference on the model generated in lr_esm_train.py.
Generates a submission file in the competition format.
"""

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lr_esm_train import (
    ModelConfig,
    MLPClassifier,
    slice_embeddings_by_pooling,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Model checkpoint (contains model weights, architecture, and GO terms)
    model_path: str = "best_model.pt"

    # Test embeddings
    test_embeddings_path: str = "input/esm1b-embeddings/test_esm1b_embeddings.npy"
    test_ids_path: str = "input/esm1b-embeddings/test_ids.pkl"

    # Output
    output_path: str = "submission.tsv"
    threshold: float = 0.01  # Minimum probability to include prediction
    batch_size: int = 256


# =============================================================================
# Data Loading
# =============================================================================

def load_test_embeddings(
    embeddings_path: str,
    ids_path: str,
    pooling: str,
) -> tuple[np.ndarray, list[str]]:
    """Load test embeddings and IDs."""
    embeddings = np.load(embeddings_path)
    with open(ids_path, "rb") as f:
        ids = pickle.load(f)

    embeddings = slice_embeddings_by_pooling(embeddings, pooling)
    print(f"Loaded {len(ids)} test proteins with embedding dim {embeddings.shape[1]}")

    return embeddings, ids


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    """
    Load checkpoint containing model weights, config, and GO terms.

    Returns dict with keys:
        - model_state_dict: model weights
        - model_config: ModelConfig used during training
        - embedding_config: EmbeddingConfig used during training
        - top_terms: list of GO term IDs
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate predictions for embeddings."""
    model.eval()
    dataset = torch.from_numpy(embeddings).float()
    all_preds = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        batch = dataset[i : i + batch_size].to(device)
        outputs = model(batch)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)

    return np.concatenate(all_preds, axis=0)


def generate_submission(
    protein_ids: list[str],
    predictions: np.ndarray,
    go_terms: list[str],
    output_path: str,
    threshold: float = 0.01,
) -> None:
    """
    Generate submission file in competition format.

    Args:
        protein_ids: List of protein IDs
        predictions: Array of shape (n_proteins, n_terms) with probabilities
        go_terms: List of GO term IDs corresponding to prediction columns
        output_path: Path to save submission TSV
        threshold: Minimum probability threshold to include prediction
    """
    rows = []

    for i, protein_id in enumerate(tqdm(protein_ids, desc="Generating submission")):
        probs = predictions[i]

        # Get all predictions above threshold
        mask = probs >= threshold
        term_indices = np.where(mask)[0]

        for idx in term_indices:
            rows.append({
                "protein_id": protein_id,
                "go_term": go_terms[idx],
                "confidence": float(probs[idx]),
            })

    df = pd.DataFrame(rows)

    # Sort by protein_id, then by confidence descending
    df = df.sort_values(["protein_id", "confidence"], ascending=[True, False])

    # Save without header (competition format)
    df.to_csv(output_path, sep="\t", index=False, header=False)

    print(f"Saved {len(df)} predictions to {output_path}")
    print(f"Unique proteins: {df['protein_id'].nunique()}")
    print(f"Avg predictions per protein: {len(df) / df['protein_id'].nunique():.1f}")


# =============================================================================
# Main
# =============================================================================

def run_inference(config: InferenceConfig):
    """Main inference function."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint (contains model config and GO terms from training)
    print(f"\nLoading checkpoint from {config.model_path}...")
    checkpoint = load_checkpoint(config.model_path, device)

    model_config: ModelConfig = checkpoint["model_config"]
    embedding_config = checkpoint["embedding_config"]
    top_terms: list[str] = checkpoint["top_terms"]

    print(f"  Model architecture: {model_config.hidden_layers}")
    print(f"  Pooling mode: {embedding_config.pooling}")
    print(f"  GO terms: {len(top_terms)}")

    # Load test embeddings with same pooling as training
    print("\nLoading test embeddings...")
    test_embeddings, test_ids = load_test_embeddings(
        config.test_embeddings_path,
        config.test_ids_path,
        embedding_config.pooling,
    )

    # Create model with same architecture as training
    input_dim = test_embeddings.shape[1]
    output_dim = len(top_terms)

    # Override dropout for inference
    inference_model_config = ModelConfig(
        hidden_layers=model_config.hidden_layers,
        dropout=0.0,
        activation=model_config.activation,
        batch_norm=model_config.batch_norm,
    )

    model = MLPClassifier(input_dim, output_dim, inference_model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run inference
    print("\nRunning inference...")
    predictions = predict(model, test_embeddings, device, config.batch_size)
    print(f"Predictions shape: {predictions.shape}")

    # Generate submission
    print("\nGenerating submission file...")
    generate_submission(
        test_ids,
        predictions,
        top_terms,
        config.output_path,
        config.threshold,
    )

    print("\nDone!")


if __name__ == "__main__":
    config = InferenceConfig(
        model_path="models/best_model.pt",
        output_path="submissions/submission.tsv",
        threshold=0.01,
        batch_size=256,
    )

    run_inference(config)
