"""
Run inference on the model generated in lr_esm_train.py.
Generates a submission file in the competition format.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import EmbeddingDataset
from models import ESMModel


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


# =============================================================================
# Submission Generation
# =============================================================================

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
    # Load model from checkpoint
    print(f"Loading model from {config.model_path}...")
    model = ESMModel.load_from_checkpoint(config.model_path)

    # Get extra data saved with checkpoint (top_terms, pooling)
    checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=False)
    top_terms: list[str] = checkpoint["top_terms"]
    pooling = checkpoint.get("pooling", "all")

    print(f"  Pooling mode: {pooling}")
    print(f"  GO terms: {len(top_terms)}")

    # Load test embeddings (no labels for inference)
    print("\nLoading test embeddings...")
    test_dataset = EmbeddingDataset(
        embeddings_path=config.test_embeddings_path,
        ids_path=config.test_ids_path,
        pooling=pooling,
        terms_path=None,  # No labels for inference
    )

    # Get embeddings as numpy array for prediction
    test_embeddings = test_dataset.embeddings.numpy()

    # Run inference
    print("\nRunning inference...")
    predictions = model.predict(test_embeddings)
    print(f"Predictions shape: {predictions.shape}")

    # Generate submission
    print("\nGenerating submission file...")
    generate_submission(
        test_dataset.ids,
        predictions,
        top_terms,
        config.output_path,
        config.threshold,
    )

    print("\nDone!")


if __name__ == "__main__":
    config = InferenceConfig(
        model_path="checkpoints/esm_20241229_120000.pt",  # Update with actual checkpoint
        output_path="submissions/submission.tsv",
        threshold=0.01,
    )

    run_inference(config)
