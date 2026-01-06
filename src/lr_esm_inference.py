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
    model_path: str

    # Test embeddings
    test_embeddings_path: str
    test_ids_path: str

    # Output
    output_path: str
    threshold: float # Minimum probability to include prediction


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
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on ESM model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="submissions/submission.tsv",
        help="Output path for submission file (default: submissions/submission.tsv)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.01,
        help="Minimum probability threshold (default: 0.01)",
    )
    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        output_path=args.output,
        threshold=args.threshold,
        test_embeddings_path="data/val/embeddings_esm2.npy",
        test_ids_path="data/val/proteins.pkl",
    )

    run_inference(config)
