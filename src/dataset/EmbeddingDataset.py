"""
EmbeddingDataset: PyTorch Dataset that loads embeddings and labels from files.
"""

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


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


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset that loads embeddings and GO term labels from files.

    Handles:
    - Loading embeddings from .npy file
    - Loading protein IDs from .pkl file
    - Applying pooling selection (mean, max, min, or all)
    - Loading and filtering GO term labels to top-k most frequent
    - Creating binary label matrix

    Example:
        dataset = EmbeddingDataset(
            embeddings_path="data/train/embeddings.npy",
            ids_path="data/train/proteins.pkl",
            pooling="mean",
            top_k=1000,
        )
        embedding, label = dataset[0]
    """

    def __init__(
        self,
        embeddings_path: str,
        ids_path: str,
        pooling: str = "all",
        top_k: int | None = 1000,
        terms_path: str | None = "input/cafa-6-protein-function-prediction/Train/train_terms.tsv",
    ):
        """
        Initialize the dataset by loading embeddings and optionally labels.

        Args:
            embeddings_path: Path to .npy file containing embeddings
            ids_path: Path to .pkl file containing protein IDs
            pooling: Pooling mode - "all", "mean", "max", or "min"
            top_k: Number of most frequent GO terms to predict (None = all terms)
            terms_path: Path to TSV file with GO term annotations (None = inference mode, no labels)
        """
        # Load embeddings
        embeddings = np.load(embeddings_path)
        embeddings = slice_embeddings_by_pooling(embeddings, pooling)

        # Load protein IDs
        with open(ids_path, "rb") as f:
            self._ids = pickle.load(f)

        print(f"Loaded {len(self._ids)} proteins with embedding dim {embeddings.shape[1]}")
        print(f"Pooling mode: {pooling}")

        # Store embeddings as tensor
        self.embeddings = torch.from_numpy(embeddings).float()

        # Load GO term labels if terms_path is provided (training mode)
        if terms_path is not None:
            terms_df = pd.read_csv(terms_path, sep="\t")

            # Get top-k most frequent terms
            term_counts = terms_df["term"].value_counts()
            if not top_k:
                self._top_terms = term_counts.index.tolist()
            else:
                self._top_terms = term_counts.head(top_k).index.tolist()

            print(f"Total unique terms: {len(term_counts)}")
            print(f"Using top {top_k} terms (min count: {term_counts.iloc[top_k-1]})")

            # Filter to top terms only
            terms_df = terms_df[terms_df["term"].isin(self._top_terms)]

            # Group by protein ID
            protein_terms = terms_df.groupby("EntryID")["term"].apply(list).to_dict()

            # Create label matrix aligned with protein IDs
            mlb = MultiLabelBinarizer(classes=self._top_terms)
            mlb.fit([self._top_terms])  # Ensure all classes are present

            # Build labels in order of protein IDs
            labels_list = [protein_terms.get(pid, []) for pid in self._ids]
            labels = mlb.transform(labels_list)

            self.labels = torch.from_numpy(labels).float()
        else:
            # Inference mode: no labels
            self._top_terms = []
            self.labels = None

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return (embedding, label) tuple or just embedding if no labels."""
        if self.labels is not None:
            return self.embeddings[idx], self.labels[idx]
        return self.embeddings[idx]

    @property
    def ids(self) -> list[str]:
        """Return list of protein IDs."""
        return self._ids

    @property
    def top_terms(self) -> list[str]:
        """Return list of GO term IDs in prediction order."""
        return self._top_terms
