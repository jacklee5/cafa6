"""
CAFA evaluation metrics for protein function prediction.

Adapted from the official CAFA evaluator:
https://github.com/claradepaolis/CAFA-evaluator-PK/blob/main/src/cafaeval/evaluation.py

The CAFA competition uses weighted F-max as the primary metric, where:
- Weights are Information Accretion (IA) values for each GO term
- F-max is the maximum F-score across all thresholds
- Final score is the average of F-max across MF, BP, and CC ontologies
"""

import numpy as np
import pandas as pd


def load_ia_weights(
    ia_path: str = "input/cafa-6-protein-function-prediction/IA.tsv",
) -> dict[str, float]:
    """
    Load Information Accretion weights from TSV file.

    Args:
        ia_path: Path to IA.tsv file with columns [GO_term, weight]

    Returns:
        Dictionary mapping GO term IDs to their IA weights
    """
    ia_df = pd.read_csv(ia_path, sep="\t", header=None, names=["term", "weight"])
    return dict(zip(ia_df["term"], ia_df["weight"]))


def get_term_weights(
    terms: list[str],
    ia_weights: dict[str, float],
    default_weight: float = 0.0,
) -> np.ndarray:
    """
    Get IA weights for a list of terms.

    Args:
        terms: List of GO term IDs
        ia_weights: Dictionary mapping GO terms to IA weights
        default_weight: Weight to use for terms not in ia_weights

    Returns:
        Array of weights aligned with terms
    """
    return np.array([ia_weights.get(t, default_weight) for t in terms])


def compute_weighted_precision_recall(
    preds: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """
    Compute weighted precision and recall at a given threshold.

    This follows the CAFA evaluation methodology where predictions and
    ground truth are weighted by Information Accretion values.

    Args:
        preds: Prediction probabilities, shape (n_samples, n_terms)
        labels: Binary ground truth labels, shape (n_samples, n_terms)
        weights: IA weights for each term, shape (n_terms,)
        threshold: Threshold for converting probabilities to binary

    Returns:
        Tuple of (weighted_precision, weighted_recall)
    """
    # Binarize predictions
    binary_preds = (preds >= threshold).astype(np.float32)

    # Apply weights
    weighted_preds = binary_preds * weights
    weighted_labels = labels * weights

    # Compute weighted intersection (true positives)
    weighted_tp = (binary_preds * labels * weights).sum(axis=1)

    # Weighted predictions and ground truth per protein
    weighted_pred_sum = weighted_preds.sum(axis=1)
    weighted_label_sum = weighted_labels.sum(axis=1)

    # Precision per protein (avoid division by zero)
    precision_per_protein = np.divide(
        weighted_tp,
        weighted_pred_sum,
        out=np.zeros_like(weighted_tp),
        where=weighted_pred_sum > 0,
    )

    # Recall per protein
    recall_per_protein = np.divide(
        weighted_tp,
        weighted_label_sum,
        out=np.zeros_like(weighted_tp),
        where=weighted_label_sum > 0,
    )

    # Macro average (average across proteins with predictions/annotations)
    n_with_preds = (weighted_pred_sum > 0).sum()
    n_with_labels = (weighted_label_sum > 0).sum()

    precision = precision_per_protein.sum() / max(n_with_preds, 1)
    recall = recall_per_protein.sum() / max(n_with_labels, 1)

    return precision, recall


def compute_f_score(precision: float, recall: float) -> float:
    """Compute F-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def weighted_fmax(
    preds: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """
    Compute weighted F-max (maximum F-score across thresholds).

    This is the primary CAFA evaluation metric.

    Args:
        preds: Prediction probabilities, shape (n_samples, n_terms)
        labels: Binary ground truth labels, shape (n_samples, n_terms)
        weights: IA weights for each term, shape (n_terms,)
        thresholds: Array of thresholds to evaluate (default: 0.01 to 0.99)

    Returns:
        Tuple of (fmax, best_threshold, precision_at_best, recall_at_best)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        precision, recall = compute_weighted_precision_recall(
            preds, labels, weights, threshold
        )
        f_score = compute_f_score(precision, recall)

        if f_score > best_f:
            best_f = f_score
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_f, best_threshold, best_precision, best_recall


def unweighted_fmax(
    preds: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """
    Compute unweighted F-max (all terms have equal weight).

    Useful for comparison or when IA weights are not available.

    Args:
        preds: Prediction probabilities, shape (n_samples, n_terms)
        labels: Binary ground truth labels, shape (n_samples, n_terms)
        thresholds: Array of thresholds to evaluate

    Returns:
        Tuple of (fmax, best_threshold, precision_at_best, recall_at_best)
    """
    weights = np.ones(preds.shape[1], dtype=np.float32)
    return weighted_fmax(preds, labels, weights, thresholds)


class CAFAMetrics:
    """
    Helper class for computing CAFA metrics during training.

    Preloads IA weights and provides convenient methods for evaluation.
    """

    def __init__(
        self,
        ia_path: str = "input/cafa-6-protein-function-prediction/IA.tsv",
        thresholds: np.ndarray | None = None,
    ):
        """
        Initialize CAFA metrics calculator.

        Args:
            ia_path: Path to IA.tsv file
            thresholds: Thresholds for F-max calculation (default: 0.01 to 0.99)
        """
        self.ia_weights = load_ia_weights(ia_path)
        self.thresholds = thresholds if thresholds is not None else np.arange(0.01, 1.0, 0.01)
        self._term_weights_cache: dict[tuple, np.ndarray] = {}

    def get_weights(self, terms: list[str]) -> np.ndarray:
        """Get cached weights for a list of terms."""
        key = tuple(terms)
        if key not in self._term_weights_cache:
            self._term_weights_cache[key] = get_term_weights(terms, self.ia_weights)
        return self._term_weights_cache[key]

    def compute_fmax(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        terms: list[str],
    ) -> dict[str, float]:
        """
        Compute both weighted and unweighted F-max.

        Args:
            preds: Prediction probabilities, shape (n_samples, n_terms)
            labels: Binary ground truth labels, shape (n_samples, n_terms)
            terms: List of GO term IDs corresponding to columns

        Returns:
            Dictionary with metrics:
                - fmax_weighted: Weighted F-max (competition metric)
                - fmax_unweighted: Unweighted F-max
                - threshold_weighted: Best threshold for weighted
                - threshold_unweighted: Best threshold for unweighted
                - precision_weighted, recall_weighted
                - precision_unweighted, recall_unweighted
        """
        weights = self.get_weights(terms)

        fmax_w, thresh_w, prec_w, rec_w = weighted_fmax(
            preds, labels, weights, self.thresholds
        )
        fmax_u, thresh_u, prec_u, rec_u = unweighted_fmax(
            preds, labels, self.thresholds
        )

        return {
            "fmax_weighted": fmax_w,
            "fmax_unweighted": fmax_u,
            "threshold_weighted": thresh_w,
            "threshold_unweighted": thresh_u,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "precision_unweighted": prec_u,
            "recall_unweighted": rec_u,
        }
