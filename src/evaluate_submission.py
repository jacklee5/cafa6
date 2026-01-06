import os
import tempfile
import pandas as pd

from os import path
from cafaeval.evaluation import cafa_eval
from utils.io import get_data_path
import time

# 1. Define paths to official files
OBO_FILE = get_data_path("cafa6", path.join("Train", "go-basic.obo"))
IA_FILE = get_data_path("cafa6", "IA.tsv")  # This is the "ia(f)" mentioned in your text
PRED_DIR = "submissions"
GT_FILE = "data/val/ground_truth.tsv"

def fast_eval(submission_path: str, th_step: float = 0.05, n_cpu: int = 4, sample_proportion: float = 0.1) -> float:
    """
    Fast evaluation by sampling a proportion of predictions.

    Args:
        submission_path: Path to the submission TSV file
        th_step: Threshold step for tau array (default 0.05)
        n_cpu: Number of CPUs to use (default 4)
        sample_proportion: Fraction of unique proteins to sample (default 0.1)

    Returns:
        Mean F-max score across subontologies
    """
    # Read the submission file
    pred_df = pd.read_csv(submission_path, sep='\t', header=None, names=['protein_id', 'go_term', 'score'])

    # Sample unique proteins
    unique_proteins = pred_df['protein_id'].unique()
    n_sample = max(1, int(len(unique_proteins) * sample_proportion))
    print(f"Sampling {n_sample} out of {len(unique_proteins)} unique proteins for fast evaluation.")
    sampled_proteins = pd.Series(unique_proteins).sample(n=n_sample, random_state=42).values

    # Filter predictions to sampled proteins
    sampled_pred_df = pred_df[pred_df['protein_id'].isin(sampled_proteins)]

    # Create temporary directory for sampled submission
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save sampled predictions
        sampled_path = os.path.join(tmp_dir, "sampled_submission.tsv")
        sampled_pred_df.to_csv(sampled_path, sep='\t', header=False, index=False)

        # Run cafa_eval on sampled data
        print("Starting evaluation on sampled data...")
        _, best_metrics_dict = cafa_eval(
            obo_file=OBO_FILE,
            pred_dir=tmp_dir,
            gt_file=GT_FILE,
            ia=IA_FILE,
            norm='cafa',
            prop='max',
            th_step=th_step,
            n_cpu=n_cpu
        )

        # Calculate mean F-max
        if 'f_w' in best_metrics_dict:
            f_max_df = best_metrics_dict['f_w']
            mean_f_max = f_max_df['f_w'].mean()
            return mean_f_max
        else:
            raise ValueError("No weighted metrics found. Check if IA file was loaded correctly.")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast evaluation of CAFA submissions")
    parser.add_argument(
        "--th-step",
        type=float,
        default=0.05,
        help="Threshold step for tau array (default: 0.05)",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=2,
        help="Number of CPUs to use (default: 4)",
    )
    parser.add_argument(
        "--sample-proportion",
        type=float,
        default=0.1,
        help="Fraction of unique proteins to sample (default: 0.1)",
    )
    args = parser.parse_args()

    submission_file = path.join(PRED_DIR, "submission.tsv")
    start_time = time.time()
    mean_fmax = fast_eval(submission_file, th_step=args.th_step, n_cpu=args.n_cpu, sample_proportion=args.sample_proportion)
    elapsed_time = time.time() - start_time
    print(f"Mean F-max (sampled): {mean_fmax:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")