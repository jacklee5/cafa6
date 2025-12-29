from os import path
from typing import Literal

BASE_DATA_PATH = path.normpath(path.join(path.dirname(__file__), "..", "..", "input"))
DATA_FOLDERS = {
    "esm1b": "esm1b-embeddings",
    "esm2": "cafa6-protein-embeddings-esm2",
    "cafa6": "cafa-6-protein-function-prediction",
}

def get_data_path(dataset: Literal["esm1b", "esm2", "cafa6"], file_path: str) -> str:
    """Load data from a specified dataset and file path."""
    full_path = path.join(BASE_DATA_PATH, DATA_FOLDERS[dataset], file_path)
    return full_path