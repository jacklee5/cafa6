import os
import pickle

import numpy as np
import pandas as pd
from ete3 import NCBITaxa
from sklearn.model_selection import train_test_split

BASE_DATA_PATH = "/home/jack2/kaggle/cafa6/input"

class DatasetPaths:
    CAFA6 = "cafa-6-protein-function-prediction"
    ESM1B = "esm1b-embeddings"

def get_data_path(dataset: str, path: str):
    return os.path.join(BASE_DATA_PATH, dataset, path)

# Load taxonomy data
taxonomy_df = pd.read_csv(
    get_data_path(DatasetPaths.CAFA6, "Train/train_taxonomy.tsv"),
    sep="\t",
    header=None,
    names=["uniprot_id", "taxon_id"],
)

# Get kingdom for each taxon (used for grouping rare species)
ncbi = NCBITaxa()

def get_rank_id(taxon_id, rank="genus"):
    lineage = ncbi.get_lineage(taxon_id)
    ranks = ncbi.get_rank(lineage)
    for taxon in ranks:
        if ranks[taxon] == rank:
            return taxon
    return None

taxonomy_df = taxonomy_df.assign(
    kingdom=taxonomy_df["taxon_id"].map(lambda taxon_id: get_rank_id(taxon_id, rank="kingdom"))
)

# Group rare species (<100 proteins) by kingdom, others by taxon_id
species_counts = taxonomy_df["taxon_id"].value_counts()
rare_species = set(species_counts[species_counts < 100].index)

def get_group(row):
    if row["taxon_id"] in rare_species:
        return f"kingdom_{row['kingdom']}"
    return str(row["taxon_id"])

taxonomy_df["group"] = taxonomy_df.apply(get_group, axis=1)

# 80/20 train/validation split stratified by group
train_proteins, val_proteins = train_test_split(
    taxonomy_df["uniprot_id"].values,
    test_size=0.2,
    random_state=42,
    stratify=taxonomy_df["group"].values,
)

print(f"Train proteins: {len(train_proteins)}")
print(f"Val proteins: {len(val_proteins)}")

# Save protein lists
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)

with open("data/train/proteins.pkl", "wb") as f:
    pickle.dump(list(train_proteins), f)

with open("data/val/proteins.pkl", "wb") as f:
    pickle.dump(list(val_proteins), f)

print("Saved proteins.pkl")

# Create and save embeddings
train_embeddings = np.load(get_data_path(DatasetPaths.ESM1B, "train_esm1b_embeddings.npy"))
embeddings_df = pd.DataFrame(train_embeddings, index=taxonomy_df["uniprot_id"])

np.save(
    "data/train/embeddings.npy",
    embeddings_df.loc[train_proteins].values,
)
np.save(
    "data/val/embeddings.npy",
    embeddings_df.loc[val_proteins].values,
)

print(f"Saved embeddings: train {len(train_proteins)}x{embeddings_df.shape[1]}, val {len(val_proteins)}x{embeddings_df.shape[1]}")
