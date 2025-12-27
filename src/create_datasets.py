import os
import pickle

import numpy as np
import pandas as pd
from ete3 import NCBITaxa
from sklearn.model_selection import train_test_split

BASE_DATA_PATH = "/home/jack2/kaggle/cafa6/input"
REGENERATE_VAL_SET = False

class DatasetPaths:
    CAFA6 = "cafa-6-protein-function-prediction"
    ESM1B = "esm1b-embeddings"
    ESM2 = "cafa6-protein-embeddings-esm2"

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
if REGENERATE_VAL_SET:
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
else:
    # Load existing protein lists
    with open("data/train/proteins.pkl", "rb") as f:
        train_proteins = pickle.load(f)

    with open("data/val/proteins.pkl", "rb") as f:
        val_proteins = pickle.load(f)

    print(f"Loaded proteins.pkl: train {len(train_proteins)}, val {len(val_proteins)}")

# Create and save ESM1B embeddings (protein order matches taxonomy_df)
print("\n--- ESM1B Embeddings ---")
train_embeddings_esm1b = np.load(get_data_path(DatasetPaths.ESM1B, "train_esm1b_embeddings.npy"))
embeddings_df_esm1b = pd.DataFrame(train_embeddings_esm1b, index=taxonomy_df["uniprot_id"])

np.save(
    "data/train/embeddings.npy",
    embeddings_df_esm1b.loc[train_proteins].values,
)
np.save(
    "data/val/embeddings.npy",
    embeddings_df_esm1b.loc[val_proteins].values,
)

print(f"Saved ESM1B embeddings: train {len(train_proteins)}x{embeddings_df_esm1b.shape[1]}, val {len(val_proteins)}x{embeddings_df_esm1b.shape[1]}")

# Create and save ESM2 embeddings (protein order from protein_ids.csv)
print("\n--- ESM2 Embeddings ---")
esm2_ids = pd.read_csv(get_data_path(DatasetPaths.ESM2, "protein_ids.csv"))
esm2_embeddings = np.load(get_data_path(DatasetPaths.ESM2, "protein_embeddings.npy"))
print(f"Loaded ESM2: {len(esm2_ids)} proteins, embedding dim {esm2_embeddings.shape[1]}")

# Create DataFrame with protein_id as index for easy lookup
embeddings_df_esm2 = pd.DataFrame(esm2_embeddings, index=esm2_ids["protein_id"])

# Extract train and val embeddings (only proteins that exist in ESM2)
train_proteins_in_esm2 = [p for p in train_proteins if p in embeddings_df_esm2.index]
val_proteins_in_esm2 = [p for p in val_proteins if p in embeddings_df_esm2.index]

print(f"Train proteins in ESM2: {len(train_proteins_in_esm2)}/{len(train_proteins)}")
print(f"Val proteins in ESM2: {len(val_proteins_in_esm2)}/{len(val_proteins)}")

np.save(
    "data/train/embeddings_esm2.npy",
    embeddings_df_esm2.loc[train_proteins_in_esm2].values,
)
np.save(
    "data/val/embeddings_esm2.npy",
    embeddings_df_esm2.loc[val_proteins_in_esm2].values,
)

# Save the protein lists for ESM2 (in case some are missing)
with open("data/train/proteins_esm2.pkl", "wb") as f:
    pickle.dump(train_proteins_in_esm2, f)
with open("data/val/proteins_esm2.pkl", "wb") as f:
    pickle.dump(val_proteins_in_esm2, f)

print(f"Saved ESM2 embeddings: train {len(train_proteins_in_esm2)}x{esm2_embeddings.shape[1]}, val {len(val_proteins_in_esm2)}x{esm2_embeddings.shape[1]}")
