# %%
import os
BASE_DATA_PATH = "/home/jack2/kaggle/cafa6/input"
class DatasetPaths:
    CAFA6 = "cafa-6-protein-function-prediction"
    ESM1B = 'esm1b-embeddings'

def get_data_path(dataset: str, path: str):
    return os.path.join(BASE_DATA_PATH, dataset, path)

# %%
import numpy as np
import os

# Load the train ESM1b embeddings
embeddings_path = '/home/jack2/kaggle/cafa6/input/esm1b-embeddings/test_esm1b_embeddings.npy'
train_embeddings = np.load(embeddings_path)
print(f"Shape of train embeddings: {train_embeddings.shape}")
print(f"Data type: {train_embeddings.dtype}")

# %%
# read fasta file
from Bio import SeqIO

fasta_file = '/home/jack2/kaggle/cafa6/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta'
def read_fasta_file(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences
test_sequences = read_fasta_file(fasta_file)
print(f"Number of sequences in the FASTA file: {len(test_sequences)}")

# %%
import pandas as pd

# find how many unique taxonomy and how many per taxonomy
train_taxonomy_path = get_data_path(DatasetPaths.CAFA6, "Train/train_taxonomy.tsv")
taxonomy_df = pd.read_csv(train_taxonomy_path, sep="\t", header=None, names=["uniprot_id", "taxon_id"])

unique_taxonomies = taxonomy_df['taxon_id'].nunique()
taxonomy_counts = taxonomy_df['taxon_id'].value_counts()


#%%
from ete3 import NCBITaxa
ncbi = NCBITaxa()
def get_rank_id(taxon_id, rank="genus"):
    lineage = ncbi.get_lineage(taxon_id)
    ranks = ncbi.get_rank(lineage)
    for taxon in ranks:
        if ranks[taxon] == rank:
            return taxon
    return None

#%%
taxonomy_df = taxonomy_df.assign(
    kingdom=taxonomy_df['taxon_id'].map(lambda taxon_id: get_rank_id(taxon_id, rank="kingdom"))
)
kingdom_counts = taxonomy_df['kingdom'].value_counts()
species_counts = taxonomy_df['taxon_id'].value_counts()
display(kingdom_counts)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.histplot(species_counts.values, bins=50, log_scale=(True, False))
plt.xlabel('Proteins per Species (log scale)')
plt.ylabel('Number of Species')
plt.title('Distribution of Protein Counts per Species')
plt.axvline(x=10, color='r', linestyle='--', label='Cutoff = 10')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Species with <= 10 proteins: {(species_counts <= 10).sum()} ({(species_counts <= 10).mean():.1%})")
print(f"Species with > 10 proteins: {(species_counts > 10).sum()}")

#%% Group rare species (< 5 proteins) by kingdom, others by taxon_id
rare_species = set(species_counts[species_counts < 100].index)

def get_group(row):
    if row['taxon_id'] in rare_species:
        return f"kingdom_{row['kingdom']}"
    return str(row['taxon_id'])

taxonomy_df['group'] = taxonomy_df.apply(get_group, axis=1)
group_counts = taxonomy_df['group'].value_counts()

plt.figure(figsize=(12, 6))
sns.histplot(group_counts.values, bins=50, log_scale=(True, False))
plt.xlabel('Proteins per Group (log scale)')
plt.ylabel('Number of Groups')
plt.title('Distribution of Protein Counts per Group (Rare Species Grouped by Kingdom)')
plt.tight_layout()
plt.show()

print(f"Original unique species: {species_counts.shape[0]}")
print(f"Groups after combining rare species: {group_counts.shape[0]}")

#%% 80/20 train/validation split stratified by group
from sklearn.model_selection import train_test_split

train_proteins, val_proteins = train_test_split(
    taxonomy_df['uniprot_id'].values,
    test_size=0.2,
    random_state=42,
    stratify=taxonomy_df['group'].values
)

print(f"Train proteins: {len(train_proteins)} ({len(train_proteins)/len(taxonomy_df):.1%})")
print(f"Val proteins: {len(val_proteins)} ({len(val_proteins)/len(taxonomy_df):.1%})")

# Save proteins to files
# os.makedirs(os.path.join(BASE_DATA_PATH, "data/train"), exist_ok=True)
# os.makedirs(os.path.join(BASE_DATA_PATH, "data/val"), exist_ok=True)

# with open(os.path.join(BASE_DATA_PATH, "data/train/proteins.txt"), "w") as f:
#     f.write("\n".join(train_proteins))

# with open(os.path.join(BASE_DATA_PATH, "data/val/proteins.txt"), "w") as f:
#     f.write("\n".join(val_proteins))

# print(f"Saved to data/[train|val]/proteins.txt")

#%% create train / val embeddings
import pandas as pd
df = pd.read_csv(get_data_path(DatasetPaths.CAFA6, "Train/train_taxonomy.tsv"), sep="\t", header=None, names=["uniprot_id", "taxon_id"])
train_embeddings = np.load(get_data_path(DatasetPaths.ESM1B, "train_esm1b_embeddings.npy"))

embeddings_df = pd.DataFrame(train_embeddings, index=df["uniprot_id"])
print(embeddings_df.shape)

# Load protein lists and save embeddings
with open("data/train/proteins.txt") as f:
    train_protein_ids = [line.strip() for line in f]
with open("data/val/proteins.txt") as f:
    val_protein_ids = [line.strip() for line in f]

np.save("data/train/embeddings.npy", embeddings_df.loc[train_protein_ids].values)
np.save("data/val/embeddings.npy", embeddings_df.loc[val_protein_ids].values)

print(f"Saved train embeddings: {len(train_protein_ids)} x {embeddings_df.shape[1]}")
print(f"Saved val embeddings: {len(val_protein_ids)} x {embeddings_df.shape[1]}")
