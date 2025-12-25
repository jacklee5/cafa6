# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for CAFA6 (Critical Assessment of Functional Annotation 6) - a protein function prediction competition. The goal is to predict Gene Ontology (GO) terms for proteins based on their sequences.

Competition: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API configured)
kaggle competitions download -c cafa-6-protein-function-prediction -p input/cafa-6-protein-function-prediction
```

## Data Structure

- `input/cafa-6-protein-function-prediction/` - Competition data
  - `Train/train_sequences.fasta` - Training protein sequences
  - `Train/train_terms.tsv` - GO term labels for training proteins
  - `Train/train_taxonomy.tsv` - Taxonomy IDs for training proteins
  - `Train/go-basic.obo` - Gene Ontology structure
  - `Test/testsuperset.fasta` - Test protein sequences
  - `IA.tsv` - Information Accretion weights for scoring
- `input/esm1b-embeddings/` - Pre-computed ESM-1b protein embeddings
  - `train_esm1b_embeddings.npy` / `train_ids.pkl`
  - `test_esm1b_embeddings.npy` / `test_ids.pkl`

## Key Libraries

- **biopython** - FASTA file parsing (`SeqIO`)
- **ete3** - NCBI taxonomy lookups (`NCBITaxa`)
- **obonet** - Parsing GO ontology (go-basic.obo)
- **numpy/pandas** - Data manipulation

## Architecture Notes from CAFA 5

Based on README findings from previous competition:
- Logistic regression on ESM-1b embeddings performed best
- Train/validation split should be done by species (taxonomy-based)
- Sample validation data proportionally to species frequency in training set
