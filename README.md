# cafa6
Code for my submissions to CAFA6 Protein Function Prediction Competition on Kaggle: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction

Download data into input folder using: `kaggle competitions download -c cafa-6-protein-function-prediction`

CAFA 5 solution findings
- Logistic regression on ESM-1b yielded best perforamnce
- They did train/val split on species
    - they sampled from each species at the rate in which they appear to form a validation dataset

## Descriptions of code
### create_datasets.py

This file creates data/[train/val]/proteins.txt as well as some initial EDA work. The proteins are split 80/20, stratified by species. For those without sufficient species representation, they are classified into a "rare species group based on kingdom"