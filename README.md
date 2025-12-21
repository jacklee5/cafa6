# cafa6
Code for my submissions to CAFA6 Protein Function Prediction Competition on Kaggle: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction

Download data into input folder using: `kaggle competitions download -c cafa-6-protein-function-prediction`

CAFA 5 solution findings
- Logistic regression on ESM-1b yielded best perforamnce
- They did train/val split on species
    - they sampled from each species at the rate in which they appear to form a validation dataset