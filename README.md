# Análise de Risco de Crédito — Comparative Study

> Technical study: comparative evaluation of three classical ML models for credit risk classification, with cross-validation on the best performer.

A focused exercise in baseline modeling for credit risk. Three models — Random Forest, Gradient Boosting and Logistic Regression — are trained on the same dataset and evaluated head-to-head on accuracy, AUC-ROC and confusion matrix. The best model is then validated with k-fold cross-validation and feature importance is examined.

This project is the *baseline study* that preceded my [credit-scoring](https://github.com/murillosezerino/credit-scoring) ensemble work — useful as a side-by-side comparison of single-model approaches versus stacking.

## What this project explores

- **Three-model comparison** on the same training data
- **Evaluation metrics** beyond accuracy: AUC-ROC, confusion matrix, comparative ROC curves
- **Cross-validation** on the best model
- **Feature importance** analysis

## Stack

`Python` · `Scikit-Learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## What's inside

```
analise-risco-credito/
├── notebooks/            # Exploration and modeling
├── src/
│   ├── preprocessing.py
│   ├── models.py         # RF, GB, LogReg
│   └── evaluation.py     # Metrics + plots
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Status

Study repository — meant as a baseline comparison, not a production model. See [credit-scoring](https://github.com/murillosezerino/credit-scoring) for the ensemble follow-up.

## Author

Murillo Sezerino — Analytics Engineer · Data Engineer
[murillosezerino.com](https://murillosezerino.com) · [LinkedIn](https://linkedin.com/in/murillosezerino)
