Dataset: 615 patients. Classes: Blood donor 533, Suspect donor 7, Hepatitis 24, Fibrosis 21, Cirrhosis 30
Missing values (imputed with the training-fold median): {'ALB': 1, 'ALP': 18, 'ALT': 1, 'CHOL': 10, 'PROT': 1}

Spearman correlation with stage (donor 0 → cirrhosis 3, suspect donors excluded):

| Biomarker | ρ | p-value |
|---|---|---|
| AST | +0.51 | 1.6e-41 |
| GGT | +0.42 | 4.3e-27 |
| BIL | +0.39 | 4.4e-23 |
| CHOL | -0.29 | 8.6e-13 |
| ALP | -0.19 | 4.8e-06 |
| CHE | -0.18 | 9.6e-06 |
| ALB | -0.17 | 2.7e-05 |
| CREA | -0.12 | 2.0e-03 |
| ALT | -0.12 | 2.5e-03 |
| PROT | +0.07 | 8.1e-02 |

Repeated stratified 5-fold CV × 10 (mean ± std, %):

| Model | Accuracy | Balanced acc. | Macro-F1 | Recall Blood donor | Recall Suspect donor | Recall Hepatitis | Recall Fibrosis | Recall Cirrhosis |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 88.3 ± 3.1 | 70.3 ± 10.9 | 61.9 ± 10.0 | 92 | 79 | 60 | 49 | 70 |
| LDA | 92.5 ± 1.7 | 58.5 ± 12.2 | 60.5 ± 11.8 | 100 | 64 | 34 | 29 | 65 |
| Random forest | 92.4 ± 1.6 | 51.4 ± 9.6 | 52.7 ± 10.6 | 99 | 17 | 27 | 33 | 80 |
| Random forest + SMOTE | 93.2 ± 1.5 | 63.5 ± 9.3 | 63.6 ± 8.8 | 99 | 62 | 43 | 35 | 79 |

Best by macro-F1: Random forest + SMOTE. Top held-out permutation importances: AST (0.111), ALB (0.056), PROT (0.050), BIL (0.045), ALP (0.035)