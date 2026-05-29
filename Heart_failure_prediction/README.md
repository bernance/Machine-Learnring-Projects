# Heart Failure Prediction

A machine learning project that predicts heart disease using clinical features, comparing multiple classification algorithms with and without PCA dimensionality reduction.

## Dataset

The [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) contains 918 records with 11 clinical features including age, sex, chest pain type, resting blood pressure, cholesterol, and more.

## Workflow

### 1. Data Preprocessing
- Removed outliers from continuous numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) using Z-score (threshold = 3)
- Label encoded binary categorical columns: `Sex`, `ExerciseAngina`, `ST_Slope`
- One-hot encoded multi-class columns: `RestingECG`, `ChestPainType`

### 2. Models Trained
All models were trained twice — with and without PCA (95% variance retained):

- Support Vector Machine (SVM) with GridSearchCV
- Random Forest with GridSearchCV
- Logistic Regression L1 (Lasso equivalent)
- Logistic Regression L2 (Ridge equivalent)

### 3. Results

| Model | Train Accuracy | Test Accuracy | Overfit Gap |
|-------|---------------|---------------|-------------|
| RF (no PCA) | 93.99% | 86.53% | 7.46% |
| SVM (no PCA) | 91.55% | 83.42% | 8.13% |
| LR L2 (no PCA) | 86.37% | 84.44% | 1.93% |
| LR L1 (no PCA) | 86.02% | 81.97% | 4.08% |
| **SVM + PCA** | **86.37%** | **85.00%** | **1.37%** |
| **LR L1 + PCA** | **86.23%** | **85.00%** | **1.23%** |
| RF + PCA | 84.84% | 83.89% | 0.95% |
| LR L2 + PCA | 86.37% | 84.44% | 1.93% |

### 4. Final Model: Logistic Regression L1 + PCA

**Why LR L1 + PCA?**
- Tied for best test accuracy (85.00%) with smallest meaningful gap (1.23%)
- Faster to train than SVM
- L1 regularization performs feature selection, ignoring less useful PCA components
- Most reliable for generalizing to unseen patient data

## Key Learnings

- **Scaling matters for SVM** — unscaled features caused extremely slow convergence
- **PCA requires fit-on-train-only** — fitting PCA on the full dataset or test set causes data leakage
- **PCA hurts Random Forest** — RF already handles feature selection internally; PCA removes that advantage
- **High training accuracy ≠ good model** — RF hit 100% train accuracy but generalized worse than LR

