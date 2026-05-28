# Hyperparameter Tuning: Grid Search vs Randomized Search

A practical comparison of two core hyperparameter optimization strategies using scikit-learn, applied to four classical ML classifiers on the Digits dataset.

---

## Overview

Hyperparameter tuning is the process of finding the optimal configuration for a machine learning model — values that aren't learned during training but set beforehand (e.g. `C` for SVM, `n_estimators` for Random Forest).

This notebook explores two systematic approaches:

- **GridSearchCV** — exhaustively tests every combination in a defined parameter grid
- **RandomizedSearchCV** — samples a fixed number of random combinations from the same grid

Both are evaluated using **4-fold cross-validation** across four classifiers.

---

## Dataset

**Scikit-learn Digits Dataset**
- 1,797 samples of 8×8 pixel handwritten digit images (0–9)
- 64 features (one per pixel)
- 10-class classification problem

---

## Models & Hyperparameters Tuned

| Model | Parameters Searched | Grid Size | RandomizedSearch `n_iter` |
|---|---|---|---|
| Support Vector Machine (SVC) | `kernel`, `gamma`, `C` | 12 | 6 |
| Random Forest Classifier | `n_estimators` | 4 | 4 |
| Decision Tree Classifier | `max_depth`, `max_leaf_nodes` | 20 | 12 |
| Logistic Regression | `solver`, `max_iter` | 8 | 8 |

---

## Project Structure

```
├── Hyperparameter_tuning_Grid_and_Randomized_search_.ipynb
└── README.md
```

---

## Requirements

```
scikit-learn
pandas
```

Install with:

```bash
pip install scikit-learn pandas
```

---

## Usage

1. Clone the repository
2. Open the notebook in Jupyter or any compatible environment
3. Run all cells sequentially

---

## Key Findings

- **GridSearchCV** guarantees finding the best combination within the defined grid but is computationally expensive — cost grows exponentially with grid size.
- **RandomizedSearchCV** is significantly faster and often finds a near-optimal result, making it practical for larger search spaces. `n_iter` is set per-model to be meaningfully less than the full grid size, ensuring genuine random sampling behavior.
- Cross-validation serves as the validation mechanism, so no separate train/test split is needed for hyperparameter search — the CV folds handle held-out evaluation internally.
- `random_state=42` is set on all `RandomizedSearchCV` calls for reproducibility.

---

## Concepts Demonstrated

- `GridSearchCV` for exhaustive hyperparameter search
- `RandomizedSearchCV` for efficient random sampling
- 4-fold cross-validation scoring
- Per-model `n_iter` tuning to avoid redundant exhaustive search
- Inspecting results via `cv_results_` and `best_params_`

---

## References

- [scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn: RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [scikit-learn: Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Hyperparameter Tuning Guide — scikit-learn docs](https://scikit-learn.org/stable/modules/grid_search.html)
