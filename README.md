# Machine Learning Projects

A collection of machine learning projects exploring classification, clustering, and NLP techniques using real-world datasets.

---

## Projects

### 1. Breast Cancer Classification
**Dataset:** Breast Cancer Wisconsin (sklearn built-in)  
**Goal:** Classify tumors as malignant or benign  
**Techniques used:**
- Support Vector Machine (SVM) with RBF and linear kernels
- Random Forest Classifier
- Cross Validation (k-fold) for model evaluation
- MinMaxScaler for feature scaling

**Results:**
| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| SVM (kernel = 'linear', C=10) | 97.14% | 96.49% |
| Random Forest (n_estimators = 40) | 100% | 95.61% |

---

### 2. Hyperparameter Tuning: Grid Search vs Randomized Search
**Dataset:** Digits (sklearn built-in)  
**Goal:** Compare GridSearchCV and RandomizedSearchCV across multiple classifiers  
**Techniques used:**
- GridSearchCV for exhaustive hyperparameter search
- RandomizedSearchCV for efficient random sampling
- 4-fold cross-validation for model evaluation
- Applied to SVM, Random Forest, Decision Tree, and Logistic Regression

**Models & Search Space:**
| Model | Parameters Searched | Grid Size | RandomizedSearch `n_iter` |
|---|---|---|---|
| Support Vector Machine (SVC) | `kernel`, `gamma`, `C` | 12 | 6 |
| Random Forest Classifier | `n_estimators` | 4 | 4 |
| Decision Tree Classifier | `max_depth`, `max_leaf_nodes` | 20 | 12 |
| Logistic Regression | `solver`, `max_iter` | 8 | 8 |

---

## Repository Structure

```
Machine-Learning-Projects/
│
├── breast_cancer_classification/
│   └── breast_cancer.ipynb
│
├── hyperparameter_tuning/
│   ├── Hyperparameter_tuning_Grid_and_Randomized_search_.ipynb
│   └── README.md
│
└── README.md
```

---
## Setup & Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

Clone the repo and open any notebook:
```bash
git clone https://github.com/bernance/machine-learning-projects.git
cd machine-learning-projects
jupyter notebook
```

---

## Tools & Libraries
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

---

## About
This repository documents my journey learning machine learning. Each project applies algorithms to real-world datasets with notes on model performance and lessons learned.
