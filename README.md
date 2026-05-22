# 🌳 Decision Tree Implementation — Telecom Customer Churn Prediction

A complete end-to-end Machine Learning project that predicts **customer churn** in a telecom company using a **Decision Tree Classifier**. This project covers the full ML pipeline — from exploratory data analysis to hyperparameter tuning with Grid Search and Randomized Search.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Transformation](#2-data-transformation)
  - [3. Label Encoding](#3-label-encoding)
  - [4. Model Building](#4-model-building)
  - [5. Evaluation Metrics](#5-evaluation-metrics)
  - [6. Cross Validation](#6-cross-validation)
  - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
  - [8. Grid Search](#8-grid-search)
  - [9. Randomized Search](#9-randomized-search)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Author](#author)

---

## Overview

**Customer churn** is when a customer stops using a company's service. Predicting churn early allows businesses to take proactive steps to retain customers. This project builds a Decision Tree classifier on telecom customer data to classify whether a customer will churn (`Yes`) or stay (`No`).

The notebook implements:
- Automated EDA using **Sweetviz**
- Feature selection and preprocessing
- **Decision Tree** with entropy-based splitting
- **Confusion Matrix** and **Classification Report**
- **K-Fold** and **Stratified K-Fold** Cross Validation
- Manual **Hyperparameter Tuning**
- **GridSearchCV** and **RandomizedSearchCV** for optimal parameter discovery

---

## Dataset

| Property        | Details                              |
|----------------|--------------------------------------|
| **File**        | `Telecom_Customer_Churn.csv`         |
| **Index**       | `customerID`                         |
| **Target**      | `Churn` (Binary: Yes / No)           |
| **Features Used** | `MultipleLines`, `tenure`, `Contract`, `OnlineSecurity`, `TechSupport` |

The dataset contains information about telecom customers such as their subscription services, account information, and whether they churned.

> 📥 You can find similar datasets on [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Tech Stack

| Library         | Purpose                                      |
|----------------|----------------------------------------------|
| `pandas`        | Data loading and manipulation                |
| `numpy`         | Numerical operations                         |
| `matplotlib`    | Plotting and visualization                   |
| `seaborn`       | Statistical visualizations (heatmaps)        |
| `sweetviz`      | Automated Exploratory Data Analysis          |
| `scikit-learn`  | ML model, metrics, cross-validation, tuning  |
| `scipy`         | Random distributions for RandomizedSearchCV  |

---

## Project Structure

```
DECISION_TREE_IMPLEMENTATION/
│
├── DECISION_TREE_IMPLEMENTATION.ipynb   # Main Jupyter Notebook
├── Telecom_Customer_Churn.csv           # Dataset
├── sweetviz_report.html                 # Auto-generated EDA report
└── README.md                            # Project documentation
```

---

## Pipeline Walkthrough

### 1. Exploratory Data Analysis

**Sweetviz** is used to generate a rich, interactive HTML report of the dataset in one line:

```python
import sweetviz as sv
report = sv.analyze(data)
report.show_html("sweetviz_report.html")
```

This report shows:
- Feature distributions
- Missing value analysis
- Correlation between features and the target (`Churn`)
- Outlier detection

A **Seaborn correlation heatmap** is also plotted across all numerical and encoded features to understand inter-feature relationships.

---

### 2. Data Transformation

- `customerID` is set as the DataFrame index (not a predictive feature).
- `TotalCharges` column is converted from object dtype to numeric, with any parsing errors filled as `0`:

```python
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0).astype(int)
```

---

### 3. Label Encoding

All categorical columns are encoded to numeric values using `LabelEncoder` from scikit-learn. Columns encoded include:

`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `Churn`

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in encode:
    data[column] = le.fit_transform(data[column])
```

---

### 4. Model Building

**Features (X):**
```
MultipleLines, tenure, Contract, OnlineSecurity, TechSupport
```

**Target (y):**
```
Churn
```

The data is split 80/20 for training and testing:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

A **Decision Tree Classifier** is trained using **Information Gain (Entropy)** as the splitting criterion:

```python
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(x_train, y_train)
```

The decision tree is visualized using `plot_tree()` at full depth and at `max_depth=2` for an interpretable view.

---

### 5. Evaluation Metrics

After prediction on the test set, the following metrics are computed:

- **Accuracy** — overall correctness
- **Precision** — of all predicted churn cases, how many actually churned
- **Recall** — of all actual churn cases, how many were caught
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visual breakdown of TP, TN, FP, FN

```python
print(classification_report(y_test, test_prediction))
```

The confusion matrix is plotted as a heatmap with labels `Not Churn` and `Churn`.

---

### 6. Cross Validation

#### Standard K-Fold (17 folds)
```python
cv_scores = cross_val_score(dt, x, y, cv=17, scoring='accuracy')
```

#### Stratified K-Fold (10 folds)
Preserves the class distribution in each fold — important for imbalanced churn data:

```python
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt, x, y, cv=skf, scoring='accuracy')
```

Outputs **mean accuracy** and **standard deviation** across folds.

#### Multi-Metric Cross Validation (5-Fold)
Cross validation is run separately for **Precision**, **Recall**, and **F1-Score** to get a robust estimate of each metric:

```python
precision_scores = cross_val_score(dt, x, y, cv=5, scoring='precision')
recall_scores    = cross_val_score(dt, x, y, cv=5, scoring='recall')
f1_scores        = cross_val_score(dt, x, y, cv=5, scoring='f1')
```

Results are visualized using **boxplots** for Accuracy, Precision, and F1-Score distributions across folds.

---

### 7. Hyperparameter Tuning (Manual)

The Decision Tree is manually tuned by setting:

```python
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=10
)
```

This helps reduce overfitting compared to a fully grown tree.

---

### 8. Grid Search

`GridSearchCV` exhaustively searches all combinations of the following parameter grid:

```python
param_grid = {
    'max_depth'        : [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 5, 10, 20],
    'max_leaf_nodes'   : [10, 20, 50]
}
```

- **Cross Validation:** 10-fold
- **Scoring:** Accuracy
- **Jobs:** `-1` (uses all CPU cores for parallel processing)

```python
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid,
                           cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
```

The best estimator is then evaluated on the test set with a full classification report.

---

### 9. Randomized Search

`RandomizedSearchCV` randomly samples `n_iter=100` combinations from a continuous parameter distribution (using `scipy.stats.randint`):

```python
param_dist = {
    'max_depth'        : randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf' : randint(1, 20),
    'max_leaf_nodes'   : randint(10, 100)
}

random_search = RandomizedSearchCV(estimator=dt_model, param_distributions=param_dist,
                                   n_iter=100, cv=5, scoring='accuracy',
                                   n_jobs=-1, verbose=2)
random_search.fit(x_train, y_train)
print("Best Hyperparameters:", random_search.best_params_)
```

This is faster than Grid Search and often finds equally good or better results.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DECISION_TREE_IMPLEMENTATION.git
cd DECISION_TREE_IMPLEMENTATION
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sweetviz scipy
```

### 3. Add the Dataset

Place `Telecom_Customer_Churn.csv` in the project root directory (or update the path in the notebook).

### 4. Run the Notebook

```bash
jupyter notebook DECISION_TREE_IMPLEMENTATION.ipynb
```

---

## Results Summary

| Method                        | Training Accuracy | Testing Accuracy |
|------------------------------|-------------------|------------------|
| Decision Tree (Default)       | High (overfits)   | Moderate         |
| Decision Tree (Manual Tuning) | Controlled        | Improved         |
| Grid Search Best Model        | Optimized         | Best (CV-tuned)  |
| Randomized Search Best Model  | Optimized         | Comparable       |



