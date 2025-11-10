# DA5401 Assignment 8 — Ensemble Learning for Complex Regression  
### **Bike Sharing Demand Prediction**

**Course:** Data Analytics (DA5401)  
**Student**: Abesech Inbasekar  
**Roll Number**: ME21B003

---

##  Overview

This repository contains the implementation for **DA5401: Assignment 8**  
**“Ensemble Learning for Complex Regression — Modeling on Bike Share Data”**.

The objective is to forecast **hourly bike rental counts (`cnt`)** using a range of ensemble methods —  
**Bagging, Boosting, and Stacking** — and compare their performance against simple baseline regressors.

Through this task, we explore how each ensemble technique manages the **bias–variance trade-off**,  
and demonstrate how combining diverse models can lead to improved predictive performance.

---

## Repository Structure

├── Assignment_8_ME21B003.ipynb # Main Colab Notebook  
├── README.md # This file

---

##  Problem Statement

Accurate forecasting of hourly bike rentals is vital for city operations and resource planning.  
The dataset exhibits strong **non-linearities** influenced by weather, time, and seasonality.

**Goal:**  
Predict total bike rentals (`cnt`) using different regression and ensemble techniques,  
evaluating their effectiveness based on **Root Mean Squared Error (RMSE)**.

---

##  Methods Implemented

| Part | Technique | Description |
|:----:|:-----------|:-------------|
| **A** | **Baseline Models** | Linear Regression & Decision Tree (max depth = 6) for benchmark RMSE |
| **B1** | **Bagging (Variance Reduction)** | Bagging Regressor using Decision Trees as base estimators (tuned) |
| **B2** | **Boosting (Bias Reduction)** | Gradient Boosting Regressor with tuned hyperparameters |
| **C** | **Stacking (Model Diversity)** | Combines KNN, Bagging, and Boosting via Ridge meta-learner |
| **D** | **Final Analysis** | Comparative RMSE table, bias–variance discussion, and visualizations |

---

##  Dataset

**Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)  
**File Used:** `hour.csv` (17,379 hourly records)

| Column | Description |
|:--------|:-------------|
| `cnt` | Total bike rentals (target) |
| `temp`, `atemp`, `hum`, `windspeed` | Continuous weather features |
| `season`, `weathersit`, `mnth`, `hr`, `weekday`, `workingday`, `holiday` | Categorical features (one-hot encoded) |
| `instant`, `dteday`, `casual`, `registered` | Dropped (index or leakage) |

---

##  Implementation Steps

1. **Data Preprocessing**
   - Dropped irrelevant and leakage columns (`instant`, `dteday`, `casual`, `registered`)
   - One-hot encoded categorical variables
   - Split data chronologically (80% train, 20% test)

2. **Baseline Models**
   - Trained Linear Regression and Decision Tree Regressor  
   - Selected the lower RMSE (Linear Regression) as the baseline

3. **Bagging**
   - Used `BaggingRegressor` with `DecisionTreeRegressor` as base estimator  
   - Tuned `n_estimators`, `max_depth`, and `max_samples`  
   - Achieved RMSE ≈ **139.67**

4. **Boosting**
   - Implemented `GradientBoostingRegressor`  
   - Tuned `learning_rate`, `max_depth`, `n_estimators`, `subsample`  
   - Achieved RMSE ≈ **81.13** (Best model)

5. **Stacking**
   - Combined **KNN**, **Bagging**, and **Boosting** as base learners  
   - Used **Ridge Regression** as meta-learner (Level-1)  
   - Achieved RMSE ≈ **90.14**

6. **Analysis**
   - Visualized RMSE comparison and conceptual bias–variance trade-off  
   - Discussed model diversity and stacking principles in depth

---

##  Results Summary

| Model | Type | RMSE | Key Observation |
|:------|:------|------:|:----------------|
| **Linear Regression** | Baseline | 133.84 | High bias, limited flexibility |
| **Decision Tree** | Single | 159.14 | Overfits, high variance |
| **Bagging Regressor (Tuned)** | Ensemble | 139.67 | Reduced variance, moderate bias |
| **Gradient Boosting (Tuned)** | Ensemble | **81.13** | Strong bias correction, best accuracy |
| **Stacking Regressor** | Ensemble | 90.14 | Balanced bias–variance via model diversity |

---

##  Key Insights

- **Bagging** stabilized predictions by reducing variance but retained the bias of shallow trees.  
- **Boosting** achieved the lowest RMSE by aggressively reducing bias through sequential error correction.  
- **Stacking** effectively blended heterogeneous learners to achieve a strong balance between bias and variance.  
- The project demonstrates how **ensemble diversity and hierarchical learning** enhance model robustness in complex regression tasks.

---

##  Visualizations

- **RMSE Comparison Bar Chart**
Plot is generated in the notebook under **Part D: Final Analysis**.

---

##  Dependencies

Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn


---

##  Authors
**Abesech Inbasekar**  
Department of Mechanical Engineering + Data Science  
**IIT Madras**

---

##  Conclusion

This project highlights how ensemble learning — through **Bagging, Boosting, and Stacking** — can  
significantly improve regression accuracy on complex, noisy, and non-linear data.

The results underscore the power of combining multiple learners to minimize both bias and variance,  
and illustrate how meta-learning through stacking can yield models that generalize more effectively  
than any individual predictor.

---


