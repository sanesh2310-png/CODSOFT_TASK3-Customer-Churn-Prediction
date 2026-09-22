# CODSOFT_TASK3 — Customer Churn Prediction

Predicts customer churn for a bank using historical customer data and classic machine learning classifiers.

## 📌 Problem Statement

Given a bank customer's demographic and account information, predict whether they will churn (leave the bank) — a moderately imbalanced binary classification problem (~20.4% churn rate).

## 📂 Dataset

[Bank Customer Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) (Kaggle)

- `Churn_Modelling.csv` — 10,000 customer records, no missing values

## 🛠️ Approach

1. **Exploratory Data Analysis**
   - Confirmed moderate class imbalance (79.63% retained vs. 20.37% churned)
   - Found a strong, non-linear relationship between number of products and churn:
     - 1 product: 27.7% churn
     - 2 products: 7.6% churn (best retention)
     - 3 products: 82.7% churn
     - 4 products: 100% churn

2. **Feature Engineering**
   - Dropped identifying columns (`RowNumber`, `CustomerId`, `Surname`)
   - One-hot encoded `Geography` and `Gender`
   - Scaled numeric features with `StandardScaler` (fit on train, applied to test only)

3. **Model Training**
   Trained and compared three classifiers:
   - Logistic Regression (with `class_weight='balanced'`)
   - Random Forest (with `class_weight='balanced'`)
   - Gradient Boosting

4. **Evaluation**
   Evaluated using precision, recall, F1-score, confusion matrix, and ROC-AUC for the churn class specifically, rather than relying on accuracy alone.

## 📊 Results

| Model | Precision (churn) | Recall (churn) | F1-score | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.39 | 0.70 | 0.50 | 0.777 | 0.71 |
| Random Forest | 0.58 | 0.64 | **0.61** | 0.860 | 0.83 |
| Gradient Boosting | **0.79** | 0.49 | 0.60 | **0.871** | **0.87** |

No single model dominates — this is a genuine precision/recall trade-off:
- **Random Forest** gives the best-balanced F1-score, catching more churners overall.
- **Gradient Boosting** achieves the highest precision and ROC-AUC — better if the priority is minimizing wasted retention offers on customers who wouldn't have churned anyway.

Recommendation depends on business priority: Random Forest if the cost of missing a churner is high; Gradient Boosting if the cost of false alarms is high.

## 📁 Repository Contents

- `CODSOFT_TASK3_Customer_Churn_Prediction.ipynb` — full notebook (EDA, feature engineering, model training, evaluation)
- `README.md` — this file

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `Churn_Modelling.csv`
3. Run all cells in order

## 🔮 Possible Improvements

- Threshold tuning to shift the precision/recall balance for Gradient Boosting
- Feature interaction terms (e.g. products × active member status)
- Hyperparameter tuning via grid search
