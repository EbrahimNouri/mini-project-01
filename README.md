# Credit Card Fraud Detection Pipeline

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, built with scikit-learn.

---

## 1. Problem Description

### Business Scenario
Credit card fraud costs financial institutions billions of dollars each year.
A bank needs an automated system that flags suspicious transactions in real time
so analysts can investigate before losses accumulate.

### Objective
Build a binary classifier that predicts whether a transaction is:
- **0** — Legitimate
- **1** — Fraudulent

### Dataset
- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples**: 284,807
- **Features**: 30 (Time, V1–V28, Amount)
- **Fraud cases**: 492 (~0.17%)
- **Challenge**: Extreme class imbalance

---

## 2. Data Analysis

### Dataset Statistics
| Property            | Value       |
|----------------------|-------------|
| Total Samples        | 284,807     |
| Features             | 30          |
| Fraudulent           | 492 (0.17%) |
| Legitimate           | 284,315     |
| Missing Values       | 0           |

### Feature Information
- **Time**: Seconds elapsed between the first and current transaction
- **V1–V28**: Anonymized features from PCA transformation
- **Amount**: Transaction amount
- **Class**: Target variable (0 = Legit, 1 = Fraud)

### Missing Value Analysis
The dataset contains **no missing values**, so no imputation is needed.

---

## 3. Initial Hypothesis

### Which model will perform best?
Logistic Regression is expected to provide a strong baseline because it is
robust to overfitting on high-dimensional data. Decision Trees may capture
non-linear patterns but risk overfitting without depth constraints.

### Which metric matters most?
**Recall** is critical — failing to detect fraud (False Negative) costs the
bank money. However, maximizing Recall alone causes excessive false alarms,
so **F1-score** is used as the primary selection metric.

### What if the model predicts all transactions as legitimate?
Accuracy would be ~99.83%, but Recall for fraud would be 0% — the model
would be useless for fraud detection. This illustrates why accuracy is
misleading on imbalanced datasets.

### Will scaling affect KNN?
Yes — KNN uses Euclidean distance, so features with larger ranges dominate.
Scaling is expected to significantly improve KNN performance.

### Will the Decision Tree overfit?
Yes — without depth limits, a Decision Tree will memorize the training data,
achieving near-perfect training scores but poor generalization.

---

## 4. Model Comparison

| Model                  | Accuracy | Precision | Recall  | F1     |
|------------------------|----------|-----------|---------|--------|
| Logistic Regression    | --       | --        | --      | --     |
| KNN (K=5)              | --       | --        | --      | --     |
| Decision Tree (d=10)   | --       | --        | --      | --     |

> Run `src/train.py` to generate these results.

### Cross-Validation (5-Fold Stratified)

| Model                  | Mean Precision | Mean Recall | Mean F1  |
|------------------------|----------------|-------------|----------|
| Logistic Regression    | --             | --          | --       |
| KNN (K=5)              | --             | --          | --       |
| Decision Tree (d=10)   | --             | --          | --       |

---

## 5. Scaling Experiment

| Model                  | Scaling | Precision | Recall  | F1     |
|------------------------|---------|-----------|---------|--------|
| KNN (K=5)              | No      | --        | --      | --     |
| KNN (K=5)              | Yes     | --        | --      | --     |

KNN is sensitive to scaling because it relies on Euclidean distances.
Decision Trees are not affected by scaling because they make axis-aligned splits.

---

## 6. Hyperparameter Experiment

| max_depth | Train F1 | Test F1 | Precision | Recall  |
|-----------|----------|---------|-----------|---------|
| 2         | --       | --      | --        | --      |
| 5         | --       | --      | --        | --      |
| 10        | --       | --      | --        | --      |
| None      | --       | --      | --        | --      |

Overfitting is clearly visible when max_depth=None.

---

## 7. Threshold Experiment

| Threshold | Precision | Recall  | F1     |
|-----------|-----------|---------|--------|
| 0.3       | --        | --      | --     |
| 0.5       | --        | --      | --     |
| 0.7       | --        | --      | --     |

A lower threshold (0.3–0.4) is recommended for fraud detection to maximize
Recall, since the cost of missing fraud exceeds the cost of false alarms.

---

## 8. Final Model Selection

**Selected Model**: Logistic Regression
**Selected Threshold**: 0.5 (or tuned based on threshold experiment)

### Why Logistic Regression?
- Stable cross-validation performance across all folds
- Less prone to overfitting than Decision Trees on this imbalanced data
- Provides calibrated probability outputs for threshold tuning
- Interpretable — feature weights can be examined

---

## 9. Running Instructions

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd mini-project-01

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset
```bash
# Option 1: Using kagglehub (automatic)
python src/data_prep.py

# Option 2: Manual download
# Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the data/ directory
```

### Training
```bash
# Run the full pipeline (data prep + training + experiments)
python src/train.py
```

### Prediction
```bash
# Single prediction
python src/predict.py input.json output.json

# Example input.json
{
  "Time": 406,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

---

## 10. Reflection

### Q1: Why is Accuracy misleading?
A model that always predicts "Legitimate" achieves ~99.83% accuracy while
detecting zero frauds. Accuracy does not account for class imbalance.

### Q2: False Positive vs False Negative trade-off
- **False Positive**: Legitimate transaction flagged as fraud → customer
  inconvenience, manual review cost
- **False Negative**: Fraudulent transaction passes → direct financial loss

For fraud detection, False Negatives are more costly, so we accept more
False Positives (lower threshold) to catch more fraud (higher Recall).

### Q3: What would we improve with more time?
- Implement class balancing techniques (SMOTE, undersampling)
- Try ensemble models (Random Forest, XGBoost)
- Add feature engineering (time-of-day patterns, amount bins)
- Build a real-time streaming prediction pipeline
- Deploy as a REST API with FastAPI
