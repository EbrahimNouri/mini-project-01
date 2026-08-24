# Credit Card Fraud Detection Pipeline

End-to-end machine learning pipeline for detecting fraudulent credit card transactions - Mini-Project 01, Machine Learning course.

---

## 1. Problem Description

### Business Scenario

A bank/payment provider wants to automatically flag potentially fraudulent credit card transactions. Each transaction is classified as:

```text
0 -> Legitimate Transaction
1 -> Fraudulent Transaction
```

The core challenge is **extreme class imbalance**: only ~0.17% of transactions are fraudulent. A trivial model predicting "legitimate" for everything reaches 99.83% accuracy while catching zero fraud - which is why Accuracy alone is meaningless here.

### Objective

Build a complete ML pipeline: data preparation -> stratified split -> leakage-free scaling -> training and comparison of four classifiers (Logistic Regression, KNN, Decision Tree, plus a bonus PyTorch MLP) -> controlled experiments -> final model selection -> reusable JSON-in/JSON-out prediction script (`src/predict.py`).

### Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - European cardholders, two days of September 2013.

| Property | Value |
|---|---|
| Samples | 284,807 |
| Features | 30 (`Time`, `V1`-`V28`, `Amount`) |
| Target | `Class` (0 = legitimate, 1 = fraud) |
| Legitimate | 284,315 |
| Fraudulent | 492 |
| Fraud ratio | 0.1727% |

`V1`-`V28` are PCA-anonymized features. Place the CSV at `data/creditcard.csv`.

---

## 2. Data Analysis

Results from the data-preparation stage:

- **Samples:** 284,807 rows x 31 columns (30 features + `Class`)
- **Missing values:** 0 in every column
- **Duplicates:** 1,081 rows (0.38%), kept: duplicate PCA signatures can correspond to genuine repeated transactions, and removing them would alter the real-world class ratio the test set should reflect
- **Class distribution:** 284,315 legitimate vs 492 fraudulent (0.1727%) -> highly imbalanced
- **Feature ranges:** `Amount` spans 0 - 25,691 while PCA features sit roughly around 0 -> distance-based models require scaling

### Preprocessing decisions

1. **Stratified train/test split** (80/20, `random_state=42`): preserves the fraud ratio in both sets (train 0.1729%, test 0.1720%).
2. **No data leakage:** the split happens *before* scaling; `StandardScaler` is fitted **on the training set only**, then applied to both splits.
3. **Encoder:** all features are already numeric, so the saved encoder is an identity passthrough `ColumnTransformer`; saved to satisfy the required artifact set.

---

## 3. Initial Hypothesis

Written before training:

> **H1 (best model):** Logistic Regression will be a solid linear baseline; a Decision Tree should capture nonlinear patterns but risks overfitting.
>
> **H2 (metric choice):** Accuracy will be misleading. Recall matters most because missing a fraud costs far more than a false alarm - but maximizing Recall alone floods investigators with false alerts, so Precision and F1 must be monitored together.
>
> **H3 (all-legitimate predictor):** A model that always predicts "legitimate" scores 99.83% accuracy yet detects nothing.
>
> **H4 (scaling):** KNN will be strongly affected by scaling because unscaled `Amount` dominates Euclidean distances; Decision Trees will be unaffected since they split on single features.

### After-training verdict

- H1: partially correct - Decision Tree beat Logistic Regression, and scaled KNN performed best overall (test F1 = 0.8587). The unpruned tree clearly overfit (Train F1 = 1.0 vs Test F1 = 0.7487). The bonus MLP confirmed strong nonlinear performance as well (F1 up to 0.81).
- H2: correct - see Final Model Selection for how the Recall/Precision trade-off drove the threshold choice.
- H3: correct.
- H4: correct and dramatic - unscaled KNN detected only 3% of frauds versus 81% when scaled.

---

## 4. Model Comparison

All models trained on the scaled training set, evaluated on the held-out test set (56,962 samples, 98 frauds).

### Test-set results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (max_iter=1000) | 0.9991 | 0.8267 | 0.6327 | 0.7168 |
| K-Nearest Neighbors (K=5) | 0.9995 | 0.9186 | 0.8061 | 0.8587 |
| Decision Tree (max_depth=10) | 0.9994 | 0.8902 | 0.7449 | 0.8111 |
| Simple MLP (PyTorch, threshold 0.5) | 0.9991 | 0.6833 | **0.8367** | 0.7523 |
| Simple MLP (threshold 0.9) | 0.9994 | 0.8144 | 0.8061 | 0.8103 |

The MLP's class weighting trades precision for recall at the default threshold; raising its threshold to 0.9 recovers a balanced P/R profile with F1 comparable to the Decision Tree.

### Confusion matrices (test set)

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 56,851 | 13 | 36 | 62 |
| KNN (K=5) | 56,857 | 7 | 19 | 79 |
| Decision Tree (depth=10) | 56,855 | 9 | 25 | 73 |
| MLP @ 0.5 | 56,826 | 38 | 16 | 82 |

False negatives are the dangerous cell: LR misses 36 of 98 frauds, KNN 19, DT 25, MLP only 16.

### 5-Fold Stratified Cross Validation (mean +/- std)

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8698 (+/-0.0263) | 0.6180 (+/-0.0645) | 0.7213 (+/-0.0510) |
| K-Nearest Neighbors (K=5) | 0.9271 (+/-0.0336) | 0.7846 (+/-0.0419) | 0.8490 (+/-0.0266) |
| Decision Tree (max_depth=10) | 0.8776 (+/-0.0301) | 0.7563 (+/-0.0562) | 0.8115 (+/-0.0387) |

(The MLP uses a dedicated stratified validation split for epoch selection instead of k-fold CV; see reports/experiments.md.)

---

## 5. Scaling Experiment (Experiment 1)

KNN (K=5) trained on raw vs standardized features:

| Model | Scaling | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| KNN | Without Scaling | 1.0000 | 0.0306 | 0.0594 |
| KNN | With Scaling | 0.9186 | 0.8061 | 0.8587 |

Full analysis in [`reports/experiments.md`](reports/experiments.md).

**Why KNN is sensitive:** KNN classifies by nearest neighbors under Euclidean distance. Unscaled `Amount` (range up to ~25,691) contributes orders of magnitude more squared-distance than PCA features, so neighborhoods are effectively determined by `Amount` alone.

**Why Decision Trees are not:** trees test one feature at a time and choose thresholds within each feature's values, so monotone rescaling never changes the split ordering.

Note the trap in the unscaled row: Precision stayed 1.0 while Recall collapsed to 3% - a system that looks "precise" while catching almost no fraud.

---

## 6. Hyperparameter Experiment (Experiment 2)

Decision Tree `max_depth` sweep (Train/Test F1 gap reveals overfitting):

| max_depth | Train F1 | Test F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 2 | 0.7931 | 0.7653 | 0.7653 | 0.7653 |
| 5 | 0.8649 | 0.8306 | 0.8941 | 0.7755 |
| 10 | 0.9220 | 0.8111 | 0.8902 | 0.7449 |
| None | 1.0000 | 0.7487 | 0.7526 | 0.7449 |

- **Overfitting occurred**: `max_depth=None` memorizes training data (Train F1 = 1.0000) and generalizes worst on test (0.7487).
- **Best balance: max_depth=5** - highest Test F1 (0.8306) with a small gap (0.034), plus the best Precision (0.8941) and Recall (0.7755).

---

## 7. Classification Threshold Experiment (Experiment 3)

Logistic Regression probabilities evaluated at different decision thresholds:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.7312 | 0.6939 | 0.7120 |
| 0.5 (default) | 0.8267 | 0.6327 | 0.7168 |
| 0.7 | 0.8310 | 0.6020 | 0.6982 |

Lowering the threshold raises Recall (+6.1pp at 0.3) and lowers Precision; raising it does the opposite. The MLP section in [`reports/experiments.md`](reports/experiments.md) repeats this analysis for the neural network, where the effect is even stronger because weighted training shifts probabilities upward.

---

## 8. Final Model Selection

**Final model: Logistic Regression, decision threshold = 0.3**

Saved artifacts (produced by `python main.py`):

```
models/model.pkl     <- trained Logistic Regression (final deployed model)
models/scaler.pkl    <- StandardScaler fitted on training data
models/encoder.pkl   <- identity passthrough ColumnTransformer
models/mlp_model.pt  <- bonus PyTorch MLP (comparison model)
```

### Why Logistic Regression despite higher raw scores elsewhere?

KNN wins on test/CV metrics and the tuned MLP reaches similar F1 to the tree, but LR was selected as the *deployed* model because:

1. **Inference cost:** KNN stores the whole 227k-row training set and scans it for every transaction; LR is a dot product - microseconds and kilobytes.
2. **Meaningful probabilities:** KNN probabilities come from discrete neighbor votes (multiples of 1/K); LR produces smooth continuous probabilities, which is exactly what threshold tuning exploits.
3. **Threshold tuning recovers Recall:** moving LR to threshold 0.3 lifts Recall from 0.63 to 0.69 while keeping Precision acceptable (0.73) and F1 essentially unchanged (0.7120 vs 0.7168). A real fraud in this dataset scored p = 0.3477 under LR - missed at 0.5, caught at 0.3.
4. **Interpretability and simplicity:** LR coefficients directly show which anonymized features push toward fraud; no extra framework (torch) needed at serving time.

The bonus MLP demonstrates the value of nonlinear models and weighting (Recall 0.84 at default threshold vs LR's 0.63) and would be the natural next candidate for deployment after calibration and proper CV.

### Why threshold 0.3?

Fraud detection has asymmetric costs: a missed fraud lets money walk away, a false alarm costs a quick review. At 0.3 the model accepts ~12 extra false alarms to catch 6 additional frauds, with F1 nearly unchanged. If business priorities shifted toward avoiding customer friction, 0.5 remains the fallback.

### Known limitations

- Cross-validation reuses scalings computed once on the full training data rather than refitting inside every fold; a sklearn `Pipeline` passed to `cross_validate` would remove this caveat.
- Duplicates were kept (Section 2).
- Small absolute fraud counts make metrics noisy between folds (LR fold Recall ranged 0.52-0.69).

---

## 9. Running Instructions

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

### Full pipeline (training)

```bash
python main.py
```

Runs everything: data prep -> split -> scaling -> LR/KNN/DT training + evaluation -> 5-fold stratified CV -> Experiments 1-3 -> saves sklearn artifacts -> trains the bonus PyTorch MLP (with overfitting analysis) -> saves `models/mlp_model.pt`. Takes several minutes (KNN cross-validation dominates).

Individual stages can also be run directly:

```bash
python src/data_prep.py       # data analysis + split + scaler/encoder artifacts
python src/train.py           # sklearn models, CV, experiments, model saving
python src/train_mlp.py       # bonus MLP only
```

### Prediction

```bash
python src/predict.py input.json    # input.json included as an example
```

Output goes to the console and to `output.json` next to the input file:

```json
{
  "prediction": "Fraud",
  "class_id": 1,
  "probability": 0.3477,
  "threshold": 0.3,
  "status": "success"
}
```

A list of transactions in the input file returns a list of predictions. Missing features produce `"status": "error"` with details instead of a guess.

---

## 10. Reflection

### Q1: Why is Accuracy a misleading metric for this dataset?

Because the classes are extremely imbalanced (99.83% legitimate). Predicting "legitimate" always scores 99.83% accuracy while detecting zero fraud. All four trained models score >= 0.999 accuracy yet differ hugely in fraud detection quality (F1 from 0.72 to 0.86). Only Precision, Recall, F1, and the confusion matrix expose what Accuracy hides.

### Q2: What is the trade-off between detecting more fraud and generating more false alarms?

Lowering the decision threshold moves the boundary: Recall rises while Precision falls. Aggressive detection spends investigation capacity on false alarms and risks customer friction; conservative detection lets fraud slip through. Since a missed fraud typically costs more than reviewing a flag, the pipeline biases toward Recall (LR at threshold 0.3): ~25 false positives per 56,962 test transactions in exchange for catching 68 of 98 frauds. The MLP sweep shows the same shape even more clearly (F1 rising from 0.61 at 0.3 to 0.81 at 0.9 as precision recovers).

### Q3: With one extra week, what would you improve?

1. Handle imbalance during training for all models: `class_weight="balanced"` or SMOTE on training folds, selecting models by PR-AUC rather than point metrics.
2. Fully leak-free CV via sklearn `Pipeline` (scaler refit inside each fold) - including CV for the MLP.
3. Probability calibration (`CalibratedClassifierCV`) so operating thresholds map to honest fraud likelihoods.
4. Wider hyperparameter search (GridSearchCV over K, tree depth, C, MLP size/lr/dropout).
5. Feature ideas: cyclical hour-of-day encoding from `Time`, log-transform or robust scaling of `Amount`.
6. Ship the optional FastAPI service around `src/predict.py` for REST scoring.
