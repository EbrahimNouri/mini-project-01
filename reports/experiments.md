# Experiment Reports — Credit Card Fraud Detection

---

## Experiment 1: Effect of Feature Scaling on KNN

### Goal
Understand whether feature scaling improves KNN performance on the fraud detection dataset.

### Setup
| Model                  | Scaling     | Precision | Recall  | F1     |
|------------------------|-------------|-----------|---------|--------|
| KNN (K=5)              | Without     | --        | --      | --     |
| KNN (K=5)              | With        | --        | --      | --     |

> Run `src/train.py` to populate this table with actual values.

### Why KNN is Sensitive to Scaling
KNN computes Euclidean distances between samples. Features with larger numeric
ranges (e.g., `Amount` can be in thousands) dominate the distance calculation,
making smaller-range features (e.g., V1–V28 centered near 0) effectively
invisible. `StandardScaler` normalizes all features to mean=0, std=1, giving
each feature equal weight in the distance metric.

### Why Decision Tree is NOT Sensitive to Scaling
Decision Trees make axis-aligned splits on individual features. The split
threshold adapts to the feature's actual range, so scaling has no effect.

---

## Experiment 2: Hyperparameter Analysis (Decision Tree max_depth)

### Goal
Determine the effect of `max_depth` on Decision Tree performance and identify overfitting.

### Results
| max_depth | Train F1 | Test F1 | Precision | Recall  |
|-----------|----------|---------|-----------|---------|
| 2         | --       | --      | --        | --      |
| 5         | --       | --      | --        | --      |
| 10        | --       | --      | --        | --      |
| None      | --       | --      | --        | --      |

> Run `src/train.py` to populate this table with actual values.

### Analysis
- **max_depth=None**: The tree memorizes the training data (Train F1 ~ 1.0)
  but generalizes poorly → classic **overfitting**.
- **max_depth=2**: Too simple → **underfitting**, misses important patterns.
- **max_depth=5 or 10**: Usually provides the best balance.

---

## Experiment 3: Classification Threshold Analysis

### Goal
Investigate how changing the decision threshold from the default 0.5 affects
the trade-off between Precision and Recall.

### Results
| Threshold | Precision | Recall  | F1     |
|-----------|-----------|---------|--------|
| 0.3       | --        | --      | --     |
| 0.5       | --        | --      | --     |
| 0.7       | --        | --      | --     |

> Run `src/train.py` to populate this table with actual values.

### Explanation
- **Threshold = 0.3** → More transactions flagged as fraud →
  Higher Recall (catch more fraud), Lower Precision (more false alarms)
- **Threshold = 0.5** → Default balanced cutoff
- **Threshold = 0.7** → Fewer transactions flagged as fraud →
  Higher Precision, Lower Recall (miss some fraud)

### Recommendation
For fraud detection, a **lower threshold (e.g., 0.3–0.4)** is recommended
because missing a fraudulent transaction (False Negative) is typically more
costly than flagging a legitimate one for manual review (False Positive).

---

## Cross-Validation Results

### 5-Fold Stratified Cross-Validation

| Model                  | Mean Precision | Mean Recall | Mean F1  |
|------------------------|----------------|-------------|----------|
| Logistic Regression    | --             | --          | --       |
| KNN (K=5)              | --             | --          | --       |
| Decision Tree (d=10)   | --             | --          | --       |

> Run `src/train.py` to populate this table with actual values.

Stratification ensures each fold preserves the ~0.17% fraud ratio, making
cross-validation results reliable for this highly imbalanced dataset.
