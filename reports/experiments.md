# Experiment Report

All experiments run via `python main.py` on the Credit Card Fraud Detection dataset (284,807 rows, 492 frauds = 0.1727%). Stratified 80/20 split (`random_state=42`): 227,845 train / 56,962 test (98 frauds). Scaler fitted on training data only - no leakage.

---

## Experiment 1: Effect of Feature Scaling

**Question:** How much does feature scaling change KNN behavior?

| Model | Scaling | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| KNN (K=5) | Without Scaling | 1.0000 | 0.0306 | 0.0594 |
| KNN (K=5) | With Scaling | 0.9186 | 0.8061 | 0.8587 |

### Observations

- Without scaling, KNN catches only **3 of 98 frauds** (Recall = 0.0306). Its perfect Precision is a trap: the model flags so few transactions that its rare flags happen to be right while missing virtually all fraud.
- With scaling, Recall jumps from 3% to **81%** and F1 from 0.06 to 0.86.

### Why is KNN sensitive to scaling?

KNN votes among nearest neighbors under Euclidean distance. `Amount` spans up to ~25,691 while PCA features live roughly in [-5, 5]. Squared-distance contributions scale with the square of the range, so unscaled `Amount` dominates the metric and neighbors are chosen almost purely by "similar amount", making the informative V-features invisible. StandardScaler (mean 0, std 1) gives every feature comparable influence.

### Why is the Decision Tree less sensitive?

A tree tests one feature at a time ("V14 <= -1.35?") and picks each threshold from within that feature's values. Scaling is a monotone remap of a single axis; it changes the numeric threshold found but never the split ordering or resulting partitions.

---

## Experiment 2: Hyperparameter Analysis (Decision Tree max_depth)

**Question:** Does overfitting occur as tree depth grows, and which depth balances bias/variance?

| max_depth | Train F1 | Test F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 2 | 0.7931 | 0.7653 | 0.7653 | 0.7653 |
| 5 | 0.8649 | 0.8306 | 0.8941 | 0.7755 |
| 10 | 0.9220 | 0.8111 | 0.8902 | 0.7449 |
| None | 1.0000 | 0.7487 | 0.7526 | 0.7449 |

### Did overfitting occur?

Yes, measurably:

- `max_depth=None`: Train F1 = 1.0000 vs Test F1 = 0.7487 -> a **0.25 memorization gap**, the classic signature of an unpruned tree fitting noise.
- The train/test gap narrows steadily with depth (depth 10: 0.111; depth 5: 0.034).

### Which value provides the best balance?

**max_depth = 5**: highest Test F1 (0.8306), best Precision (0.8941) and Recall (0.7755), with only a small gap. Depth 2 underfits; depth 10 already trades generalization for training fit; `None` generalizes worst despite perfect training performance.

---

## Experiment 3: Impact of Classification Threshold

**Question:** How does moving the decision threshold away from 0.5 reshape the Precision/Recall trade-off for Logistic Regression?

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.7312 | 0.6939 | 0.7120 |
| 0.5 (default) | 0.8267 | 0.6327 | 0.7168 |
| 0.7 | 0.8310 | 0.6020 | 0.6982 |

### What happens to Recall when the threshold decreases?

It increases: more transactions clear the lower bar (Recall 0.6020 -> 0.6939 going from 0.7 to 0.3, i.e., 68 vs 59 frauds caught out of 98).

### What happens to Precision?

It decreases: the extra flagged transactions include false alarms (Precision 0.8310 -> 0.7312 across the same range).

### Recommended threshold

**0.3.** Missing fraud costs more than reviewing a false alarm. Going 0.5 -> 0.3 buys +6.1pp Recall (62 -> 68 frauds caught) for ~12 extra false alarms total, while F1 moves only -0.005. Concrete demonstration: an actual fraudulent transaction scored p = 0.3477 under LR - silently approved at 0.5, correctly flagged at 0.3.

---

## Bonus Experiment: Simple MLP (PyTorch)

Notebook Model 4 (Bonus). Purpose per the project definition: explore nonlinear decision boundaries and analyze overfitting behavior.

### Setup

- Architecture: `30 -> 64 -> 32 -> 1`, ReLU activations, Dropout(0.2)
- Loss: `BCEWithLogitsLoss` with `pos_weight = sqrt(n_legit/n_fraud) ~= 24`
- Optimizer: Adam, lr = 5e-4, batch size 2048, up to 60 epochs, seed 42, CPU
- Validation: stratified 10% slice of the training set used for epoch selection (best val F1); test set untouched until final evaluation
- Saved artifact: `models/mlp_model.pt`

### Why not the full imbalance ratio as pos_weight?

The first attempt used the full ratio (~577). Two failure modes appeared:

1. With batch size 512, most batches contained zero frauds while the rare fraud-carrying batches produced enormous gradients - validation loss exploded after epoch 1.
2. The weighted validation loss became dominated by only ~39 fraud samples, making "best epoch" selection meaningless (it picked an undertrained epoch; test F1 collapsed to 0.095).

Using the square root of the ratio (~24) keeps a strong recall boost without distorting the boundary; validation is monitored with **unweighted** BCE so epoch selection stays interpretable. Batch size 2048 ensures every batch sees several fraud examples.

### Results (test set)

At default threshold 0.5 (best val-F1 weights, epoch 5):

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.9991 | 0.6833 | 0.8367 | 0.7523 |

Confusion matrix: TN 56,826 / FP 38 / FN 16 / TP 82 - only 16 missed frauds vs LR's 36.

### Threshold sweep

Weighted training shifts predicted probabilities upward, so higher thresholds become useful:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.4645 | 0.8673 | 0.6050 |
| 0.4 | 0.6296 | 0.8673 | 0.7296 |
| 0.5 | 0.6833 | 0.8367 | 0.7523 |
| 0.6 | 0.7257 | 0.8367 | 0.7773 |
| 0.7 | 0.7664 | 0.8367 | 0.8000 |
| 0.8 | 0.8000 | 0.8163 | 0.8081 |
| 0.9 | 0.8144 | 0.8061 | 0.8103 |

Best F1 = **0.8103 at threshold 0.9** (Precision 0.81 / Recall 0.81) - comparable to the Decision Tree and approaching KNN, while catching 79 of 98 frauds.

### Overfitting analysis

- Training (weighted) loss fell monotonically 0.40 -> 0.011 over 60 epochs.
- Unweighted validation loss reached its minimum 0.0078 at epoch 47 and ended at 0.0079 - essentially flat, so **no meaningful overfitting** occurred, in contrast to the unpruned Decision Tree's 0.25 Train/Test gap.
- Best-val-F1 epoch selection plus Dropout(0.2) keep the model honest; Train F1 0.799 vs Test F1 0.752 shows only a modest gap (+0.047).
- Takeaway: with moderate class weighting, sufficiently large batches, dropout, and validation-based checkpointing, a small MLP trains stably on this highly imbalanced problem and delivers the strongest Recall of all four models.

---

## Cross-Validation Reference

5-Fold Stratified CV (shuffle, seed 42) for the sklearn models:

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8698 (+/-0.0263) | 0.6180 (+/-0.0645) | 0.7213 (+/-0.0510) |
| K-Nearest Neighbors (K=5) | 0.9271 (+/-0.0336) | 0.7846 (+/-0.0419) | 0.8490 (+/-0.0266) |
| Decision Tree (max_depth=10) | 0.8776 (+/-0.0301) | 0.7563 (+/-0.0562) | 0.8115 (+/-0.0387) |

The MLP uses a held-out validation split instead of k-fold CV (epoch-wise checkpointing needs a fixed validation set); wrapping it in cross-validation would be a natural next step.

Note: scalings are computed once on the full training data rather than refit inside each fold; a sklearn `Pipeline` would remove this minor caveat entirely.
