from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from .data_prep import run_data_pipeline
except ImportError:
    from data_prep import run_data_pipeline

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

SEED = 42
EPOCHS = 40
BATCH_SIZE = 2048
LEARNING_RATE = 0.0005
VALIDATION_FRACTION = 0.1
THRESHOLD = 0.5
REPORT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_tensors(X, y):
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    return X_t, y_t


@torch.no_grad()
def predict_proba(model, X_t, batch_size=8192):
    model.eval()
    probs = []
    for start in range(0, len(X_t), batch_size):
        probs.append(torch.sigmoid(model(X_t[start:start + batch_size])))
    return torch.cat(probs).numpy()


@torch.no_grad()
def average_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        total += criterion(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)


def metrics_at_threshold(y_true, proba, threshold=THRESHOLD):
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
    }


def train_mlp_model(X_train_scaled, X_test_scaled, y_train, y_test) -> dict:
    print("\n" + "=" * 60)
    print("BONUS MODEL: SIMPLE MLP (PyTorch)")
    print("=" * 60)

    set_seeds()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=SEED,
    )

    X_tr_t, y_tr_t = make_tensors(X_tr, y_tr)
    X_val_t, y_val_t = make_tensors(X_val, y_val)
    X_test_t, _ = make_tensors(X_test_scaled, y_test)
    X_full_train_t, _ = make_tensors(X_train_scaled, y_train)

    n_pos = float((y_tr_t == 1).sum())
    n_neg = float((y_tr_t == 0).sum())
    pos_weight_value = n_neg / n_pos

    print(f"\nTraining rows   : {len(X_tr_t)} (fraud: {int(n_pos)})")
    print(f"Validation rows : {len(X_val_t)} (fraud: {int((y_val_t == 1).sum())})")
    print(f"Test rows       : {len(X_test_t)} (fraud: {int(np.asarray(y_test).sum())})")
    print(f"BCE pos_weight  : {pos_weight_value:.1f} (= n_legit / n_fraud)")
    print(f"Architecture    : {X_tr_t.shape[1]} -> 64 -> 32 -> 1 (ReLU + Dropout(0.2))")

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)

    model = FraudMLP(X_tr_t.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]))
    eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    print(f"\n{'Epoch':>5} | {'Train Loss*':>11} | {'Val Loss':>8} | {'Val F1':>7}")
    print("-" * 45)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)
        val_loss = average_loss(model, val_loader, eval_criterion)
        val_f1 = f1_score(
            np.asarray(y_val), (predict_proba(model, X_val_t) >= THRESHOLD).astype(int)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"{epoch:>5} | {train_loss:>11.4f} | {val_loss:>8.4f} | {val_f1:>7.4f}{marker}")

    model.load_state_dict(best_state)

    train_metrics = metrics_at_threshold(
        np.asarray(y_train), predict_proba(model, X_full_train_t)
    )
    test_proba = predict_proba(model, X_test_t)
    test_metrics = metrics_at_threshold(np.asarray(y_test), test_proba)

    accuracy = test_metrics["accuracy"]
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]
    f1 = test_metrics["f1"]
    cm = test_metrics["confusion"]
    tn, fp, fn, tp = cm.ravel()

    print(f"\nMLP TEST SET EVALUATION at default threshold {THRESHOLD} "
          f"(best val-F1 weights, epoch {best_epoch})")
    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(cm)
    print()
    print(f"True Negatives  (correctly predicted legit): {tn}")
    print(f"False Positives (predicted fraud, actually legit): {fp}")
    print(f"False Negatives (predicted legit, actually fraud): {fn}  <-- DANGEROUS!")
    print(f"True Positives  (correctly predicted fraud): {tp}")
    print()
    print("Classification Report:")
    print(classification_report(np.asarray(y_test), test_metrics["y_pred"],
                                target_names=["Legitimate", "Fraud"]))

    print("THRESHOLD SWEEP (weighted training shifts probabilities upward):")
    print("\n| Threshold | Precision | Recall  | F1     |")
    print("|-----------|-----------|---------|--------|")
    sweep = []
    for thr in REPORT_THRESHOLDS:
        m = metrics_at_threshold(np.asarray(y_test), test_proba, threshold=thr)
        sweep.append({"threshold": thr,
                      "precision": m["precision"],
                      "recall": m["recall"],
                      "f1": m["f1"]})
        print(f"| {thr:>9} | {m['precision']:.4f}   | {m['recall']:.4f} | {m['f1']:.4f} |")
    best_sweep = max(sweep, key=lambda s: s["f1"])
    print(f"\nBest F1 on test: {best_sweep['f1']:.4f} at threshold {best_sweep['threshold']}")

    final = history[-1]
    min_val_loss = min(h["val_loss"] for h in history)
    min_val_loss_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    print("\nOVERFITTING ANALYSIS:")
    print("  - Train loss is the weighted objective "
          f"(pos_weight={pos_weight_value:.0f}); validation is monitored")
    print("    with UNWEIGHTED BCE so it stays interpretable.")
    print(f"  - Unweighted val loss minimum: {min_val_loss:.4f} at epoch "
          f"{min_val_loss_epoch}; final epoch: {final['val_loss']:.4f}")
    if final["val_loss"] > min_val_loss * 1.05:
        print("  - Val loss rose after its minimum -> overfitting late in training;")
        print("    saved weights come from the best-val-F1 epoch instead.")
    else:
        print("  - Val loss stayed near its minimum -> no strong overfitting.")
    print(f"  - Train F1 {train_metrics['f1']:.4f} vs Test F1 {f1:.4f} "
          f"(gap {train_metrics['f1'] - f1:+.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "mlp_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X_tr_t.shape[1]),
            "threshold": THRESHOLD,
            "pos_weight": pos_weight_value,
            "best_epoch": best_epoch,
        },
        model_path,
    )
    print(f"\nMLP saved to: {model_path}")

    return {
        "model_name": "Simple MLP (PyTorch)",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "best_epoch": best_epoch,
        "history": history,
    }


if __name__ == "__main__":
    data_path = MODEL_DIR.parent / "data" / "creditcard.csv"
    splits = run_data_pipeline(str(data_path))
    X_train_scaled, X_test_scaled, _, _, y_train, y_test = splits
    train_mlp_model(X_train_scaled, X_test_scaled, y_train, y_test)
