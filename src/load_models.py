from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from .train_mlp import FraudMLP, predict_proba as mlp_predict_proba
except ImportError:
    from train_mlp import FraudMLP, predict_proba as mlp_predict_proba

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

SKLEARN_MODEL_FILES = {
    "Logistic Regression": "model.pkl",
    "KNN (K=5)": "knn_model.pkl",
    "Decision Tree (max_depth=10)": "dt_model.pkl",
}
MLP_FILE = "mlp_model.pt"
MLP_THRESHOLD = 0.5


def missing_model_files() -> list:
    missing = [f for f in SKLEARN_MODEL_FILES.values()
               if not (MODEL_DIR / f).exists()]
    if not (MODEL_DIR / MLP_FILE).exists():
        missing.append(MLP_FILE)
    return missing


def all_models_exist() -> bool:
    missing = missing_model_files()
    if missing:
        print("Missing model files:", ", ".join(f"models/{m}" for m in missing))
        return False
    return True


def load_all_models() -> dict:
    print("\n" + "=" * 60)
    print("LOADING SAVED MODELS")
    print("=" * 60)

    models = {}
    for name, fname in SKLEARN_MODEL_FILES.items():
        models[name] = joblib.load(MODEL_DIR / fname)
        print(f"  [OK] {name:<28} <- models/{fname}")

    checkpoint = torch.load(MODEL_DIR / MLP_FILE)
    mlp = FraudMLP(checkpoint["input_dim"])
    mlp.load_state_dict(checkpoint["state_dict"])
    mlp.eval()
    models["Simple MLP"] = mlp
    print(f"  [OK] {'Simple MLP':<28} <- models/{MLP_FILE} "
          f"(trained epoch {checkpoint.get('best_epoch', '?')})")

    return models


def load_scaler():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    print("  [OK] StandardScaler               <- models/scaler.pkl")
    return scaler


def _metric_row(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )


def evaluate_loaded_models(models: dict, X_test_scaled, y_test) -> None:
    y_test_np = np.asarray(y_test)

    print("\n" + "=" * 60)
    print("FAST EVALUATION OF LOADED MODELS (test set)")
    print("=" * 60)
    print("\n| Model                        | Accuracy | Precision | Recall  | F1     |")
    print("|------------------------------|----------|-----------|---------|--------|")

    for name, model in models.items():
        if name == "Simple MLP":
            X_t = torch.as_tensor(np.asarray(X_test_scaled, dtype=np.float32))
            proba = mlp_predict_proba(model, X_t)
            acc, p, r, f1 = _metric_row(
                y_test_np, (proba >= MLP_THRESHOLD).astype(int)
            )
        else:
            acc, p, r, f1 = _metric_row(y_test_np, model.predict(X_test_scaled))
        print(
            f"| {name:<28} | {acc:.4f}   | {p:.4f}    | {r:.4f}  | {f1:.4f} |"
        )

    print("\n(MLP evaluated at its default threshold "
          f"{MLP_THRESHOLD}; see reports/experiments.md for the sweep)")
