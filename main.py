# [Section 2.1] Standard ML pipeline flow: raw data -> prep -> split -> scale -> train -> evaluate -> predict.
import sys
import os
import numpy as np
import pyfiglet


# Add the src/ directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_prep import run_data_pipeline
from src.train import run_training_pipeline
from src.train_mlp import train_mlp_model

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_full_pipeline():
    """Run the complete fraud detection pipeline: train everything and save."""
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 60)

    (
        X_train_scaled,
        X_test_scaled,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
    ) = run_data_pipeline()

    print(pyfiglet.figlet_format("SCALED"))
    run_training_pipeline(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_raw,
        X_test_raw,
    )

    print(pyfiglet.figlet_format("MLP"))
    train_mlp_model(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Saved artifacts:")
    print("  - models/model.pkl     (final Logistic Regression)")
    print("  - models/knn_model.pkl (K-Nearest Neighbors K=5)")
    print("  - models/dt_model.pkl  (Decision Tree max_depth=10)")
    print("  - models/scaler.pkl    (fitted StandardScaler)")
    print("  - models/encoder.pkl   (fitted passthrough encoder)")
    print("  - models/mlp_model.pt  (bonus PyTorch MLP)")
    print()
    print("Fast next run: python main.py fast")

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
    )


def run_fast_mode():
    """Reuse saved models: load from disk, quick evaluation, prediction demo.

    First time (or after deleting models/) it trains and saves first,
    then continues exactly like any other fast run.
    """
    from src.load_models import all_models_exist, evaluate_loaded_models, load_all_models

    if not all_models_exist():
        print("\nSaved models not found -> training them now "
              "(next 'fast' run will skip this)...")
        splits = run_full_pipeline()
    else:
        splits = run_data_pipeline()

    (
        X_train_scaled,
        X_test_scaled,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
    ) = splits

    models = load_all_models()
    evaluate_loaded_models(models, X_test_scaled, y_test)

    sample_input = os.path.join(PROJECT_DIR, "input.json")
    if not os.path.exists(sample_input):
        print(f"\nNo sample input found at {sample_input} - skipping demo.")
        return 0

    import json

    import pandas as pd
    import torch

    from src.load_models import load_scaler, mlp_predict_proba
    from src.predict import FEATURE_COLUMNS

    print("\n" + "=" * 60)
    print("PREDICTION DEMO: input.json (all four loaded models)")
    print("=" * 60)

    with open(sample_input, "r") as f:
        sample = json.load(f)
    X_sample = pd.DataFrame([{c: sample[c] for c in FEATURE_COLUMNS}])
    X_scaled = load_scaler().transform(X_sample)

    for name, model in models.items():
        if name == "Simple MLP":
            X_t = torch.as_tensor(np.asarray(X_scaled, dtype=np.float32))
            proba = float(mlp_predict_proba(model, X_t)[0])
        else:
            proba = float(model.predict_proba(X_scaled)[0][1])
        class_id = int(proba >= 0.5)
        result = {
            "prediction": "Fraud" if class_id else "Legitimate",
            "class_id": class_id,
            "probability": round(proba, 4),
            "threshold": 0.5,
            "status": "success",
        }
        print(f"{name:<30} -> {json.dumps(result)}")

    return 0


def print_usage():
    print(
        "Usage:\n"
        "  python main.py        full training pipeline (always retrains)\n"
        "  python main.py fast   reuse saved models (trains first only if files missing)"
    )


def main(argv):
    command = argv[1].lower() if len(argv) > 1 else ""

    if command == "":
        run_full_pipeline()
        return 0
    if command == "fast":
        return run_fast_mode()

    print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
    #TODO Ensemble Learning: KNN + MLP
