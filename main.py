import sys
import os
import pyfiglet


# Add the src/ directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_prep import run_data_pipeline
from src.train import run_training_pipeline
from src.train_mlp import train_mlp_model


def main():
    """Run the complete fraud detection pipeline."""
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
    print("  - models/scaler.pkl    (fitted StandardScaler)")
    print("  - models/encoder.pkl   (fitted passthrough encoder)")
    print("  - models/mlp_model.pt  (bonus PyTorch MLP)")
    print()
    print("Run predictions:")
    print("  python src/predict.py input.json")


if __name__ == "__main__":
    main()
