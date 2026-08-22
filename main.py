"""
Main entry point for the Credit Card Fraud Detection Pipeline.

Run this file to execute the complete pipeline:
  1. Download/load the dataset
  2. Analyze data quality
  3. Split and scale features
  4. Fit encoder and save it
  5. Train and evaluate models
  6. Run all experiments
  7. Save the final model

All artifacts are saved in models/:
  models/model.pkl    — trained classifier
  models/encoder.pkl  — fitted encoder (passthrough for numeric features)
  models/scaler.pkl   — fitted StandardScaler
"""

import sys
import os

# Add the src/ directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_prep import run_data_pipeline
from src.train import run_training_pipeline


def main():
    """Run the complete fraud detection pipeline."""
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 60)

    # Step 1: Data preparation (load, analyze, split, encode, scale)
    # This saves models/encoder.pkl and models/scaler.pkl
    X_train_scaled, X_test_scaled, y_train, y_test = run_data_pipeline()

    # Step 2: Model training, evaluation, experiments, and model saving
    # This saves models/model.pkl and verifies all 3 artifacts exist
    run_training_pipeline(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Saved artifacts:")
    print("  - models/model.pkl    (trained classifier)")
    print("  - models/encoder.pkl  (fitted passthrough encoder)")
    print("  - models/scaler.pkl   (fitted StandardScaler)")
    print()
    print("Run predictions:")
    print("  python src/predict.py input.json output.json")


if __name__ == "__main__":
    main()
