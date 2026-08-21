"""
Prediction Module for Credit Card Fraud Detection.

This module loads a trained model, encoder, and scaler, receives input data
as JSON, performs prediction, and returns a JSON result.

The prediction pipeline mirrors the training pipeline exactly:
  Input JSON → Load artifacts → Encode → Scale → Predict → Output JSON

Usage (CLI):
    python src/predict.py input.json output.json

Usage (Python):
    from src.predict import predict_transaction
    result = predict_transaction(input_dict)
"""

import os
import sys
import json
import numpy as np
import joblib


# Default paths for saved artifacts (all must be in models/)
MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

# All feature columns the model expects (excluding 'Class')
FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
]


def load_artifacts(
    model_path: str = MODEL_PATH,
    encoder_path: str = ENCODER_PATH,
    scaler_path: str = SCALER_PATH,
):
    """
    Load the saved model, encoder, and scaler from disk.

    All three artifacts must exist in models/:
      - model.pkl   — the trained classifier
      - encoder.pkl — the fitted ColumnTransformer (passthrough for this dataset)
      - scaler.pkl  — the fitted StandardScaler

    Raises FileNotFoundError if any artifact is missing.
    """
    # Check that all required artifacts exist before loading
    for path, name in [
        (model_path, "model"),
        (encoder_path, "encoder"),
        (scaler_path, "scaler"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name.capitalize()} not found at {path}. "
                f"Run data_prep.py and train.py first."
            )

    # Load all three artifacts from disk
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    return model, encoder, scaler


def predict_transaction(data: dict, threshold: float = 0.5) -> dict:
    """
    Predict whether a single transaction is fraudulent.

    Preprocessing pipeline (must match training):
      1. Validate input features
      2. Build feature array from JSON data
      3. Apply the fitted encoder (passthrough for numeric features)
      4. Apply the fitted scaler (StandardScaler)
      5. Feed scaled features into the trained model
      6. Apply custom threshold to predicted probability

    Args:
        data: A dictionary with all 30 feature values (Time, V1..V28, Amount).
        threshold: Classification threshold for converting probability to label.

    Returns:
        A dictionary with prediction results:
          - prediction: "Fraud" or "Legitimate"
          - class_id: 1 or 0
          - probability: predicted probability of fraud
          - threshold: the threshold used
          - status: "success" or "error"
    """
    # Load the trained model, encoder, and scaler
    model, encoder, scaler = load_artifacts()

    # Validate that all required features are present in the input
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        return {
            "prediction": "Error",
            "class_id": -1,
            "probability": 0.0,
            "threshold": threshold,
            "status": "error",
            "message": f"Missing features: {missing}",
        }

    # Build the feature vector in the correct column order
    # Convert to DataFrame to maintain column names (required by encoder)
    import pandas as pd
    input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])

    # Step 1: Apply the fitted encoder (passthrough for this dataset)
    # This transforms the raw features using the same encoder fit during training
    encoded_features = encoder.transform(input_df)

    # Step 2: Apply the fitted scaler
    # This normalizes features using the same scaler fit during training
    scaled_features = scaler.transform(encoded_features)

    # Step 3: Get predicted probability of fraud (class 1) from the model
    fraud_probability = model.predict_proba(scaled_features)[0][1]

    # Step 4: Apply custom threshold to decide the class label
    class_id = 1 if fraud_probability >= threshold else 0
    prediction_label = "Fraud" if class_id == 1 else "Legitimate"

    return {
        "prediction": prediction_label,
        "class_id": class_id,
        "probability": round(float(fraud_probability), 4),
        "threshold": threshold,
        "status": "success",
    }


def main():
    """
    CLI entry point: python src/predict.py input.json output.json
    """
    if len(sys.argv) != 3:
        print("Usage: python src/predict.py <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load input data from JSON file
    with open(input_path, "r") as f:
        input_data = json.load(f)

    # Run prediction (applies encoder → scaler → model)
    result = predict_transaction(input_data)

    # Write result to output JSON file
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']}")
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
