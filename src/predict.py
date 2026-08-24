# [Section 14] Prediction script: JSON input -> JSON output using the saved model pipeline.
import os
import sys
import json
import numpy as np
import joblib


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


# [Section 14] One transaction -> encoder -> scaler -> model -> probability -> class label.
def predict_transaction(data: dict, threshold: float = 0.5) -> dict:

    model, encoder, scaler = load_artifacts()

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

    import pandas as pd
    input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])

    encoded_features = encoder.transform(input_df)

    scaled_features = scaler.transform(encoded_features)

    fraud_probability = model.predict_proba(scaled_features)[0][1]

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
    if len(sys.argv) not in (3, 4):
        print("Usage: python src/predict.py <input.json> <output.json> [threshold]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 0.5

    with open(input_path, "r") as f:
        input_data = json.load(f)

    # Run prediction (applies encoder => scaler => model)
    result = predict_transaction(input_data, threshold=threshold)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']}")
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
