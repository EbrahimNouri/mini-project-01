import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def download_data() -> str:
    try:
        import kagglehub
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = os.path.join(path, "creditcard.csv")
        print(f"Dataset downloaded to: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place creditcard.csv in the data/ directory.")
        return "data/creditcard.csv"


# [Phase 1] Load the dataset and show its structure (shape, head, dtypes).
def load_data(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Number of Samples : {df.shape[0]}")
    print(f"Number of Features: {df.shape[1]}")
    print()

    print("First 5 rows:")
    print(df.head())
    print()

    print("Data Types:")
    print(df.dtypes)
    print()

    return df


# [Phase 1 + Phase 2] Analyze missing values, duplicates and class distribution.
def analyze_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nTotal Missing Values: {total_missing}")
    if total_missing > 0:
        print("Columns with missing values:")
        print(missing[missing > 0])
    else:
        print("No missing values found in any column.")

    n_duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {n_duplicates}")
    if n_duplicates > 0:
        print(f"Percentage of duplicates: {n_duplicates / len(df) * 100}%")

    print("\nClass Distribution:")
    class_counts = df["Class"].value_counts()
    print()
    print(class_counts)
    print()

    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_ratio = fraud_count / len(df) * 100

    print(f"Legitimate Transactions : {legit_count}")
    print(f"Fraudulent Transactions: {fraud_count}")
    print(f"Fraud Ratio             : {fraud_ratio:.4f}%")
    print()
    print("NOTE: The dataset is highly imbalanced. Accuracy alone is misleading!")
    print("=" * 60)


# [Section 15] Encoder artifact (identity passthrough) saved for the prediction pipeline.
def fit_encoder(X_train: pd.DataFrame,
                save_path: str = "models/encoder.pkl"
                ) -> ColumnTransformer:
    print("\n" + "=" * 60)
    print("ENCODER FITTING")
    print("=" * 60)

    feature_columns = list(X_train.columns)

    encoder = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", feature_columns)
        ],
        remainder="drop"
    )

    encoder.fit(X_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(encoder, save_path)
    print(f"Encoder saved to: {save_path}")
    print(f"Features handled: {len(feature_columns)} (all passed through)")

    return encoder


# [Phase 2] Stratified train/test split to preserve the fraud ratio.
def prepare_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Print split summary
    print(f"Training set size : {X_train.shape[0]} samples")
    print(f"Test set size     : {X_test.shape[0]} samples")
    print(f"Training fraud %  : {y_train.mean() * 100:.4f}%")
    print(f"Test fraud %      : {y_test.mean() * 100:.4f}%")

    return X_train, X_test, y_train, y_test


# [Phase 2] Feature scaling fitted on train only AFTER split (no data leakage).
def scale_feature(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: str = "models/scaler.pkl"
):
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to: {save_path}")

    return X_train_scaled, X_test_scaled


def run_data_pipeline(csv_path: str = "data/creditcard.csv"):
    df = load_data(csv_path)
    analyze_data(df)
    X_train, X_test, y_train, y_test = prepare_and_split(df)
    fit_encoder(X_train)
    X_train_scaled, X_test_scaled = scale_feature(X_train, X_test)
    print("\nData preparation complete!")
    print("Saved artifacts: models/encoder.pkl, models/scaler.pkl")
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_data_pipeline()
