"""
Data Preparation Module for Credit Card Fraud Detection.

This module handles all data-related operations:
  1. Downloading the dataset from Kaggle
  2. Loading and exploring the data
  3. Analyzing class distribution and data quality
  4. Fitting a passthrough encoder (identity transform for numeric features)
  5. Splitting data into train/test sets (stratified)
  6. Scaling features (fit on train, transform both)

The correct order is critical to avoid Data Leakage:
  Split → Fit Encoder on Train → Fit Scaler on Train → Transform Train & Test

All preprocessing artifacts are saved in models/:
  models/encoder.pkl
  models/scaler.pkl
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# ---------------------------------------------------------------------------
# 1. Download the dataset from Kaggle (only once)
# ---------------------------------------------------------------------------
def download_data() -> str:
    """
    Download the Credit Card Fraud dataset from Kaggle.

    Returns the path to the downloaded CSV file.
    Uses kagglehub to handle authentication and caching automatically.
    """
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


# ---------------------------------------------------------------------------
# 2. Load the dataset and perform basic analysis
# ---------------------------------------------------------------------------
def load_data(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load the creditcard.csv dataset into a Pandas DataFrame.

    Also prints a quick summary:
      - Shape (rows, columns)
      - First few rows
      - Data types
      - Descriptive statistics
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Print basic information about the dataset
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Number of Samples : {df.shape[0]}")
    print(f"Number of Features: {df.shape[1]}")
    print()

    # Show the first 5 rows so we can inspect the data
    print("First 5 rows:")
    print(df.head())
    print()

    # Show data types and non-null counts
    print("Data Types:")
    print(df.dtypes)
    print()

    return df


# ---------------------------------------------------------------------------
# 3. Analyze data quality (missing values, duplicates, class balance)
# ---------------------------------------------------------------------------
def analyze_data(df: pd.DataFrame) -> None:
    """
    Perform a thorough data quality analysis:
      - Missing values per column
      - Duplicate rows
      - Class distribution (fraud vs legitimate)
      - Class imbalance ratio
    """
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    # --- Missing values ---
    # Count missing values in every column
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nTotal Missing Values: {total_missing}")
    if total_missing > 0:
        print("Columns with missing values:")
        print(missing[missing > 0])
    else:
        print("No missing values found in any column.")

    # --- Duplicate rows ---
    # Duplicate rows can inflate metrics, so we check and report them
    n_duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {n_duplicates}")
    if n_duplicates > 0:
        print(f"Percentage of duplicates: {n_duplicates / len(df) * 100:.2f}%")

    # --- Class distribution ---
    # The target column is 'Class': 0 = Legitimate, 1 = Fraud
    print("\nClass Distribution:")
    class_counts = df["Class"].value_counts()
    print(class_counts)
    print()

    # Calculate the fraud ratio to understand the imbalance
    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_ratio = fraud_count / len(df) * 100

    print(f"Legitimate Transactions : {legit_count}")
    print(f"Fraudulent Transactions: {fraud_count}")
    print(f"Fraud Ratio             : {fraud_ratio:.4f}%")
    print()
    print("NOTE: The dataset is highly imbalanced. Accuracy alone is misleading!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 4. Fit a passthrough encoder (identity transform for all numeric features)
# ---------------------------------------------------------------------------
def fit_encoder(
    X_train: pd.DataFrame,
    save_path: str = "models/encoder.pkl",
) -> ColumnTransformer:
    """
    Fit a ColumnTransformer that passes all features through unchanged.

    This dataset has no categorical features (V1-V28 are PCA-transformed),
    so the encoder is a passthrough.  This satisfies the project requirement
    for models/encoder.pkl and makes the pipeline easily extendable if
    categorical features are added later.

    Args:
        X_train: Training features (DataFrame).
        save_path: Where to save the fitted encoder.

    Returns:
        The fitted ColumnTransformer.
    """
    print("\n" + "=" * 60)
    print("ENCODER FITTING")
    print("=" * 60)

    # Get all feature column names from the training DataFrame
    feature_columns = list(X_train.columns)

    # Create a ColumnTransformer with a passthrough (identity) transformer.
    # All columns are passed through without any transformation.
    # This serves as a placeholder — if categorical features are added later,
    # we can replace 'passthrough' with OneHotEncoder or OrdinalEncoder.
    encoder = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", feature_columns),
        ],
        remainder="drop",  # Drop any columns not specified (safety measure)
    )

    # Fit the encoder on training data only
    encoder.fit(X_train)

    # Save the fitted encoder to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(encoder, save_path)
    print(f"Encoder saved to: {save_path}")
    print(f"Features handled: {len(feature_columns)} (all passed through)")

    return encoder


# ---------------------------------------------------------------------------
# 5. Prepare features and target, then split (stratified)
# ---------------------------------------------------------------------------
def prepare_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Separate features (X) and target (y), then split into train/test.

    Uses stratified splitting so that the proportion of fraud cases is
    preserved in both the training and test sets.  This is essential for
    imbalanced datasets.

    Args:
        df: The full DataFrame (must contain a 'Class' column).
        test_size: Fraction of data reserved for testing (default 0.2).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    # Separate features (all columns except 'Class') from the target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified train/test split ensures the same class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,           # Preserve class distribution
        random_state=random_state,
    )

    # Print split summary
    print(f"Training set size : {X_train.shape[0]} samples")
    print(f"Test set size     : {X_test.shape[0]} samples")
    print(f"Training fraud %  : {y_train.mean() * 100:.4f}%")
    print(f"Test fraud %      : {y_test.mean() * 100:.4f}%")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. Fit a scaler on TRAINING data, transform both sets
# ---------------------------------------------------------------------------
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: str = "models/scaler.pkl",
):
    """
    Apply StandardScaler to features.

    IMPORTANT — to avoid Data Leakage:
      1. Fit the scaler ONLY on the training data.
      2. Use that same fitted scaler to transform both train and test.

    This ensures that information from the test set never influences
    the preprocessing step.

    Args:
        X_train: Training features.
        X_test:  Test features.
        save_path: Where to save the fitted scaler.

    Returns:
        X_train_scaled, X_test_scaled (as NumPy arrays)
    """
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit on training data ONLY, then transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same fitted scaler to transform the test data
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler so we can use it later for predictions
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to: {save_path}")

    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# 7. Convenience function — run the entire preparation pipeline
# ---------------------------------------------------------------------------
def run_data_pipeline(csv_path: str = "data/creditcard.csv"):
    """
    Execute the full data preparation pipeline in one call:
      1. Load data
      2. Analyze data quality
      3. Split into train/test
      4. Fit encoder and save to models/encoder.pkl
      5. Scale features and save to models/scaler.pkl

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Step 1: Load the raw dataset
    df = load_data(csv_path)

    # Step 2: Analyze data quality (missing values, duplicates, class balance)
    analyze_data(df)

    # Step 3: Split into train and test sets (stratified)
    X_train, X_test, y_train, y_test = prepare_and_split(df)

    # Step 4: Fit encoder on training data and save it
    fit_encoder(X_train)

    # Step 5: Scale features (fit on train only) and save scaler
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print("\nData preparation complete!")
    print("Saved artifacts: models/encoder.pkl, models/scaler.pkl")
    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------------
# Allow running this module directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_data_pipeline()
