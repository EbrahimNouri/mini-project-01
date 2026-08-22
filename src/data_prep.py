import os
import pandas as pd


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

    n_dplidates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {n_dplidates}")
    if n_dplidates > 0:
        print(f"Percentage of duplicates: {n_dplidates / len(df) * 100}%")

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
