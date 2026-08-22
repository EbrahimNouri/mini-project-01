
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

