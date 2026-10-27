from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"


def load_data(file_path=DATA_PATH):
    """Load a CSV file into a Pandas DataFrame."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Please place sample_sales.csv inside the data folder."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


if __name__ == "__main__":
    df = load_data()

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Dataset shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Column names ---")
    print(df.columns.tolist())