import os
import sys
import pandas as pd

def ensure_output_dir(path: str) -> str:
    """Ensure output directory exists and return its path."""
    os.makedirs(path, exist_ok=True)
    return path

def load_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV with basic error handling and column cleanup."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print(f"Loaded data: {csv_path}")
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)
