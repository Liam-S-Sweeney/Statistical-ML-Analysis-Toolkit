import pandas as pd
import numpy as np
from config import DATA_PATH, MISSING_CODES

def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Load raw dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {path.resolve()}\n\n"
            "Place the dataset inside the 'data_files' folder (or update DATA_PATH in config.py).\n"
        )
    return pd.read_csv(path)

def clean_numeric(df: pd.DataFrame, missing_codes=MISSING_CODES) -> pd.DataFrame:
    """Convert to numeric and replace missing codes with NaN."""
    return (
        df.apply(pd.to_numeric, errors="coerce")
          .replace(missing_codes, np.nan)
    )

def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """Convenience function: load and clean dataset."""
    raw = load_raw(path)
    return clean_numeric(raw)