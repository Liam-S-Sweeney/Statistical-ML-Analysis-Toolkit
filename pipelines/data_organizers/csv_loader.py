import numpy as np
import pandas as pd

from pipelines.data_organizers import session_data
from pipelines.data_organizers.file_pathways import MAIN_CSV


def load_raw(path=MAIN_CSV) -> pd.DataFrame:
    """Load raw dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {path.resolve()}\n\n"
            "Upload a CSV in the app, or place one in 'files/master_csvs/'.\n"
        )
    return pd.read_csv(path, low_memory=False)


def clean_numeric(df: pd.DataFrame, missing_codes=None) -> pd.DataFrame:
    """Convert to numeric and replace missing codes with NaN."""
    if missing_codes is None:
        missing_codes = session_data.get_setting("MISSING_CODES")
    numeric_cols = df.select_dtypes(include="number").columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].replace(list(missing_codes), np.nan)
    return df


def load_clean(path=None) -> pd.DataFrame:
    """
    Return the cleaned active dataset.

    Resolution order:
      1. A DataFrame uploaded in this session (set via session_data)
      2. An explicit `path` argument
      3. The configured CSV on disk (local / dev use)

    Every pipeline module calls this, so honouring the session store here is
    what lets the whole toolkit run on uploaded data without any further
    pipeline changes.
    """
    session_df = session_data.get_active_df()
    if session_df is not None and path is None:
        return clean_numeric(session_df)

    return clean_numeric(load_raw(path or MAIN_CSV))
