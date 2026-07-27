import numpy as np
import pandas as pd
import logging
from config import IMPOSSIBLE_ZERO_VARS
from pipelines.data_organizers.csv_loader import load_clean
from typing import Sequence

def clean_impossible_var(
    clean_df: pd.DataFrame,
    *cols: str,
    impossible_zero_vars: Sequence[str] = IMPOSSIBLE_ZERO_VARS,
    replace_val: int = 0,
    ):
    """
    Replaces all specified variables that contain any replacement value ("0" by default) that do not make sense with Nan.
    """
    logger = logging.getLogger(__name__)
    
    df = clean_df[list(cols)].copy()

    impossible_zero_cols = [c for c in impossible_zero_vars if c in df.columns]
    for col in impossible_zero_cols:
        proportion = (df[col] == 0).mean()
        if proportion > 0:
            logger.info(f"{col}: {proportion:.1%} zeros detected — replacing with NaN")
            df[col] = df[col].replace(replace_val, np.nan)

    df = df.dropna(subset=list(cols), how='any')
    
    logger.info(f"Final df shape after cleaning: {df.shape}")
    return df

def endo_exo_clean_impossible_var(*cols, 
    clean_df=None, 
    impossible_zero_vars=IMPOSSIBLE_ZERO_VARS,
    replace_val: int = 0,
    ):
    """
    Replaces all specified variables that contain any replacement value ("0" by default) that do not make sense with Nan.
    """
    logger = logging.getLogger(__name__)
    if clean_df is None:
        clean_df = load_clean()
    
    flats_cols = []
    for col in cols:
        if isinstance(col, list):
            flats_cols.extend(col)
        else:
            flats_cols.append(col)

    df = clean_df[flats_cols].copy()
    df = df.dropna(how='any')
    
    impossible_zero_cols = [c for c in impossible_zero_vars if c in df.columns]
    for col in impossible_zero_cols:
        proportion = (df[col] == 0).mean()
        if proportion > 0:
            logger.info(f"{col}: {proportion:.1%} zeros detected — replacing with NaN")
            df[col] = df[col].replace(replace_val, np.nan)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=flats_cols, how='any')
    logger.info(f"Final df shape after cleaning: {df.shape}")
    return df