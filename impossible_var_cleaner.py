import numpy as np
from config import IMPOSSIBLE_ZERO_VARS
from data_loader import load_clean

def clean_impossible_var(clean_df=load_clean(), *cols, impossible_zero_vars=IMPOSSIBLE_ZERO_VARS):
    df = clean_df[list(cols)].dropna().copy()

    impossible_zero_cols = [c for c in impossible_zero_vars if c in df.columns]

    threshold = 0
    skewed_val = 0

    for col in impossible_zero_cols:
        proportion = (df[col] == 0).mean()
        if proportion > threshold:
            print(f"{col}: {proportion:.1%} {skewed_val} detected — replacing with NaN")
            df[col] = df[col].replace(0, np.nan)

    df = df.dropna()
    print(f"Final df shape after cleaning: {df.shape}")
    return df
###