import pandas as pd
from data_loader import load_clean
from cdfs import compute_descriptives_for_series
from config import DATA_DIR, SINGLE_VAR_DESCRIPTIVES_PATH

def master_descriptive_csv_generator():
    df = load_clean()
    rows = [
        compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        for col in df.columns
        ]
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "master_descriptives.csv", index=False)

def all_single_var_descriptive_csv_generator():
    df = load_clean()
    out_dir = SINGLE_VAR_DESCRIPTIVES_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        row = compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        pd.DataFrame([row]).to_csv(out_dir / f"{col}_descriptive.csv", index=False)