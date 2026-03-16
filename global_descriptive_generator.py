import pandas as pd
from pathlib import Path
from data_loader import load_clean
from cdfs import compute_descriptives_for_series

def master_descriptive_csv_generator():
    df = load_clean()
    rows = [
        compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        for col in df.columns
        ]
    pd.DataFrame(rows).to_csv(Path("data_files") / "master_descriptives.csv", index=False)

def all_single_var_descriptive_csv_generator():
    df = load_clean()
    out_dir = Path("single_var_descriptives")
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        row = compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        pd.DataFrame([row]).to_csv(out_dir / f"{col}_descriptive.csv", index=False)