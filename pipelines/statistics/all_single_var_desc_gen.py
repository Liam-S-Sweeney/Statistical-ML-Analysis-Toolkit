import pandas as pd
from pipelines.data_organizers.csv_loader import load_clean
from pipelines.utility.cdfs import compute_descriptives_for_series
from pipelines.data_organizers.file_pathways import ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER

def all_single_var_descriptive_generator():
    df = load_clean()
    out_dir = ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        row = compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        pd.DataFrame([row]).to_csv(out_dir / f"{col}_descriptive.csv", index=False)
        