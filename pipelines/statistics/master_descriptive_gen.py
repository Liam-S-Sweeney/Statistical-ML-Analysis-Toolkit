import pandas as pd
from config import MAIN_CSV_NAME
from pipelines.data_organizers.csv_loader import load_clean
from pipelines.utility.cdfs import compute_descriptives_for_series
from pipelines.data_organizers.file_pathways import MASTER_VAR_DESC_OUTPUT_FOLDER

def master_descriptive_csv_generator():
    df = load_clean()
    rows = [
        compute_descriptives_for_series(df[col], col, df.columns.get_loc(col))
        for col in df.columns
        ]
    
    out_dir = MASTER_VAR_DESC_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{MAIN_CSV_NAME.split('.')[0]}_master_descriptives.csv", index=False)