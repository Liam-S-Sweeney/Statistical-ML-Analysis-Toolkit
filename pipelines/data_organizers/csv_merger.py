import json
import os
import pyreadstat
import pandas as pd
import re
from pathlib import Path
from config import ID_VAR
from pipelines.data_organizers.file_pathways import UNMERGED_CSVS_FOLDER, MASTER_CSVS_FOLDER

def merge_csv(convert_dir=UNMERGED_CSVS_FOLDER, output_path = MASTER_CSVS_FOLDER):
    all_csvs_name = ""
    df_ls = []
    for file in sorted(convert_dir.glob('*.csv')):
        wave_label = file.stem
        temp = pd.read_csv(file, low_memory=False)

        if ID_VAR not in temp.columns:
            print(f"WARNING: {file.name} has no '{ID_VAR}' column — skipping")
            continue

        temp.columns = [
            col if col == ID_VAR else f"{wave_label}_{col}"
            for col in temp.columns
        ]
        df_ls.append(temp)
        if not df_ls:
            raise ValueError(f"No files contained '{ID_VAR}' — check column name")
        all_csvs_name = all_csvs_name + Path(file).stem + '_'

    from functools import reduce
    merged = reduce(lambda l, r: pd.merge(l, r, on=ID_VAR, how='outer'), df_ls)

    print(f"Merged shape: {merged.shape}")
    print(f"Subjects: {merged[ID_VAR].nunique()}")
    print(f"Full coverage (no NaN): {merged.dropna().shape[0]} rows")

    merged.to_csv(output_path / f"{all_csvs_name}merged_raw.csv", index=False)
    print(f"Saved to {MASTER_CSVS_FOLDER}")

