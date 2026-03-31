import json
import os
import pyreadstat
import pandas as pd
import re
from config import UNMERGED_DIR, DATA_DIR

JOIN_KEY = 'ID1C'

def merge_csv(convert_dir=UNMERGED_DIR, output_path = DATA_DIR):
    df_ls = []
    for file in sorted(convert_dir.glob('*.csv')):
        wave_label = file.stem
        temp = pd.read_csv(file, low_memory=False)

        if JOIN_KEY not in temp.columns:
            print(f"WARNING: {file.name} has no '{JOIN_KEY}' column — skipping")
            continue

        temp.columns = [
            col if col == JOIN_KEY else f"{wave_label}_{col}"
            for col in temp.columns
        ]
        df_ls.append(temp)
        if not df_ls:
            raise ValueError(f"No files contained '{JOIN_KEY}' — check column name")

    from functools import reduce
    merged = reduce(lambda l, r: pd.merge(l, r, on=JOIN_KEY, how='outer'), df_ls)

    print(f"Merged shape: {merged.shape}")
    print(f"Subjects: {merged[JOIN_KEY].nunique()}")
    print(f"Full coverage (no NaN): {merged.dropna().shape[0]} rows")

    merged.to_csv(output_path / "merged_raw_df.csv", index=False)
    print("Saved to merged_raw_df.csv")


merge_csv()
# from pathlib import Path
# import pandas as pd

# for f in Path('unmerged_data_files').glob('*.csv'):
#     df = pd.read_csv(f, nrows=0)  # just headers
#     print(f.name, list(df.columns[:5]))  # first 5 cols of each file
