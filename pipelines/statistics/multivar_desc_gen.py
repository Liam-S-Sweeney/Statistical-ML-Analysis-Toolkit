import pandas as pd
from pipelines.data_organizers.csv_loader import load_clean
from pipelines.utility.cdfs import compute_descriptives_for_series
from pipelines.data_organizers.impossible_var_cleaner import clean_impossible_var
from pipelines.data_organizers.file_pathways import MULTI_VAR_ANALYSIS_OUTPUT_FOLDER

def explore_multi_variables(*cols):
    clean_df = load_clean()

    # build df from selected cols 
    df = clean_impossible_var(clean_df, *cols)

    if df.empty:
        print(f"No rows with data for: {cols}. Are these from different waves?")
        return

    descriptive_dict = [
        compute_descriptives_for_series(df[col], name=col, position=df.columns.get_loc(col))
        for col in cols
        ]
    
    out_dir = MULTI_VAR_ANALYSIS_OUTPUT_FOLDER
    cols_title = '-'.join(cols)
    pd.DataFrame(descriptive_dict).to_csv(out_dir / f"{cols_title}-multivar_desc.csv",index=False)
    
     
