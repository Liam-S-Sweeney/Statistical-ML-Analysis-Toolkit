from pathlib import Path

DATA_DIR = Path("data_files")
DATA_PATH = DATA_DIR / "merged_raw_df.csv"

CONVERT_DIR = Path('raw_files')

UNMERGED_DIR = Path('unmerged_data_files')

ID_VAR='ID1C'

SINGLE_VAR_DESCRIPTIVES_PATH = Path('single_var_descriptives')
MULTIVARIATE_ANALYSIS_PATH = Path('multivariate_analysis')
OUTPUT_PNGS_PATH = Path('output_pngs')
CORRELATIONAL_OUTPUT = Path('correlational_output')

MISSING_CODES = [-99, -999, -9999]
IMPOSSIBLE_ZERO_VARS = [
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI'
    ] 

HUE_COL = 'status'
SIZE_COL = 'status'
PALETTE = 'status'
DX = 'Outcome'
