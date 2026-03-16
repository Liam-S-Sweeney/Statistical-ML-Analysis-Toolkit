from pathlib import Path

DATA_DIR = Path("data_files")
DATA_PATH = DATA_DIR / "diabetes.csv"
SINGLE_VAR_DESCRIPTIVES_PATH = Path('single_var_descriptives')
MULTIVARIATE_ANALYSIS_PATH = Path('multivariate_analysis')
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
