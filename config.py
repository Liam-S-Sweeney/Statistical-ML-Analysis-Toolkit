"""
UPDATE THIS TO NO LONGER BE ABOUT LAB VVV
"""

MAIN_CSV_NAME = 'SB_W1_W2_W3_W4_merged_raw.csv'

ID_VAR = 'ID1C'

MISSING_CODES = [-99, -999, -9999]
IMPOSSIBLE_ZERO_VARS = [
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI'
    ] 

HUE_COL = None
SIZE_COL = None
PALETTE = None

# MRA Specific
DATA_THRESHOLD = 1

# GMM Specific
DX = 'W4_day_al_30_C4'
DPI = 300
K_MIN = 1
K_MAX = 11
N_INIT = 10
RAND_STATE = 0
REG_COVAR = 1e-3

# LDA Specific
CV = 5