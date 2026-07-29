"""
Packaged defaults.

These are fallbacks only. At runtime the app writes user-chosen values into
`pipelines.data_organizers.session_data`, and the pipelines resolve from there,
so nothing in this file is dataset-specific any more.
"""

# Optional local-development CSV. Leave as-is for the hosted app; the file
# uploader takes precedence whenever a dataset has been uploaded.
MAIN_CSV_NAME = 'dataset.csv'

ID_VAR = ''

MISSING_CODES = [-99, -999, -9999]
IMPOSSIBLE_ZERO_VARS = []

HUE_COL = None
SIZE_COL = None
PALETTE = None

# Moderated regression
DATA_THRESHOLD = 1

# GMM
DX = ''
DPI = 300
K_MIN = 1
K_MAX = 11
N_INIT = 10
RAND_STATE = 0
REG_COVAR = 1e-3

# LDA
CV = 5
