from pathlib import Path
from config import MAIN_CSV_NAME

######

""" ALERT """
# This file connects all pathways:
#     - Changing this will likely compromise many features
""" ALERT """

######

""" MAIN FOLDER """
MAIN_FOLDER = Path(__file__).parent.parent.parent

""" Primary Folder Pathways """
FILES_FOLDER = MAIN_FOLDER / 'files'
OUTPUTS_FOLDER = MAIN_FOLDER / 'outputs'
PIPELINES_FOLDER = MAIN_FOLDER / 'pipelines'
APP_STYLES_FOLDER = MAIN_FOLDER / 'app_styles'

""" Secondary Folder Pathways """
# Files
MASTER_CSVS_FOLDER = FILES_FOLDER / 'master_csvs'
NON_CSVS_FOLDER = FILES_FOLDER / 'non_csvs'
UNMERGED_CSVS_FOLDER = FILES_FOLDER / 'unmerged_csvs'

# Outputs
ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'all_var_desc_analysis_output'
FIGURE_PNGS_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'figure_pngs_output'
GMM_ANALYSIS_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'gmm_analysis_output'
MASTER_VAR_DESC_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'master_var_desc_output'
MULTI_VAR_ANALYSIS_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'multi_var_analysis_output'
REGRESSION_ANALYSIS_OUTPUT_FOLDER = OUTPUTS_FOLDER / 'regression_analysis_output'
# Pipelines
DATA_ORGANIZERS_FOLDER = PIPELINES_FOLDER / 'data_organizers'
ML_FOLDER = PIPELINES_FOLDER / 'ml'
STATISTICS_FOLDER = PIPELINES_FOLDER / 'statistics'
UTILITY_FOLDER = PIPELINES_FOLDER / 'utility'

""" Tertiary Folders Pathway """
# Statistics
PNG_GENERATORS = STATISTICS_FOLDER / 'png_generators'
# Figure PNGs
BIC_AIC_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'bic_aic_vis'
CEV_PCA_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'cev_pca'
DESC_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'desc_vis'
GMM_HM_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'gmm_hm_vis'
GMM_PP_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'gmm_pp_vis'
HM_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'hm_vis'
PG_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'pg_vis'
PP_VIS = FIGURE_PNGS_OUTPUT_FOLDER / 'pp_vis'

""" Tertiary File Pathways """
# Master CSVs
MAIN_CSV = MASTER_CSVS_FOLDER / MAIN_CSV_NAME

# All Essential Folders
RUNTIME_FOLDERS = [
    MASTER_CSVS_FOLDER,
    NON_CSVS_FOLDER,
    UNMERGED_CSVS_FOLDER,
    ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER,
    FIGURE_PNGS_OUTPUT_FOLDER,
    GMM_ANALYSIS_OUTPUT_FOLDER,
    MULTI_VAR_ANALYSIS_OUTPUT_FOLDER,
    REGRESSION_ANALYSIS_OUTPUT_FOLDER,
    BIC_AIC_VIS,
    CEV_PCA_VIS,
    DESC_VIS,
    GMM_HM_VIS,
    GMM_PP_VIS,
    HM_VIS,
    PG_VIS,
    PP_VIS,
]
