from statsmodels.miscmodels.ordinal_model import OrderedModel
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER

def run_olr(endo, exo, id_var=ID_VAR):
    df = endo_exo_clean_impossible_var(endo, *exo, id_var)

    # Ordinal Logistic Regression
    olr_result = OrderedModel(
        endog=df[endo],
        exog=df[exo],
        distr='logit',
    ).fit(method='bfgs')

    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    endo_str = endo[0] if isinstance(endo, list) else endo
    cols_title = '-'.join([endo_str] + exo)

    with open(out_dir / f"{cols_title}-olr.txt", 'w') as f:
        f.write(olr_result.summary().as_text())
    return None

