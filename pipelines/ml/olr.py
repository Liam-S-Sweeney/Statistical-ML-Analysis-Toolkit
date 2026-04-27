import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER

def run_olr(endo, exo, id_var=ID_VAR):
    endo_str = endo[0] if isinstance(endo, list) else endo
    df = endo_exo_clean_impossible_var(endo_str, *exo, id_var)
    
    endog_series = df[endo_str].astype('category')
    endog_ordered = pd.Categorical(endog_series, categories=sorted(endog_series.unique()), ordered=True)

    # Ordinal Logistic Regression
    olr_result = OrderedModel(
        endog=endog_ordered,
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

