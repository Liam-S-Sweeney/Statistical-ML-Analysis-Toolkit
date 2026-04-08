import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER

def run_ols(endo, exo):
    df = endo_exo_clean_impossible_var(endo, *exo)

    X = sm.add_constant(df[exo])
    y = df[endo]

    # Ordinary Least Squares
    ols_result = OLS(y, X).fit()

    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    endo_str = endo[0] if isinstance(endo, list) else endo
    cols_title = '-'.join([endo_str] + exo)

    with open(out_dir / f"{cols_title}-ols.txt", 'w') as f:
        f.write(ols_result.summary().as_text())
    return None

