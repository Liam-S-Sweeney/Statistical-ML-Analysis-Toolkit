import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan

from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var


def run_ols(endo, exo):
    df = endo_exo_clean_impossible_var(endo, *exo)

    X = sm.add_constant(df[exo])
    y = df[endo]

    # Ordinary Least Squares
    ols_result = OLS(endog=y, exog=X).fit()

    # Non-normality + Heteroscedasticity Tests
    warnings = []

    sw_stat, sw_p = stats.shapiro(ols_result.resid)
    if sw_p < 0.05:
        warnings.append(f"[WARNING] --> Residuals are non-normal (Shapiro-Wilk p={round(float(sw_p), 4)})")

    bp_lm, bp_lm_p, bp_f, bp_fp = het_breuschpagan(ols_result.resid, X)
    if bp_lm_p < 0.05:
        warnings.append(f"[WARNING] --> Heteroscedasticity detected (Breusch-Pagan p={round(float(bp_lm_p), 4)})")

    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    endo_str = endo[0] if isinstance(endo, list) else endo
    cols_title = '-'.join([endo_str] + exo)

    with open(out_dir / f"{cols_title}-ols.txt", 'w') as f:
        f.write(ols_result.summary().as_text())
        if warnings:
            f.write('\n\n Assumption Checks: \n')
            for w in warnings:
                f.write(f'{w}\n')
    return None

