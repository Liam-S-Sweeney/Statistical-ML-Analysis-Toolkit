import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER
from pipelines.utility.dichotomize_count_var import dichotomize_count_var

def run_blr(
    endo: list | list[str],
    exo: list | list[str],
    id_var: str = ID_VAR
):
    """
    Binary Logistic Regression:\n
    - Models probability of binary outcome (0/1)\n
    - Appropriate for dichotomized count DVs (any use vs. none)\n
    - Outputs model summary, odds ratios, and assumption checks\n
    
    Parameters\n
    ----------\n
    endo        : binary outcome variable (0 = no use, 1 = any use)\n
    exo         : predictor variable(s)\n
    """
    endo_var = endo[0] if isinstance(endo, list) else endo
    exo = [exo] if isinstance(exo, str) else exo
    df = endo_exo_clean_impossible_var(endo_var, *exo, id_var)

    df[endo_var] = dichotomize_count_var(df[endo_var], var_name=endo_var, threshold=1)

    X = sm.add_constant(df[exo])
    y = df[endo_var]

    # Validate outcome is binary
    unique_vals = sorted(y.dropna().unique())
    if not all(v in [0, 1] for v in unique_vals):
        raise ValueError(
            f"[BLR ERROR] '{endo_var}' is not binary. "
            f"Unique values found: {unique_vals}. "
            "Dichotomize before running BLR (0 = no use, 1 = any use)."
        )

    # EPP Calculations
    n_events = int(y.sum())
    n_nonevents = int(len(y) - n_events)
    k = X.shape[1] - 1                      # exclude constant
    epp = min(n_events, n_nonevents) / k    # Peduzzi convention: minority class

    # Assumption checks
    warnings = []
    pct_ones = float(y.mean())

    if pct_ones > 0.90:
        warnings.append(
            f"[WARNING] Outcome is severely imbalanced "
            f"({pct_ones:.1%} positive).\n"
            "Odds ratios may be unstable — interpret with caution."
        )

    if epp < 5:
        warnings.append(
            f"[WARNING] EPP = {epp:.1f} (events = {n_events}, predictors = {k}). "
            "Below minimum of 5 for stable ML estimation — treat as non-estimable.\n"
        )
    elif epp < 10:
        warnings.append(
            f"[NOTE] EPP = {epp:.1f} (events = {n_events}, predictors = {k}). "
            "Below conventional threshold of 10; estimates exploratory.\n"
        )

    # Fit BLR
    try:
        result = sm.Logit(endog=y, exog=X).fit(
            method='bfgs',
            maxiter=1000,
            disp=False
        )
    except Exception as e:
        with open(
            REGRESSION_ANALYSIS_OUTPUT_FOLDER
            / f"{'-'.join([endo_var] + exo)}-blr.txt", 'w', encoding='utf-8'
        ) as f:
            f.write(f"[BLR ERROR] Model failed: {str(e)}\n")
        return None

    # Odds Ratios + CIs
    odds_ratios = np.exp(result.params)
    ci = np.exp(result.conf_int())
    ci.columns = ['OR_CI_low', 'OR_CI_high']
    or_df = pd.concat([odds_ratios.rename('OR'), ci], axis=1)

    # Output
    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = '-'.join([endo_var] + exo)

    with open(out_dir / f"{cols_title}-blr.txt", 'w', encoding='utf-8') as f:
        f.write(result.summary().as_text())
        f.write("\n\nOdds Ratios (exp(coef)) with 95% CIs:\n")
        f.write(or_df.to_string())
        f.write(f"\n\nModel Fit:\n")
        f.write(f"AIC: {round(result.aic, 3)}\n")
        f.write(f"BIC: {round(result.bic, 3)}\n")
        f.write(f"Pseudo R-squared (McFadden): {round(result.prsquared, 3)}\n")
        f.write(f"Log-Likelihood: {round(result.llf, 3)}\n")
        f.write(f"LLR p-value: {round(result.llr_pvalue, 4)}\n")
        f.write(f"\nEPP: {epp:.1f} (events = {n_events}, predictors = {k})")
        if warnings:
            f.write("\n\nAssumption Checks:\n")
            for w in warnings:
                f.write(f"{w}\n")

    return None