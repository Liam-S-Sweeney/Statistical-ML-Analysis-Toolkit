import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import DATA_THRESHOLD, ID_VAR
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.utility import dichotomize_count_var, log_ssa, probe_vals


def run_mblr(
        endo: str | list[str],             # Y - dependent / predicted
        focal_var: str | list[str],        # X - independent / main predictor
        moderator_var: str | list[str],    # W - alters strength or direction of relationship between X and Y
        id_var: str = ID_VAR,
        data_threshold: int = DATA_THRESHOLD
    ):
    """
    Moderated regression: endo ~ focal + moderator + focal*moderator\n
    Outputs model summary, simple slopes, and assumption checks.\n
    \n
    Parameters\n
    ----------\n
    endo            : outcome variable\n
    focal_var       : the predictor whose effect you want to understand (e.g. pain perception)\n
    moderator_var   : the variable that changes the focal effect (e.g. executive control)\n
    """
    endo = endo[0] if isinstance(endo, list) else endo
    focal_var = focal_var[0] if isinstance(focal_var, list) else focal_var
    moderator_var = moderator_var[0] if isinstance(moderator_var, list) else moderator_var
    exo = [focal_var, moderator_var]
    df = endo_exo_clean_impossible_var(endo, *exo, id_var)

    # Dichotomize vars based on config threshold
    df[endo] = dichotomize_count_var.dichotomize_count_var(df[endo], var_name=endo, threshold=data_threshold)

    # Mean-Centering
    df[f'{focal_var}_c'] = df[focal_var] - df[focal_var].mean()
    df[f'{moderator_var}_c'] = df[moderator_var] - df[moderator_var].mean()
    df[f'{focal_var}_x_{moderator_var}_c'] = df[f'{focal_var}_c'] * df[f'{moderator_var}_c']

    X = sm.add_constant(df[[f'{focal_var}_c', f'{moderator_var}_c', f'{focal_var}_x_{moderator_var}_c']])
    y = df[endo]

    # Attempt Logistic Regression for 'result'
    try:
        result = sm.Logit(endog=y, exog=X).fit(method='bfgs', maxiter=1000, disp=False)
    except Exception as e:
        out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{endo}-{focal_var}-{moderator_var}-mra.txt", 'w', encoding='utf-8') as f:
            f.write(f"[MRA ERROR] Model failed: {e}\n")
        return None

    # Variance Inflation Factor (VIF) Testing
    vif_vars = [f'{focal_var}_c', f'{moderator_var}_c', f'{focal_var}_x_{moderator_var}_c']
    vif_matrix = df[vif_vars].values
    vif_data = pd.DataFrame({
        'Variable': vif_vars,
        'VIF': [variance_inflation_factor(vif_matrix, i) for i in range(len(vif_vars))]
    })

    # Predictions + Confusion Matrix
    yhat = result.predict(X)
    prediction = list(map(round, yhat))
    cm = confusion_matrix(y, prediction , labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=['Actual 0', 'Actual 1'],
        columns=['Predicted 0', 'Predicted 1']
    )

    # EPP Calculations
    n_events = int(y.sum())
    n_nonevents = int(len(y) - n_events)
    k = X.shape[1] - 1                      # exclude constant
    epp = min(n_events, n_nonevents) / k    # Peduzzi convention: minority class



    # Assumption Checks
    warnings = []
    # Check for complete separation
    pct_ones = float(y.mean())
    if pct_ones > 0.90:
        warnings.append(
            f"[WARNING] Severely imbalanced outcome ({pct_ones:.1%} positive). "
            "Risk of complete separation — interpret with caution.\n"
            )

    if not result.mle_retvals.get('converged', False):
        warnings.append("[WARNING] Model did not converge. Interpret all results with caution.\n")

    interaction_var = f'{focal_var}_x_{moderator_var}_c'

    interaction_p = result.pvalues[interaction_var]
    if interaction_p >= 0.05:
        warnings.append(
            f"[NOTE] Interaction term is non-significant (p={interaction_p:.4f}).\n"
            "*Simple slopes are exploratory only.\n"
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


    # Empirical Separation Detection
    max_se = np.nanmax(result.bse) if np.isfinite(result.bse).any() else np.inf
    separated = (not np.isfinite(result.bse).all()) or (max_se > 10) or (result.prsquared > 0.99)
    if separated:
        warnings.append(
            f"[WARNING] Evidence of complete/quasi-complete separation "
            f"(max SE = {max_se:.1f}, pseudo R² = {result.prsquared:.3f}).\n"
        )

    # Simple Slopes Analysis (only for estimable models)
    slopes_df = None
    pv, pm = probe_vals.p_vm(df[f'{moderator_var}_c'])
    if epp >= 5 and not separated:
        try:
            slopes_df = log_ssa.simple_slopes_logistic(
                result=result,
                focal_var=f'{focal_var}_c',
                moderator_var=f'{moderator_var}_c',
                interaction_var=interaction_var,
                probe_vals=pv,
                df=df,
            )
        except Exception as e:
            slopes_df = None
            warnings.append(f"[WARNING] Simple slopes computation failed: {e}\n")

    # Output
    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = f'{endo}-{focal_var}-{moderator_var}-mra'

    with open(out_dir / f"{cols_title}.txt", 'w', encoding='utf-8') as f:
        f.write(result.summary().as_text())
        f.write(
            "\n\n--- Variance Inflation Factors ---\n"
            f"{vif_data.to_string(index=False)}\n"
        )
        f.write(
            "\n\n--- Confusion Matrix ---\n"
            f"{cm_df}\n"
            "\n\n--- Test Accuracy ---\n"
            f"{accuracy_score(y, prediction)}\n"
        )
        if slopes_df is not None:
            f.write(
                "\n\n--- Simple Slopes Analysis ---\n"
                f"Probe method: {pm}\n"
                f"Focal predictor: {focal_var} (mean-centered)\n"
                f"Moderator: {moderator_var} (mean-centered; probe values in centered units)\n\n"
            )
            f.write(slopes_df.to_string(index=False))
        else:
            f.write("\n\n--- Simple Slopes Analysis ---\n")
            f.write("SSA suppressed: model non-estimable or slope computation failed (see Assumption Checks).\n")
        f.write(f"\nEPP: {epp:.1f} (events = {n_events}, predictors = {k})")
        if warnings:
            f.write("\n\n--- Assumption Checks ---\n")
            for w in warnings:
                f.write(f"{w}\n")

    return None
