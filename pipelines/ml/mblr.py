import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import ID_VAR, DATA_THRESHOLD
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER
from pipelines.utility import dichotomize_count_var, log_ssa, probe_vals

def run_mblr(
        endo:list,              # Y - dependent / predicted
        focal_var: list,        # X - independent / main predictor
        moderator_var: list,    # W - alters strength or direction of relationship between X and Y
        id_var: str = ID_VAR,
        data_threshold: int = DATA_THRESHOLD
    ):
    """
    Moderated regression: endo ~ focal + moderator + focal*moderator\n
    Outputs model summary, simple slopes, and assumption checks.\n
    \n
    Parameters\n
    ----------\n
    endo        : outcome variable\n
    focal_var   : the predictor whose effect you want to understand (e.g. pain perception)\n
    moderator_var : the variable that changes the focal effect (e.g. executive control)\n
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
    
    result = sm.Logit(endog=y, exog=X).fit(method='bfgs', maxiter=1000, disp=False)

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

    # Assumption Checks
    warnings = []
    # Check for complete separation
    pct_ones = float(y.mean())
    if pct_ones > 0.90 or pct_ones < 0.10:
        warnings.append(
            f"[WARNING] Severely imbalanced outcome ({pct_ones:.1%} positive). "
            "Risk of complete separation — interpret with caution."
            )

    if not result.mle_retvals.get('converged', False):
        warnings.append("[WARNING] Model did not converge. Interpret all results with caution.")

    interaction_var = f'{focal_var}_x_{moderator_var}_c'

    interaction_p = result.pvalues[interaction_var]
    if interaction_p >= 0.05:
        warnings.append(
            f"[NOTE] Interaction term is non-significant (p={interaction_p:.4f}).\n"
            "*Simple slopes are exploratory only."
        )


    # Simple Slopes Analysis
    pv, pm = probe_vals.p_vm(df[f'{moderator_var}_c'])
    try:
        slopes_df = log_ssa.simple_slopes_logistic(
        result=result,
        focal_var=f'{focal_var}_c',
        moderator_var=f'{moderator_var}_c',
        interaction_var=interaction_var,
        probe_vals=pv,
        df=df,
        endo_var=endo
        )
    except ValueError as e:
        return e


    # Output
    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = f'{endo}-{focal_var}-{moderator_var}-mra'

    with open(out_dir / f"{cols_title}.txt", 'w') as f:
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
        f.write(
            "\n\n--- Simple Slopes Analysis ---\n"
            f"Probe method: {pm}\n"
            f"Focal predictor: {focal_var} (mean-centered)\n"
            f"Moderator: {moderator_var} (mean-centered; probe values in centered units)\n\n"
        )
        f.write(slopes_df.to_string(index=False))
        if warnings:
            f.write("\n\n--- Assumption Checks ---\n")
            for w in warnings:
                f.write(f"{w}\n")

    return None