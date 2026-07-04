import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import t as t_dist
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER
from pipelines.utility import ols_ssa, probe_vals

def run_mols(
        endo:list,              # Y - dependent / predicted
        focal_var: list,        # X - independent / main predictor
        moderator_var: list,    # W - alters strength or direction of relationship between X and Y
        id_var: str = ID_VAR
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

    interaction_var = f"{focal_var}_x_{moderator_var}"
    
    # Mean-Centering
    df[f'{focal_var}_c'] = df[focal_var]    - df[focal_var].mean()
    df[f'{moderator_var}_c'] = df[moderator_var] - df[moderator_var].mean()
    df[interaction_var] = df[f'{focal_var}_c'] * df[f'{moderator_var}_c']

    X = sm.add_constant(df[[f'{focal_var}_c', f'{moderator_var}_c', interaction_var]])
    y = df[endo]

    result = sm.OLS(endog=y, exog=X).fit()

    # Assumption Checks
    warnings = []
    sw_stat, sw_p = stats.shapiro(result.resid)
    sw_p = float(sw_p)
    if sw_p < 0.05:
        warnings.append(
            f"[WARNING] Residuals non-normal (Shapiro-Wilk p={sw_p:.4f}). "
            "Interpret with caution."
        )

    _, bp_p, _, _ = het_breuschpagan(result.resid, X)
    bp_p = float(bp_p)
    if bp_p < 0.05:
        warnings.append(
            f"[WARNING] Heteroscedasticity detected (Breusch-Pagan p={bp_p:.4f}). "
            "Consider robust SEs."
        )

    interaction_p = result.pvalues[interaction_var]
    if interaction_p >= 0.05:
        warnings.append(
            f"[NOTE] Interaction term is non-significant (p={interaction_p:.4f}).\n"
            "*Simple slopes are exploratory only."
        )

    # Simple Slopes Vars
    b_focal = float(result.params[f'{focal_var}_c'])
    b_interaction = float(result.params[interaction_var])
    cov_focal = float(result.cov_params().loc[f'{focal_var}_c', f'{focal_var}_c'])
    cov_interaction = float(result.cov_params().loc[interaction_var, interaction_var])
    cov_cross = float(result.cov_params().loc[f'{focal_var}_c', interaction_var])
    df_resid = float(result.df_resid)
    


    # Simple Slopes Analysis
    pv, pm = probe_vals.p_vm(df[f'{moderator_var}_c'])
    slopes_df = ols_ssa.simple_slopes(
        b_focal=b_focal,
        b_interaction=b_interaction,
        cov_focal=cov_focal,
        cov_interaction=cov_interaction,
        cov_cross=cov_cross,
        df_resid= df_resid,
        moderator_var=moderator_var,
        probe_vals=pv,
    )

    # Output
    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = f'{endo}-{focal_var}-{moderator_var}-mra'

    with open(out_dir / f"{cols_title}.txt", 'w') as f:
        f.write(result.summary().as_text()) 
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