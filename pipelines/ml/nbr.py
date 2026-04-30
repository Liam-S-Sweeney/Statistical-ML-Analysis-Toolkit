import numpy as np
import statsmodels.api as sm
from config import ID_VAR
from pipelines.data_organizers.impossible_var_cleaner import endo_exo_clean_impossible_var
from pipelines.data_organizers.file_pathways import REGRESSION_ANALYSIS_OUTPUT_FOLDER

def run_nbr(
    endo: list,
    exo: list,
    id_var: str = ID_VAR
):
    """
    Negative Binomial Regression:\n
    - Similar to Poisson, characterizes count data where majority of data points are clusterd towards lower values\n
    - Used to model count data where the variance is higher than the mean\n
    Outputs model summary and assumption checks
    """
    endo_var = endo[0] if isinstance(endo, list) else endo

    df = endo_exo_clean_impossible_var(endo_var, *exo, id_var)
    X = sm.add_constant(df[exo])
    y = df[endo_var]

    print(y.describe())
    print(y.unique())
    print("Negative values:", (y < 0).sum())
    print("Non-integers:", (y != y.astype(int)).sum())

    # Validate outcome is appropriate for NB
    if (y < 0).any():
        raise ValueError(
            f"[NBR ERROR] '{endo_var}' contains negative values. "
            "Negative binomial requires non-negative integer counts. "
            "This variable may be standardized — use the raw version."
        )
    if (y != y.astype(int)).any():
        raise ValueError(
            f"[NBR ERROR] '{endo_var}' contains non-integer values. "
            "Negative binomial requires count data."
        )

    # Negative Binomial
    result = sm.NegativeBinomial(endog=y, exog=X).fit(method='bfgs', maxiter=1000, disp=False)

    # Assumption checks
    warnings = []

    mean_y = float(y.mean())
    var_y = float(y.var())
    if var_y <= mean_y:
        warnings.append(
            f"[NOTE] Variance ({round(var_y, 4)}) <= Mean ({round(mean_y, 4)}). "
            "Data may not be overdispersed."
        )

    pct_zeros = float((y == 0).mean())
    if pct_zeros > 0.5:
        warnings.append(
            f"[WARNING] {pct_zeros:.1%} of outcome values are zero."
        )

    # Output
    out_dir = REGRESSION_ANALYSIS_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_title = '-'.join([endo_var] + exo)

    irr = np.exp(result.params)
    with open(out_dir / f"{cols_title}-nb.txt", 'w') as f:
        f.write(result.summary().as_text())
        f.write("\n\nIncidence Rate Ratios (exp(coef)):\n")
        f.write(irr.to_string())
        if warnings:
            f.write("\n\nAssumption Checks:\n")
            for w in warnings:
                f.write(f"{w}\n")
    return None