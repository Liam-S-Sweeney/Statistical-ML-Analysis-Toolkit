import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


def simple_slopes(
        b_focal: float,
        b_interaction: float,
        cov_focal: float,
        cov_interaction: float,
        cov_cross: float,
        df_resid: float,
        moderator_var: str,
        probe_vals: list[float],
) -> pd.DataFrame:
    """
    For Outcome (Y) ~ focal predictor (X) + moderator (W) + focal * moderator (X*W)\n
    Slope of focal predictor at each probe value of moderator = b_focal + b_interaction * probe_val\n
    SE via delta method using model vcov
    """
    rows=[]

    for z_val in probe_vals:
        slope = b_focal + b_interaction * z_val
        var_slope = cov_focal + 2 * z_val * cov_cross + z_val**2 * cov_interaction

        se = np.sqrt(max(var_slope, 0))
        t_stat = slope / se
        p_val = float(t_dist.sf(np.abs(t_stat), df=df_resid) * 2)
        ci_low = slope - se * float(t_dist.ppf(0.975, df=df_resid))
        ci_high = slope + se * float(t_dist.ppf(0.975, df=df_resid))

        rows.append({
            f"{moderator_var}_val": round(z_val, 4),
            "simple_slope": round(slope, 4),
            'SE': round(se, 4),
            't': round(t_stat, 4),
            'p': round(p_val, 4),
            "CI_low": round(ci_low, 4),
            "CI_high": round(ci_high, 4),
            "sig": p_val < 0.05,
        })
    return pd.DataFrame(rows)
