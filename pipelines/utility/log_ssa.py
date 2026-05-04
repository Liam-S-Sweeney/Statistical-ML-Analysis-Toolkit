import numpy as np
import pandas as pd
from scipy.stats import norm


def simple_slopes_logistic(
    result,
    focal_var: str,
    moderator_var: str,
    interaction_var: str,
    probe_vals: list,
    df: pd.DataFrame,
    endo_var: str
) -> pd.DataFrame:
    """
    Simple Slopes Analysis for Logistic MRA.\n
    Tests whether the focal predictor slope is significant\n
    at each probe value of the moderator.\n
    """
    focal_sd = float(df[focal_var].std())
    focal_probes = {
        'focal_low (-1SD)':  -focal_sd,
        'focal_mean':         0.0,
        'focal_high (+1SD)':  focal_sd,
    }

    cov = result.cov_params().loc[
        ['const', focal_var, moderator_var, interaction_var],
        ['const', focal_var, moderator_var, interaction_var]
    ].values

    rows = []
    for mod_val in probe_vals:

        # Simple slope at this moderator value
        simple_slope = (
            result.params[focal_var]
            + result.params[interaction_var] * mod_val
        )

        # Gradient for slope SE — only focal and interaction terms matter
        grad_slope = np.array([
            0,        # const
            1,        # focal
            0,        # moderator
            mod_val   # interaction
        ])
        var_slope = grad_slope @ cov @ grad_slope
        se_slope = np.sqrt(var_slope)

        z = simple_slope / se_slope
        p = 2 * (1 - norm.cdf(abs(z)))

        for focal_label, focal_val in focal_probes.items():

            # Predicted log-odds and probability at this combination
            log_odds = (
                result.params['const']
                + result.params[focal_var] * focal_val
                + result.params[moderator_var] * mod_val
                + result.params[interaction_var] * focal_val * mod_val
            )
            prob = 1 / (1 + np.exp(-log_odds))

            # Delta method CI for predicted probability
            grad_pred = np.array([
                1,
                focal_val,
                mod_val,
                focal_val * mod_val
            ])
            var_log_odds = grad_pred @ cov @ grad_pred
            se_log_odds = np.sqrt(var_log_odds)

            ci_low_prob  = 1 / (1 + np.exp(-(log_odds - 1.96 * se_log_odds)))
            ci_high_prob = 1 / (1 + np.exp(-(log_odds + 1.96 * se_log_odds)))

            rows.append({
                'moderator_val':  round(mod_val, 4),
                'focal_level':    focal_label,
                'simple_slope':   round(simple_slope, 4),
                'se_slope':       round(se_slope, 4),
                'OR':             round(np.exp(simple_slope), 4),
                'z':              round(z, 4),
                'p':              round(p, 4),
                'sig':            p < 0.05,
                'log_odds':       round(log_odds, 4),
                'prob':           round(prob, 4),
                'CI_low_prob':    round(ci_low_prob, 4),
                'CI_high_prob':   round(ci_high_prob, 4),
            })

    return pd.DataFrame(rows)