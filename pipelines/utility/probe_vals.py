import pandas as pd
import scipy.stats as stats


def p_vm(series: pd.Series) -> tuple[list[float], str]:
    """
    Returns probe values and the method used.\n
    Uses +/-1 SD if moderator is normal, quartiles if skewed.
    """
    sw_stat, sw_p = stats.shapiro(series.dropna())
    if sw_p > 0.05:
        mean, sd = series.mean(), series.std()
        return [mean - sd, mean, mean + sd], "mean +/-1 SD"
    else:
        q25, q50, q75 = series.quantile([0.25, 0.5, 0.75])
        return [q25, q50, q75], "25th-50th-75th percentiles"
