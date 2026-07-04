import pandas as pd
import numpy as np
import logging


def dichotomize_count_var(
    series: pd.Series,
    var_name: str = None,
    threshold: int = 0
) -> pd.Series:
    """
    Dichotomizes a count variable into binary (0/1) for use in BLR.\n
    \n
    Parameters\n
    ----------\n
    series      : raw count variable\n
    var_name    : optional name for logging purposes\n
    threshold   : values above this become 1 (default: 0, meaning any use = 1)\n
    \n
    Returns\n
    -------\n
    pd.Series   : binary series (0 = False, 1 = True)\n
    """
    logger = logging.getLogger(__name__)
    name = var_name or series.name or "variable"

    # Validate input is count data
    non_na = pd.to_numeric(series, errors='coerce').dropna()
    if (non_na < 0).any():
        raise ValueError(
            f"[DICHOTOMIZE ERROR] '{name}' contains negative values. "
            "Expected non-negative count data."
        )
    if (non_na != non_na.astype(int)).any():
        raise ValueError(
            f"[DICHOTOMIZE ERROR] '{name}' contains non-integer values. "
            "Expected count data."
        )

    # Recode
    dichotomized = (series > threshold).astype(float)
    dichotomized[series.isna()] = np.nan

    # Summary log
    n_total = int(non_na.size)
    n_any_use = int((non_na > threshold).sum())
    n_no_use = n_total - n_any_use
    pct_use = n_any_use / n_total * 100

    logger.info(
        f"[DICHOTOMIZE] '{name}' | "
        f"threshold > {threshold} | "
        f"N = {n_total} | "
        f"No use (0): {n_no_use} ({100 - pct_use:.1f}%) | "
        f"Any use (1): {n_any_use} ({pct_use:.1f}%)"
    )

    return dichotomized