import numpy as np
import pandas as pd
import scipy.stats as stats


def _shape_cutoff(n: int):
    """Kim (2013) critical |z| by sample size; None means use absolute values instead."""
    if n < 50:
        return 1.96
    if n <= 300:
        return 3.29
    return None  # n > 300: judge |skew| > 2 or excess kurtosis > 4


def compute_descriptives_for_series(series: pd.Series, name: str, position: int) -> dict:
    non_na = pd.to_numeric(series, errors='coerce').dropna()
    values = non_na.to_numpy(dtype=float) if not non_na.empty else np.array([])

    out = {
        "position": position,
        "variable_name": name,
        "mean": np.nan, "median": np.nan, "mode": np.nan,
        "var": np.nan, "std": np.nan, "range": np.nan,
        "q1": np.nan, "q3": np.nan, "iqr": np.nan,
        "skew": np.nan, "se_skew": np.nan, "z_skew": np.nan,          # NEW
        "kurtosis": np.nan, "se_kurt": np.nan, "z_kurt": np.nan,      # NEW
        "skew_nonnormal": np.nan, "kurt_nonnormal": np.nan,           # NEW
        "count": np.nan,
        "min": np.nan, "max": np.nan, "se_mean": np.nan,
        "sw_stat": np.nan, "sw_p": np.nan, "normal_sw": np.nan,
    }

    if non_na.empty:
        return out

    n = int(values.size)                                              # NEW

    # central tendency
    out["mean"] = round(non_na.mean(), 3)
    out["median"] = round(non_na.median(), 3)

    m = non_na.mode()
    out["mode"] = m.iloc[0] if not m.empty else np.nan

    # variability
    sd = non_na.std(ddof=1)                                           # CHANGED: keep unrounded
    out["var"] = round(non_na.var(ddof=1), 3)
    out["std"] = round(sd, 3)
    out["range"] = non_na.max() - non_na.min()
    out["q1"] = np.percentile(values, 25)
    out["q3"] = np.percentile(values, 75)
    out["iqr"] = out["q3"] - out["q1"]

    # shape
    has_variance = not np.isclose(values.std(ddof=0), 0.0, atol=1e-8)
    cutoff = _shape_cutoff(n)                                         # NEW

    if n >= 3 and has_variance:
        g1 = stats.skew(values, bias=False)
        se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))   # NEW
        z_skew = g1 / se_skew                                                # NEW
        out["skew"] = round(g1, 3)
        out["se_skew"] = round(se_skew, 3)                                   # NEW
        out["z_skew"] = round(z_skew, 3)                                     # NEW
        out["skew_nonnormal"] = (abs(z_skew) > cutoff) if cutoff else (abs(g1) > 2)  # NEW

        if n >= 4:                                                           # NEW: kurtosis SE needs n > 3
            g2 = stats.kurtosis(values, fisher=True, bias=False)             # CHANGED: fisher explicit
            se_kurt = 2 * se_skew * np.sqrt((n**2 - 1) / ((n - 3) * (n + 5)))  # NEW
            z_kurt = g2 / se_kurt                                            # NEW
            out["kurtosis"] = round(g2, 3)
            out["se_kurt"] = round(se_kurt, 3)                               # NEW
            out["z_kurt"] = round(z_kurt, 3)                                 # NEW
            out["kurt_nonnormal"] = (abs(z_kurt) > cutoff) if cutoff else (abs(g2) > 4)  # NEW

    # misc
    out["count"] = n
    out["min"] = float(non_na.min())
    out["max"] = float(non_na.max())
    out["se_mean"] = round(float(sd / np.sqrt(n)), 3)                 # CHANGED: uses unrounded SD

    # normality <- Only valid for n < 5000
    if 3 <= n <= 5000:
        sw_stat, sw_p = stats.shapiro(values)
        out["sw_stat"] = round(float(sw_stat), 3)
        out["sw_p"] = round(float(sw_p), 3)
        out["normal_sw"] = sw_p > 0.05

    return out
