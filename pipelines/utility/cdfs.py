from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_descriptives_for_series(series: pd.Series, name: str, position: int) -> dict:
    non_na = pd.to_numeric(series, errors='coerce').dropna()
    values = non_na.to_numpy(dtype=float) if not non_na.empty else np.array([])

    out = {
        "position": position,
        "variable_name": name,
        "mean": np.nan, "median": np.nan, "mode": np.nan,
        "var": np.nan, "std": np.nan, "range": np.nan,
        "q1": np.nan, "q3": np.nan, "iqr": np.nan,
        "skew": np.nan, "kurtosis": np.nan,
        "count": np.nan,
        "min": np.nan, "max": np.nan, "se_mean": np.nan,
        "sw_stat": np.nan, "sw_p": np.nan, "normal_sw": np.nan,
    }

    if non_na.empty:
        return out

    # central tendency
    out["mean"] = round(non_na.mean(),3)
    out["median"] = round(non_na.median(),3)
    
    m = non_na.mode()
    out["mode"] = m.iloc[0] if not m.empty else np.nan

    # variability
    out["var"] = round(non_na.var(ddof=1),3)
    out["std"] = round(non_na.std(ddof=1),3)
    out["range"] = non_na.max() - non_na.min()
    out["q1"] = np.percentile(values, 25)
    out["q3"] = np.percentile(values, 75)
    out["iqr"] = out["q3"] - out["q1"]

    # shape
    if values.size >= 3 and not np.isclose(values.std(ddof=0), 0.0, atol=1e-8):
        out["skew"] = round(stats.skew(values, bias=False),3)
        out["kurtosis"] = round(stats.kurtosis(values, bias=False),3)

    # misc
    out["count"] = int(values.size)
    out["min"] = float(non_na.min())
    out["max"] = float(non_na.max())
    out["se_mean"] = round(float(out["std"] / np.sqrt(out["count"])) if out["count"] > 0 else np.nan,3)

    # normality <- Only valid for n < 5000
    if 3 <= values.size <= 5000:
        sw_stat, sw_p = stats.shapiro(values)
        out["sw_stat"] = round(float(sw_stat),3)
        out["sw_p"] = round(float(sw_p),3)
        out["normal_sw"] = sw_p > 0.05

    return out