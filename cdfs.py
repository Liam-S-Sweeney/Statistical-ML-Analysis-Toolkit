from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_descriptives_for_series(series: pd.Series, name: str, position: int) -> dict:
    non_na = series.dropna()
    non_na = pd.to_numeric(non_na, errors='coerce').dropna()
    values = non_na.to_numpy(dtype=float) if not non_na.empty else np.array([])

    out = {
        "position": position,
        "variable_name": name,
        "mean": np.nan, "median": np.nan, "mode": np.nan,
        "var": np.nan, "std": np.nan, "range": np.nan,
        "q1": np.nan, "q3": np.nan, "iqr": np.nan,
        "skew": np.nan, "kurtosis": np.nan,
        "count": np.nan, "frequency": np.nan,
        "min": np.nan, "max": np.nan, "se_mean": np.nan,
    }

    if non_na.empty:
        return out

    # central tendency
    out["mean"] = non_na.mean()
    out["median"] = non_na.median()
    
    m = non_na.mode()
    out["mode"] = m.iloc[0] if not m.empty else np.nan

    # variability
    out["var"] = non_na.var()
    out["std"] = non_na.std()
    out["range"] = non_na.max() - non_na.min()
    out["q1"] = np.percentile(values, 25)
    out["q3"] = np.percentile(values, 75)
    out["iqr"] = out["q3"] - out["q1"]

    # shape
    if values.size >= 3 and not np.isclose(values.std(ddof=0), 0.0, atol=1e-8):
        out["skew"] = stats.skew(values)
        out["kurtosis"] = stats.kurtosis(values)

    # frequency
    unique_values, counts = np.unique(values, return_counts=True)
    if len(unique_values) > 20:
        out["frequency"] = "CONTINUOUS"
    else:
        percents = np.round((counts / counts.sum()) * 100, 2)
        out["frequency"] = [(float(uv), int(ct), float(p)) for uv, ct, p in zip(unique_values, counts, percents)]

    # misc
    out["count"] = int(values.size)
    out["min"] = float(non_na.min())
    out["max"] = float(non_na.max())
    out["se_mean"] = float(out["std"] / np.sqrt(out["count"])) if out["count"] > 0 else np.nan

    return out