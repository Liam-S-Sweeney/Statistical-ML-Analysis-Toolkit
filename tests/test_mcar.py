from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.utility.mcar import em_mvn, little_mcar_test
 

AIRQUALITY = Path(__file__).parent.parent / "files" / "master_csvs" / "airquality.csv"


def _correlated_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array(
        [
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.3],
            [0.2, 0.1, 0.3, 1.0],
        ]
    )
    X = rng.multivariate_normal(np.zeros(4), cov, size=n)
    return pd.DataFrame(X, columns=["a", "b", "c", "d"])


@pytest.mark.skipif(not AIRQUALITY.exists(), reason="airquality.csv not found")
def test_matches_r_naniar_on_airquality():
    # Reference: naniar::mcar_test(airquality) in R
    # -> statistic 35.1, df 14, p 0.00142, 4 patterns
    df = pd.read_csv(AIRQUALITY).drop(columns="rownames")
    result = little_mcar_test(df)
    assert result.statistic == pytest.approx(35.106, abs=1e-3)
    assert result.df == 14
    assert result.p_value == pytest.approx(0.001418, abs=1e-5)
    assert result.n_patterns == 4


def test_mcar_data_is_not_rejected():
    df = _correlated_data(2000, seed=1)
    rng = np.random.default_rng(2)
    mask = rng.random(df.shape) < 0.15  # missing purely at random
    df = df.mask(mask)
    result = little_mcar_test(df)
    assert result.p_value > 0.05


def test_mar_data_is_rejected():
    df = _correlated_data(2000, seed=3)
    # 'b' goes missing whenever 'a' is high: depends on an observed variable.
    df.loc[df["a"] > 0.5, "b"] = np.nan
    result = little_mcar_test(df)
    assert result.p_value < 0.001


def test_degrees_of_freedom():
    df = _correlated_data(200, seed=4)
    df.loc[:19, "a"] = np.nan            # pattern: a missing (3 observed)
    df.loc[20:39, ["b", "c"]] = np.nan   # pattern: b, c missing (2 observed)
    result = little_mcar_test(df)
    # complete (4) + 3 + 2 observed, minus p = 4
    assert result.df == 4 + 3 + 2 - 4
    assert result.n_patterns == 3


def test_em_on_complete_data_equals_ml_estimates():
    X = _correlated_data(500, seed=5).to_numpy()
    em = em_mvn(X)
    assert em.converged
    np.testing.assert_allclose(em.mean, X.mean(axis=0))
    np.testing.assert_allclose(em.cov, np.cov(X, rowvar=False, bias=True))


def test_rows_missing_everything_are_dropped():
    df = _correlated_data(300, seed=6)
    df.loc[:9, "a"] = np.nan
    df.loc[10:12, :] = np.nan
    result = little_mcar_test(df)
    assert result.n_dropped_all_missing == 3
    assert result.n_used == 297


def test_no_missing_data_raises():
    with pytest.raises(ValueError, match="No missing values"):
        little_mcar_test(_correlated_data(100, seed=7))


def test_non_numeric_column_raises():
    df = _correlated_data(100, seed=8)
    df["group"] = "x"
    with pytest.raises(ValueError, match="Non-numeric"):
        little_mcar_test(df, columns=["a", "b", "group"])


def test_constant_column_raises():
    df = _correlated_data(100, seed=9)
    df["d"] = 1.0
    df.loc[:9, "a"] = np.nan
    with pytest.raises(ValueError, match="zero variance"):
        little_mcar_test(df)