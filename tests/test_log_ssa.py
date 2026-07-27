import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from pipelines.utility.log_ssa import simple_slopes_logistic


class FakeResult:
    """Minimal stand-in for a fitted statsmodels Logit result.
    Exposes only what simple_slopes_logistic touches: .params and .cov_params()."""
    def __init__(self, params: pd.Series, cov: pd.DataFrame):
        self.params = params
        self._cov = cov
    def cov_params(self):
        return self._cov


@pytest.fixture
def known_model():
    params = pd.Series({'const': -0.5, 'X_c': 0.8, 'W_c': 0.3, 'X_x_W_c': 0.2})
    cov = pd.DataFrame(
        [[0.04, 0.00, 0.00, 0.000],
         [0.00, 0.01, 0.00, 0.005],
         [0.00, 0.00, 0.02, 0.000],
         [0.00, 0.005, 0.00, 0.030]],
        index=['const', 'X_c', 'W_c', 'X_x_W_c'],
        columns=['const', 'X_c', 'W_c', 'X_x_W_c'],
    )
    return FakeResult(params, cov)


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'X_c': rng.normal(0, 1, 200),
        'W_c': rng.normal(0, 1, 200),
        'endo': rng.integers(0, 2, 200),
    })


def test_simple_slope_matches_hand_calc(known_model, fake_df):
    out = simple_slopes_logistic(
        result=known_model, focal_var='X_c', moderator_var='W_c',
        interaction_var='X_x_W_c', probe_vals=[1.0], df=fake_df,
    )
    row = out.iloc[0]
    assert row['simple_slope'] == pytest.approx(1.0, abs=1e-4)
    assert row['se_slope'] == pytest.approx(0.223607, abs=1e-4)
    assert row['z'] == pytest.approx(4.472136, abs=1e-4)
    assert row['p'] == pytest.approx(0.000008, abs=1e-5)
    assert row['OR'] == pytest.approx(np.exp(1.0), abs=1e-4)
    assert bool(row['sig']) is True


def test_slope_constant_across_focal_levels(known_model, fake_df):
    out = simple_slopes_logistic(
        result=known_model, focal_var='X_c', moderator_var='W_c',
        interaction_var='X_x_W_c', probe_vals=[1.0], df=fake_df,
    )
    assert out['simple_slope'].nunique() == 1
    assert out['se_slope'].nunique() == 1


def test_zero_moderator_gives_focal_coefficient(known_model, fake_df):
    out = simple_slopes_logistic(
        result=known_model, focal_var='X_c', moderator_var='W_c',
        interaction_var='X_x_W_c', probe_vals=[0.0], df=fake_df,
    )
    assert out.iloc[0]['simple_slope'] == pytest.approx(0.8, abs=1e-4)