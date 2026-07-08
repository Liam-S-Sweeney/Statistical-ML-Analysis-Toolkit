import numpy as np
import pandas as pd
import pytest
from pipelines.utility.ols_ssa import simple_slopes


# Same coefficients/covariance as the logistic test so slope & SE are directly
# comparable and hand-verifiable; df_resid fixed at 196.
KNOWN = dict(
    b_focal=0.8, b_interaction=0.2,
    cov_focal=0.01, cov_interaction=0.03, cov_cross=0.005,
    df_resid=196.0, moderator_var='W',
)


def test_simple_slope_matches_hand_calc():
    out = simple_slopes(**KNOWN, probe_vals=[1.0])
    row = out.iloc[0]
    assert row['simple_slope'] == pytest.approx(1.0, abs=1e-4)
    assert row['SE'] == pytest.approx(0.223607, abs=1e-4)
    assert row['t'] == pytest.approx(4.472136, abs=1e-4)
    # function rounds p to 4dp, so a true p of 1.3e-5 is reported as 0.0;
    # assert the meaningful property (highly significant) rather than the floored value
    assert row['p'] < 0.0001
    assert row['CI_low'] == pytest.approx(0.559016, abs=1e-4)
    assert row['CI_high'] == pytest.approx(1.440984, abs=1e-4)
    assert bool(row['sig']) is True


def test_zero_moderator_gives_focal_coefficient():
    out = simple_slopes(**KNOWN, probe_vals=[0.0])
    assert out.iloc[0]['simple_slope'] == pytest.approx(0.8, abs=1e-4)


def test_ci_is_symmetric_about_slope():
    out = simple_slopes(**KNOWN, probe_vals=[1.0])
    row = out.iloc[0]
    midpoint = (row['CI_low'] + row['CI_high']) / 2
    assert midpoint == pytest.approx(row['simple_slope'], abs=1e-4)


def test_one_row_per_probe_value():
    out = simple_slopes(**KNOWN, probe_vals=[-1.0, 0.0, 1.0])
    assert len(out) == 3