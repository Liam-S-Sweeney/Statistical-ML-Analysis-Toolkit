"""Little's (1988) test of Missing Completely At Random (MCAR).

Usage
-----
    from mcar import little_mcar_test

    result = little_mcar_test(df, columns=["age", "bmi", "glucose"])
    print(result.summary())
    result.patterns  # per-pattern breakdown as a DataFrame

How the test works
------------------
1. Estimate the mean vector and covariance matrix of all variables by
   maximum likelihood with the EM algorithm, assuming multivariate normality.
   EM uses every observed value, not just complete cases.
2. Group rows by missingness pattern (which columns are observed).
3. For each pattern j, compare the mean of its observed variables against
   the EM mean for those same variables, scaled by the EM covariance:

       d2_j = n_j * (ybar_j - mu_j)' Sigma_j^{-1} (ybar_j - mu_j)

4. Sum over patterns. Under MCAR, d2 ~ chi-square with
   df = sum(k_j) - p, where k_j is the number of observed variables in
   pattern j and p is the total number of variables.

A small p-value means pattern means differ more than chance allows, so the
data are not MCAR. A large p-value does NOT prove MCAR. The test has limited
power, only looks at means, and can miss MAR/MNAR mechanisms that leave
means unchanged.

Reference: Little, R. J. A. (1988). A test of missing completely at random
for multivariate data with missing values. JASA, 83(404), 1198-1202.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class EMResult:
    """Maximum-likelihood mean and covariance from the EM algorithm."""

    mean: np.ndarray
    cov: np.ndarray
    n_iter: int
    converged: bool


@dataclass(frozen=True)
class MCARTestResult:
    """Output of Little's MCAR test."""

    statistic: float
    df: int
    p_value: float
    n_patterns: int
    n_used: int
    n_dropped_all_missing: int
    columns: list[str]
    patterns: pd.DataFrame = field(repr=False)
    em: EMResult = field(repr=False)

    def is_mcar_rejected(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def summary(self, alpha: float = 0.05) -> str:
        lines = [
            "Little's MCAR test",
            f"  chi-square = {self.statistic:.3f}, df = {self.df}, "
            f"p = {self.p_value:.4g}",
            f"  {self.n_patterns} missingness patterns, "
            f"{self.n_used} rows used",
        ]
        if self.n_dropped_all_missing:
            lines.append(
                f"  ({self.n_dropped_all_missing} rows dropped: all "
                "selected variables missing)"
            )
        if not self.em.converged:
            lines.append(
                f"  WARNING: EM did not converge in {self.em.n_iter} "
                "iterations; treat the result with caution."
            )
        if self.is_mcar_rejected(alpha):
            lines.append(
                f"  p < {alpha}: reject MCAR. Missingness is related to the "
                "data, so complete-case analysis may be biased."
            )
        else:
            lines.append(
                f"  p >= {alpha}: no evidence against MCAR. This does not "
                "prove MCAR; the test has limited power."
            )
        return "\n".join(lines)


def em_mvn(
    X: np.ndarray, max_iter: int = 1000, tol: float = 1e-6
) -> EMResult:
    """ML estimates of mean and covariance for multivariate normal data
    with missing values (NaN), via the EM algorithm.

    Every row must have at least one observed value.
    """
    n, p = X.shape
    observed = ~np.isnan(X)

    # Starting values: available-case means and a diagonal covariance.
    mu = np.nanmean(X, axis=0)
    sigma = np.diag(np.nanvar(X, axis=0))

    # Process rows in pattern groups so each matrix solve happens once per
    # pattern instead of once per row.
    pattern_masks, pattern_ids = np.unique(
        observed, axis=0, return_inverse=True
    )
    pattern_ids = pattern_ids.ravel()
    groups = [
        (mask, X[pattern_ids == k]) for k, mask in enumerate(pattern_masks)
    ]

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))

        for obs_mask, rows in groups:
            mis_mask = ~obs_mask
            x_obs = rows[:, obs_mask]
            x_hat = np.empty_like(rows)
            x_hat[:, obs_mask] = x_obs
            cond_cov = np.zeros((p, p))

            if mis_mask.any():
                s_oo = sigma[np.ix_(obs_mask, obs_mask)]
                s_mo = sigma[np.ix_(mis_mask, obs_mask)]
                # Regression coefficients of missing on observed:
                # B = S_mo @ inv(S_oo), computed with solve for stability.
                B = np.linalg.solve(s_oo, s_mo.T).T
                # E-step: expected missing values given observed ones.
                x_hat[:, mis_mask] = mu[mis_mask] + (x_obs - mu[obs_mask]) @ B.T
                # Conditional covariance of the missing block. It adds the
                # uncertainty that plugging in x_hat alone would ignore.
                cond_cov[np.ix_(mis_mask, mis_mask)] = (
                    sigma[np.ix_(mis_mask, mis_mask)] - B @ s_mo.T
                )

            sum_x += x_hat.sum(axis=0)
            sum_xx += x_hat.T @ x_hat + len(rows) * cond_cov

        # M-step: ML (divide by n) mean and covariance.
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)

        change = max(
            np.max(np.abs(new_mu - mu)), np.max(np.abs(new_sigma - sigma))
        )
        mu, sigma = new_mu, new_sigma
        if change < tol:
            converged = True
            break

    return EMResult(mean=mu, cov=sigma, n_iter=n_iter, converged=converged)


def little_mcar_test(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> MCARTestResult:
    """Run Little's MCAR test on numeric columns of a DataFrame.

    Parameters
    ----------
    data : DataFrame with NaN for missing values.
    columns : columns to test. Defaults to all numeric columns. Categorical
        variables should be left out (or dummy-coded); the test assumes
        multivariate normality.
    max_iter, tol : EM convergence settings.

    Raises
    ------
    ValueError if columns are non-numeric, constant, entirely missing, or
    if there is no missing data to test.
    """
    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()
    else:
        non_numeric = [
            c for c in columns if not pd.api.types.is_numeric_dtype(data[c])
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns can't be tested: {non_numeric}. "
                "Drop them or dummy-code them first."
            )

    if len(columns) < 2:
        raise ValueError("Little's test needs at least 2 numeric columns.")

    X = data[columns].to_numpy(dtype=float)

    empty_cols = [c for c, empty in zip(columns, np.isnan(X).all(axis=0)) if empty]
    if empty_cols:
        raise ValueError(f"Columns with no observed values: {empty_cols}")

    all_missing_rows = np.isnan(X).all(axis=1)
    n_dropped = int(all_missing_rows.sum())
    X = X[~all_missing_rows]
    n, p = X.shape

    if not np.isnan(X).any():
        raise ValueError("No missing values in the selected columns.")

    constant_cols = [
        c for c, v in zip(columns, np.nanvar(X, axis=0)) if v == 0
    ]
    if constant_cols:
        raise ValueError(
            f"Columns with zero variance make the covariance singular: "
            f"{constant_cols}"
        )

    em = em_mvn(X, max_iter=max_iter, tol=tol)
    if not em.converged:
        warnings.warn(
            f"EM did not converge in {em.n_iter} iterations.", RuntimeWarning
        )

    observed = ~np.isnan(X)
    pattern_masks, pattern_ids = np.unique(
        observed, axis=0, return_inverse=True
    )
    pattern_ids = pattern_ids.ravel()

    d2 = 0.0
    df = 0
    rows_out = []
    for k, obs_mask in enumerate(pattern_masks):
        rows = X[pattern_ids == k][:, obs_mask]
        n_k = len(rows)
        diff = rows.mean(axis=0) - em.mean[obs_mask]
        sigma_k = em.cov[np.ix_(obs_mask, obs_mask)]
        contribution = float(n_k * diff @ np.linalg.solve(sigma_k, diff))
        d2 += contribution
        df += int(obs_mask.sum())
        rows_out.append(
            {
                "missing": ", ".join(
                    c for c, o in zip(columns, obs_mask) if not o
                )
                or "(none)",
                "n": n_k,
                "n_observed_vars": int(obs_mask.sum()),
                "d2_contribution": contribution,
            }
        )
    df -= p

    patterns = (
        pd.DataFrame(rows_out)
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    return MCARTestResult(
        statistic=d2,
        df=df,
        p_value=float(stats.chi2.sf(d2, df)),
        n_patterns=len(pattern_masks),
        n_used=n,
        n_dropped_all_missing=n_dropped,
        columns=list(columns),
        patterns=patterns,
        em=em,
    )