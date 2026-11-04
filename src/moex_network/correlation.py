"""Returns, correlation and the correlation-based metric distance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_i(t) = ln P_i(t) - ln P_i(t-1)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of log-returns."""
    return returns.corr(method="pearson")


def correlation_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna (1999) distance d_ij = sqrt(2 (1 - rho_ij)).

    A proper metric: d = 0 for rho = 1, d = sqrt(2) for rho = 0, d = 2 for rho = -1.
    """
    values = np.sqrt(np.clip(2.0 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def distance_to_correlation(d):
    """Inverse of :func:`correlation_to_distance`: rho = 1 - d^2 / 2."""
    return 1.0 - np.asarray(d) ** 2 / 2.0


def pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Upper triangle of a correlation matrix as a long table (a, b, rho)."""
    iu = np.triu_indices_from(corr.values, k=1)
    cols = corr.columns
    return pd.DataFrame({
        "a": cols[iu[0]],
        "b": cols[iu[1]],
        "rho": corr.values[iu],
    })
