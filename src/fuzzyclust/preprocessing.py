# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data preparation helpers.

Scaling is intentionally *not* part of the clustering estimators: use
:class:`sklearn.preprocessing.StandardScaler` / :class:`~sklearn.preprocessing.MinMaxScaler`
in a :class:`sklearn.pipeline.Pipeline`. The functions here reproduce the
MATLAB toolbox conveniences (``clust_normalize`` / ``clust_denormalize``) and
build regression matrices for dynamic-system identification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__all__ = ["normalize", "make_regressors", "lagged_matrix"]


def normalize(X, method: str = "var"):
    r"""Scale the data and return the scaler for inverse transformation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    method : {"var", "range"}, default="var"
        ``"var"``: zero mean and unit variance per feature
        (:math:`\tilde{x}_{kj} = (x_{kj}-\bar{x}_j)/\sigma_j`);
        ``"range"``: linear scaling of every feature to :math:`[0, 1]`.

    Returns
    -------
    X_scaled : ndarray or DataFrame (same type as the input)
    scaler : fitted scikit-learn scaler
        ``scaler.inverse_transform`` undoes the operation (MATLAB ``clust_denormalize``).

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.preprocessing import normalize
    >>> X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    >>> Xs, scaler = normalize(X, "range")
    >>> Xs.min(axis=0), Xs.max(axis=0)
    (array([0., 0.]), array([1., 1.]))
    """
    if method == "var":
        scaler = StandardScaler()
    elif method == "range":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'var' or 'range'.")
    if isinstance(X, pd.DataFrame):
        scaler.set_output(transform="pandas")
    return scaler.fit_transform(X), scaler


def lagged_matrix(x, lags: int, include_current: bool = False) -> np.ndarray:
    r"""Delay-embedding of a univariate series.

    Builds rows :math:`[x_{k-1}, \dots, x_{k-\text{lags}}]` (or
    :math:`[x_k, x_{k-1},\dots]` if ``include_current``) for every
    :math:`k\ge\text{lags}`.

    Returns
    -------
    ndarray of shape (len(x) - lags, lags + include_current)
    """
    x = np.asarray(x, dtype=float).ravel()
    T = x.shape[0]
    start = 0 if include_current else 1
    cols = [x[lags - d : T - d] for d in range(start, lags + 1)]
    return np.column_stack(cols)


def make_regressors(y, u=None, n_a: int = 1, n_b: int = 0, n_k: int = 1):
    r"""Regression matrix and target of a NARX model

    .. math::
        y_k = f\big(y_{k-1},\dots,y_{k-n_a},\; u_{k-n_k},\dots,u_{k-n_k-n_b+1}\big)

    Parameters
    ----------
    y : array-like of shape (T,) or (T, n_outputs)
        Output sequence.
    u : array-like of shape (T,) or (T, n_inputs), optional
        Input sequence (``None`` for an autonomous / time-series model).
    n_a : int, default=1
        Number of lagged outputs.
    n_b : int, default=0
        Number of lagged inputs (``0`` if ``u`` is ``None``).
    n_k : int, default=1
        Input delay (dead time) in samples.

    Returns
    -------
    Phi : ndarray of shape (T - max_lag, n_regressors)
        Rows :math:`\boldsymbol{\varphi}_k`; lagged outputs first (all outputs at
        lag 1, then lag 2, ...), then lagged inputs.
    target : ndarray of shape (T - max_lag,) or (T - max_lag, n_outputs)
        :math:`y_k` aligned with the rows of ``Phi``.
    names : list of str
        Human readable regressor names, e.g. ``"y1(k-1)"``, ``"u1(k-2)"``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.preprocessing import make_regressors
    >>> y = np.arange(6.0); u = np.arange(6.0) * 10
    >>> Phi, target, names = make_regressors(y, u, n_a=2, n_b=1, n_k=1)
    >>> names
    ['y(k-1)', 'y(k-2)', 'u(k-1)']
    >>> Phi[0], float(target[0])
    (array([ 1.,  0., 10.]), 2.0)
    """
    Y = np.asarray(y, dtype=float)
    Y2 = Y[:, None] if Y.ndim == 1 else Y
    T, ny = Y2.shape
    if u is not None:
        U = np.asarray(u, dtype=float)
        U2 = U[:, None] if U.ndim == 1 else U
        nu = U2.shape[1]
    else:
        U2, nu, n_b = None, 0, 0
    max_lag = max(n_a, n_b + n_k - 1 if n_b > 0 else 0)
    if max_lag < 1:
        raise ValueError("At least one lagged regressor is required.")
    cols, names = [], []
    for d in range(1, n_a + 1):
        for j in range(ny):
            cols.append(Y2[max_lag - d : T - d, j])
            names.append(f"y{j + 1 if ny > 1 else ''}(k-{d})")
    for d in range(n_k, n_k + n_b):
        for j in range(nu):
            cols.append(U2[max_lag - d : T - d, j])
            names.append(f"u{j + 1 if nu > 1 else ''}(k-{d})")
    Phi = np.column_stack(cols)
    target = Y2[max_lag:]
    if Y.ndim == 1:
        target = target[:, 0]
    return Phi, target, names
