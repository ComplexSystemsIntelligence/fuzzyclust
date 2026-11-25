# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data helpers for the incomplete-data algorithms: planting missing values."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_mcar", "make_incomplete_blobs"]


def make_mcar(X, missing_rate: float = 0.2, random_state=None, return_mask: bool = False):
    r"""Remove values *missing completely at random* (MCAR) from a data matrix.

    Every entry is deleted independently with probability ``missing_rate``
    and replaced by ``NaN``. Two repairs keep the result usable by every
    strategy of :class:`~fuzzyclust.incomplete.IncompleteFCM`: a sample
    that would lose all its features keeps one randomly chosen value, and
    a feature that would be lost everywhere keeps one randomly chosen
    sample. The deletion mechanism is the one of the experiments of
    Hathaway and Bezdek (2001).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Complete data.
    missing_rate : float, default=0.2
        Probability of every entry being missing, in ``[0, 1)``.
    random_state : int, RandomState instance or None
    return_mask : bool, default=False
        Also return the boolean mask of the deleted entries.

    Returns
    -------
    X_missing : ndarray of shape (n_samples, n_features)
        Copy of ``X`` with ``NaN`` at the deleted entries.
    mask : ndarray of bool of shape (n_samples, n_features)
        Only when ``return_mask=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.incomplete import make_mcar
    >>> X = np.arange(20.0).reshape(10, 2)
    >>> Xm, mask = make_mcar(X, missing_rate=0.3, random_state=0, return_mask=True)
    >>> bool(np.isnan(Xm).sum() == mask.sum()), bool((~mask).any(axis=1).all())
    (True, True)
    """
    X = np.array(X, dtype=float, copy=True)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features).")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1).")
    rng = check_random_state(random_state)
    N, n = X.shape
    mask = rng.rand(N, n) < missing_rate
    # every sample keeps at least one value
    for k in np.flatnonzero(mask.all(axis=1)):
        mask[k, rng.randint(n)] = False
    # every feature keeps at least one value
    for j in np.flatnonzero(mask.all(axis=0)):
        mask[rng.randint(N), j] = False
    X[mask] = np.nan
    return (X, mask) if return_mask else X


def make_incomplete_blobs(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_features: int = 2,
    missing_rate: float = 0.2,
    spread: float = 1.0 / 1.5,
    box: float = 10.0,
    random_state=None,
):
    r"""Isotropic Gaussian clusters with values missing completely at random.

    A convenience wrapper around :func:`fuzzyclust.datasets.make_fuzzy_blobs`
    and :func:`make_mcar` that also returns the complete data, so that the
    recovered prototypes and the imputed values can be compared with the
    truth.

    Parameters
    ----------
    n_clusters : int, default=3
    n_samples_per_cluster : int, default=100
    n_features : int, default=2
    missing_rate : float, default=0.2
    spread : float, default=1/1.5
    box : float, default=10.0
        See :func:`fuzzyclust.datasets.make_fuzzy_blobs`.
    random_state : int, RandomState instance or None

    Returns
    -------
    X_missing : ndarray of shape (n_samples, n_features)
        Data with ``NaN`` for the missing values.
    y : ndarray of shape (n_samples,)
        Index of the generating cluster.
    X_complete : ndarray of shape (n_samples, n_features)
        The complete data.
    """
    from ..datasets import make_fuzzy_blobs

    rng = check_random_state(random_state)
    X, y = make_fuzzy_blobs(
        n_clusters=n_clusters, n_samples_per_cluster=n_samples_per_cluster, n_features=n_features,
        spread=spread, box=box, random_state=rng,
    )
    Xm = make_mcar(X, missing_rate=missing_rate, random_state=rng)
    return Xm, y, X
