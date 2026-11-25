# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
"""Shared numerical helpers used across fuzzyclust.

All helpers follow the notation of :mod:`fuzzyclust` (see ``DESIGN.md``):
``U`` is the fuzzy partition stored as ``(n_samples, n_clusters)`` (i.e.
:math:`\\mathbf{U}^T`), ``V`` the prototypes ``(n_clusters, n_features)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: Small constant added to squared distances before taking negative powers.
#: Identical to the ``1e-10`` used in the original MATLAB toolbox.
EPS = 1e-10


def memberships_from_distances(d2: np.ndarray, m: float) -> np.ndarray:
    r"""Compute fuzzy memberships from squared distances.

    Implements the first-order optimality condition of the c-means functional

    .. math::
        \mu_{ik} = \frac{1}{\sum_{j=1}^{c} \left( D_{ik} / D_{jk} \right)^{2/(m-1)}} .

    Parameters
    ----------
    d2 : ndarray of shape (n_samples, n_clusters)
        Squared distances :math:`D^2_{ik}` of every sample to every prototype.
    m : float
        Fuzziness exponent :math:`m>1`. For ``m == 1`` the (possibilistic
        maximum-likelihood style) inverse-distance rule :math:`1/D^2` is used,
        mirroring the MATLAB toolbox.

    Returns
    -------
    U : ndarray of shape (n_samples, n_clusters)
        Row-stochastic membership matrix.
    """
    d2 = np.asarray(d2, dtype=float)
    if m > 1.0:
        inv = (d2 + EPS) ** (-1.0 / (m - 1.0))
    else:
        inv = (d2 + EPS) ** (-1.0)
    return inv / inv.sum(axis=1, keepdims=True)


def normalize_rows(U: np.ndarray) -> np.ndarray:
    """Rescale rows so that memberships of each sample sum to one."""
    U = np.asarray(U, dtype=float)
    s = U.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return U / s


def fuzzy_centers(X: np.ndarray, Um: np.ndarray) -> np.ndarray:
    r"""Weighted means :math:`\mathbf{v}_i = \sum_k \mu_{ik}^m \mathbf{x}_k / \sum_k \mu_{ik}^m`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    Um : ndarray of shape (n_samples, n_clusters)
        Memberships raised to the power ``m``.
    """
    s = Um.sum(axis=0)
    s = np.where(s == 0, EPS, s)
    return (Um.T @ X) / s[:, None]


def fuzzy_covariances(X: np.ndarray, Um: np.ndarray, V: np.ndarray) -> np.ndarray:
    r"""Fuzzy covariance matrices

    .. math::
        \mathbf{F}_i = \frac{\sum_{k=1}^{N} \mu_{ik}^m
            (\mathbf{x}_k-\mathbf{v}_i)(\mathbf{x}_k-\mathbf{v}_i)^T}
                            {\sum_{k=1}^{N} \mu_{ik}^m}.

    Returns
    -------
    F : ndarray of shape (n_clusters, n_features, n_features)
    """
    c = V.shape[0]
    n = X.shape[1]
    F = np.empty((c, n, n))
    s = Um.sum(axis=0)
    for i in range(c):
        xv = X - V[i]
        F[i] = (xv * Um[:, i][:, None]).T @ xv / max(s[i], EPS)
    return F


def sq_euclidean(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances ``(n_samples, n_clusters)`` between rows of X and V."""
    # ||x||^2 + ||v||^2 - 2 x.v, clipped at zero for numerical safety
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    v2 = np.einsum("ij,ij->i", V, V)[None, :]
    d2 = x2 + v2 - 2.0 * X @ V.T
    np.maximum(d2, 0.0, out=d2)
    return d2


def sq_mahalanobis(X: np.ndarray, V: np.ndarray, A: np.ndarray) -> np.ndarray:
    r"""Squared inner-product distances
    :math:`(\mathbf{x}_k-\mathbf{v}_i)^T \mathbf{A}_i (\mathbf{x}_k-\mathbf{v}_i)`.

    Parameters
    ----------
    A : ndarray of shape (n_clusters, n_features, n_features) or (n_features, n_features)
        Norm-inducing matrix per cluster (or one shared matrix).
    """
    c = V.shape[0]
    d2 = np.empty((X.shape[0], c))
    for i in range(c):
        Ai = A if A.ndim == 2 else A[i]
        xv = X - V[i]
        d2[:, i] = np.einsum("kj,jl,kl->k", xv, Ai, xv)
    np.maximum(d2, 0.0, out=d2)
    return d2


def harden(U: np.ndarray) -> np.ndarray:
    """Hard labels (argmax over clusters) of a membership matrix."""
    return np.asarray(U).argmax(axis=1)


def labels_to_membership(labels: ArrayLike, n_clusters: int | None = None) -> np.ndarray:
    """Convert hard labels to a crisp (0/1) membership matrix ``(n_samples, n_clusters)``."""
    labels = np.asarray(labels, dtype=int)
    c = int(labels.max()) + 1 if n_clusters is None else n_clusters
    U = np.zeros((labels.shape[0], c))
    U[np.arange(labels.shape[0]), labels] = 1.0
    return U


def check_membership(U: ArrayLike, n_samples: int | None = None) -> np.ndarray:
    """Validate a membership matrix: 2-D, entries in [0, 1], rows summing to one."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("Membership matrix must be 2-D (n_samples, n_clusters).")
    if n_samples is not None and U.shape[0] != n_samples:
        raise ValueError(f"Membership matrix has {U.shape[0]} rows, expected {n_samples}.")
    if np.any(U < -1e-12) or np.any(U > 1 + 1e-12):
        raise ValueError("Memberships must lie in [0, 1].")
    return U


def regularize_covariance(
    F: np.ndarray, F0: np.ndarray | None = None, gamma: float = 0.0, beta: float = 1e15
) -> np.ndarray:
    r"""Regularize a (fuzzy) covariance matrix as in Babuška, van der Veen & Kaymak (2002).

    Two independent devices are applied:

    1. shrinkage toward a scaled identity (``gamma``):
       :math:`\mathbf{F}_i \leftarrow (1-\gamma)\mathbf{F}_i + \gamma\,\det(\mathbf{F}_0)^{1/n}\mathbf{I}`,
       where :math:`\mathbf{F}_0` is the covariance of the whole data set;
    2. condition-number bounding (``beta``): eigenvalues smaller than
       :math:`\lambda_{\max}/\beta` are raised to that value.

    Parameters
    ----------
    F : ndarray of shape (n_features, n_features)
    F0 : ndarray of shape (n_features, n_features), optional
        Covariance of the whole data set (needed if ``gamma > 0``).
    gamma : float, default=0.0
    beta : float, default=1e15
    """
    n = F.shape[0]
    if gamma > 0.0:
        if F0 is None:
            raise ValueError("F0 is required when gamma > 0.")
        scale = np.linalg.det(F0) ** (1.0 / n)
        F = (1.0 - gamma) * F + gamma * scale * np.eye(n)
    # symmetric eigen-decomposition
    w, Q = np.linalg.eigh((F + F.T) / 2.0)
    wmax = w.max()
    if wmax <= 0:  # degenerate
        return np.eye(n) * EPS
    if np.isfinite(beta) and (wmax / max(w.min(), EPS)) > beta:
        w = np.maximum(w, wmax / beta)
        F = (Q * w) @ Q.T
    return F
