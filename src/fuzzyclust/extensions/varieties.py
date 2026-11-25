# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-varieties / fuzzy c-lines / fuzzy c-elliptotypes."""

from __future__ import annotations

import numpy as np

from .._utils import EPS, fuzzy_centers, sq_euclidean
from ..base import BaseFuzzyClusterer


class FuzzyCVarieties(BaseFuzzyClusterer):
    r"""Fuzzy c-varieties (FCV) clustering: linear-variety prototypes (Bezdek et al.).

    The prototype of cluster :math:`i` is an :math:`r`-dimensional linear
    variety (affine subspace) :math:`V_i=\{\mathbf{v}_i+\sum_{j=1}^{r}t_j\mathbf{s}_{ij}\}`
    through the center :math:`\mathbf{v}_i` spanned by orthonormal directions
    :math:`\mathbf{s}_{i1},\dots,\mathbf{s}_{ir}`. With :math:`r=1` the
    prototypes are lines (*fuzzy c-lines*, FCL), with :math:`r=0` points
    (FCM), with :math:`r=n-1` hyperplanes. The distance of a sample to a
    cluster is its squared orthogonal distance to the variety, so data lying
    along lines or planes are grouped by the subspace they belong to rather
    than by their position along it -- a sharper tool for linear structure
    than Gustafson--Kessel, whose ellipsoids still measure the position
    along the line. Because a line is infinitely long, two groups of points
    that happen to be collinear are merged; the *fuzzy c-elliptotypes*
    (FCE) compromise ``alpha`` mixes in the ordinary point distance to keep
    the clusters local.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    n_dims : int, default=1
        Dimension :math:`r` of the varieties, :math:`0\le r<n`.
    alpha : float, default=0.0
        Elliptotype weight :math:`\alpha\in[0,1]` of the point distance:
        :math:`D^2 = \alpha\|\mathbf{x}_k-\mathbf{v}_i\|^2 + (1-\alpha)D^2_{\mathrm{FCV}}`.
        ``0`` gives pure c-varieties, ``1`` FCM.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
        Number of random restarts; the run with the lowest objective is
        kept. Lines through a crossing have several local optima, so
        restarts are advisable.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centers :math:`\mathbf{v}_i` of the varieties (fuzzy means).
    directions_ : ndarray of shape (n_clusters, n_dims, n_features)
        Orthonormal direction vectors :math:`\mathbf{s}_{ij}` (rows).
    scatter_matrices_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy scatter matrices :math:`\mathbf{S}_i` whose leading eigenvectors
        are the directions.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices :math:`\mathbf{F}_i=\mathbf{S}_i/\sum_k\mu_{ik}^m`
        (same eigenvectors; for the ellipse plots of the toolbox).
    eigenvalues_ : ndarray of shape (n_clusters, n_features)
        Eigenvalues of :math:`\mathbf{S}_i` in descending order; the first
        :math:`r` belong to the directions, the rest measure the scatter
        *around* the variety.
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    FCV minimizes the c-means functional :math:`J=\sum_i\sum_k\mu_{ik}^m D^2_{ik}`
    with the squared orthogonal distance to the variety,

    .. math::
        D^2_{ik} = \|\mathbf{x}_k-\mathbf{v}_i\|^2
                   - \sum_{j=1}^{r}\left((\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{s}_{ij}\right)^2 .

    The necessary conditions are the FCM membership formula, the fuzzy mean
    for :math:`\mathbf{v}_i` (unchanged from FCM), and for the directions the
    :math:`r` unit eigenvectors belonging to the largest eigenvalues of the
    fuzzy scatter matrix

    .. math::
        \mathbf{S}_i = \sum_{k=1}^{N}\mu_{ik}^m(\mathbf{x}_k-\mathbf{v}_i)(\mathbf{x}_k-\mathbf{v}_i)^T ,

    i.e. a membership-weighted principal component analysis per cluster.
    Bezdek et al. [2]_ also consider convex combinations
    :math:`\sum_r\alpha_r J_r` of the FCV functionals of different
    dimension; the two-term combination of :math:`r=0` and :math:`r=1`
    (``alpha`` above) is the fuzzy c-elliptotypes algorithm, whose
    prototypes are "elliptotypes" between a point and a line and whose
    distance, for the same :math:`\mathbf{v}_i` and :math:`\mathbf{s}_{i1}`,
    is simply the convex combination of the two distances.

    References
    ----------
    .. [1] J. C. Bezdek, C. Coray, R. Gunderson, J. Watson, "Detection and
           characterization of cluster substructure. I. Linear structure:
           fuzzy c-lines", SIAM J. Appl. Math. 40(2), 339--357, 1981.
    .. [2] J. C. Bezdek, C. Coray, R. Gunderson, J. Watson, "Detection and
           characterization of cluster substructure. II. Fuzzy c-varieties
           and convex combinations thereof", SIAM J. Appl. Math. 40(2),
           358--372, 1981.

    Examples
    --------
    >>> from fuzzyclust.extensions import FuzzyCVarieties
    >>> from fuzzyclust.datasets import make_lines
    >>> X, y = make_lines(n_lines=3, random_state=0)
    >>> fcv = FuzzyCVarieties(n_clusters=3, n_dims=1, n_init=3, random_state=0).fit(X)
    >>> fcv.directions_.shape
    (3, 1, 2)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        n_dims: int = 1,
        alpha: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_dims = n_dims
        self.alpha = alpha

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("FuzzyCVarieties requires m > 1.")
        if not isinstance(self.n_dims, (int, np.integer)) or self.n_dims < 0:
            raise ValueError("n_dims must be a non-negative integer.")
        if self.n_dims >= X.shape[1]:
            raise ValueError(f"n_dims={self.n_dims} must be smaller than n_features={X.shape[1]}.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")

    def _update_prototypes(self, X, Um):
        c, r = self.n_clusters, self.n_dims
        n = X.shape[1]
        V = fuzzy_centers(X, Um)
        S = np.empty((c, n, n))
        directions = np.empty((c, r, n))
        eigenvalues = np.empty((c, n))
        for i in range(c):
            xv = X - V[i]
            S[i] = (xv * Um[:, i][:, None]).T @ xv
            w, Q = np.linalg.eigh(S[i])
            order = np.argsort(w)[::-1]
            eigenvalues[i] = w[order]
            directions[i] = Q[:, order[:r]].T
        self.cluster_centers_ = V
        self.scatter_matrices_ = S
        self.covariances_ = S / np.maximum(Um.sum(axis=0), EPS)[:, None, None]
        self.directions_ = directions
        self.eigenvalues_ = eigenvalues

    def _distances(self, X):
        d2_point = sq_euclidean(X, self.cluster_centers_)
        if self.n_dims == 0:
            return d2_point
        d2 = np.empty_like(d2_point)
        for i in range(self.n_clusters):
            proj = (X - self.cluster_centers_[i]) @ self.directions_[i].T  # (N, r)
            d2[:, i] = d2_point[:, i] - np.einsum("kj,kj->k", proj, proj)
        np.maximum(d2, 0.0, out=d2)
        if self.alpha > 0.0:
            d2 = self.alpha * d2_point + (1.0 - self.alpha) * d2
        return d2
