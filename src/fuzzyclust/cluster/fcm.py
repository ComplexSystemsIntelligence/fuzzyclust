# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-means (FCM) clustering."""

from __future__ import annotations

import numpy as np

from .._utils import fuzzy_centers, fuzzy_covariances, sq_mahalanobis
from ..base import BaseFuzzyClusterer


class FCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means clustering (Dunn, Bezdek).

    Every sample belongs to every cluster with a membership degree
    :math:`\mu_{ik}\in[0,1]`, :math:`\sum_i\mu_{ik}=1`. Prototypes are
    membership-weighted means and the distance is a fixed inner-product norm,
    so FCM detects (hyper)spherical clusters of roughly equal size.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    norm : {"euclidean", "diagonal", "mahalanobis"} or ndarray, default="euclidean"
        Norm-inducing matrix :math:`\mathbf{A}` shared by all clusters (an
        explicit matrix has shape ``(n_features, n_features)``): identity,
        the inverse diagonal variances of the data, the inverse of the data
        covariance (Mahalanobis norm), or an explicit matrix.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
        Number of random restarts; the run with the lowest final :math:`J` is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes :math:`\mathbf{v}_i`.
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition :math:`\mathbf{U}^T`.
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
        :math:`D_{ik\mathbf{A}}` at convergence.
    norm_matrix_ : ndarray of shape (n_features, n_features)
        The norm-inducing matrix :math:`\mathbf{A}` actually used.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices :math:`\mathbf{F}_i` of the final partition
        (not used by the algorithm, provided for analysis and plotting).
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    FCM minimizes the c-means functional

    .. math::
        J(\mathbf{X};\mathbf{U},\mathbf{V}) = \sum_{i=1}^{c}\sum_{k=1}^{N}
        \mu_{ik}^m\, D^2_{ik\mathbf{A}}, \qquad
        D^2_{ik\mathbf{A}} = (\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{A}(\mathbf{x}_k-\mathbf{v}_i)

    by Picard iteration through the necessary conditions

    .. math::
        \mu_{ik} = \frac{1}{\sum_{j=1}^{c}\left(D_{ik\mathbf{A}}/D_{jk\mathbf{A}}\right)^{2/(m-1)}},
        \qquad
        \mathbf{v}_i = \frac{\sum_{k=1}^{N}\mu_{ik}^m\mathbf{x}_k}{\sum_{k=1}^{N}\mu_{ik}^m}.

    References
    ----------
    .. [1] J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in
           detecting compact well-separated clusters", J. Cybernetics 3, 1973.
    .. [2] J. C. Bezdek, "Pattern Recognition with Fuzzy Objective Function
           Algorithms", Plenum Press, 1981.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> fcm = FCM(n_clusters=3, m=2.0, random_state=0).fit(X)
    >>> float(fcm.membership_.sum(axis=1).round(6).min())
    1.0
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        norm="euclidean",
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
        self.norm = norm

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("FCM requires m > 1 (use KMeans for a hard partition).")
        n = X.shape[1]
        if isinstance(self.norm, str):
            if self.norm == "euclidean":
                A = np.eye(n)
            elif self.norm == "diagonal":
                var = X.var(axis=0)
                A = np.diag(1.0 / np.where(var > 0, var, 1.0))
            elif self.norm == "mahalanobis":
                A = np.linalg.pinv(np.cov(X, rowvar=False))
            else:
                raise ValueError("norm must be 'euclidean', 'diagonal', 'mahalanobis' or a matrix.")
        else:
            A = np.asarray(self.norm, dtype=float)
            if A.shape != (n, n):
                raise ValueError(f"norm matrix must have shape ({n}, {n}).")
        self.norm_matrix_ = A

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um)

    def _distances(self, X):
        return sq_mahalanobis(X, self.cluster_centers_, self.norm_matrix_)

    def _finalize(self, X, Um):
        self.covariances_ = fuzzy_covariances(X, Um, self.cluster_centers_)
