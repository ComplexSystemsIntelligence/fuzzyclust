# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Gustafson–Kessel clustering with adaptive distance norms."""

from __future__ import annotations

import numpy as np

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, regularize_covariance, sq_mahalanobis
from ..base import BaseFuzzyClusterer


class GK(BaseFuzzyClusterer):
    r"""Gustafson–Kessel (GK) fuzzy clustering.

    Each cluster has its own norm-inducing matrix :math:`\mathbf{A}_i`
    derived from the fuzzy covariance matrix of the cluster, so clusters of
    different orientation and elongation (ellipsoids) can be detected within
    one data set. The volume of each cluster is fixed by :math:`\rho_i`.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    rho : float or array-like of shape (n_clusters,), default=None
        Cluster volumes :math:`\rho_i` (``None`` means all equal to one).
    gamma : float, default=0.0
        Covariance regularization weight :math:`\gamma\in[0,1]`: shrinks every
        :math:`\mathbf{F}_i` toward a scaled identity. Recommended for small or
        nearly collinear clusters (Babuška, van der Veen & Kaymak, 2002).
    beta : float, default=1e15
        Upper bound on the condition number of :math:`\mathbf{F}_i`
        (eigenvalues below :math:`\lambda_{\max}/\beta` are raised).
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
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
        :math:`D_{ik\mathbf{A}_i}` at convergence.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices :math:`\mathbf{F}_i` (after regularization).
    norm_matrices_ : ndarray of shape (n_clusters, n_features, n_features)
        Norm-inducing matrices :math:`\mathbf{A}_i`.
    eigenvalues_ : ndarray of shape (n_clusters, n_features)
        Eigenvalues of :math:`\mathbf{F}_i` in ascending order (cluster shape).
    eigenvectors_ : ndarray of shape (n_clusters, n_features, n_features)
        Corresponding eigenvectors (columns); the first column is the normal
        direction of the "flattest" extent of the cluster, useful for
        extracting local linear models.
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    GK minimizes

    .. math::
        J(\mathbf{X};\mathbf{U},\mathbf{V},\{\mathbf{A}_i\}) =
        \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m D^2_{ik\mathbf{A}_i},\qquad
        D^2_{ik\mathbf{A}_i} = (\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{A}_i(\mathbf{x}_k-\mathbf{v}_i)

    subject to the volume constraint :math:`\det\mathbf{A}_i=\rho_i`, which gives

    .. math::
        \mathbf{A}_i = \left[\rho_i\det(\mathbf{F}_i)\right]^{1/n}\mathbf{F}_i^{-1},\qquad
        \mathbf{F}_i = \frac{\sum_k\mu_{ik}^m(\mathbf{x}_k-\mathbf{v}_i)(\mathbf{x}_k-\mathbf{v}_i)^T}
                            {\sum_k\mu_{ik}^m}.

    The membership and prototype updates are those of FCM. The numerically
    robust variant (regularization by ``gamma`` and ``beta``) follows [2]_.

    References
    ----------
    .. [1] D. E. Gustafson, W. C. Kessel, "Fuzzy clustering with a fuzzy
           covariance matrix", Proc. IEEE CDC, 1979.
    .. [2] R. Babuška, P. J. van der Veen, U. Kaymak, "Improved covariance
           estimation for Gustafson–Kessel clustering", Proc. FUZZ-IEEE, 2002.

    Examples
    --------
    >>> from fuzzyclust.cluster import GK
    >>> from fuzzyclust.datasets import make_ellipses
    >>> X, _ = make_ellipses(random_state=0)
    >>> gk = GK(n_clusters=3, random_state=0).fit(X)
    >>> gk.covariances_.shape
    (3, 2, 2)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        rho=None,
        gamma: float = 0.0,
        beta: float = 1e15,
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
        self.rho = rho
        self.gamma = gamma
        self.beta = beta

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("GK requires m > 1.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if self.rho is None:
            self.rho_ = np.ones(self.n_clusters)
        else:
            rho = np.broadcast_to(np.asarray(self.rho, dtype=float), (self.n_clusters,)).copy()
            if np.any(rho <= 0):
                raise ValueError("rho must be positive.")
            self.rho_ = rho
        self._F0 = np.cov(X, rowvar=False) if X.shape[1] > 1 else np.atleast_2d(X.var())

    def _update_prototypes(self, X, Um):
        n = X.shape[1]
        V = fuzzy_centers(X, Um)
        F = fuzzy_covariances(X, Um, V)
        A = np.empty_like(F)
        for i in range(self.n_clusters):
            Fi = regularize_covariance(F[i], self._F0, self.gamma, self.beta)
            det = max(np.linalg.det(Fi), EPS)
            A[i] = (self.rho_[i] * det) ** (1.0 / n) * np.linalg.pinv(Fi)
            F[i] = Fi
        self.cluster_centers_ = V
        self.covariances_ = F
        self.norm_matrices_ = A

    def _distances(self, X):
        return sq_mahalanobis(X, self.cluster_centers_, self.norm_matrices_)

    def _finalize(self, X, Um):
        w, Q = np.linalg.eigh(self.covariances_)
        self.eigenvalues_ = w
        self.eigenvectors_ = Q
