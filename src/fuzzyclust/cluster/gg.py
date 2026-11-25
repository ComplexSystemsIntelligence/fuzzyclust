# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Gath–Geva (fuzzy maximum likelihood estimation) clustering."""

from __future__ import annotations

import numpy as np

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, regularize_covariance
from ..base import BaseFuzzyClusterer

_LOG_CAP = 700.0  # exp(700) is the largest finite double


class GG(BaseFuzzyClusterer):
    r"""Gath–Geva (GG) fuzzy clustering.

    The distance of a sample to a cluster is the inverse of a (scaled) Gaussian
    density, so clusters are interpreted as Gaussian components with their own
    covariance **and** prior probability. GG therefore handles clusters of
    different shape, size and density, at the price of being more sensitive to
    initialization than FCM or GK (the exponential distance converges fast to
    a nearby local optimum). It is usually initialized with FCM or GK.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent. ``m == 1`` gives the original fuzzy maximum
        likelihood estimation (FMLE) algorithm of Bezdek and Dunn; ``m == 2``
        is recommended because the exponential distance already produces
        rather crisp partitions.
    gamma : float, default=0.0
        Covariance regularization weight (shrinkage toward scaled identity),
        see :class:`~fuzzyclust.cluster.GK`.
    beta : float, default=1e15
        Condition-number bound of the covariance matrices.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        Because of the sensitivity to initialization, passing the membership
        matrix of a fitted FCM/GK model is the recommended practice.
    n_init : int, default=1
        Number of random restarts; the run with the lowest final :math:`J` is
        kept (ignored when ``init`` is an array).
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
        :math:`D_{ik}` at convergence (square root of the GG distance).
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices :math:`\mathbf{F}_{wi}`.
    priors_ : ndarray of shape (n_clusters,)
        Prior probabilities :math:`\alpha_i`.
    eigenvalues_, eigenvectors_
        Eigen-decomposition of ``covariances_`` (ascending eigenvalues).
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    The GG distance is

    .. math::
        D^2_{ik} = \frac{\sqrt{\det\mathbf{F}_{wi}}}{\alpha_i}
        \exp\!\left(\tfrac12(\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{F}_{wi}^{-1}(\mathbf{x}_k-\mathbf{v}_i)\right),

    with the fuzzy covariance and the prior

    .. math::
        \mathbf{F}_{wi} = \frac{\sum_k\mu_{ik}^{w}(\mathbf{x}_k-\mathbf{v}_i)(\mathbf{x}_k-\mathbf{v}_i)^T}
                              {\sum_k\mu_{ik}^{w}},\qquad
        \alpha_i = \frac1N\sum_{k=1}^{N}\mu_{ik},

    where the covariance weighting exponent :math:`w` equals ``m`` (``w = 2``
    by default, as in the MATLAB toolbox). Memberships and prototypes are
    updated with the FCM formulas. :math:`D^2_{ik}` is the reciprocal of the
    posterior-weighted Gaussian density, so :math:`\mu_{ik}` can be read as the
    posterior probability of cluster :math:`i` given :math:`\mathbf{x}_k`.
    All exponentials are evaluated in the log domain for numerical stability.

    References
    ----------
    .. [1] I. Gath, A. B. Geva, "Unsupervised optimal fuzzy clustering",
           IEEE Trans. Pattern Anal. Mach. Intell. 11(7), 1989.
    .. [2] J. C. Bezdek, J. C. Dunn, "Optimal fuzzy partitions: a heuristic for
           estimating the parameters in a mixture of normal distributions",
           IEEE Trans. Computers, 1975.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM, GG
    >>> from fuzzyclust.datasets import make_different_densities
    >>> X, _ = make_different_densities(random_state=0)
    >>> fcm = FCM(n_clusters=3, random_state=0).fit(X)
    >>> gg = GG(n_clusters=3, init=fcm.membership_).fit(X)
    >>> float(gg.priors_.sum().round(6))
    1.0
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
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
        self.gamma = gamma
        self.beta = beta

    def _check_params(self, X):
        super()._check_params(X)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        self._F0 = np.cov(X, rowvar=False) if X.shape[1] > 1 else np.atleast_2d(X.var())

    def _update_prototypes(self, X, Um):
        N = X.shape[0]
        V = fuzzy_centers(X, Um)
        F = fuzzy_covariances(X, Um, V)
        # priors use the un-powered memberships (eq. alpha_i = 1/N sum_k mu_ik)
        U = Um ** (1.0 / self.m) if self.m != 1.0 else Um
        priors = U.sum(axis=0) / N
        inv = np.empty_like(F)
        logdet = np.empty(self.n_clusters)
        for i in range(self.n_clusters):
            Fi = regularize_covariance(F[i], self._F0, self.gamma, self.beta)
            F[i] = Fi
            sign, ld = np.linalg.slogdet(Fi)
            logdet[i] = ld if sign > 0 else np.log(EPS)
            inv[i] = np.linalg.pinv(Fi)
        self.cluster_centers_ = V
        self.covariances_ = F
        self.priors_ = np.maximum(priors, EPS)
        self._inv_cov = inv
        self._logdet = logdet

    def _log_distances(self, X):
        c = self.n_clusters
        logd2 = np.empty((X.shape[0], c))
        for i in range(c):
            xv = X - self.cluster_centers_[i]
            maha = np.einsum("kj,jl,kl->k", xv, self._inv_cov[i], xv)
            logd2[:, i] = 0.5 * self._logdet[i] - np.log(self.priors_[i]) + 0.5 * maha
        return logd2

    def _distances(self, X):
        return np.exp(np.minimum(self._log_distances(X), _LOG_CAP))

    def _memberships(self, d2):
        # stable: mu_ik proportional to exp(-log(d2)/(m-1)) (softmax over clusters)
        logd2 = np.log(d2 + EPS)
        expo = 1.0 / (self.m - 1.0) if self.m > 1.0 else 1.0
        logits = -expo * logd2
        logits -= logits.max(axis=1, keepdims=True)
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    def _finalize(self, X, Um):
        w, Q = np.linalg.eigh(self.covariances_)
        self.eigenvalues_ = w
        self.eigenvectors_ = Q
