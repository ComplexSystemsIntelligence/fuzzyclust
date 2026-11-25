# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy clustering with attribute (feature) weights: SCAD and FCM with global attribute weights."""

from __future__ import annotations

import numpy as np

from .._utils import EPS, fuzzy_centers, fuzzy_covariances
from ..base import BaseFuzzyClusterer

_VARIANTS = ("scad1", "scad2")


def _featurewise_sq_distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Per-feature squared distances ``(n_samples, n_clusters, n_features)``."""
    return (X[:, None, :] - V[None, :, :]) ** 2


class SCAD(BaseFuzzyClusterer):
    r"""Simultaneous clustering and attribute discrimination (Frigui & Nasraoui, 2004).

    Every cluster :math:`i` has its own vector of attribute weights
    :math:`\mathbf{w}_i=(w_{i1},\dots,w_{in})`, :math:`\sum_j w_{ij}=1`,
    that is learned together with the partition: the distance of a sample
    to a prototype is the weighted sum of the per-feature squared
    distances, and features that scatter the members of a cluster widely
    are down-weighted for that cluster. The weights make a cluster
    sensitive to the features that define it and let different clusters
    live in different subspaces; an irrelevant noise feature receives a
    small weight in every cluster. Two forms exist. *SCAD-1* uses the
    weights linearly and keeps them away from the trivial crisp solution
    (all weight on the feature with the smallest scatter) by a quadratic
    regularization term with a cluster-specific coefficient
    :math:`\delta_i`, which may drive some weights to exactly zero.
    *SCAD-2* raises the weights to an exponent :math:`q>1` (like the
    fuzziness exponent for the memberships), needs no :math:`\delta_i` and
    never gives zero weights.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    variant : {"scad1", "scad2"}, default="scad1"
        The regularized (SCAD-1) or the exponent (SCAD-2) form.
    q : float, default=2.0
        Discrimination exponent of SCAD-2 (:math:`q>1`); the closer to one,
        the sharper the weight vectors.
    delta : float, array-like of shape (n_clusters,) or None, default=None
        Regularization coefficients :math:`\delta_i` of SCAD-1. ``None``
        re-estimates them from the current partition at every iteration as
        :math:`\delta_i=K\sum_k\mu_{ik}^m\sum_jw_{ij}d^2_{ikj}/\sum_jw_{ij}^2`
        (the balance between the two terms of the functional, as proposed by
        the authors); a number or an array fixes them.
    delta_scale : float, default=1.0
        Factor :math:`K` of the estimated :math:`\delta_i` (SCAD-1).
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    feature_weights_ : ndarray of shape (n_clusters, n_features)
        Attribute weights :math:`w_{ij}` (rows sum to one).
    delta_ : ndarray of shape (n_clusters,)
        Coefficients :math:`\delta_i` at convergence (SCAD-1; zeros for SCAD-2).
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_``
        and ``transform`` return the weighted distances.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices of the final partition (unweighted).

    Notes
    -----
    With the per-feature squared distances :math:`d^2_{ikj}=(x_{kj}-v_{ij})^2`,
    SCAD-1 minimizes

    .. math::
        J_1=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\sum_{j=1}^{n}w_{ij}\,d^2_{ikj}
            +\sum_{i=1}^{c}\delta_i\sum_{j=1}^{n}w_{ij}^2,
        \qquad \sum_j w_{ij}=1,

    whose weight update is

    .. math::
        w_{ij}=\frac1n+\frac{1}{2\delta_i}\sum_{k=1}^{N}\mu_{ik}^m
        \Bigl(\frac1n\sum_{l=1}^{n}d^2_{ikl}-d^2_{ikj}\Bigr):

    the weight of feature :math:`j` in cluster :math:`i` is raised above
    :math:`1/n` when the members of the cluster scatter less along :math:`j`
    than on average; negative values are set to zero and the row is
    renormalized. SCAD-2 minimizes
    :math:`J_2=\sum_i\sum_k\mu_{ik}^m\sum_jw_{ij}^q d^2_{ikj}` and gives

    .. math::
        w_{ij}=\frac{1}{\sum_{l=1}^{n}\Bigl(\sum_k\mu_{ik}^md^2_{ikj}\big/
        \sum_k\mu_{ik}^md^2_{ikl}\Bigr)^{1/(q-1)}} .

    In both forms the memberships follow the FCM formula with the weighted
    distance :math:`D^2_{ik}=\sum_jw_{ij}d^2_{ikj}` (resp. :math:`w_{ij}^q`)
    and the centers are the :math:`\mu^m`-weighted means, since every
    coordinate of the center minimizes its own weighted sum of squares.
    The weights reward features with a small within-cluster scatter, so
    the features should be on comparable scales (standardize first): a
    feature with a much smaller variance than the others attracts the
    weight regardless of its relevance, because slicing it into :math:`c`
    pieces gives a tiny scatter.

    References
    ----------
    .. [1] H. Frigui, O. Nasraoui, "Unsupervised learning of prototypes and
           attribute weights", Pattern Recognition 37(3), 567--581, 2004.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.extensions import SCAD
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> rng = np.random.RandomState(0)
    >>> Xn = np.c_[X, 3.0 * rng.randn(X.shape[0])]       # an irrelevant noise feature
    >>> scad = SCAD(n_clusters=3, random_state=0).fit(Xn)
    >>> bool((scad.feature_weights_[:, 2] < 0.2).all())
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        variant: str = "scad1",
        q: float = 2.0,
        delta=None,
        delta_scale: float = 1.0,
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
        self.variant = variant
        self.q = q
        self.delta = delta
        self.delta_scale = delta_scale

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("SCAD requires m > 1.")
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}.")
        if self.q <= 1.0:
            raise ValueError("q must be > 1.")
        if self.delta_scale <= 0:
            raise ValueError("delta_scale must be positive.")
        if self.delta is not None:
            d = np.asarray(self.delta, dtype=float)
            if d.ndim > 1 or (d.ndim == 1 and d.shape[0] != self.n_clusters) or np.any(d <= 0):
                raise ValueError("delta must be a positive scalar or an array of n_clusters positive values.")

    def _init_state(self, X, rng):
        U = super()._init_state(X, rng)
        self.feature_weights_ = np.full((self.n_clusters, X.shape[1]), 1.0 / X.shape[1])
        self.delta_ = np.zeros(self.n_clusters)
        return U

    def _update_prototypes(self, X, Um):
        c, n = self.n_clusters, X.shape[1]
        V = fuzzy_centers(X, Um)
        d2f = _featurewise_sq_distances(X, V)  # (N, c, n)
        scatter = np.einsum("ki,kij->ij", Um, d2f)  # (c, n): weighted scatter per cluster and feature
        W = self.feature_weights_
        if self.variant == "scad2":
            expo = 1.0 / (self.q - 1.0)
            ratio = (scatter[:, :, None] + EPS) / (scatter[:, None, :] + EPS)  # (c, n, n): s_ij / s_il
            W_new = 1.0 / np.sum(ratio**expo, axis=2)
            delta = np.zeros(c)
        else:
            if self.delta is None:
                delta = self.delta_scale * np.sum(W * scatter, axis=1) / np.maximum(np.sum(W**2, axis=1), EPS)
            else:
                delta = np.broadcast_to(np.asarray(self.delta, dtype=float), (c,)).copy()
            delta = np.maximum(delta, EPS)
            W_new = 1.0 / n + (scatter.mean(axis=1, keepdims=True) - scatter) / (2.0 * delta[:, None])
            W_new = np.maximum(W_new, 0.0)
            W_new /= np.maximum(W_new.sum(axis=1, keepdims=True), EPS)
        self.cluster_centers_ = V
        self.feature_weights_ = W_new
        self.delta_ = delta

    def _distances(self, X):
        d2f = _featurewise_sq_distances(X, self.cluster_centers_)
        W = self.feature_weights_ ** self.q if self.variant == "scad2" else self.feature_weights_
        return np.einsum("kij,ij->ki", d2f, W)

    def _objective(self, Um, d2):
        J = float(np.sum(Um * d2))
        if self.variant == "scad1":
            J += float(np.sum(self.delta_[:, None] * self.feature_weights_**2))
        return J

    def _finalize(self, X, Um):
        self.covariances_ = fuzzy_covariances(X, Um, self.cluster_centers_)


class FeatureWeightedFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means with global attribute weights (after Huang et al., 2005).

    One weight vector :math:`\boldsymbol\alpha=(\alpha_1,\dots,\alpha_n)`,
    :math:`\sum_j\alpha_j=1`, shared by all clusters, scales the features in
    the distance, and the weights are optimized together with the partition.
    A feature along which the clusters are poorly separated---so that the
    members scatter widely about their prototypes---receives a small weight,
    and the clustering effectively runs in the subspace of the informative
    features. The exponent :math:`t>1` on the weights plays the role of the
    fuzziness exponent: the closer to one, the more the weight concentrates
    on the single best feature.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    t : float, default=2.0
        Weight exponent :math:`t>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    feature_weights_ : ndarray of shape (n_features,)
        Global attribute weights :math:`\alpha_j` (sum to one).
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_``
        and ``transform`` return the weighted distances.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices of the final partition (unweighted).

    Notes
    -----
    The functional is

    .. math::
        J=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\sum_{j=1}^{n}\alpha_j^t\,(x_{kj}-v_{ij})^2,
        \qquad\sum_{j=1}^{n}\alpha_j=1,

    and the Lagrange conditions give, next to the FCM membership formula
    with the weighted distance and the fuzzy means for the centers,

    .. math::
        \alpha_j=\frac{1}{\sum_{l=1}^{n}\Bigl(\sum_{i,k}\mu_{ik}^md^2_{ikj}\big/
        \sum_{i,k}\mu_{ik}^md^2_{ikl}\Bigr)^{1/(t-1)}},

    the same form as the membership formula with the total scatter along a
    feature in place of the distance to a prototype. The weights are the
    diagonal of a norm-inducing matrix :math:`\mathbf{A}=\diag(\alpha_j^t)`
    learned from the data, so the method sits between FCM with the
    diagonal norm (fixed, from the feature variances) and GK (full,
    cluster-specific). As with SCAD the features should be standardized
    first, because a feature of small variance attracts the weight. Keller
    and Klawonn use the same kind of weights per cluster.

    References
    ----------
    .. [1] J. Z. Huang, M. K. Ng, H. Rong, Z. Li, "Automated variable
           weighting in k-means type clustering", IEEE Transactions on
           Pattern Analysis and Machine Intelligence 27(5), 657--668, 2005.
    .. [2] A. Keller, F. Klawonn, "Fuzzy clustering with weighting of data
           variables", International Journal of Uncertainty, Fuzziness and
           Knowledge-Based Systems 8(6), 735--746, 2000.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.extensions import FeatureWeightedFCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> Xn = np.c_[X, 3.0 * np.random.RandomState(0).randn(X.shape[0])]
    >>> wfcm = FeatureWeightedFCM(n_clusters=3, random_state=0).fit(Xn)
    >>> int(wfcm.feature_weights_.argmin())
    2
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        t: float = 2.0,
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
        self.t = t

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("FeatureWeightedFCM requires m > 1.")
        if self.t <= 1.0:
            raise ValueError("t must be > 1.")

    def _init_state(self, X, rng):
        U = super()._init_state(X, rng)
        self.feature_weights_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return U

    def _update_prototypes(self, X, Um):
        V = fuzzy_centers(X, Um)
        scatter = np.einsum("ki,kij->j", Um, _featurewise_sq_distances(X, V))  # (n,)
        expo = 1.0 / (self.t - 1.0)
        ratio = (scatter[:, None] + EPS) / (scatter[None, :] + EPS)
        self.cluster_centers_ = V
        self.feature_weights_ = 1.0 / np.sum(ratio**expo, axis=1)

    def _distances(self, X):
        d2f = _featurewise_sq_distances(X, self.cluster_centers_)
        return d2f @ (self.feature_weights_**self.t)

    def _finalize(self, X, Um):
        self.covariances_ = fuzzy_covariances(X, Um, self.cluster_centers_)


__all__ = ["SCAD", "FeatureWeightedFCM"]
