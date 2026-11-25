# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Possibilistic fuzzy c-means (PFCM) clustering."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer
from .pcm import seed_partition


class PFCM(BaseFuzzyClusterer):
    r"""Possibilistic fuzzy c-means clustering (Pal, Pal, Keller and Bezdek).

    PFCM combines the two views of cluster membership. Every sample gets a
    *probabilistic* membership :math:`\mu_{ik}` as in FCM (rows sum to one;
    a relative quantity that says which cluster the sample belongs to
    *more*) **and** a *typicality* :math:`t_{ik}` as in PCM (an absolute
    quantity that says how representative the sample is of the cluster).
    The memberships keep the prototypes apart and so avoid the coincident
    clusters of PCM, the typicalities make the prototypes insensitive to
    outliers. The weights :math:`a` and :math:`b` set the relative influence
    of the two in the prototype update.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Exponent of the probabilistic memberships, :math:`m>1`.
    eta : float, default=2.0
        Exponent :math:`\eta>1` of the typicalities (the paper's
        :math:`\eta`; it plays the role of :math:`m` in PCM).
    a : float, default=1.0
        Weight of the memberships in the objective (:math:`a\ge0`).
    b : float, default=1.0
        Weight of the typicalities (:math:`b\ge0`, not both zero). With
        ``b > a`` the prototypes follow the typicalities and become more
        robust to outliers; the paper recommends :math:`b>a` for noisy data
        and warns that a very large :math:`b` brings back the coincident
        clusters of PCM.
    gamma : None, float or array-like of shape (n_clusters,), default=None
        Possibilistic scales :math:`\gamma_i`. ``None`` estimates them from
        the initial FCM partition, see Notes.
    gamma_scale : float, default=1.0
        Multiplier :math:`K` of the estimated scales.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"fcm", "random", "kmeans++"} or ndarray, default="fcm"
        As in :class:`~fuzzyclust.extensions.PCM`: an FCM run (or an
        explicit membership matrix) provides the initial partition and the
        scales.
    n_init : int, default=1
        Number of restarts; the run with the lowest objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
        Probabilistic memberships :math:`\mu_{ik}` (rows sum to one).
    typicality_ : ndarray of shape (n_samples, n_clusters)
        Typicalities :math:`t_{ik}` (rows do not sum to one).
    labels_ : ndarray of shape (n_samples,)
        ``membership_.argmax(axis=1)``.
    gamma_ : ndarray of shape (n_clusters,)
        Scales :math:`\gamma_i` used.
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    PFCM minimizes

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\left(a\,\mu_{ik}^m + b\,t_{ik}^{\eta}\right)D^2_{ik}
          + \sum_{i=1}^{c}\gamma_i\sum_{k=1}^{N}(1-t_{ik})^{\eta}

    subject to :math:`\sum_i\mu_{ik}=1` (no constraint on the typicalities),
    which gives the FCM membership formula for :math:`\mu_{ik}`, the PCM
    typicality formula with the scale :math:`\gamma_i/b`, and a prototype
    update weighted by both:

    .. math::
        t_{ik} = \frac{1}{1+\left(b\,D^2_{ik}/\gamma_i\right)^{1/(\eta-1)}},\qquad
        \mathbf{v}_i = \frac{\sum_k\left(a\mu_{ik}^m+b\,t_{ik}^{\eta}\right)\mathbf{x}_k}
                            {\sum_k\left(a\mu_{ik}^m+b\,t_{ik}^{\eta}\right)} .

    The scales are set from the FCM partition as in PCM,
    :math:`\gamma_i = K\sum_k\mu_{ik}^m D^2_{ik}/\sum_k\mu_{ik}^m`. With
    :math:`b=0` PFCM reduces to FCM, with :math:`a=0` to PCM.
    :meth:`predict_proba` returns the probabilistic memberships,
    :meth:`predict_typicality` the typicalities of new samples.

    References
    ----------
    .. [1] N. R. Pal, K. Pal, J. M. Keller, J. C. Bezdek, "A possibilistic
           fuzzy c-means clustering algorithm", IEEE Trans. Fuzzy Systems
           13(4), 517--530, 2005.

    Examples
    --------
    >>> from fuzzyclust.extensions import PFCM
    >>> from fuzzyclust.datasets import make_noisy_blobs
    >>> X, _ = make_noisy_blobs(random_state=0)
    >>> pfcm = PFCM(n_clusters=3, a=1.0, b=2.0, random_state=0).fit(X)
    >>> pfcm.typicality_.shape
    (330, 3)
    >>> bool(abs(pfcm.membership_.sum(axis=1) - 1).max() < 1e-9)
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        eta: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        gamma=None,
        gamma_scale: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="fcm",
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
        self.eta = eta
        self.a = a
        self.b = b
        self.gamma = gamma
        self.gamma_scale = gamma_scale

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0 or self.eta <= 1.0:
            raise ValueError("PFCM requires m > 1 and eta > 1.")
        if self.a < 0 or self.b < 0 or self.a + self.b <= 0:
            raise ValueError("a and b must be non-negative and not both zero.")
        if self.gamma_scale <= 0:
            raise ValueError("gamma_scale must be positive.")
        if self.gamma is not None:
            g = np.asarray(self.gamma, dtype=float)
            if g.ndim > 1 or (g.ndim == 1 and g.shape[0] != self.n_clusters) or np.any(g <= 0):
                raise ValueError("gamma must be a positive scalar or an array of n_clusters positive values.")

    def _init_state(self, X, rng):
        U0, d2 = seed_partition(X, rng, self.n_clusters, self.m, self.tol, self.max_iter, self.init)
        Um = U0**self.m
        if self.gamma is None:
            gamma = self.gamma_scale * (Um * d2).sum(axis=0) / np.maximum(Um.sum(axis=0), EPS)
        else:
            gamma = np.broadcast_to(np.asarray(self.gamma, dtype=float), (self.n_clusters,)).copy()
        self.gamma_ = np.maximum(gamma, EPS)
        self.cluster_centers_ = fuzzy_centers(X, Um)
        return U0

    def _typicality(self, d2):
        expo = 1.0 / (self.eta - 1.0)
        return 1.0 / (1.0 + (self.b * (d2 + EPS) / self.gamma_[None, :]) ** expo)

    def _update_prototypes(self, X, Um):
        # the typicalities belong to the distances of the current prototypes (same as U)
        T = self._typicality(sq_euclidean(X, self.cluster_centers_))
        W = self.a * Um + self.b * T**self.eta
        self.cluster_centers_ = fuzzy_centers(X, W)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)

    def _objective(self, Um, d2):
        T = self._typicality(d2)
        fit = np.sum((self.a * Um + self.b * T**self.eta) * d2)
        penalty = np.sum(self.gamma_[None, :] * (1.0 - T) ** self.eta)
        return float(fit + penalty)

    def _finalize(self, X, Um):
        self.typicality_ = self._typicality(self._distances(X))

    def predict_typicality(self, X):
        """Typicalities :math:`t_{ik}` of (new) samples, shape ``(n_samples, n_clusters)``."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self._typicality(self._distances(X))
