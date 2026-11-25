# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Possibilistic c-means (PCM) clustering."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, normalize_rows, sq_euclidean
from ..base import BaseFuzzyClusterer
from ..cluster import FCM


def _typicality(d2: np.ndarray, eta: np.ndarray, m: float) -> np.ndarray:
    r"""Possibilistic typicality :math:`t_{ik} = 1/(1 + (D^2_{ik}/\eta_i)^{1/(m-1)})`."""
    expo = 1.0 / (m - 1.0)
    return 1.0 / (1.0 + ((d2 + EPS) / eta[None, :]) ** expo)


def seed_partition(X, rng, n_clusters, m, tol, max_iter, init):
    """Probabilistic (FCM) partition and its squared distances that seed a possibilistic run.

    ``init="fcm"`` runs FCM from a random start, ``"random"``/``"kmeans++"`` run
    FCM with that initialization, an ``(n_clusters, n_features)`` array is used
    as initial FCM prototypes and an ``(n_samples, n_clusters)`` array directly
    as the seed partition. Returns ``(U0, d2)``.
    """
    if not isinstance(init, str):
        init = np.asarray(init, dtype=float)
        if init.shape == (X.shape[0], n_clusters):
            U0 = normalize_rows(init)
            V0 = fuzzy_centers(X, U0**m)
            return U0, sq_euclidean(X, V0)
        if init.shape != (n_clusters, X.shape[1]):
            raise ValueError(
                "init array must have shape (n_clusters, n_features) or (n_samples, n_clusters), "
                f"got {init.shape}."
            )
    fcm_init = "random" if (isinstance(init, str) and init == "fcm") else init
    fcm = FCM(n_clusters=n_clusters, m=m, tol=tol, max_iter=max_iter, init=fcm_init, random_state=rng).fit(X)
    return fcm.membership_, fcm.distances_ ** 2


class PCM(BaseFuzzyClusterer):
    r"""Possibilistic c-means clustering (Krishnapuram and Keller).

    FCM constrains the memberships of every sample to sum to one over the
    clusters, so a membership is a *relative* quantity: an outlier far from
    all prototypes still receives memberships such as :math:`1/c` in every
    cluster and pulls every prototype toward itself. PCM drops the
    constraint. The membership :math:`t_{ik}` -- called *typicality* --
    depends only on the distance of the sample to prototype :math:`i`
    relative to a cluster scale :math:`\eta_i`, not on the distances to the
    other prototypes. Outliers are atypical of *all* clusters and no longer
    distort the prototypes, which makes PCM robust to noise. The price is
    that the clusters do not interact: PCM is essentially :math:`c`
    independent mode-seeking procedures, and two prototypes may converge to
    the same dense region (the *coincident clusters* problem) when the
    scales :math:`\eta_i` are too large or the groups are close. The
    recommended practice of the original papers, followed here by default,
    is to initialize PCM with FCM and to estimate :math:`\eta_i` from the FCM
    partition.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Possibilistic exponent :math:`m>1`. In PCM it controls how fast the
        typicality decays with the distance (it does not make the partition
        "fuzzier" as in FCM); the follow-up paper [2]_ recommends values
        around 1.5, which give sharper clusters and more robustness than 2.
    eta : None, float or array-like of shape (n_clusters,), default=None
        Scale parameters :math:`\eta_i` -- the squared distance at which the
        typicality equals 0.5. ``None`` estimates them from the initial FCM
        partition with the factor ``eta_scale``, see Notes.
    eta_scale : float, default=1.0
        Multiplier :math:`K` of the estimated scales (only used when
        ``eta=None``). :math:`K>1` widens the zone of influence of every
        cluster and favors coincident clusters, :math:`K<1` narrows it.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"fcm", "random", "kmeans++"} or ndarray, default="fcm"
        ``"fcm"`` runs :class:`~fuzzyclust.cluster.FCM` (random start) and
        uses its partition for :math:`\eta_i` and the initial typicalities;
        ``"random"``/``"kmeans++"`` do the same with the corresponding FCM
        initialization. An array of shape ``(n_samples, n_clusters)`` is used
        directly as initial (probabilistic) partition, e.g. the
        ``membership_`` of a fitted FCM or GK model.
    n_init : int, default=1
        Number of restarts (each with its own FCM run); the run with the
        lowest objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes :math:`\mathbf{v}_i`.
    typicality_ : ndarray of shape (n_samples, n_clusters)
        Typicalities :math:`t_{ik}\in[0,1]`. **Rows do not sum to one.**
    membership_ : ndarray of shape (n_samples, n_clusters)
        The same matrix as ``typicality_`` (PCM has no probabilistic
        membership); kept so that the generic tools of the toolbox work.
    labels_ : ndarray of shape (n_samples,)
        ``typicality_.argmax(axis=1)``.
    eta_ : ndarray of shape (n_clusters,)
        Scales :math:`\eta_i` actually used.
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    Compared with the FCM functional :math:`\sum_i\sum_k\mu_{ik}^m D^2_{ik}`,
    PCM minimizes

    .. math::
        J(\mathbf{X};\mathbf{T},\mathbf{V}) = \sum_{i=1}^{c}\sum_{k=1}^{N} t_{ik}^m D^2_{ik}
        + \sum_{i=1}^{c}\eta_i\sum_{k=1}^{N}(1-t_{ik})^m

    without the constraint :math:`\sum_i t_{ik}=1`; the second term
    penalizes the trivial solution :math:`t_{ik}=0`. The necessary
    conditions are

    .. math::
        t_{ik} = \frac{1}{1+\left(D^2_{ik}/\eta_i\right)^{1/(m-1)}},\qquad
        \mathbf{v}_i = \frac{\sum_k t_{ik}^m\mathbf{x}_k}{\sum_k t_{ik}^m}:

    the prototype update is that of FCM with typicalities in place of
    memberships, the typicality replaces the FCM membership formula. Since
    :math:`J` is a sum of :math:`c` independent terms, the clusters can be
    found one at a time, and nothing prevents two of them from being
    identical. The scales are estimated from a probabilistic partition
    :math:`\mathbf{U}` (FCM) as the fuzzy intra-cluster squared distance

    .. math::
        \eta_i = K\,\frac{\sum_k\mu_{ik}^m D^2_{ik}}{\sum_k\mu_{ik}^m},\qquad K=1 ,

    and kept fixed during the iterations, as recommended in [2]_. The
    original paper [1]_ also proposes the mean squared distance of the
    samples whose membership exceeds an :math:`\alpha`-cut and, as an
    optional second phase, re-estimating :math:`\eta_i` in this way from
    the converged typicalities and running PCM a second time; both can be
    done by passing ``eta`` explicitly. Because the rows
    of :math:`\mathbf{T}` are not normalized, :meth:`predict_proba`
    **returns typicalities** (as does :meth:`predict_typicality`), and
    ``labels_`` of samples with very small typicality in every cluster
    should be interpreted with care: compare ``typicality_.max(axis=1)``
    against a threshold to find outliers.

    References
    ----------
    .. [1] R. Krishnapuram, J. M. Keller, "A possibilistic approach to
           clustering", IEEE Trans. Fuzzy Systems 1(2), 98--110, 1993.
    .. [2] R. Krishnapuram, J. M. Keller, "The possibilistic c-means
           algorithm: insights and recommendations", IEEE Trans. Fuzzy
           Systems 4(3), 385--393, 1996.

    Examples
    --------
    >>> from fuzzyclust.extensions import PCM
    >>> from fuzzyclust.datasets import make_noisy_blobs
    >>> X, _ = make_noisy_blobs(random_state=0)
    >>> pcm = PCM(n_clusters=3, m=1.5, random_state=0).fit(X)
    >>> pcm.typicality_.shape, pcm.eta_.shape
    ((330, 3), (3,))
    >>> bool(pcm.typicality_.sum(axis=1).max() <= 3.0)
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        eta=None,
        eta_scale: float = 1.0,
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
        self.eta_scale = eta_scale

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("PCM requires m > 1.")
        if self.eta_scale <= 0:
            raise ValueError("eta_scale must be positive.")
        if self.eta is not None:
            eta = np.asarray(self.eta, dtype=float)
            if eta.ndim > 1 or (eta.ndim == 1 and eta.shape[0] != self.n_clusters) or np.any(eta <= 0):
                raise ValueError("eta must be a positive scalar or an array of n_clusters positive values.")

    def _init_state(self, X, rng):
        U0, d2 = seed_partition(X, rng, self.n_clusters, self.m, self.tol, self.max_iter, self.init)
        Um = U0**self.m
        if self.eta is None:
            eta = self.eta_scale * (Um * d2).sum(axis=0) / np.maximum(Um.sum(axis=0), EPS)
        else:
            eta = np.broadcast_to(np.asarray(self.eta, dtype=float), (self.n_clusters,)).copy()
        self.eta_ = np.maximum(eta, EPS)
        self.cluster_centers_ = fuzzy_centers(X, Um)
        return self._memberships(d2)

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        return _typicality(d2, self.eta_, self.m)

    def _objective(self, Um, d2):
        T = Um ** (1.0 / self.m)
        penalty = np.sum(self.eta_[None, :] * (1.0 - T) ** self.m)
        return float(np.sum(Um * d2) + penalty)

    def _finalize(self, X, Um):
        self.typicality_ = self.membership_

    def predict_typicality(self, X):
        """Typicalities :math:`t_{ik}` of (new) samples, shape ``(n_samples, n_clusters)``.

        Identical to :meth:`predict_proba` (PCM has no probabilistic membership).
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self._memberships(self._distances(X))
