# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Noise clustering (Davé) -- FCM with an additional noise prototype."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, sq_euclidean
from ..base import BaseFuzzyClusterer


class NoiseClustering(BaseFuzzyClusterer):
    r"""Noise clustering: fuzzy c-means with a virtual noise cluster (Davé).

    A fictitious *noise prototype* is added to the :math:`c` real clusters.
    It has no position: it is defined to be at the same distance
    :math:`\delta` from every sample. A sample far from all real prototypes
    is closer to the noise prototype than to any real one and therefore
    receives most of its membership in the noise cluster, so that it no
    longer influences the real prototypes. The memberships in the real
    clusters sum to less than one; the remainder is the noise membership.
    Compared with PCM the real clusters still interact through the
    probabilistic constraint, so coincident clusters do not occur; Davé
    and Krishnapuram [2]_ show that PCM is in fact noise clustering with
    one noise cluster *per* real cluster and :math:`\delta_i^2=\eta_i`,
    and relate both to robust M-estimators.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of real clusters :math:`c` (the noise cluster is extra).
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    delta : float or None, default=None
        Noise distance :math:`\delta`. ``None`` estimates it from the data
        at every iteration as :math:`\delta^2 = \lambda\,\overline{D^2}`,
        the scaled mean squared distance of all samples to all current
        prototypes (Davé's rule, see Notes).
    delta_scale : float, default=0.5
        Davé's multiplier :math:`\lambda` of the mean squared distance (used
        when ``delta=None``). Smaller values declare more samples noise.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
        Number of random restarts; the run with the lowest objective is
        kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes of the real clusters.
    membership_ : ndarray of shape (n_samples, n_clusters)
        Memberships in the real clusters; rows sum to :math:`1-\mu_{*k}`.
    noise_membership_ : ndarray of shape (n_samples,)
        Membership :math:`\mu_{*k}` in the noise cluster.
    labels_ : ndarray of shape (n_samples,)
        ``argmax`` over the real clusters, or ``-1`` where the noise
        membership exceeds every real membership.
    delta_ : float
        Noise distance actually used (at the last iteration).
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    Noise clustering minimizes the FCM functional with a :math:`(c+1)`-th
    cluster at constant distance :math:`\delta`,

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m D^2_{ik} + \sum_{k=1}^{N}\mu_{*k}^m\,\delta^2,
        \qquad \mu_{*k} = 1-\sum_{i=1}^{c}\mu_{ik}.

    The memberships follow from the FCM formula extended by the noise
    term,

    .. math::
        \mu_{ik} = \frac{1}{\sum_{j=1}^{c}\left(D^2_{ik}/D^2_{jk}\right)^{1/(m-1)}
                           + \left(D^2_{ik}/\delta^2\right)^{1/(m-1)}},

    and the prototypes are the usual weighted means of the real clusters:
    only the membership formula changes with respect to FCM. Davé proposed
    to tie :math:`\delta` to the scale of the data,

    .. math::
        \delta^2 = \lambda\,\frac{1}{Nc}\sum_{i=1}^{c}\sum_{k=1}^{N}D^2_{ik},

    with a user-chosen :math:`\lambda`; :math:`\delta` acts as a radius
    beyond which a sample is "more noise than member". When ``delta`` is
    estimated the functional changes slightly from iteration to iteration,
    so the objective history need not be monotone. :meth:`predict_proba`
    returns the memberships in the real clusters (rows sum to
    :math:`1-\mu_{*k}`), :meth:`predict_noise_membership` the noise
    membership, and :meth:`predict` labels noise-dominated samples ``-1``.

    References
    ----------
    .. [1] R. N. Davé, "Characterization and detection of noise in
           clustering", Pattern Recognition Letters 12(11), 657--664, 1991.
    .. [2] R. N. Davé, R. Krishnapuram, "Robust clustering methods: a unified
           view", IEEE Trans. Fuzzy Systems 5(2), 270--293, 1997.

    Examples
    --------
    >>> from fuzzyclust.extensions import NoiseClustering
    >>> from fuzzyclust.datasets import make_noisy_blobs
    >>> X, y = make_noisy_blobs(random_state=0)
    >>> nc = NoiseClustering(n_clusters=3, delta_scale=0.2, random_state=0).fit(X)
    >>> nc.membership_.shape, nc.noise_membership_.shape
    ((330, 3), (330,))
    >>> bool((nc.labels_ == -1).sum() > 0)
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        delta=None,
        delta_scale: float = 0.5,
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
        self.delta = delta
        self.delta_scale = delta_scale

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("NoiseClustering requires m > 1.")
        if self.delta is not None and self.delta <= 0:
            raise ValueError("delta must be positive.")
        if self.delta_scale <= 0:
            raise ValueError("delta_scale must be positive.")

    def _estimate_delta(self, X):
        if self.delta is not None:
            return float(self.delta)
        d2 = sq_euclidean(X, self.cluster_centers_)
        return float(np.sqrt(max(self.delta_scale * d2.mean(), EPS)))

    def _init_state(self, X, rng):
        U = super()._init_state(X, rng)
        # a provisional delta so that the first membership step is defined
        self.cluster_centers_ = fuzzy_centers(X, U**self.m)
        self.delta_ = self._estimate_delta(X)
        return U

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um)
        self.delta_ = self._estimate_delta(X)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        expo = -1.0 / (self.m - 1.0)
        inv = (d2 + EPS) ** expo
        inv_noise = (self.delta_**2 + EPS) ** expo
        return inv / (inv.sum(axis=1, keepdims=True) + inv_noise)

    def _objective(self, Um, d2):
        U = Um ** (1.0 / self.m)
        noise = np.clip(1.0 - U.sum(axis=1), 0.0, 1.0)
        return float(np.sum(Um * d2) + np.sum(noise**self.m) * self.delta_**2)

    @staticmethod
    def _labels_from(U):
        noise = 1.0 - U.sum(axis=1)
        labels = U.argmax(axis=1)
        labels[noise > U.max(axis=1)] = -1
        return labels

    def _finalize(self, X, Um):
        U = self.membership_
        self.noise_membership_ = np.clip(1.0 - U.sum(axis=1), 0.0, 1.0)
        self.labels_ = self._labels_from(U)

    def predict(self, X):
        """Hard labels of (new) samples; ``-1`` marks noise-dominated samples."""
        return self._labels_from(self.predict_proba(X))

    def predict_noise_membership(self, X):
        """Noise membership :math:`\\mu_{*k}` of (new) samples, shape ``(n_samples,)``."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return np.clip(1.0 - self._memberships(self._distances(X)).sum(axis=1), 0.0, 1.0)
