# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Weighted fuzzy c-means (wFCM), the building block of the very-large-data algorithms."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import _num_samples, check_is_fitted, validate_data

from .._utils import fuzzy_centers, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer

__all__ = ["WeightedFCM", "extend_partition"]


def extend_partition(X, V, m: float = 2.0, chunk_size: int | None = None):
    r"""Memberships of (many) samples in fixed prototypes (the *extension* step).

    One FCM membership step with the prototypes held fixed,

    .. math::
        \mu_{ik}=\frac{1}{\sum_{j=1}^{c}\big(D_{ik}/D_{jk}\big)^{2/(m-1)}},\qquad
        D^2_{ik}=\norm{\mathbf{x}_k-\mathbf{v}_i}^2 ,

    evaluated chunk by chunk so that the working memory stays
    :math:`O(\text{chunk size}\cdot c)`. This is how rseFCM, spFCM and
    oFCM assign the whole data set to the clusters they found on samples
    or chunks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    V : array-like of shape (n_clusters, n_features)
    m : float, default=2.0
    chunk_size : int or None, default=None
        Samples per chunk; ``None`` processes all samples at once.

    Returns
    -------
    U : ndarray of shape (n_samples, n_clusters)
    d2 : ndarray of shape (n_samples, n_clusters)
        Squared Euclidean distances.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.large import extend_partition
    >>> X = np.array([[0.0], [1.0], [9.0], [10.0]])
    >>> U, d2 = extend_partition(X, np.array([[0.5], [9.5]]))
    >>> U.argmax(axis=1)
    array([0, 0, 1, 1])
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    N = X.shape[0]
    if chunk_size is None or chunk_size >= N:
        d2 = sq_euclidean(X, V)
        return memberships_from_distances(d2, m), d2
    U = np.empty((N, V.shape[0]))
    d2 = np.empty((N, V.shape[0]))
    for start in range(0, N, int(chunk_size)):
        stop = min(start + int(chunk_size), N)
        d2[start:stop] = sq_euclidean(X[start:stop], V)
        U[start:stop] = memberships_from_distances(d2[start:stop], m)
    return U, d2


class WeightedFCM(BaseFuzzyClusterer):
    r"""Weighted fuzzy c-means (wFCM): fuzzy c-means with a weight per sample.

    Every sample :math:`\mathbf{x}_k` carries a non-negative weight
    :math:`w_k` that counts how many observations it stands for. A sample
    with weight 2 has exactly the effect of two identical samples, so a
    weighted prototype (a cluster center summarizing a chunk of data) can
    be clustered together with raw samples. wFCM is the building block of
    the single-pass and online algorithms for very large data of Havens
    et al. (2012) and, with the gray-level histogram as weights, of the
    fast image segmentation algorithms of :mod:`fuzzyclust.image`.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
        Random restarts; the run with the lowest final :math:`J` is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
    sample_weight_ : ndarray of shape (n_samples,)
        The weights used in the fit.
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; :math:`J` is the
        weighted functional below.

    Notes
    -----
    wFCM minimizes the weighted c-means functional

    .. math::
        J_w(\mathbf{X};\mathbf{U},\mathbf{V})=\sum_{i=1}^{c}\sum_{k=1}^{N}
        w_k\,\mu_{ik}^m\,\norm{\mathbf{x}_k-\mathbf{v}_i}^2

    by alternating the unchanged FCM membership formula (the weights cancel
    in the per-sample optimality condition) with the weighted prototype
    update

    .. math::
        \mathbf{v}_i=\frac{\sum_{k=1}^{N} w_k\,\mu_{ik}^m\,\mathbf{x}_k}
                          {\sum_{k=1}^{N} w_k\,\mu_{ik}^m} .

    With all weights equal to one wFCM is FCM; with integer weights it is
    FCM on the data set in which sample :math:`k` is repeated :math:`w_k`
    times.

    References
    ----------
    .. [1] T. C. Havens, J. C. Bezdek, C. Leckie, L. O. Hall,
           M. Palaniswami, "Fuzzy c-means algorithms for very large data",
           IEEE Transactions on Fuzzy Systems 20(6), 1130--1146, 2012.
    .. [2] J. C. Bezdek, "Pattern Recognition with Fuzzy Objective Function
           Algorithms", Plenum Press, 1981.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.large import WeightedFCM
    >>> X = np.array([[0.0], [1.0], [9.0], [10.0]])
    >>> w = np.array([1.0, 3.0, 1.0, 1.0])
    >>> wfcm = WeightedFCM(n_clusters=2, random_state=0).fit(X, sample_weight=w)
    >>> np.sort(wfcm.cluster_centers_.ravel()).round(2)
    array([0.75, 9.5 ])
    """

    def fit(self, X, y=None, sample_weight=None):
        """Compute the weighted clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
        sample_weight : array-like of shape (n_samples,), default=None
            Non-negative weights :math:`w_k`; ``None`` means all ones.

        Returns
        -------
        self : object
        """
        n = _num_samples(X)
        if sample_weight is None:
            w = np.ones(n)
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
            if w.shape != (n,):
                raise ValueError(f"sample_weight has shape {w.shape}, expected ({n},).")
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
            if w.sum() <= 0:
                raise ValueError("sample_weight must not be all zero.")
        self._w = w
        super().fit(X)
        self.sample_weight_ = w
        return self

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("WeightedFCM requires m > 1.")

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        # weight-aware version of the base initialization, so that a sample of
        # weight w is treated exactly like w copies of it (and weight 0 like absence)
        c, n = self.n_clusters, X.shape[1]
        w = self._w
        if isinstance(self.init, str) and self.init == "random":
            present = w > 0
            mean = (w[:, None] * X).sum(axis=0) / w.sum()
            half_range = np.abs(X[present] - mean).max(axis=0)
            V = 2.0 * half_range * (rng.rand(c, n) - 0.5) + mean
            return self._init_memberships(sq_euclidean(X, V))
        if isinstance(self.init, str) and self.init == "kmeans++":
            from sklearn.cluster import kmeans_plusplus

            V, _ = kmeans_plusplus(X, n_clusters=c, sample_weight=w, random_state=rng)
            return self._init_memberships(sq_euclidean(X, V))
        return super()._init_state(X, rng)

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um * self._w[:, None])

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _objective(self, Um, d2):
        if Um.shape[0] == self._w.shape[0]:
            return float(np.sum(self._w[:, None] * Um * d2))
        return float(np.sum(Um * d2))

    def score(self, X, y=None, sample_weight=None):
        """Negative weighted functional :math:`-J_w` on ``X`` (higher is better)."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        d2 = self._distances(X)
        U = self._memberships(d2)
        return -float(np.sum(w[:, None] * U**self.m * d2))
