# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Random sample and extend fuzzy c-means (rseFCM)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from .._utils import sq_euclidean
from ..base import BaseFuzzyClusterer
from ..cluster import FCM
from .weighted import extend_partition

__all__ = ["RandomSampleFCM"]


def _resolve_sample_size(sample_size, N: int, c: int) -> int:
    if isinstance(sample_size, (int, np.integer)) and not isinstance(sample_size, bool):
        n_s = int(sample_size)
    elif isinstance(sample_size, float):
        if not 0.0 < sample_size <= 1.0:
            raise ValueError("A float sample_size must lie in (0, 1].")
        n_s = int(np.ceil(sample_size * N))
    else:
        raise ValueError("sample_size must be an int (number of samples) or a float (fraction).")
    n_s = min(n_s, N)
    if n_s < c:
        raise ValueError(f"sample_size resolves to {n_s} samples, fewer than n_clusters={c}.")
    return n_s


class RandomSampleFCM(BaseFuzzyClusterer):
    r"""Random sample and extend FCM (rseFCM) for very large data.

    The simplest way to cluster more data than FCM can iterate over: draw
    a random sample of :math:`n_s\ll N` observations, run FCM on the
    sample, and *extend* the result to all :math:`N` samples with one
    membership step from the fixed prototypes (:func:`extend_partition`).
    Havens et al. (2012) use rseFCM as the baseline of their study of FCM
    for very large data: it is the fastest of the schemes, and when the
    clusters are well represented in a random sample its partition is
    practically that of FCM on the full data.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    sample_size : int or float, default=0.1
        Size :math:`n_s` of the random sample: a number of samples or a
        fraction of ``n_samples``.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
        Random restarts of the FCM run on the sample.
    chunk_size : int or None, default=None
        Samples per chunk of the extension step (memory); ``None`` extends
        all samples at once.
    random_state : int, RandomState instance or None, default=None
        Controls the sample and the initialization.
    verbose : int, default=0

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_sample,)
        Row indices of the sample.
    fcm_ : :class:`~fuzzyclust.cluster.FCM`
        The clusterer fitted on the sample.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
        Extended partition of all samples.
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
    objective_ : float
        :math:`J` of the extended partition on the full data.
    objective_history_ : ndarray
        :math:`J` per iteration of the FCM run on the sample.
    n_iter_ : int
        Iterations of the FCM run on the sample.

    Notes
    -----
    rseFCM has no approximation guarantee: a cluster that is rare in the
    data may be missed by the sample. Its cost is one FCM run on
    :math:`n_s` samples plus one pass over the data,
    :math:`O(n_s c\,l + N c)`.

    References
    ----------
    .. [1] T. C. Havens, J. C. Bezdek, C. Leckie, L. O. Hall,
           M. Palaniswami, "Fuzzy c-means algorithms for very large data",
           IEEE Transactions on Fuzzy Systems 20(6), 1130--1146, 2012.

    Examples
    --------
    >>> from fuzzyclust.large import RandomSampleFCM, make_large_blobs
    >>> X, y = make_large_blobs(n_samples=5000, n_clusters=3, random_state=0)
    >>> rse = RandomSampleFCM(n_clusters=3, sample_size=500, random_state=0).fit(X)
    >>> rse.sample_indices_.shape, rse.membership_.shape
    ((500,), (5000, 3))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        sample_size=0.1,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        chunk_size: int | None = None,
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
        self.sample_size = sample_size
        self.chunk_size = chunk_size

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("RandomSampleFCM requires m > 1.")
        if self.chunk_size is not None and (
            not isinstance(self.chunk_size, (int, np.integer)) or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer or None.")

    def fit(self, X, y=None):
        """Cluster a random sample with FCM and extend the partition to all samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        self._check_params(X)
        rng = check_random_state(self.random_state)
        N = X.shape[0]
        n_s = _resolve_sample_size(self.sample_size, N, self.n_clusters)
        idx = np.sort(rng.choice(N, size=n_s, replace=False))
        init = self.init
        if isinstance(init, np.ndarray) and init.shape == (N, self.n_clusters):
            init = init[idx]  # a full membership init is restricted to the sample
        fcm = FCM(
            n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
            init=init, n_init=self.n_init, random_state=rng, verbose=self.verbose,
        ).fit(X[idx])
        self.sample_indices_ = idx
        self.fcm_ = fcm
        self.cluster_centers_ = fcm.cluster_centers_
        U, d2 = extend_partition(X, fcm.cluster_centers_, self.m, self.chunk_size)
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.distances_ = np.sqrt(d2)
        self.objective_ = float(np.sum(U**self.m * d2))
        self.objective_history_ = fcm.objective_history_
        self.n_iter_ = fcm.n_iter_
        return self

    # ------------------------------------------------------------------ base API hooks
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)
