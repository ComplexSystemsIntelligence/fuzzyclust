# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Hard partitioning algorithms: k-means and k-medoids.

They are included so that hard and fuzzy methods share one interface (and one
alternating-optimization loop); for large-scale production use of k-means
:class:`sklearn.cluster.KMeans` is faster.
"""

from __future__ import annotations

import numpy as np

from .._utils import labels_to_membership, sq_euclidean
from ..base import BaseFuzzyClusterer, _EuclideanPrototypeMixin


class KMeans(_EuclideanPrototypeMixin, BaseFuzzyClusterer):
    r"""Hard c-means (k-means) clustering with the fuzzyclust interface.

    The algorithm alternates between assigning every sample to its nearest
    prototype and recomputing each prototype as the mean of its members.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    tol : float, default=1e-4
        Termination tolerance on the change of the (crisp) partition.
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
        Crisp partition (one-hot rows).
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
    objective_ : float
        Within-cluster sum of squares.
    objective_history_ : ndarray
    n_iter_ : int

    Notes
    -----
    Minimizes the within-cluster sum of squares

    .. math::
        J(\mathbf{X};\mathbf{U},\mathbf{V}) = \sum_{i=1}^{c}\sum_{\mathbf{x}_k\in A_i}
        \|\mathbf{x}_k-\mathbf{v}_i\|^2, \qquad
        \mathbf{v}_i = \frac{1}{N_i}\sum_{\mathbf{x}_k\in A_i}\mathbf{x}_k .

    It is the :math:`m\to 1` limit of fuzzy c-means. The membership matrix is
    an element of the hard partitioning space :math:`M_{hc}`
    (:math:`\mu_{ik}\in\{0,1\}`, :math:`\sum_i\mu_{ik}=1`).

    References
    ----------
    .. [1] J. MacQueen, "Some methods for classification and analysis of
           multivariate observations", Proc. 5th Berkeley Symp., 1967.
    .. [2] B. Balasko, J. Abonyi, B. Feil, "Fuzzy Clustering and Data Analysis
           Toolbox for use with MATLAB", 2005.

    Examples
    --------
    >>> from fuzzyclust.cluster import KMeans
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, y = make_fuzzy_blobs(n_clusters=3, random_state=0)
    >>> km = KMeans(n_clusters=3, random_state=0).fit(X)
    >>> km.labels_.shape
    (300,)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=1.0,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )

    def _memberships(self, d2):
        return labels_to_membership(d2.argmin(axis=1), self.n_clusters)

    def _update_prototypes(self, X, Um):
        counts = Um.sum(axis=0)
        V = (Um.T @ X) / np.where(counts == 0, 1.0, counts)[:, None]
        if np.any(counts == 0) and hasattr(self, "cluster_centers_"):
            # keep previous prototype of empty clusters (avoids collapse to origin)
            V[counts == 0] = self.cluster_centers_[counts == 0]
        self.cluster_centers_ = V


class KMedoids(BaseFuzzyClusterer):
    r"""Hard k-medoids clustering (prototypes are data points).

    Useful when the mean of a cluster is not a meaningful object, e.g. the
    samples are discrete states of a system. Two medoid rules are available:
    the exact medoid of every cluster (the member with minimal total Euclidean
    distance to the other members, the PAM criterion) or, as in the MATLAB
    toolbox, the member closest to the cluster mean.

    Parameters
    ----------
    n_clusters : int, default=3
    medoid : {"exact", "nearest_mean"}, default="exact"
        Rule used to select the medoid of each cluster.
    tol, max_iter, init, n_init, random_state, verbose
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        The medoids (rows of ``X``).
    medoid_indices_ : ndarray of shape (n_clusters,)
        Row indices of the medoids in the training data.
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    Every sample is assigned to its nearest medoid, and the medoids are
    constrained to the data, :math:`\mathbf{v}_i\in\mathbf{X}`. The two rules
    minimize different criteria over the members of each cluster
    :math:`A_i`:

    * ``"exact"`` minimizes the sum of (unsquared) distances,
      :math:`\sum_{\mathbf{x}_k\in A_i}\|\mathbf{x}_k-\mathbf{v}_i\|`;
    * ``"nearest_mean"`` minimizes the sum of squared distances,
      :math:`\sum_{\mathbf{x}_k\in A_i}\|\mathbf{x}_k-\mathbf{v}_i\|^2`,
      because this sum equals :math:`|A_i|\,\|\mathbf{v}_i-\bar{\mathbf{x}}_{A_i}\|^2`
      plus a constant.

    In both cases ``objective_`` and ``objective_history_`` report the
    within-cluster sum of squared distances to the medoids,
    :math:`J=\sum_i\sum_{\mathbf{x}_k\in A_i}\|\mathbf{x}_k-\mathbf{v}_i\|^2`.

    References
    ----------
    .. [1] L. Kaufman, P. J. Rousseeuw, "Finding Groups in Data", Wiley, 1990.

    Examples
    --------
    >>> from fuzzyclust.cluster import KMedoids
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> kmed = KMedoids(n_clusters=3, random_state=0).fit(X)
    >>> kmed.medoid_indices_.shape
    (3,)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        medoid: str = "exact",
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=1.0,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.medoid = medoid

    def _memberships(self, d2):
        return labels_to_membership(d2.argmin(axis=1), self.n_clusters)

    def _update_prototypes(self, X, Um):
        if self.medoid not in ("exact", "nearest_mean"):
            raise ValueError("medoid must be 'exact' or 'nearest_mean'.")
        labels = Um.argmax(axis=1)
        idx = np.empty(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            members = np.flatnonzero(labels == i)
            if members.size == 0:
                idx[i] = self.medoid_indices_[i] if hasattr(self, "medoid_indices_") else 0
                continue
            Xi = X[members]
            if self.medoid == "exact":
                D = np.sqrt(sq_euclidean(Xi, Xi))
                idx[i] = members[D.sum(axis=1).argmin()]
            else:
                mean = Xi.mean(axis=0, keepdims=True)
                idx[i] = members[sq_euclidean(Xi, mean)[:, 0].argmin()]
        self.medoid_indices_ = idx
        self.cluster_centers_ = X[idx].copy()

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)
