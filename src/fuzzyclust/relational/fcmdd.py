# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-medoids (FCMdd) with the linear-complexity and trimmed variants.

Krishnapuram, Joshi, Nasraoui and Yi (2001) cluster relational data with
prototypes that are *objects*: the medoid of a fuzzy cluster is the object
that minimizes the membership-weighted sum of dissimilarities to the
cluster. Unlike relational c-means the method accepts any dissimilarity
(no Euclidean assumption), the prototypes are interpretable, and the
search for the medoid can be restricted to the few objects with the
highest memberships, which makes an iteration linear in :math:`N`.
"""

from __future__ import annotations

import numpy as np

from .._utils import memberships_from_distances
from ..base import BaseFuzzyClusterer
from .dissimilarity import check_dissimilarity_matrix, pairwise_dissimilarity

__all__ = ["FuzzyCMedoids"]


class FuzzyCMedoids(BaseFuzzyClusterer):
    r"""Fuzzy c-medoids clustering of relational data (Krishnapuram et al., 2001).

    The prototypes are medoids -- actual objects -- and the objective is the
    membership-weighted sum of the dissimilarities of the objects to their
    medoids. Because only dissimilarities between objects enter, the
    algorithm works on any dissimilarity matrix (dynamic-time-warping,
    Gower, edit distances, web-session dissimilarities of the original
    paper) without the Euclidean assumption of relational c-means, and
    the result is a set of representative objects that can be shown and
    interpreted directly. Two variants are available: the *linear* FCMdd of
    the paper, which searches the new medoid of a cluster only among the
    ``n_candidates`` objects with the highest memberships in it, and a
    *trimmed* variant (FCTMdd) in the spirit of the paper's robust RFCMdd,
    which ignores, for every candidate medoid, the share ``trim`` of the
    objects that fit it worst, to make the medoids robust to outliers.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    metric : str, default="euclidean"
        ``"precomputed"``: ``fit(R)`` receives the :math:`N\times N`
        dissimilarity matrix (symmetric, zero diagonal, non-negative). Any
        other value is a metric name of :func:`scipy.spatial.distance.pdist`
        applied to the rows of ``X``; the dissimilarities are used *as
        they are* (not squared), since the algorithm does not need
        squared distances.
    n_candidates : int or None, default=None
        ``None``: the medoid of a cluster is searched among all :math:`N`
        objects (:math:`O(cN^2)` per iteration). An integer :math:`p`
        restricts the search to the :math:`p` objects with the highest
        memberships in the cluster -- the linear-complexity FCMdd of the
        paper, :math:`O(cpN)` per iteration. Small values speed the
        algorithm up at the risk of a worse local optimum.
    trim : float, default=0.0
        Share of the objects (``0 <= trim < 1``) that are *trimmed* when
        the medoids are updated: for every candidate medoid only the
        :math:`K=\lceil(1-\text{trim})N\rceil` smallest weighted
        dissimilarities :math:`\mu_{ik}^m r(\mathbf{x}_k,\mathbf{x}_q)`
        are summed (fuzzy c-trimmed medoids, FCTMdd). ``0`` is plain FCMdd.
        This is a simpler rule than the RFCMdd of the paper, which ranks
        the objects by a harmonic-mean-type dissimilarity to all medoids,
        keeps the best-fitting ones and computes memberships and medoids
        from them only; here the memberships of all objects are kept.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=100
    init : {"kmeans++", "random"} or array-like of int, default="kmeans++"
        ``"random"`` picks :math:`c` distinct random objects as initial
        medoids (the paper's start), ``"kmeans++"`` seeds them by
        :math:`D^2`-weighted sampling on the dissimilarities; an integer
        array gives the initial medoid indices.
    n_init : int, default=1
        Random restarts; the run with the lowest :math:`J` is kept. The
        paper recommends several restarts, the result depending on the start.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    medoid_indices_ : ndarray of shape (n_clusters,)
        Row indices of the medoids among the training objects.
    cluster_centers_ : ndarray of shape (n_clusters, n_features) or (n_clusters, n_samples)
        The medoids: ``X[medoid_indices_]`` with a feature-based metric, or
        the rows of the dissimilarity matrix of the medoids (their
        dissimilarities to all objects) with ``metric="precomputed"``.
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        The matrix :math:`\mathbf{R}` the algorithm ran on.
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
        Dissimilarities :math:`r(\mathbf{x}_k,\mathbf{v}_i)` of the objects
        to the medoids (not square-rooted: the dissimilarity itself plays
        the role of :math:`D^2_{ik}`; :meth:`transform` returns the same).
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    The objective is

    .. math::
        J=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\,r(\mathbf{x}_k,\mathbf{v}_i),\qquad
        \mathbf{v}_i\in\{\mathbf{x}_1,\dots,\mathbf{x}_N\},

    minimized by alternating the membership update (the FCM formula with
    the dissimilarity in place of the squared distance; :math:`\mu_{ik}=1`
    when :math:`\mathbf{x}_k=\mathbf{v}_i`)

    .. math::
        \mu_{ik}=\frac{\big(1/r(\mathbf{x}_k,\mathbf{v}_i)\big)^{1/(m-1)}}
                      {\sum_{j=1}^{c}\big(1/r(\mathbf{x}_k,\mathbf{v}_j)\big)^{1/(m-1)}}

    and the medoid update

    .. math::
        \mathbf{v}_i=\mathbf{x}_q,\qquad
        q=\argmin_{j\in C_i}\sum_{k=1}^{N}\mu_{ik}^m\,r(\mathbf{x}_k,\mathbf{x}_j),

    where the candidate set :math:`C_i` is all objects (full search) or the
    ``n_candidates`` objects with the largest :math:`\mu_{ik}` (linear
    FCMdd). The iteration stops when the medoids no longer change (the
    partition is then fixed too) or after ``max_iter`` iterations. If two
    clusters would select the same object, the second takes its best
    *unused* candidate, so that the medoids stay distinct. With
    ``trim > 0`` the sum in the medoid update runs over the :math:`K`
    smallest terms only; the objective reported is the same trimmed sum.
    New objects are assigned from their dissimilarities to the training
    objects (``(n_new, n_samples)`` matrix with ``metric="precomputed"``,
    feature vectors otherwise): only the columns of the medoids are used.

    References
    ----------
    .. [1] R. Krishnapuram, A. Joshi, O. Nasraoui, L. Yi, "Low-complexity
           fuzzy relational clustering algorithms for web mining", IEEE
           Transactions on Fuzzy Systems 9(4), 595--607, 2001.
    .. [2] L. Kaufman, P. J. Rousseeuw, "Finding Groups in Data: An
           Introduction to Cluster Analysis", Wiley, 1990.

    Examples
    --------
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.relational import FuzzyCMedoids, pairwise_dissimilarity
    >>> X, y = make_fuzzy_blobs(random_state=0)
    >>> R = pairwise_dissimilarity(X)
    >>> fcmdd = FuzzyCMedoids(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    >>> fcmdd.medoid_indices_.shape, fcmdd.membership_.shape
    ((3,), (300, 3))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        n_candidates: int | None = None,
        trim: float = 0.0,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 100,
        init="kmeans++",
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
        self.metric = metric
        self.n_candidates = n_candidates
        self.trim = trim

    # ------------------------------------------------------------------ data handling
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("Fuzzy c-medoids requires m > 1.")
        if not isinstance(self.metric, str):
            raise ValueError("metric must be a string (a SciPy metric name or 'precomputed').")
        if self.n_candidates is not None and (
            not isinstance(self.n_candidates, (int, np.integer)) or self.n_candidates < 1
        ):
            raise ValueError("n_candidates must be None or a positive integer.")
        if not 0.0 <= self.trim < 1.0:
            raise ValueError("trim must lie in [0, 1).")
        if self.metric == "precomputed":
            R = check_dissimilarity_matrix(X)
        else:
            R = pairwise_dissimilarity(X, metric=self.metric)
        self._X_fit = X
        self.dissimilarity_matrix_ = R
        self._n_keep = int(np.ceil((1.0 - self.trim) * X.shape[0]))

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        c, N = self.n_clusters, X.shape[0]
        R = self.dissimilarity_matrix_
        init = self.init
        if isinstance(init, str):
            if init == "random":
                idx = rng.choice(N, size=c, replace=False)
            elif init == "kmeans++":
                idx = [int(rng.randint(N))]
                dmin = R[:, idx[0]].copy()
                while len(idx) < c:
                    dmin[idx] = 0.0
                    probs = dmin / dmin.sum() if dmin.sum() > 0 else np.full(N, 1.0 / N)
                    j = int(rng.choice(N, p=probs))
                    idx.append(j)
                    dmin = np.minimum(dmin, R[:, j])
                idx = np.asarray(idx)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an index array.")
        else:
            idx = np.asarray(init).ravel()
            if idx.shape != (c,) or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError("init array must hold n_clusters integer medoid indices.")
            if idx.min() < 0 or idx.max() >= N:
                raise ValueError("medoid indices out of range.")
        self._set_medoids(X, np.asarray(idx, dtype=int))
        return self._memberships(R[:, self.medoid_indices_])

    def _set_medoids(self, X, idx):
        self.medoid_indices_ = idx
        if self.metric == "precomputed":
            self.cluster_centers_ = self.dissimilarity_matrix_[idx].copy()
        else:
            self.cluster_centers_ = X[idx].copy()

    def _candidate_cost(self, R, Um_i, candidates):
        """Weighted (and possibly trimmed) dissimilarity sum of every candidate medoid."""
        W = R[:, candidates] * Um_i[:, None]  # (N, n_cand): mu_ik^m r(x_k, x_q)
        if self._n_keep < R.shape[0]:
            W = np.partition(W, self._n_keep - 1, axis=0)[: self._n_keep]
        return W.sum(axis=0)

    def _update_prototypes(self, X, Um):
        R = self.dissimilarity_matrix_
        c, N = self.n_clusters, X.shape[0]
        idx = np.empty(c, dtype=int)
        taken = np.zeros(N, dtype=bool)
        for i in range(c):
            if self.n_candidates is None or self.n_candidates >= N:
                cand = np.arange(N)
            else:
                cand = np.argpartition(-Um[:, i], self.n_candidates - 1)[: self.n_candidates]
            cost = self._candidate_cost(R, Um[:, i], cand)
            cost[taken[cand]] = np.inf
            if np.all(np.isinf(cost)):  # every candidate already taken: fall back to all free objects
                cand = np.flatnonzero(~taken)
                cost = self._candidate_cost(R, Um[:, i], cand)
            q = int(cand[int(np.argmin(cost))])
            idx[i] = q
            taken[q] = True
        self._set_medoids(X, idx)

    def _distances(self, X):
        if X is self._X_fit:
            return self.dissimilarity_matrix_[:, self.medoid_indices_]
        if self.metric == "precomputed":
            R = check_dissimilarity_matrix(X, n_objects=self.dissimilarity_matrix_.shape[0], square=False)
            return R[:, self.medoid_indices_]
        Xm = self._X_fit[self.medoid_indices_]
        return pairwise_dissimilarity(X, Xm, metric=self.metric)

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)

    def _objective(self, Um, d2):
        W = Um * d2
        if self._n_keep < W.shape[0]:
            W = np.partition(W, self._n_keep - 1, axis=0)[: self._n_keep]
        return float(np.sum(W))

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Cluster the objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples) or (n_samples, n_features)
            The dissimilarity matrix when ``metric="precomputed"``,
            otherwise feature vectors.
        y : Ignored

        Returns
        -------
        self : object
        """
        super().fit(X, y)
        self.distances_ = self.distances_**2  # undo the square root of the base class
        return self

    def predict_proba(self, X):
        """Memberships of (new) objects, from their dissimilarities to the medoids.

        Parameters
        ----------
        X : array-like of shape (n_new, n_samples) or (n_new, n_features)
            With ``metric="precomputed"`` the dissimilarities of the new
            objects to all ``n_samples`` training objects, otherwise
            feature vectors.

        Returns
        -------
        U : ndarray of shape (n_new, n_clusters)
        """
        return super().predict_proba(X)

    def transform(self, X):
        r"""Dissimilarities :math:`r(\mathbf{x}_k,\mathbf{v}_i)` of (new) objects to the medoids."""
        return super().transform(X) ** 2

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.positive_only = self.metric == "precomputed"  # dissimilarities are >= 0
        return tags
