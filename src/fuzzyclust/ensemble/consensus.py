# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Consensus of soft clusterings: library, co-association matrix and consensus functions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, labels_to_membership, memberships_from_distances, sq_euclidean
from ..cluster import FCM
from ..validity import fuzzy_rand_index

__all__ = [
    "FuzzyClusterEnsemble",
    "coassociation_matrix",
    "evidence_accumulation",
    "soft_mcla",
    "information_theoretic_kmeans",
    "cspa",
    "mcla",
    "stability_curve",
]


# --------------------------------------------------------------------------- building blocks
def coassociation_matrix(memberships, similarity: str = "cosine") -> np.ndarray:
    r"""Soft co-association (similarity) matrix of a library of fuzzy partitions.

    For partition :math:`q` with membership vectors
    :math:`\boldsymbol{\mu}^{(q)}_k=[\mu^{(q)}_{1k},\dots,\mu^{(q)}_{c_qk}]^T`
    the similarity of two objects is the cosine of their membership
    vectors (``similarity="cosine"``) or their plain inner product
    (``"product"``), and the co-association is the average over the library,

    .. math::
        s_{kl} = \frac{1}{Q}\sum_{q=1}^{Q}
                 \frac{\langle\boldsymbol{\mu}^{(q)}_k,\boldsymbol{\mu}^{(q)}_l\rangle}
                      {\|\boldsymbol{\mu}^{(q)}_k\|\,\|\boldsymbol{\mu}^{(q)}_l\|}
        \quad\text{or}\quad
        s_{kl} = \frac{1}{Q}\sum_{q=1}^{Q}\sum_{i=1}^{c_q}\mu^{(q)}_{ik}\mu^{(q)}_{il} .

    For crisp partitions both are the fraction of partitions that put
    :math:`k` and :math:`l` into the same cluster -- the evidence
    accumulation matrix of Fred and Jain. For soft partitions the inner
    product is the probability that two independent draws from the
    membership vectors fall into the same cluster; it is below one even
    on the diagonal (an object at the border of two clusters is only
    "half" together with itself). The cosine normalizes every partition's
    similarity so that an object is fully co-associated with itself and
    with every object that has the same membership vector, which keeps the
    scale of the matrix independent of the fuzziness and of the number of
    clusters of the members.

    Parameters
    ----------
    memberships : sequence of array-like of shape (n_samples, c_q)
    similarity : {"cosine", "product"}, default="cosine"

    Returns
    -------
    S : ndarray of shape (n_samples, n_samples)
        Symmetric, entries in :math:`[0,1]`; unit diagonal for ``"cosine"``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.ensemble import coassociation_matrix
    >>> U1 = np.eye(2)[[0, 0, 1, 1]]
    >>> U2 = np.eye(2)[[0, 1, 1, 1]]
    >>> coassociation_matrix([U1, U2]).round(2)
    array([[1. , 0.5, 0. , 0. ],
           [0.5, 1. , 0.5, 0.5],
           [0. , 0.5, 1. , 1. ],
           [0. , 0.5, 1. , 1. ]])
    """
    if similarity not in ("cosine", "product"):
        raise ValueError("similarity must be 'cosine' or 'product'.")
    Us = [np.asarray(U, dtype=float) for U in memberships]
    if not Us:
        raise ValueError("The library is empty.")
    N = Us[0].shape[0]
    S = np.zeros((N, N))
    for U in Us:
        if U.ndim != 2 or U.shape[0] != N:
            raise ValueError("All partitions must be membership matrices of the same objects.")
        if np.any(U < 0):
            raise ValueError("Memberships must be non-negative.")
        if similarity == "cosine":
            U = _row_normalize(U)
        S += U @ U.T
    S /= len(Us)
    S = np.clip((S + S.T) / 2.0, 0.0, 1.0)
    if similarity == "cosine":
        np.fill_diagonal(S, 1.0)
    return S


def _row_normalize(U):
    """Rows scaled to unit Euclidean length (zero rows stay zero)."""
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), EPS)


def _cross_coassociation(Us_new, Us_fit, similarity):
    """Co-association ``(n_new, n_fit)`` between new and training objects."""
    S = np.zeros((Us_new[0].shape[0], Us_fit[0].shape[0]))
    for A, B in zip(Us_new, Us_fit, strict=True):
        if similarity == "cosine":
            A, B = _row_normalize(A), _row_normalize(B)
        S += A @ B.T
    return np.clip(S / len(Us_fit), 0.0, 1.0)


def _soft_from_hard(S, labels, m=2.0):
    """Soft memberships for a hard consensus.

    The mean co-association row of every consensus cluster serves as its
    prototype in the co-association space, and every object receives the
    fuzzy c-means memberships of its own row to these prototypes -- the
    same rule as the ``"fcm"`` consensus, so all consensus functions
    return memberships on one scale.
    """
    H = labels_to_membership(np.asarray(labels))
    H = H[:, H.sum(axis=0) > 0]
    V = fuzzy_centers(S, H)
    return memberships_from_distances(sq_euclidean(S, V), m), V


def _fcm_rows(S, c, rng, m=2.0, n_init=3, tol=1e-4, max_iter=200):
    """Plain fuzzy c-means on the rows of the co-association matrix (Euclidean, fast).

    The rows live in an :math:`N`-dimensional space, where fuzzy c-means
    from a random partition tends to the trivial fixed point (all
    prototypes at the grand mean, memberships :math:`1/c`). The prototypes
    are therefore seeded with k-means++ on the rows, which places them in
    different blocks of the matrix.
    """
    from sklearn.cluster import kmeans_plusplus

    best = None
    for _ in range(n_init):
        V0, _ = kmeans_plusplus(S, n_clusters=c, random_state=rng)
        U = memberships_from_distances(sq_euclidean(S, V0), m)
        for _ in range(max_iter):
            V = fuzzy_centers(S, U**m)
            d2 = sq_euclidean(S, V)
            U_new = memberships_from_distances(d2, m)
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < tol:
                break
        J = float(np.sum(U**m * d2))
        if best is None or J < best[0]:
            best = (J, U, V)
    return best[1], best[2]


def _lifetimes(Z, N, max_clusters):
    """Lifetime of every number of clusters in a dendrogram (Fred and Jain, 2005)."""
    heights = Z[:, 2]
    kmax = min(max_clusters, N)
    life = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        lo = heights[N - k - 1] if k < N else 0.0  # merge that creates the k-cluster partition
        hi = heights[N - k] if k > 1 else max(1.0, heights[-1])  # merge that destroys it
        life[k] = hi - lo
    return life[1:]  # index 0 -> one cluster


def evidence_accumulation(S, n_clusters: int | None = None, method: str = "average", max_clusters: int = 20):
    r"""Evidence accumulation clustering of a co-association matrix (Fred and Jain, 2005).

    The co-association :math:`s_{kl}` is turned into the distance
    :math:`1-s_{kl}` and clustered hierarchically; the dendrogram is cut
    into ``n_clusters`` groups, or -- when ``n_clusters`` is ``None`` -- at
    the number of clusters with the longest *lifetime*, the range of
    thresholds over which that number of clusters persists.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples)
    n_clusters : int or None, default=None
    method : {"average", "single", "complete", "ward"}, default="average"
        Linkage of :func:`scipy.cluster.hierarchy.linkage` (the paper uses
        single and average link).
    max_clusters : int, default=20
        Largest number of clusters considered by the lifetime criterion.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    Z : ndarray
        The linkage matrix.
    lifetimes : ndarray of shape (max_clusters,)
        Lifetime of :math:`k=1,\dots` clusters (``lifetimes[k - 1]``).

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.ensemble import coassociation_matrix, evidence_accumulation
    >>> U = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    >>> labels, Z, life = evidence_accumulation(coassociation_matrix([U]))
    >>> int(labels[0] != labels[-1]), int(np.argmax(life) + 1)
    (1, 2)
    """
    S = np.asarray(S, dtype=float)
    N = S.shape[0]
    D = np.clip(1.0 - S, 0.0, None)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    life = _lifetimes(Z, N, max_clusters)
    if n_clusters is None:
        n_clusters = int(np.argmax(life)) + 1
    # cut_tree returns exactly n_clusters groups also when merge heights tie
    # (fcluster's "maxclust" cannot split tied merges and may return fewer)
    labels = cut_tree(Z, n_clusters=int(min(n_clusters, N))).ravel().astype(int)
    return labels, Z, life


def soft_mcla(memberships, n_clusters: int, random_state=None):
    r"""Soft meta-clustering algorithm, sMCLA (Punera and Ghosh, 2008).

    Every cluster of every partition in the library is a *hyperedge*: a
    vector of :math:`N` memberships. The hyperedges are grouped into
    ``n_clusters`` meta-clusters by their cosine similarity (the original
    partitions the meta-graph with METIS; here spectral clustering of the
    similarity matrix is used), each meta-cluster is collapsed into its
    mean membership vector, and the consensus membership of an object is
    its collapsed membership normalized over the meta-clusters.

    Parameters
    ----------
    memberships : sequence of array-like of shape (n_samples, c_q)
    n_clusters : int
    random_state : int, RandomState instance or None

    Returns
    -------
    U : ndarray of shape (n_samples, n_clusters)
        Consensus memberships.
    meta_labels : ndarray of shape (sum of c_q,)
        Meta-cluster of every hyperedge, in library order.
    """
    from sklearn.cluster import SpectralClustering

    H = np.hstack([np.asarray(U, dtype=float) for U in memberships])  # (N, R)
    R = H.shape[1]
    if n_clusters > R:
        raise ValueError("More consensus clusters requested than clusters in the library.")
    norms = np.linalg.norm(H, axis=0)
    Hn = H / np.maximum(norms, EPS)
    sim = np.clip(Hn.T @ Hn, 0.0, 1.0)
    if n_clusters == 1:
        meta = np.zeros(R, dtype=int)
    else:
        with warnings.catch_warnings():  # tiny meta-graphs make ARPACK fall back to a dense solver
            warnings.simplefilter("ignore", RuntimeWarning)
            meta = SpectralClustering(
                n_clusters=n_clusters, affinity="precomputed", random_state=random_state,
                assign_labels="discretize",
            ).fit_predict(sim + 1e-6)
    U = np.zeros((H.shape[0], n_clusters))
    for i in range(n_clusters):
        members = meta == i
        if members.any():
            U[:, i] = H[:, members].mean(axis=1)
    U = np.maximum(U, EPS)
    return U / U.sum(axis=1, keepdims=True), meta


def information_theoretic_kmeans(
    memberships, n_clusters: int, max_iter: int = 100, n_init: int = 10, random_state=None
):
    r"""Information-theoretic k-means consensus, ITK (Punera and Ghosh, 2008).

    The concatenated membership vectors of an object, divided by the number
    of partitions, form a probability distribution :math:`\mathbf{p}_k`
    over all clusters of the library. ITK is k-means with the
    Kullback--Leibler divergence :math:`\mathrm{KL}(\mathbf{p}_k\|\mathbf{m}_i)`
    as distortion, whose centroids :math:`\mathbf{m}_i` are the arithmetic
    means of the assigned distributions. Like k-means it finds local
    optima; ``n_init`` k-means++ seedings (with the KL divergence) are run
    and the one with the smallest total divergence is kept.

    Parameters
    ----------
    memberships : sequence of array-like of shape (n_samples, c_q)
    n_clusters : int
    max_iter : int, default=100
    n_init : int, default=10
    random_state : int, RandomState instance or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (n_clusters, sum of c_q)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.ensemble import information_theoretic_kmeans
    >>> U = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    >>> labels, _ = information_theoretic_kmeans([U, U[:, ::-1]], 2, random_state=0)
    >>> int(labels[0] != labels[-1])
    1
    """
    rng = check_random_state(random_state)
    P = np.hstack([np.asarray(U, dtype=float) for U in memberships])
    P = P / P.sum(axis=1, keepdims=True)
    N = P.shape[0]
    if not 1 <= n_clusters <= N:
        raise ValueError("n_clusters must lie in [1, n_samples].")
    logP = np.log(P + EPS)
    ent = np.sum(P * logP, axis=1)  # sum p log p

    def kl(M):
        return ent[:, None] - P @ np.log(M + EPS).T  # KL(p_k || m_i)

    best = None
    for _ in range(max(1, int(n_init))):
        idx = [int(rng.randint(N))]  # k-means++ seeding with the KL divergence
        while len(idx) < n_clusters:
            d = np.maximum(kl(P[idx]).min(axis=1), 0.0)
            d[idx] = 0.0
            probs = d / d.sum() if d.sum() > 0 else np.full(N, 1.0 / N)
            idx.append(int(rng.choice(N, p=probs)))
        M = P[idx].copy()
        labels = np.full(N, -1)
        for _ in range(max_iter):
            new = kl(M).argmin(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for i in range(n_clusters):
                if np.any(labels == i):
                    M[i] = P[labels == i].mean(axis=0)
        cost = float(kl(M)[np.arange(N), labels].sum())
        if best is None or cost < best[0]:
            best = (cost, labels.copy(), M.copy())
    return best[1], best[2]


def cspa(labels_list, n_clusters: int, random_state=None):
    """Cluster-based similarity partitioning, CSPA (Strehl and Ghosh, 2002), for hard partitions.

    The co-association matrix of the crisp partitions is partitioned into
    ``n_clusters`` groups (spectral clustering stands in for METIS).

    Parameters
    ----------
    labels_list : sequence of array-like of shape (n_samples,)
    n_clusters : int
    random_state : int, RandomState instance or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    from sklearn.cluster import SpectralClustering

    S = coassociation_matrix([labels_to_membership(lab) for lab in labels_list])
    return SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=random_state, assign_labels="discretize"
    ).fit_predict(S + 1e-6)


def mcla(labels_list, n_clusters: int, random_state=None):
    """Meta-clustering algorithm, MCLA (Strehl and Ghosh, 2002), for hard partitions.

    The crisp clusters are the hyperedges; :func:`soft_mcla` on their 0/1
    indicator vectors followed by the arg-max gives the consensus labels.

    Parameters
    ----------
    labels_list : sequence of array-like of shape (n_samples,)
    n_clusters : int
    random_state : int, RandomState instance or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    U, _ = soft_mcla([labels_to_membership(lab) for lab in labels_list], n_clusters, random_state)
    return U.argmax(axis=1)


# --------------------------------------------------------------------------- the estimator
class FuzzyClusterEnsemble(ClusterMixin, BaseEstimator):
    r"""Consensus of a library of soft clusterings (Punera and Ghosh, 2008).

    A single run of an alternating-optimization clusterer depends on its
    initialization, its number of clusters and the particular sample it
    saw. An *ensemble* fits many such runs -- different random starts,
    sub-samples, feature subsets or numbers of clusters -- and combines
    their partitions into one consensus that is more stable and often
    more accurate than any member. Soft partitions carry more information
    than hard ones (how confidently an object was assigned), and the
    consensus functions of this class use them directly instead of
    hardening first.

    Parameters
    ----------
    n_clusters : int or None, default=3
        Number of consensus clusters. ``None`` picks it by the lifetime
        criterion of the co-association dendrogram (Fred and Jain, 2005).
    base_estimator : estimator with ``predict_proba``, default=None
        Clusterer of the library; default :class:`~fuzzyclust.cluster.FCM`.
        It is cloned for every member and receives its own ``random_state``.
    n_estimators : int, default=10
        Size :math:`Q` of the library.
    base_n_clusters : int, sequence of int or None, default=None
        Number of clusters of the members: one value, or a sequence from
        which every member draws at random (a library over several
        :math:`c` is the paper's recipe for robustness). ``None`` uses
        ``n_clusters`` (which must then be given).
    sample_fraction : float, default=1.0
        Share of the samples every member is fitted on (drawn without
        replacement); the member's partition of *all* samples is obtained
        with its ``predict_proba``.
    feature_fraction : float, default=1.0
        Share of the features every member sees (at least one).
    consensus : {"fcm", "spectral", "hierarchical", "smcla", "itk"}, default="fcm"
        Consensus function: ``"fcm"`` clusters the rows of the
        co-association matrix with fuzzy c-means (a soft cluster-based
        similarity partitioning); ``"spectral"`` partitions the
        co-association graph by spectral clustering (sCSPA with a graph
        partitioner); ``"hierarchical"`` is evidence accumulation
        (average or single link on :math:`1-s_{kl}`); ``"smcla"`` the soft
        meta-clustering algorithm; ``"itk"`` information-theoretic
        k-means on the concatenated memberships.
    linkage : {"average", "single", "complete"}, default="average"
        Linkage of the hierarchical consensus and of the lifetime criterion.
    similarity : {"cosine", "product"}, default="cosine"
        Similarity of two membership vectors in the co-association matrix
        (:func:`coassociation_matrix`).
    max_clusters : int, default=20
        Largest number of clusters the lifetime criterion considers.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    base_estimators_ : list
        The fitted members.
    base_memberships_ : list of ndarray of shape (n_samples, c_q)
        Partition of every member over all samples.
    base_feature_subsets_ : list of ndarray
        Feature indices every member used.
    coassociation_ : ndarray of shape (n_samples, n_samples)
        Soft co-association matrix :math:`s_{kl}`.
    consensus_membership_ : ndarray of shape (n_samples, ``n_clusters_``)
        Consensus partition (``membership_`` is an alias). For the hard
        consensus functions (spectral, hierarchical, ITK) it is the fuzzy
        c-means membership (:math:`m=2`) of every object's co-association
        row to the mean rows of the consensus clusters.
    labels_ : ndarray of shape (n_samples,)
    n_clusters_ : int
    lifetimes_ : ndarray of shape (max_clusters,)
        Lifetime of :math:`k=1,\dots` clusters in the dendrogram (computed
        when ``n_clusters`` is ``None`` or ``consensus="hierarchical"``).
    linkage_ : ndarray
        Linkage matrix of the co-association dendrogram (same condition).
    meta_cluster_labels_ : ndarray
        Meta-cluster of every library cluster (``consensus="smcla"``).
    consensus_centers_ : ndarray of shape (``n_clusters_``, n_samples)
        Prototypes of the consensus clusters in the co-association space:
        the fuzzy c-means prototypes (``consensus="fcm"``) or the mean rows
        of the clusters (hard consensus functions); not set for ``"smcla"``.
    base_agreement_ : float
        Mean fuzzy Rand index between the members and the consensus.

    Notes
    -----
    The co-association of two objects is the library average of the
    cosines (or inner products) of their membership vectors,
    :math:`s_{kl}=\frac1Q\sum_q\cos\big(\boldsymbol{\mu}^{(q)}_k,\boldsymbol{\mu}^{(q)}_l\big)`
    (:func:`coassociation_matrix`). The members need not agree on the
    number or the labels of the clusters: only co-memberships are
    compared, which is why ensembles over several :math:`c` work.
    New samples (:meth:`predict_proba`) receive memberships from every
    member, their co-association with the training samples follows, and
    the consensus function's own rule maps it to the consensus clusters.

    References
    ----------
    .. [1] K. Punera, J. Ghosh, "Consensus-based ensembles of soft
           clusterings", Applied Artificial Intelligence 22(7--8),
           780--810, 2008.
    .. [2] A. Strehl, J. Ghosh, "Cluster ensembles -- a knowledge reuse
           framework for combining multiple partitions", Journal of
           Machine Learning Research 3, 583--617, 2002.
    .. [3] A. L. N. Fred, A. K. Jain, "Combining multiple clusterings
           using evidence accumulation", IEEE Transactions on Pattern
           Analysis and Machine Intelligence 27(6), 835--850, 2005.

    Examples
    --------
    >>> from fuzzyclust.ensemble import FuzzyClusterEnsemble
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, y = make_fuzzy_blobs(random_state=0)
    >>> ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=5, random_state=0).fit(X)
    >>> ens.coassociation_.shape, ens.consensus_membership_.shape
    ((300, 300), (300, 3))
    """

    def __init__(
        self,
        n_clusters: int | None = 3,
        base_estimator=None,
        n_estimators: int = 10,
        base_n_clusters=None,
        sample_fraction: float = 1.0,
        feature_fraction: float = 1.0,
        consensus: str = "fcm",
        linkage: str = "average",
        similarity: str = "cosine",
        max_clusters: int = 20,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.base_n_clusters = base_n_clusters
        self.sample_fraction = sample_fraction
        self.feature_fraction = feature_fraction
        self.consensus = consensus
        self.linkage = linkage
        self.similarity = similarity
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ validation
    def _check_params(self, X):
        N, n = X.shape
        if self.n_clusters is not None and (
            not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1 or self.n_clusters > N
        ):
            raise ValueError("n_clusters must be None or an integer in [1, n_samples].")
        if not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not 0.0 < self.sample_fraction <= 1.0 or not 0.0 < self.feature_fraction <= 1.0:
            raise ValueError("sample_fraction and feature_fraction must lie in (0, 1].")
        if self.consensus not in ("fcm", "spectral", "hierarchical", "smcla", "itk"):
            raise ValueError("consensus must be 'fcm', 'spectral', 'hierarchical', 'smcla' or 'itk'.")
        if self.linkage not in ("average", "single", "complete"):
            raise ValueError("linkage must be 'average', 'single' or 'complete'.")
        if self.similarity not in ("cosine", "product"):
            raise ValueError("similarity must be 'cosine' or 'product'.")
        if self.base_estimator is not None and not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("base_estimator must implement predict_proba (soft memberships of all samples).")
        if self.base_n_clusters is None and self.n_clusters is None:
            raise ValueError("Give base_n_clusters when n_clusters is None.")
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be >= 1.")

    def _member_n_clusters(self, rng):
        if self.base_n_clusters is None:
            return int(self.n_clusters)
        bc = np.atleast_1d(np.asarray(self.base_n_clusters, dtype=int))
        if np.any(bc < 1):
            raise ValueError("base_n_clusters must be positive integers.")
        return int(bc[rng.randint(bc.size)])

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Fit the library and derive the consensus partition.

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
        N, n = X.shape
        base = FCM() if self.base_estimator is None else self.base_estimator
        n_rows = max(2, int(round(self.sample_fraction * N)))
        n_cols = max(1, int(round(self.feature_fraction * n)))
        self.base_estimators_, self.base_memberships_, self.base_feature_subsets_ = [], [], []
        for q in range(self.n_estimators):
            c_q = min(self._member_n_clusters(rng), n_rows)
            rows = rng.choice(N, size=n_rows, replace=False) if n_rows < N else np.arange(N)
            cols = np.sort(rng.choice(n, size=n_cols, replace=False)) if n_cols < n else np.arange(n)
            est = clone(base)
            params = est.get_params()
            if "n_clusters" in params:
                est.set_params(n_clusters=c_q)
            if "random_state" in params:
                est.set_params(random_state=int(rng.randint(np.iinfo(np.int32).max)))
            est.fit(X[rows][:, cols])
            U = est.predict_proba(X[:, cols])
            self.base_estimators_.append(est)
            self.base_memberships_.append(np.asarray(U, dtype=float))
            self.base_feature_subsets_.append(cols)
            if self.verbose:
                print(f"member {q + 1}/{self.n_estimators}: c = {c_q}, {n_rows} samples, {n_cols} features")
        self.coassociation_ = coassociation_matrix(self.base_memberships_, self.similarity)
        self._consensus(rng)
        self.base_agreement_ = float(
            np.mean([fuzzy_rand_index(U, self.consensus_membership_) for U in self.base_memberships_])
        )
        return self

    def _consensus(self, rng):
        S = self.coassociation_
        N = S.shape[0]
        c = self.n_clusters
        if c is None or self.consensus == "hierarchical":
            labels_h, Z, life = evidence_accumulation(
                S, c, method=self.linkage, max_clusters=self.max_clusters
            )
            self.linkage_, self.lifetimes_ = Z, life
            if c is None:
                c = int(np.argmax(life)) + 1
                if self.verbose:
                    print(f"lifetime criterion: {c} clusters")
        self.n_clusters_ = int(c)
        if self.consensus == "fcm":
            U, V = _fcm_rows(S, c, rng)
        elif self.consensus == "spectral":
            from sklearn.cluster import SpectralClustering

            if c == 1:
                labels = np.zeros(N, dtype=int)
            else:
                labels = SpectralClustering(
                    n_clusters=c, affinity="precomputed", random_state=rng, assign_labels="discretize"
                ).fit_predict(S + 1e-6)
            U, V = _soft_from_hard(S, labels)
        elif self.consensus == "hierarchical":
            U, V = _soft_from_hard(S, labels_h)
        elif self.consensus == "smcla":
            U, meta = soft_mcla(self.base_memberships_, c, random_state=rng)
            self.meta_cluster_labels_ = meta
        else:  # itk
            labels, _ = information_theoretic_kmeans(self.base_memberships_, c, random_state=rng)
            U, V = _soft_from_hard(S, labels)
        if self.consensus != "smcla":
            self.consensus_centers_ = V
        self.n_clusters_ = U.shape[1]
        self.consensus_membership_ = U
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)

    # ------------------------------------------------------------------ inference
    def _new_coassociation(self, X):
        """Co-association of new samples with the training samples."""
        Us_new = [est.predict_proba(X[:, cols]) for est, cols in
                  zip(self.base_estimators_, self.base_feature_subsets_, strict=True)]
        return _cross_coassociation(Us_new, self.base_memberships_, self.similarity)

    def predict_proba(self, X):
        """Consensus memberships of (new) samples.

        Returns
        -------
        U : ndarray of shape (n_samples, ``n_clusters_``)
        """
        check_is_fitted(self, "consensus_membership_")
        X = validate_data(self, X, dtype=float, reset=False)
        if self.consensus == "smcla":
            H = np.hstack([est.predict_proba(X[:, cols]) for est, cols in
                           zip(self.base_estimators_, self.base_feature_subsets_, strict=True)])
            U = np.zeros((X.shape[0], self.n_clusters_))
            for i in range(self.n_clusters_):
                members = self.meta_cluster_labels_ == i
                if members.any():
                    U[:, i] = H[:, members].mean(axis=1)
            U = np.maximum(U, EPS)
            return U / U.sum(axis=1, keepdims=True)
        S_new = self._new_coassociation(X)
        return memberships_from_distances(sq_euclidean(S_new, self.consensus_centers_), 2.0)

    def predict(self, X):
        """Hard consensus labels of (new) samples."""
        return self.predict_proba(X).argmax(axis=1)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


def stability_curve(
    X,
    n_estimators_grid=(1, 2, 5, 10, 20),
    n_repeats: int = 5,
    random_state=None,
    index=fuzzy_rand_index,
    **ensemble_params,
) -> pd.DataFrame:
    r"""Reproducibility of the consensus partition as the library grows.

    For every library size, ``n_repeats`` independent ensembles are fitted
    and all pairs of their consensus partitions are compared with the
    fuzzy Rand index (or another ``index``); the mean and standard deviation of the agreement is
    the *stability* of the consensus at that size. A single run
    (``n_estimators=1``) gives the stability of the base clusterer itself.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_estimators_grid : sequence of int, default=(1, 2, 5, 10, 20)
    n_repeats : int, default=5
    random_state : int, RandomState instance or None
    index : callable, default=:func:`fuzzyclust.validity.fuzzy_rand_index`
        Comparison of two membership matrices, e.g.
        :func:`~fuzzyclust.validity.fuzzy_adjusted_rand_index`.
    **ensemble_params
        Passed to :class:`FuzzyClusterEnsemble`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``n_estimators`` with columns ``mean`` and ``std`` of the
        pairwise agreement (``index``) between repeats.

    Examples
    --------
    >>> from fuzzyclust.ensemble import stability_curve
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(n_samples_per_cluster=30, random_state=0)
    >>> table = stability_curve(X, n_estimators_grid=(1, 3), n_repeats=2, n_clusters=3, random_state=0)
    >>> list(table.columns), list(table.index)
    (['mean', 'std'], [1, 3])
    """
    rng = check_random_state(random_state)
    rows = {}
    for Q in n_estimators_grid:
        parts = [
            FuzzyClusterEnsemble(n_estimators=int(Q), random_state=rng, **ensemble_params)
            .fit(X)
            .consensus_membership_
            for _ in range(n_repeats)
        ]
        vals = [
            index(parts[a], parts[b]) for a in range(n_repeats) for b in range(a + 1, n_repeats)
        ]
        rows[int(Q)] = {
            "mean": float(np.mean(vals)) if vals else np.nan,
            "std": float(np.std(vals)) if vals else np.nan,
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "n_estimators"
    return df
