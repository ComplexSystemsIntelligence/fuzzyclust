# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Collaborative fuzzy clustering (Pedrycz, 2002)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state

from .._utils import EPS, memberships_from_distances, sq_euclidean
from ..cluster import FCM
from ..validity import fuzzy_rand_index

__all__ = ["CollaborativeFCM", "align_partitions"]


def align_partitions(U_ref, U):
    """Permute the clusters of ``U`` to match those of ``U_ref`` (maximal membership overlap).

    Returns the permutation ``perm`` such that ``U[:, perm]`` is aligned, found
    with the Hungarian algorithm on the overlap matrix ``U_ref.T @ U``.
    """
    overlap = np.asarray(U_ref).T @ np.asarray(U)
    rows, cols = linear_sum_assignment(-overlap)
    perm = np.empty(overlap.shape[1], dtype=int)
    perm[rows] = cols
    return perm


class CollaborativeFCM(BaseEstimator):
    r"""Horizontal collaborative fuzzy clustering (Pedrycz, 2002).

    The same :math:`N` objects are described at :math:`P` *data sites* by
    different sets of features -- departments of a company, laboratories
    measuring different variables, sensors of different kinds -- and the
    raw data must stay where they are (confidentiality, size, legacy
    formats). Each site runs its own fuzzy c-means; the sites then
    *collaborate* by exchanging only their partition matrices and re-run
    the clustering with an extra term that penalizes the difference
    between their own memberships and those of the other sites, weighted
    by a collaboration matrix :math:`\alpha[ii,jj]`. The structures found
    at the sites are pulled toward each other without any data leaving a
    site; how far they move tells how much the sites agree.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`, the same at every site (the sites
        must talk about the same clusters; after the independent phase the
        clusters of the sites are matched by the Hungarian algorithm on
        their membership overlap).
    alpha : float or array-like of shape (n_sites, n_sites), default=0.5
        Collaboration matrix :math:`\alpha[ii,jj]\ge0`, the weight site
        :math:`ii` gives to the partition of site :math:`jj`; a scalar
        applies to all pairs. ``0`` gives :math:`P` independent FCM runs.
    m : float, default=2.0
        The paper's updates are derived for :math:`m=2`; other values raise.
    tol : float, default=1e-4
        Termination tolerance on :math:`\max|\Delta\mu|` over all sites,
        used in both phases.
    max_iter : int, default=300
        Maximal number of collaboration iterations (and of FCM iterations in the first phase).
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization of the independent FCM runs.
    n_init : int, default=1
        Random restarts of the independent FCM runs (lowest :math:`J` kept).
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    n_sites_ : int
    alpha_ : ndarray of shape (n_sites, n_sites)
        The collaboration matrix used (zero diagonal).
    memberships_ : list of ndarray of shape (n_samples, n_clusters)
        Partition :math:`\mathbf{U}[ii]^T` of every site after collaboration.
    cluster_centers_ : list of ndarray of shape (n_clusters, n_features_ii)
        Prototypes of every site in its own feature space.
    labels_ : list of ndarray of shape (n_samples,)
    independent_memberships_, independent_centers_ : list
        The results of the independent FCM runs (phase 1, clusters aligned).
    objective_ : ndarray of shape (n_sites,)
        Final collaborative objective :math:`Q[ii]` of every site.
    objective_history_ : ndarray of shape (``n_iter_``, n_sites)
    n_iter_ : int
        Number of collaboration iterations.
    partition_distances_ : ndarray of shape (n_sites, n_sites)
        :math:`\frac1N\sum_{i,k}(\mu_{ik}[ii]-\mu_{ik}[jj])^2` between the
        collaborative partitions; ``independent_partition_distances_`` is
        the same for phase 1.
    collaboration_change_ : ndarray of shape (n_sites,)
        Mean absolute membership change of every site from phase 1 to the
        collaborative result -- how much the site gave in.
    consensus_membership_ : ndarray of shape (n_samples, n_clusters)
        Row-normalized mean of the site partitions after collaboration.

    Notes
    -----
    Site :math:`ii` minimizes, with its own squared distances
    :math:`D_{ik}^2[ii]` to its prototypes,

    .. math::
        Q[ii] = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2[ii]\,D_{ik}^2[ii]
              + \sum_{jj\ne ii}\alpha[ii,jj]\sum_{i=1}^{c}\sum_{k=1}^{N}
                \big(\mu_{ik}[ii]-\mu_{ik}[jj]\big)^2 D_{ik}^2[ii] .

    The necessary conditions give, with the collaboration weight
    :math:`\bar\alpha[ii]=\sum_{jj\ne ii}\alpha[ii,jj]` and the communicated memberships
    :math:`\varphi_{ik}[ii]=\sum_{jj\ne ii}\alpha[ii,jj]\mu_{ik}[jj]\big/\big(1+\bar\alpha[ii]\big)`,

    .. math::
        \mu_{ik}[ii] = \varphi_{ik}[ii] + \frac{1-\sum_{l=1}^{c}\varphi_{lk}[ii]}
                                            {\sum_{l=1}^{c} D_{ik}^2[ii]/D_{lk}^2[ii]},
        \qquad
        \mathbf{v}_i[ii] = \frac{\sum_k w_{ik}[ii]\,\mathbf{x}_k[ii]}{\sum_k w_{ik}[ii]},
        \quad
        w_{ik}[ii] = \mu_{ik}^2[ii]+\sum_{jj\ne ii}\alpha[ii,jj]\big(\mu_{ik}[ii]-\mu_{ik}[jj]\big)^2 :

    the membership is a convex combination of the other sites' memberships
    and the site's own FCM membership, with weight
    :math:`\sum\alpha/(1+\sum\alpha)` on the collaborators. The algorithm
    has two phases: independent FCM at every site, then synchronous
    collaboration iterations in which every site updates its prototypes
    and memberships from the partitions the other sites communicated in
    the previous round. New objects described at all sites receive
    memberships by iterating the membership formula with the fitted
    prototypes (:meth:`predict_proba`).

    References
    ----------
    .. [1] W. Pedrycz, "Collaborative fuzzy clustering", Pattern Recognition
           Letters 23(14), 1675--1686, 2002.

    Examples
    --------
    >>> from fuzzyclust.collaborative import CollaborativeFCM, make_collaborative_sites
    >>> Xs, y = make_collaborative_sites(random_state=0)
    >>> cfcm = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(Xs)
    >>> len(cfcm.memberships_), cfcm.memberships_[0].shape
    (2, (300, 3))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        alpha=0.5,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ validation
    def _check_sites(self, Xs, fitted=False):
        if isinstance(Xs, np.ndarray) and Xs.ndim == 2:
            Xs = [Xs]
        if not isinstance(Xs, (list, tuple)) or len(Xs) == 0:
            raise ValueError("Give the data sites as a list of arrays [X_1, ..., X_P] sharing the rows.")
        Xs = [check_array(X, dtype=float, ensure_min_samples=2) for X in Xs]
        N = Xs[0].shape[0]
        if any(X.shape[0] != N for X in Xs):
            raise ValueError("All sites must describe the same objects: equal numbers of rows.")
        if fitted:
            if len(Xs) != self.n_sites_:
                raise ValueError(f"Expected {self.n_sites_} sites, got {len(Xs)}.")
            for X, V in zip(Xs, self.cluster_centers_, strict=True):
                if X.shape[1] != V.shape[1]:
                    raise ValueError("A site has a different number of features than at fit time.")
        return Xs

    def _check_params(self, Xs):
        c = self.n_clusters
        P = len(Xs)
        if not isinstance(c, (int, np.integer)) or c < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if c > Xs[0].shape[0]:
            raise ValueError("n_clusters must not exceed the number of objects.")
        if self.m != 2.0:
            raise ValueError("CollaborativeFCM implements the m = 2 updates of the paper.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        A = np.asarray(self.alpha, dtype=float)
        if A.ndim == 0:
            A = np.full((P, P), float(A))
        if A.shape != (P, P):
            raise ValueError(f"alpha must be a scalar or a ({P}, {P}) matrix.")
        if np.any(A < 0):
            raise ValueError("Collaboration weights must be non-negative.")
        A = A.copy()
        np.fill_diagonal(A, 0.0)
        return A

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Run the independent phase at every site, then the collaboration.

        Parameters
        ----------
        X : list of array-like, each of shape (n_samples, n_features_ii)
            The data sites; the rows of all arrays are the same objects.
        y : Ignored

        Returns
        -------
        self : object
        """
        Xs = self._check_sites(X)
        A = self._check_params(Xs)
        P = len(Xs)
        self.n_sites_ = P
        self.alpha_ = A
        rng = check_random_state(self.random_state)
        # phase 1: independent FCM at every site
        Us, Vs = [], []
        for X in Xs:
            fcm = FCM(
                n_clusters=self.n_clusters, m=2.0, tol=self.tol, max_iter=self.max_iter, init=self.init,
                n_init=self.n_init, random_state=rng,
            ).fit(X)
            Us.append(fcm.membership_)
            Vs.append(fcm.cluster_centers_)
        for p in range(1, P):  # the sites must agree on the labels of the clusters
            perm = align_partitions(Us[0], Us[p])
            Us[p] = Us[p][:, perm]
            Vs[p] = Vs[p][perm]
        self.independent_memberships_ = [U.copy() for U in Us]
        self.independent_centers_ = [V.copy() for V in Vs]
        # phase 2: collaboration
        history = []
        n_iter = 0
        if np.any(A > 0):
            for it in range(1, self.max_iter + 1):
                Us_new, Vs_new, Q = [], [], []
                for ii in range(P):
                    V = self._prototypes(Xs[ii], Us, A, ii)
                    d2 = sq_euclidean(Xs[ii], V)
                    U = self._collaborative_memberships(d2, Us, A, ii)
                    Q.append(self._objective(U, Us, d2, A, ii))
                    Us_new.append(U)
                    Vs_new.append(V)
                history.append(Q)
                delta = max(np.max(np.abs(Un - Uo)) for Un, Uo in zip(Us_new, Us, strict=True))
                Us, Vs = Us_new, Vs_new
                n_iter = it
                if self.verbose:
                    print(f"collaboration iter {it:4d}  Q = {np.round(Q, 4)}")
                if delta < self.tol:
                    break
        else:  # alpha = 0: the independent runs are the result
            Q = [float(np.sum(U**2 * sq_euclidean(X, V))) for X, U, V in zip(Xs, Us, Vs, strict=True)]
            history.append(Q)
        self.memberships_ = Us
        self.cluster_centers_ = Vs
        self.labels_ = [U.argmax(axis=1) for U in Us]
        self.objective_history_ = np.asarray(history)
        self.objective_ = self.objective_history_[-1]
        self.n_iter_ = n_iter
        self.partition_distances_ = self._pairwise_distances(Us)
        self.independent_partition_distances_ = self._pairwise_distances(self.independent_memberships_)
        self.collaboration_change_ = np.array(
            [float(np.mean(np.abs(U - U0))) for U, U0 in zip(Us, self.independent_memberships_, strict=True)]
        )
        mean = np.mean(Us, axis=0)
        self.consensus_membership_ = mean / mean.sum(axis=1, keepdims=True)
        return self

    @staticmethod
    def _pairwise_distances(Us):
        P = len(Us)
        N = Us[0].shape[0]
        D = np.zeros((P, P))
        for a in range(P):
            for b in range(a + 1, P):
                D[a, b] = D[b, a] = float(np.sum((Us[a] - Us[b]) ** 2)) / N
        return D

    @staticmethod
    def _collaborative_memberships(d2, Us, A, ii):
        inv = 1.0 / (d2 + EPS)
        fcm = inv / inv.sum(axis=1, keepdims=True)
        others = [jj for jj in range(len(Us)) if jj != ii and A[ii, jj] > 0]
        if not others:
            return fcm
        s = sum(A[ii, jj] for jj in others)
        phi = sum(A[ii, jj] * Us[jj] for jj in others) / (1.0 + s)
        return phi + (1.0 - phi.sum(axis=1, keepdims=True)) * fcm

    @staticmethod
    def _prototypes(X, Us, A, ii):
        U = Us[ii]
        W = U**2
        for jj in range(len(Us)):
            if jj != ii and A[ii, jj] > 0:
                W = W + A[ii, jj] * (U - Us[jj]) ** 2
        s = np.maximum(W.sum(axis=0), EPS)
        return (W.T @ X) / s[:, None]

    @staticmethod
    def _objective(U, Us, d2, A, ii):
        Q = float(np.sum(U**2 * d2))
        for jj in range(len(Us)):
            if jj != ii and A[ii, jj] > 0:
                Q += A[ii, jj] * float(np.sum((U - Us[jj]) ** 2 * d2))
        return Q

    # ------------------------------------------------------------------ inference
    def predict_proba(self, X, max_iter: int = 50):
        """Memberships of new objects at every site, with the fitted prototypes and collaboration.

        Parameters
        ----------
        X : list of array-like
            The new objects described at every site (same order of sites as at fit time).
        max_iter : int, default=50
            Iterations of the collaborative membership formula (starting
            from the independent FCM memberships of every site).

        Returns
        -------
        Us : list of ndarray of shape (n_samples, n_clusters)
        """
        Xs = self._check_sites(X, fitted=True)
        d2s = [sq_euclidean(Xi, V) for Xi, V in zip(Xs, self.cluster_centers_, strict=True)]
        Us = [memberships_from_distances(d2, 2.0) for d2 in d2s]
        if np.any(self.alpha_ > 0):
            for _ in range(max_iter):
                Us_new = [
                    self._collaborative_memberships(d2s[ii], Us, self.alpha_, ii) for ii in range(len(Xs))
                ]
                delta = max(np.max(np.abs(a - b)) for a, b in zip(Us_new, Us, strict=True))
                Us = Us_new
                if delta < self.tol:
                    break
        return Us

    def predict(self, X):
        """Hard labels of new objects at every site."""
        return [U.argmax(axis=1) for U in self.predict_proba(X)]

    def transform(self, X):
        """Distances of new objects to the prototypes of every site."""
        Xs = self._check_sites(X, fitted=True)
        return [np.sqrt(sq_euclidean(Xi, V)) for Xi, V in zip(Xs, self.cluster_centers_, strict=True)]

    def site_agreement(self, index=fuzzy_rand_index):
        """Pairwise agreement of the site partitions.

        Parameters
        ----------
        index : callable, default=:func:`fuzzyclust.validity.fuzzy_rand_index`
            Comparison ``index(U_a, U_b) -> float`` of two membership matrices,
            e.g. :func:`~fuzzyclust.validity.fuzzy_adjusted_rand_index`.

        Returns
        -------
        agreement : ndarray of shape (n_sites, n_sites)
        """
        P = self.n_sites_
        S = np.eye(P)
        for a in range(P):
            for b in range(a + 1, P):
                S[a, b] = S[b, a] = index(self.memberships_[a], self.memberships_[b])
        return S
