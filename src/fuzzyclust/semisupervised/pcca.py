# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Pairwise-constrained fuzzy clustering (Grira, Crucianu and Boujemaa, 2006)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer

__all__ = ["PairwiseConstrainedFCM", "check_pairwise_constraints", "constraint_satisfaction"]


def check_pairwise_constraints(must_link=None, cannot_link=None, n_samples: int | None = None):
    """Validate lists of pairwise constraints and return them as integer arrays ``(n_pairs, 2)``.

    Pairs are unordered; a pair that appears in both lists is an error.
    """
    out = []
    for name, pairs in (("must_link", must_link), ("cannot_link", cannot_link)):
        P = np.asarray(pairs if pairs is not None else [], dtype=int).reshape(-1, 2)
        if P.size and n_samples is not None and (P.min() < 0 or P.max() >= n_samples):
            raise ValueError(f"{name} contains sample indices outside [0, {n_samples}).")
        if P.size and np.any(P[:, 0] == P[:, 1]):
            raise ValueError(f"{name} contains a pair of identical indices.")
        out.append(np.sort(P, axis=1))
    M, C = out
    if M.size and C.size:
        both = set(map(tuple, M)) & set(map(tuple, C))
        if both:
            raise ValueError(f"Pairs {sorted(both)[:3]} appear as must-link and as cannot-link.")
    return M, C


def constraint_satisfaction(labels, must_link=None, cannot_link=None):
    """Fraction of satisfied must-link and cannot-link pairs under hard ``labels``.

    Returns
    -------
    must_link : float
        Share of must-link pairs placed in the same cluster (``nan`` if none).
    cannot_link : float
        Share of cannot-link pairs placed in different clusters (``nan`` if none).

    Examples
    --------
    >>> from fuzzyclust.semisupervised import constraint_satisfaction
    >>> constraint_satisfaction([0, 0, 1, 1], must_link=[(0, 1), (1, 2)], cannot_link=[(0, 3)])
    (0.5, 1.0)
    """
    labels = np.asarray(labels)
    M, C = check_pairwise_constraints(must_link, cannot_link, labels.shape[0])
    ml = float(np.mean(labels[M[:, 0]] == labels[M[:, 1]])) if M.size else float("nan")
    cl = float(np.mean(labels[C[:, 0]] != labels[C[:, 1]])) if C.size else float("nan")
    return ml, cl


class PairwiseConstrainedFCM(BaseFuzzyClusterer):
    r"""Fuzzy clustering with pairwise constraints, PCCA (Grira, Crucianu, and Boujemaa).

    Knowledge about the data comes as *pairs*: two samples that must end up
    in the same cluster (must-link) or in different clusters (cannot-link).
    Such constraints are far easier to collect than class labels -- a user
    only has to say whether two images belong together -- and they need no
    agreed set of classes. The algorithm adds to the c-means functional a
    penalty for violated constraints, and (optionally) the competitive
    agglomeration term of Frigui and Krishnapuram, which lets the number of
    clusters shrink from an over-estimate to the number the data and the
    constraints support.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c` (the initial, maximal number when
        ``agglomeration > 0``).
    alpha : float or "auto", default=1.0
        Weight :math:`\alpha\ge0` of the constraint penalty, in the units of
        the squared distances. ``"auto"`` renews it at every iteration as
        :math:`\sum_{ik}\mu_{ik}^2D_{ik}^2/|\mathcal{M}\cup\mathcal{C}|`, so
        that the few constraints weigh as much as the data. The published
        rule, :math:`\alpha = N\sum_{ik}\mu_{ik}^2D_{ik}^2 /
        \big(|\mathcal{M}\cup\mathcal{C}|\sum_{ik}\mu_{ik}^2\big)`, also
        divides by the mean squared membership; the two coincide for a crisp
        partition.
    agglomeration : float, default=0.0
        :math:`\eta_0` of the competitive agglomeration term; ``0`` keeps the
        number of clusters fixed (pure pairwise-constrained FCM).
    agglomeration_decay : float, default=10.0
        Time constant :math:`\tau` of the schedule :math:`\eta(l)=\eta_0 e^{-l/\tau}`.
    min_cardinality : float, default=1.0
        A cluster whose fuzzy cardinality :math:`N_i=\sum_k\mu_{ik}` falls
        below this value is discarded (only when ``agglomeration > 0``).
    m : float, default=2.0
        The paper's updates are derived for :math:`m=2`; other values raise.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (``n_clusters_``, n_features)
    membership_ : ndarray of shape (n_samples, ``n_clusters_``)
    labels_ : ndarray of shape (n_samples,)
    n_clusters_ : int
        Number of clusters at convergence (``n_clusters`` unless clusters were discarded).
    must_link_, cannot_link_ : ndarray of shape (n_pairs, 2)
        The validated constraints.
    must_link_satisfied_, cannot_link_satisfied_ : float
        Share of constraints satisfied by the hard partition.
    alpha_ : float
        Penalty weight at the last iteration.
    cardinalities_ : ndarray of shape (``n_clusters_``,)
        Fuzzy cardinalities :math:`N_i` of the final clusters.
    covariances_ : ndarray of shape (``n_clusters_``, n_features, n_features)
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    With the sets :math:`\mathcal{M}` of must-link and :math:`\mathcal{C}`
    of cannot-link pairs the objective is

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2 D_{ik}^2
          + \alpha\Big[\sum_{(k,l)\in\mathcal{M}}\sum_{i=1}^{c}\sum_{j\ne i}\mu_{ik}\mu_{jl}
          + \sum_{(k,l)\in\mathcal{C}}\sum_{i=1}^{c}\mu_{ik}\mu_{il}\Big]
          - \beta\sum_{i=1}^{c}N_i^2,\qquad N_i=\sum_{k=1}^{N}\mu_{ik},

    the bracket being the expected number of violated constraints and the
    last term the competitive agglomeration bias. The necessary conditions
    give a membership that is the FCM membership plus two corrections,

    .. math::
        \mu_{ik} = \frac{1/D_{ik}^2}{\sum_j 1/D_{jk}^2}
          + \frac{\alpha}{2D_{ik}^2}\big(\bar{C}_k - C_{ik}\big)
          + \frac{\beta}{D_{ik}^2}\big(N_i - \bar{N}_k\big),

    where :math:`C_{ik}=\sum_{l:(k,l)\in\mathcal{M}}(1-\mu_{il})+\sum_{l:(k,l)\in\mathcal{C}}\mu_{il}`
    is the violation cost of placing :math:`k` in cluster :math:`i`, and
    :math:`\bar C_k`, :math:`\bar N_k` are the :math:`1/D^2_{jk}`-weighted
    means of :math:`C_{jk}` and :math:`N_j` over the clusters. A cluster that
    violates fewer constraints than average gains membership; a cluster
    larger than average gains membership when :math:`\beta>0`. The
    corrections can push a membership below zero; as in competitive
    agglomeration, negative values are clipped and the row renormalized.
    The agglomeration weight follows Frigui and Krishnapuram's schedule
    :math:`\beta(l)=\eta_0e^{-l/\tau}\sum_{ik}\mu_{ik}^2D_{ik}^2/\sum_iN_i^2`,
    and clusters with :math:`N_i<` ``min_cardinality`` are removed. The
    prototypes are the usual FCM means. New samples carry no constraints
    and receive FCM memberships.

    References
    ----------
    .. [1] N. Grira, M. Crucianu, N. Boujemaa, "Fuzzy clustering with
           pairwise constraints for knowledge-driven image categorization",
           IEE Proceedings -- Vision, Image and Signal Processing 153(3),
           299--304, 2006.
    .. [2] H. Frigui, R. Krishnapuram, "Clustering by competitive
           agglomeration", Pattern Recognition 30(7), 1109--1119, 1997.

    Examples
    --------
    >>> from fuzzyclust.semisupervised import PairwiseConstrainedFCM, make_pairwise_constraints
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, y = make_fuzzy_blobs(random_state=0)
    >>> ml, cl = make_pairwise_constraints(y, n_must_link=20, n_cannot_link=20, random_state=0)
    >>> pcca = PairwiseConstrainedFCM(n_clusters=3, random_state=0).fit(X, must_link=ml, cannot_link=cl)
    >>> pcca.membership_.shape, pcca.must_link_.shape
    ((300, 3), (20, 2))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        alpha=1.0,
        agglomeration: float = 0.0,
        agglomeration_decay: float = 10.0,
        min_cardinality: float = 1.0,
        m: float = 2.0,
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
        self.alpha = alpha
        self.agglomeration = agglomeration
        self.agglomeration_decay = agglomeration_decay
        self.min_cardinality = min_cardinality

    # ------------------------------------------------------------------ validation
    def _check_params(self, X):
        super()._check_params(X)
        if self.m != 2.0:
            raise ValueError("PairwiseConstrainedFCM implements the m = 2 updates of the paper.")
        if isinstance(self.alpha, str):
            if self.alpha != "auto":
                raise ValueError("alpha must be a non-negative float or 'auto'.")
        elif self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.agglomeration < 0:
            raise ValueError("agglomeration must be non-negative.")
        if self.agglomeration_decay <= 0:
            raise ValueError("agglomeration_decay must be positive.")
        if self.min_cardinality < 0:
            raise ValueError("min_cardinality must be non-negative.")

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None, must_link=None, cannot_link=None):
        """Cluster ``X`` under pairwise constraints.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
        must_link : sequence of (int, int), optional
            Pairs of sample indices that belong to the same cluster.
        cannot_link : sequence of (int, int), optional
            Pairs of sample indices that belong to different clusters.

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        self._check_params(X)
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        N = X.shape[0]
        M, C = check_pairwise_constraints(must_link, cannot_link, N)
        self.must_link_, self.cannot_link_ = M, C
        self._A_must = self._adjacency(M, N)
        self._A_cannot = self._adjacency(C, N)
        self._n_constraints = M.shape[0] + C.shape[0]
        rng = check_random_state(self.random_state)
        n_init = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(n_init):
            result = self._fit_single(X, rng)
            if best is None or result["history"][-1] < best["history"][-1]:
                best = result
        self.cluster_centers_ = best["V"]
        self.membership_ = best["U"]
        self.labels_ = best["U"].argmax(axis=1)
        self.distances_ = np.sqrt(best["d2"])
        self.objective_history_ = np.asarray(best["history"])
        self.objective_ = best["history"][-1]
        self.n_iter_ = best["n_iter"]
        self.alpha_ = best["alpha"]
        self.n_clusters_ = best["U"].shape[1]
        self.cardinalities_ = best["U"].sum(axis=0)
        self.covariances_ = fuzzy_covariances(X, best["U"] ** 2, best["V"])
        self.must_link_satisfied_, self.cannot_link_satisfied_ = constraint_satisfaction(self.labels_, M, C)
        return self

    @staticmethod
    def _adjacency(P, N):
        if P.size == 0:
            return None
        rows = np.r_[P[:, 0], P[:, 1]]
        cols = np.r_[P[:, 1], P[:, 0]]
        return coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(N, N)).tocsr()

    def _violation_cost(self, U):
        """:math:`C_{ik}` -- expected number of constraints violated by placing ``k`` in ``i``."""
        Cik = np.zeros_like(U)
        if self._A_must is not None:
            Cik += self._A_must @ (1.0 - U)
        if self._A_cannot is not None:
            Cik += self._A_cannot @ U
        return Cik

    def _fit_single(self, X, rng):
        U = self._init_state(X, rng)
        history = []
        alpha = 0.0
        for it in range(1, self.max_iter + 1):
            V = fuzzy_centers(X, U**2)
            d2 = sq_euclidean(X, V)
            inv = 1.0 / (d2 + EPS)
            inv_sum = inv.sum(axis=1, keepdims=True)
            fcm = inv / inv_sum
            J_fcm = float(np.sum(U**2 * d2))
            Ni = U.sum(axis=0)
            # constraint penalty
            if self._n_constraints:
                alpha = J_fcm / self._n_constraints if isinstance(self.alpha, str) else float(self.alpha)
                Cik = self._violation_cost(U)
                Cbar = (inv * Cik).sum(axis=1, keepdims=True) / inv_sum
                corr_c = 0.5 * alpha * inv * (Cbar - Cik)
                violations = float(np.sum(U * Cik)) / 2.0
            else:
                Cik = np.zeros_like(U)
                corr_c = 0.0
                violations = 0.0
            # competitive agglomeration bias
            if self.agglomeration > 0:
                eta = self.agglomeration * np.exp(-it / self.agglomeration_decay)
                beta = eta * J_fcm / max(np.sum(Ni**2), EPS)
                Nbar = (inv * Ni[None, :]).sum(axis=1, keepdims=True) / inv_sum
                corr_b = beta * inv * (Ni[None, :] - Nbar)
            else:
                beta = 0.0
                corr_b = 0.0
            U_new = fcm + corr_c + corr_b
            np.maximum(U_new, 0.0, out=U_new)
            U_new /= np.maximum(U_new.sum(axis=1, keepdims=True), EPS)
            history.append(J_fcm + alpha * violations - beta * float(np.sum(Ni**2)))
            if self.verbose:
                print(f"iter {it:4d}  J = {history[-1]:.6g}  clusters = {U.shape[1]}")
            # discard empty clusters (competitive agglomeration)
            if self.agglomeration > 0 and U_new.shape[1] > 1:
                keep = U_new.sum(axis=0) >= self.min_cardinality
                if not keep.all():
                    if not keep.any():
                        keep[U_new.sum(axis=0).argmax()] = True
                    U_new = U_new[:, keep]
                    U_new /= np.maximum(U_new.sum(axis=1, keepdims=True), EPS)
                    U = U_new
                    continue
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol:
                break
        V = fuzzy_centers(X, U**2)
        d2 = sq_euclidean(X, V)
        return {"U": U, "V": V, "d2": d2, "history": history, "n_iter": it, "alpha": alpha}

    # ------------------------------------------------------------------ base hooks
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        self.cluster_centers_ = fuzzy_centers(X, Um)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        return memberships_from_distances(d2, 2.0)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters_)], dtype=object)
