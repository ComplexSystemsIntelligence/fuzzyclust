# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy co-clustering of objects and items: FCCM and Fuzzy CoDoK."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS

__all__ = ["FCCM", "FCoDoK"]


class _BaseFuzzyCoClustering(ClusterMixin, TransformerMixin, BaseEstimator):
    r"""Shared machinery of the fuzzy co-clustering algorithms (not public).

    Both algorithms maximize the *aggregation degree*
    :math:`\sum_{i=1}^{c}\sum_{k=1}^{N}\sum_{j=1}^{M} u_{ik} w_{ij} r_{kj}`
    of a non-negative object--item matrix :math:`\mathbf{R}=[r_{kj}]`
    under the constraints :math:`\sum_i u_{ik}=1` (every object is shared
    among the co-clusters) and :math:`\sum_j w_{ij}=1` (every co-cluster
    distributes one unit of item membership), with a regularizer that
    keeps both membership sets from becoming crisp. They differ in the
    regularizer only.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        lambda_u: float = 1.0,
        lambda_w: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.lambda_u = lambda_u
        self.lambda_w = lambda_w
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ hooks
    def _object_memberships(self, A):
        """Object memberships ``(n_objects, c)`` from the aggregations ``A = R @ W.T``."""
        raise NotImplementedError

    def _item_memberships(self, B):
        """Item memberships ``(c, n_items)`` from the aggregations ``B = U.T @ R``."""
        raise NotImplementedError

    def _regularizer(self, U, W):
        raise NotImplementedError

    # ------------------------------------------------------------------ checks
    def _check_params(self, R):
        c = self.n_clusters
        if not isinstance(c, (int, np.integer)) or c < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if c > R.shape[0]:
            raise ValueError(f"n_clusters={c} must be <= n_objects={R.shape[0]}.")
        if self.lambda_u <= 0 or self.lambda_w <= 0:
            raise ValueError("lambda_u and lambda_w must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if np.any(R < 0):
            raise ValueError(
                "Negative values in data: the object-item matrix must be non-negative "
                "(counts, frequencies, TF-IDF weights)."
            )

    def _init_state(self, R, rng):
        N, c = R.shape[0], self.n_clusters
        init = self.init
        if isinstance(init, str):
            if init == "random":
                U = rng.rand(N, c) + EPS
            elif init == "kmeans++":
                from sklearn.cluster import kmeans_plusplus

                V, _ = kmeans_plusplus(R, n_clusters=c, random_state=rng)
                d2 = ((R[:, None, :] - V[None]) ** 2).sum(axis=2)
                U = 1.0 / (d2 + EPS)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
        else:
            U = np.asarray(init, dtype=float)
            if U.shape != (N, c):
                raise ValueError(f"init array must have shape ({N}, {c}).")
        return U / U.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        r"""Co-cluster the rows (objects) and columns (items) of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_objects, n_items)
            Non-negative object--item matrix :math:`\mathbf{R}` (counts,
            frequencies, TF-IDF weights, cooccurrences).
        y : Ignored

        Returns
        -------
        self : object
        """
        R = validate_data(self, X, dtype=float, ensure_min_samples=2)
        self._check_params(R)
        rng = check_random_state(self.random_state)
        n_init = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for _ in range(n_init):
            result = self._fit_single(R, rng)
            if best is None or result["history"][-1] > best["history"][-1]:  # maximization
                best = result
        self.membership_ = best["U"]
        self.item_membership_ = best["W"].T
        self.labels_ = best["U"].argmax(axis=1)
        self.item_labels_ = best["W"].argmax(axis=0)
        self.aggregation_ = best["aggregation"]
        self.objective_history_ = np.asarray(best["history"])
        self.objective_ = best["history"][-1]
        self.n_iter_ = best["n_iter"]
        return self

    def _fit_single(self, R, rng):
        U = self._init_state(R, rng)
        history = []
        for it in range(1, self.max_iter + 1):
            W = self._item_memberships(U.T @ R)  # (c, M)
            A = R @ W.T  # (N, c)
            U_new = self._object_memberships(A)
            agg = float(np.sum(U_new * A))
            history.append(agg + self._regularizer(U_new, W))
            if self.verbose:
                print(f"iter {it:4d}  J = {history[-1]:.6g}  aggregation = {agg:.6g}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol:
                break
        return {"U": U, "W": W, "aggregation": agg, "history": history, "n_iter": it}

    # ------------------------------------------------------------------ inference
    def transform(self, X):
        r"""Aggregation :math:`a_{ik}=\sum_j w_{ij}r_{kj}` of (new) rows with every co-cluster's item profile.

        Returns
        -------
        A : ndarray of shape (n_objects, n_clusters)
        """
        check_is_fitted(self, "item_membership_")
        R = validate_data(self, X, dtype=float, reset=False)
        return R @ self.item_membership_

    def predict_proba(self, X):
        """Object memberships of (new) rows given the fitted item memberships."""
        return self._object_memberships(self.transform(X))

    def predict(self, X):
        """Hard co-cluster label of (new) rows."""
        return self.predict_proba(X).argmax(axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def score(self, X, y=None):
        """Aggregation degree of ``X`` with the fitted item memberships (higher is better)."""
        A = self.transform(X)
        return float(np.sum(self._object_memberships(A) * A))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "item_membership_")
        return np.asarray([f"cocluster_{i}" for i in range(self.n_clusters)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.input_tags.positive_only = True
        return tags


class FCCM(_BaseFuzzyCoClustering):
    r"""Fuzzy clustering for categorical multivariate data, FCCM (Oh, Honda, and Ichihashi, 2001).

    A document--term matrix, a user--item purchase table or any other
    non-negative *cooccurrence* matrix :math:`\mathbf{R}` has two sets of
    entities: objects (rows) and items (columns). Co-clustering looks for
    pairs of an object group and an item group that cooccur strongly -- a
    topic together with its vocabulary. FCCM assigns a fuzzy membership to
    both the objects and the items of every co-cluster and maximizes the
    *aggregation degree* of the pairs, regularized by the entropies of both
    membership sets so that the solution stays fuzzy. Unlike k-means type
    clustering of the rows it needs no distance between sparse count
    vectors, and it returns, per cluster, *which items* characterize it.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of co-clusters :math:`c`.
    lambda_u : float, default=1.0
        Entropy weight :math:`\lambda_u>0` of the object memberships. It is
        measured in the units of a row's aggregation :math:`a_{ik}=\sum_j
        w_{ij}r_{kj}`, which is a weighted *mean* of the row's entries over
        the items of the co-cluster -- so a value comparable to the typical
        entry of :math:`\mathbf{R}` gives a moderately fuzzy partition, a
        much smaller value a nearly crisp one.
    lambda_w : float, default=1.0
        Entropy weight :math:`\lambda_w>0` of the item memberships, in the
        units of a column's aggregation :math:`b_{ij}=\sum_k u_{ik}r_{kj}`,
        which is a weighted *sum* over the objects of the co-cluster and
        therefore grows with the number of objects.
    tol : float, default=1e-4
        Termination tolerance on :math:`\max|\Delta u_{ik}|`.
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray of shape (n_objects, n_clusters), default="random"
    n_init : int, default=1
        Random restarts; the run with the largest objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    membership_ : ndarray of shape (n_objects, n_clusters)
        Object memberships :math:`u_{ik}` (rows sum to one).
    item_membership_ : ndarray of shape (n_items, n_clusters)
        Item memberships :math:`w_{ij}`; every *column* (co-cluster) sums to
        one, so the entries are the share of the co-cluster's item mass
        and are small when the vocabulary of a cluster is large.
    labels_ : ndarray of shape (n_objects,)
        Hard object labels.
    item_labels_ : ndarray of shape (n_items,)
        Co-cluster in which every item has its largest membership.
    aggregation_ : float
        Final aggregation degree :math:`\sum_{ikj}u_{ik}w_{ij}r_{kj}`.
    objective_, objective_history_, n_iter_
        Regularized objective (maximized) and its history.

    Notes
    -----
    With object memberships :math:`u_{ik}` (cluster :math:`i`, object
    :math:`k`) and item memberships :math:`w_{ij}` (item :math:`j`), FCCM
    maximizes

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\sum_{j=1}^{M} u_{ik}w_{ij}r_{kj}
            - \lambda_u\sum_{i=1}^{c}\sum_{k=1}^{N} u_{ik}\log u_{ik}
            - \lambda_w\sum_{i=1}^{c}\sum_{j=1}^{M} w_{ij}\log w_{ij}

    subject to :math:`\sum_{i}u_{ik}=1` and :math:`\sum_{j}w_{ij}=1`. Without
    the entropy terms the maximum is crisp and degenerate (one cluster
    takes everything); the terms are the entropy regularization of
    Miyamoto and Mukaidono, and the necessary conditions are soft-max
    assignments

    .. math::
        u_{ik} = \frac{\exp\big(\sum_j w_{ij}r_{kj}/\lambda_u\big)}
                      {\sum_{l=1}^{c}\exp\big(\sum_j w_{lj}r_{kj}/\lambda_u\big)},\qquad
        w_{ij} = \frac{\exp\big(\sum_k u_{ik}r_{kj}/\lambda_w\big)}
                      {\sum_{l=1}^{M}\exp\big(\sum_k u_{ik}r_{kl}/\lambda_w\big)},

    iterated alternately until the object memberships settle. New rows
    receive memberships through the first formula with the fitted item
    memberships (:meth:`predict_proba`).

    References
    ----------
    .. [1] C.-H. Oh, K. Honda, H. Ichihashi, "Fuzzy clustering for
           categorical multivariate data", Proc. Joint 9th IFSA World
           Congress and 20th NAFIPS International Conference, 2154--2159, 2001.
    .. [2] S. Miyamoto, M. Mukaidono, "Fuzzy c-means as a regularization and
           maximum entropy approach", Proc. 7th IFSA World Congress,
           vol. 2, 86--92, 1997.

    Examples
    --------
    >>> from fuzzyclust.coclustering import FCCM, make_block_matrix
    >>> R, rows, cols = make_block_matrix(random_state=0)
    >>> fccm = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    >>> fccm.membership_.shape, fccm.item_membership_.shape
    ((120, 3), (60, 3))
    """

    def _object_memberships(self, A):
        return softmax(A / self.lambda_u, axis=1)

    def _item_memberships(self, B):
        return softmax(B / self.lambda_w, axis=1)

    def _regularizer(self, U, W):
        hu = -float(np.sum(U * np.log(U + EPS)))
        hw = -float(np.sum(W * np.log(W + EPS)))
        return self.lambda_u * hu + self.lambda_w * hw


class FCoDoK(_BaseFuzzyCoClustering):
    r"""Fuzzy co-clustering of documents and keywords, Fuzzy CoDoK (Kummamuru, Dhawale, Krishnapuram, 2003).

    The same aggregation objective as :class:`FCCM`, regularized by the
    *squared* memberships instead of their entropies. The quadratic
    regularizer makes the updates linear -- a membership is the uniform
    value plus a term proportional to how much the entity's aggregation
    with the cluster exceeds the average -- so that the method scales to
    large document collections and is less prone to the numerical
    overflow of the exponentials; memberships that would become negative
    are clipped and the row (or the cluster's item vector) renormalized.

    Parameters
    ----------
    n_clusters : int, default=3
    lambda_u : float, default=1.0
        Weight :math:`T_u>0` of the quadratic regularizer of the object
        memberships; larger values give fuzzier objects (same units as in
        :class:`FCCM`).
    lambda_w : float, default=1.0
        Weight :math:`T_w>0` for the item memberships.
    tol, max_iter, init, n_init, random_state, verbose
        See :class:`FCCM`.

    Attributes
    ----------
    membership_, item_membership_, labels_, item_labels_, aggregation_
        See :class:`FCCM`.
    objective_, objective_history_, n_iter_
        See :class:`FCCM`.

    Notes
    -----
    Fuzzy CoDoK maximizes

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\sum_{j=1}^{M} u_{ik}w_{ij}r_{kj}
            - T_u\sum_{i=1}^{c}\sum_{k=1}^{N} u_{ik}^2
            - T_w\sum_{i=1}^{c}\sum_{j=1}^{M} w_{ij}^2

    subject to :math:`\sum_i u_{ik}=1`, :math:`\sum_j w_{ij}=1`. With the
    aggregations :math:`a_{ik}=\sum_j w_{ij}r_{kj}` and
    :math:`b_{ij}=\sum_k u_{ik}r_{kj}` the necessary conditions read

    .. math::
        u_{ik} = \frac{1}{c} + \frac{1}{2T_u}\Big(a_{ik}-\frac1c\sum_{l=1}^{c}a_{lk}\Big),\qquad
        w_{ij} = \frac{1}{M} + \frac{1}{2T_w}\Big(b_{ij}-\frac1M\sum_{l=1}^{M}b_{il}\Big),

    followed by clipping at zero and renormalization.

    References
    ----------
    .. [1] K. Kummamuru, A. Dhawale, R. Krishnapuram, "Fuzzy co-clustering of
           documents and keywords", Proc. 12th IEEE International Conference
           on Fuzzy Systems (FUZZ-IEEE), vol. 2, 772--777, 2003.

    Examples
    --------
    >>> from fuzzyclust.coclustering import FCoDoK, make_block_matrix
    >>> R, rows, cols = make_block_matrix(random_state=0)
    >>> codok = FCoDoK(n_clusters=3, lambda_u=2.0, lambda_w=50.0, random_state=0).fit(R)
    >>> codok.membership_.shape
    (120, 3)
    """

    @staticmethod
    def _clip_normalize(U):
        U = np.maximum(U, 0.0)
        s = U.sum(axis=1, keepdims=True)
        return U / np.maximum(s, EPS)

    def _object_memberships(self, A):
        c = A.shape[1]
        U = 1.0 / c + (A - A.mean(axis=1, keepdims=True)) / (2.0 * self.lambda_u)
        return self._clip_normalize(U)

    def _item_memberships(self, B):
        M = B.shape[1]
        W = 1.0 / M + (B - B.mean(axis=1, keepdims=True)) / (2.0 * self.lambda_w)
        return self._clip_normalize(W)

    def _regularizer(self, U, W):
        return -self.lambda_u * float(np.sum(U**2)) - self.lambda_w * float(np.sum(W**2))
