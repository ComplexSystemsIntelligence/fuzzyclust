# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Base class for all prototype-based (hard and fuzzy) clustering estimators.

The alternating-optimization (Picard) loop shared by k-means, fuzzy c-means,
Gustafson–Kessel, Gath–Geva and their derivatives is implemented once here.
Concrete algorithms only define how prototypes are updated and how distances
are measured.

Notation (see ``DESIGN.md``): the data matrix :math:`\mathbf{X}\in\mathbb{R}^{N\times n}`
has samples :math:`\mathbf{x}_k` as rows; the partition matrix
:math:`\mathbf{U}=[\mu_{ik}]\in\mathbb{R}^{c\times N}` collects the membership
of sample :math:`k` in cluster :math:`i`. In code the partition is stored
transposed as ``membership_`` of shape ``(n_samples, n_clusters)`` so that it
aligns with scikit-learn's ``predict_proba`` layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._utils import EPS, fuzzy_centers, memberships_from_distances, sq_euclidean


class BaseFuzzyClusterer(ClusterMixin, TransformerMixin, BaseEstimator, ABC):
    r"""Abstract base of the prototype-based clustering algorithms.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness (weighting) exponent :math:`m>1`. Larger values give fuzzier
        partitions; :math:`m\to 1` approaches a hard partition.
    tol : float, default=1e-4
        Termination tolerance :math:`\epsilon` on
        :math:`\max_{ik} |\mu_{ik}^{(l)} - \mu_{ik}^{(l-1)}|`.
    max_iter : int, default=300
        Maximum number of alternating-optimization iterations.
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization. ``"random"`` draws random prototypes inside the
        bounding box of the data and derives the initial partition from them
        (MATLAB toolbox behavior). ``"kmeans++"`` uses the k-means++ seeding.
        An array of shape ``(n_clusters, n_features)`` is used as initial
        prototypes, an array of shape ``(n_samples, n_clusters)`` as initial
        partition.
    n_init : int, default=1
        Number of random restarts. The alternating optimization converges to a
        local optimum of :math:`J`; with ``n_init > 1`` the run with the lowest
        final objective is kept (ignored when ``init`` is an array).
    random_state : int, RandomState instance or None, default=None
        Controls the random initialization.
    verbose : int, default=0
        Print the objective at every iteration when ``> 0``.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes :math:`\mathbf{v}_i`.
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition :math:`\mathbf{U}^T` (rows sum to one).
    labels_ : ndarray of shape (n_samples,)
        Hard labels, ``membership_.argmax(axis=1)``.
    distances_ : ndarray of shape (n_samples, n_clusters)
        Distances :math:`D_{ik}` of the training samples to the prototypes at
        convergence (algorithm specific norm).
    objective_ : float
        Final value of the objective :math:`J`.
    objective_history_ : ndarray of shape (``n_iter_``,)
        :math:`J` after every iteration.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Defined when ``X`` has feature names (e.g. a :class:`pandas.DataFrame`).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _update_prototypes(self, X: np.ndarray, Um: np.ndarray) -> None:
        """Update ``cluster_centers_`` (and any shape parameters) from ``U**m``."""

    @abstractmethod
    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Squared distances :math:`D^2_{ik}` of ``X`` to the current prototypes."""

    def _memberships(self, d2: np.ndarray) -> np.ndarray:
        """Partition from squared distances (fuzzy by default; k-means overrides)."""
        return memberships_from_distances(d2, self.m)

    def _objective(self, Um: np.ndarray, d2: np.ndarray) -> float:
        return float(np.sum(Um * d2))

    def _init_state(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Return the initial membership matrix ``(n_samples, n_clusters)``."""
        c, n = self.n_clusters, X.shape[1]
        init = self.init
        if isinstance(init, str):
            if init == "random":
                mean = X.mean(axis=0)
                half_range = np.abs(X - mean).max(axis=0)
                V = 2.0 * half_range * (rng.rand(c, n) - 0.5) + mean
            elif init == "kmeans++":
                from sklearn.cluster import kmeans_plusplus

                V, _ = kmeans_plusplus(X, n_clusters=c, random_state=rng)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
            return self._init_memberships(sq_euclidean(X, V))
        init = np.asarray(init, dtype=float)
        if init.shape == (c, n):
            return self._init_memberships(sq_euclidean(X, init))
        if init.shape == (X.shape[0], c):
            return init / init.sum(axis=1, keepdims=True)
        raise ValueError(
            "init array must have shape (n_clusters, n_features) or (n_samples, n_clusters), "
            f"got {init.shape}."
        )

    def _init_memberships(self, d2: np.ndarray) -> np.ndarray:
        """Initial partition from Euclidean squared distances to initial prototypes."""
        if self.m > 1.0:
            return memberships_from_distances(d2, self.m)
        return self._memberships(d2)

    def _check_params(self, X: np.ndarray) -> None:
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= n_samples={X.shape[0]}."
            )
        if self.m < 1.0:
            raise ValueError("m must be >= 1 (m > 1 for fuzzy partitions).")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Compute the clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. :class:`pandas.DataFrame` input is accepted and its
            column names are stored in ``feature_names_in_``.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        self._check_params(X)
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        rng = check_random_state(self.random_state)
        n_init = 1 if not isinstance(self.init, str) else self.n_init

        best = None
        for run in range(n_init):
            result = self._fit_single(X, rng)
            if best is None or result["history"][-1] < best["history"][-1]:
                best = result
            if self.verbose and n_init > 1:
                print(f"restart {run + 1}/{n_init}: J = {result['history'][-1]:.6g}")
        self.__dict__.update(best["state"])
        self.membership_ = best["U"]
        self.labels_ = best["U"].argmax(axis=1)
        self.distances_ = np.sqrt(best["d2"])
        self.objective_history_ = np.asarray(best["history"])
        self.objective_ = best["history"][-1]
        self.n_iter_ = best["n_iter"]
        self._finalize(X, best["U"] ** self.m)
        return self

    def _fit_single(self, X: np.ndarray, rng) -> dict:
        """One run of the alternating optimization from a fresh initialization."""
        U = self._init_state(X, rng)
        history = []
        for it in range(1, self.max_iter + 1):
            Um = U**self.m
            self._update_prototypes(X, Um)
            d2 = self._distances(X)
            U_new = self._memberships(d2)
            history.append(self._objective(Um, d2))
            if self.verbose:
                print(f"iter {it:4d}  J = {history[-1]:.6g}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol:
                break
        # snapshot of every fitted attribute the prototype update created
        state = {k: v for k, v in self.__dict__.items() if k.endswith("_") and not k.startswith("_")}
        state.update(
            {
                k: v
                for k, v in self.__dict__.items()
                if k.startswith("_") and not k.startswith("__") and k not in ("_F0",)
            }
        )
        return {"U": U, "d2": d2, "history": history, "n_iter": it, "state": state}

    def _finalize(self, X: np.ndarray, Um: np.ndarray) -> None:
        """Hook for post-processing (e.g. storing covariances) after convergence."""

    # ------------------------------------------------------------------ inference
    def predict_proba(self, X):
        """Membership degrees of (new) samples in the fitted clusters.

        Equivalent to ``clusteval`` of the MATLAB toolbox: the distances to the
        fitted prototypes (with their fitted norms) are converted into
        memberships with the algorithm's own membership formula.

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self._memberships(self._distances(X))

    def predict(self, X):
        """Hard cluster labels of (new) samples (argmax of :meth:`predict_proba`)."""
        return self.predict_proba(X).argmax(axis=1)

    def transform(self, X):
        """Distances :math:`D_{ik}` of samples to the fitted prototypes.

        Returns
        -------
        D : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return np.sqrt(self._distances(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def score(self, X, y=None):
        """Negative objective :math:`-J` on ``X`` (higher is better)."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        d2 = self._distances(X)
        U = self._memberships(d2)
        return -self._objective(U**self.m, d2)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class _EuclideanPrototypeMixin:
    """Prototype update = fuzzy mean, distance = Euclidean (used by FCM and k-means)."""

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)


__all__ = ["BaseFuzzyClusterer", "EPS"]
