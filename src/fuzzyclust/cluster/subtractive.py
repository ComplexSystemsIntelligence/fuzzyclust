# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Density-based estimation of the number and location of clusters: mountain method, subtractive clustering.

Both methods compute a density-like *potential* at candidate points, select
the point of highest potential as a cluster center, subtract the influence
of that center from the potential of its neighborhood and repeat until the
remaining potential is small. They need no number of clusters and no
initialization; their result---a set of centers and their number---is the
natural starting point for the alternating-optimization algorithms and for
the rule-number question of fuzzy model identification.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, sq_euclidean


def _scaled_sq_distances(A: np.ndarray, B: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Squared distances ``(len(A), len(B))`` with every feature divided by its radius."""
    return sq_euclidean(A / radii, B / radii)


class _DensityPeaks(ClusterMixin, TransformerMixin, BaseEstimator):
    """Shared inference methods of the two potential-based estimators."""

    def _check_radii(self, n):
        radii = np.asarray(self.radii, dtype=float)
        if radii.ndim == 0:
            radii = np.full(n, float(radii))
        if radii.shape != (n,) or np.any(radii <= 0):
            raise ValueError(f"radii must be a positive scalar or an array of {n} positive values.")
        if self.squash <= 0:
            raise ValueError("squash must be positive.")
        return radii

    def _normalize(self, X):
        """Min--max scale to the unit hypercube (fitted bounds); constant features are left at zero."""
        span = np.where(self._span > 0, self._span, 1.0)
        return (X - self._lo) / span

    def _d2_to_centers(self, X):
        """Radius-scaled squared distances of (raw) samples to the centers."""
        return _scaled_sq_distances(self._normalize(X), self._centers_unit, self.radii_)

    def predict_proba(self, X):
        r"""Gaussian membership :math:`\exp(-4D^2_{ik}/r_a^2)` in every center, normalized per sample.

        Returns
        -------
        U : ndarray of shape (n_samples, ``n_clusters_``)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        U = np.exp(-4.0 * self._d2_to_centers(X))
        return U / np.maximum(U.sum(axis=1, keepdims=True), EPS)

    def predict(self, X):
        """Index of the nearest center (in the radius-scaled metric) of (new) samples."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self._d2_to_centers(X).argmin(axis=1)

    def transform(self, X):
        """Euclidean distances to the centers (original units), shape ``(n_samples, n_clusters_)``."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return np.sqrt(sq_euclidean(X, self.cluster_centers_))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters_)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class SubtractiveClustering(_DensityPeaks):
    r"""Subtractive clustering (Chiu, 1994): cluster centers from the potential of the data points.

    Every data point is a candidate center. Its *potential* is the sum of
    Gaussian influences of all other points within a neighborhood radius
    :math:`r_a`; the point of highest potential becomes the first center,
    its influence (with a somewhat larger radius :math:`r_b`) is subtracted
    from the potential of all points, and the next highest potential is
    examined. A candidate is accepted when its remaining potential is still
    a large fraction of the first one, rejected---and the procedure
    stopped---when it is a small fraction, and in between accepted only if
    it is far enough from the existing centers. The method is the
    data-point version of the mountain method (:class:`MountainClustering`),
    which it frees from the exponential cost of a grid, and it is the
    classical estimator of the number of rules of a Takagi--Sugeno model
    identified in the product space of inputs and output. The data are
    scaled to the unit hypercube internally, so the radii are fractions of
    the data range.

    Parameters
    ----------
    radii : float or array-like of shape (n_features,), default=0.5
        Neighborhood radius :math:`r_a` of a center as a fraction of the
        range of every feature (one value per feature allowed). Smaller
        radii give more clusters. Chiu recommends :math:`0.2`--:math:`0.5`
        for his examples; :math:`0.5` is the usual default.
    squash : float, default=1.25
        Squash factor :math:`\eta`: the potentials are revised with the radius
        :math:`r_b=\eta\,r_a>r_a` so that no center is placed close to an
        existing one.
    accept_ratio : float, default=0.5
        Acceptance ratio :math:`\bar\varepsilon`: a candidate with potential
        above :math:`\bar\varepsilon P^*_1` is accepted outright.
    reject_ratio : float, default=0.15
        Rejection ratio :math:`\underline\varepsilon`: a candidate with
        potential below :math:`\underline\varepsilon P^*_1` is rejected and
        the search ends.
    max_clusters : int or None, default=None
        Upper bound on the number of centers (``None``: no bound).

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (``n_clusters_``, n_features)
        The selected data points, in the original units.
    center_indices_ : ndarray of shape (``n_clusters_``,)
        Row indices of the centers in the training data.
    n_clusters_ : int
        Number of centers found.
    potentials_ : ndarray of shape (``n_clusters_``,)
        Potential :math:`P^*_i` of every center at the moment of its selection,
        relative to the first (:math:`P^*_1=1`).
    initial_potentials_ : ndarray of shape (n_samples,)
        Potential of every training sample before any subtraction (a density estimate).
    membership_ : ndarray of shape (n_samples, ``n_clusters_``)
        Normalized Gaussian memberships of the training samples (``predict_proba``).
    labels_ : ndarray of shape (n_samples,)
        Nearest center of every training sample.
    radii_ : ndarray of shape (n_features,)
        Radii actually used (in the unit hypercube).
    n_features_in_, feature_names_in_

    Notes
    -----
    With the data scaled to :math:`[0,1]^n` and :math:`\alpha=4/r_a^2`,
    :math:`\beta=4/r_b^2`, the potential of sample :math:`k` is

    .. math::
        P_k=\sum_{l=1}^{N}\exp\bigl(-\alpha\,\|\mathbf{x}_k-\mathbf{x}_l\|^2\bigr),

    (with a per-feature radius the squared distance is
    :math:`\sum_j(x_{kj}-x_{lj})^2/r_j^2` and :math:`\alpha=4`). After the
    selection of the center :math:`\mathbf{x}^*_i` with potential
    :math:`P^*_i`, the potentials are revised,

    .. math::
        P_k\leftarrow P_k-P^*_i\exp\bigl(-\beta\,\|\mathbf{x}_k-\mathbf{x}^*_i\|^2\bigr),

    and the candidate with the largest remaining potential is accepted if
    :math:`P^*>\bar\varepsilon P^*_1`, rejected (end) if
    :math:`P^*<\underline\varepsilon P^*_1`, and otherwise accepted only if
    :math:`d_{\min}/r_a+P^*/P^*_1\ge1`, where :math:`d_{\min}` is its
    distance to the nearest existing center; a candidate that fails this
    test has its potential set to zero and the next one is examined. The
    returned centers can be passed as ``init`` to :class:`~fuzzyclust.cluster.FCM`
    or :class:`~fuzzyclust.cluster.GK` together with ``n_clusters_``, and
    ``n_clusters_`` of a run on the product-space data ``[X, y]`` is a
    rule-number estimate for :class:`~fuzzyclust.models.TakagiSugeno`.

    References
    ----------
    .. [1] S. L. Chiu, "Fuzzy model identification based on cluster
           estimation", Journal of Intelligent and Fuzzy Systems 2(3),
           267--278, 1994.
    .. [2] R. R. Yager, D. P. Filev, "Approximate clustering via the mountain
           method", IEEE Transactions on Systems, Man, and Cybernetics 24(8),
           1279--1284, 1994.

    Examples
    --------
    >>> from fuzzyclust.cluster import SubtractiveClustering, FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(n_clusters=4, random_state=0)
    >>> sub = SubtractiveClustering(radii=0.3).fit(X)
    >>> int(sub.n_clusters_)
    4
    >>> fcm = FCM(n_clusters=sub.n_clusters_, init=sub.cluster_centers_).fit(X)
    >>> fcm.n_iter_ < 20
    True
    """

    def __init__(
        self,
        radii=0.5,
        squash: float = 1.25,
        accept_ratio: float = 0.5,
        reject_ratio: float = 0.15,
        max_clusters=None,
    ):
        self.radii = radii
        self.squash = squash
        self.accept_ratio = accept_ratio
        self.reject_ratio = reject_ratio
        self.max_clusters = max_clusters

    def fit(self, X, y=None):
        """Estimate the cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        N, n = X.shape
        radii = self._check_radii(n)
        if not 0.0 <= self.reject_ratio <= self.accept_ratio <= 1.0:
            raise ValueError("Need 0 <= reject_ratio <= accept_ratio <= 1.")
        if self.max_clusters is not None and self.max_clusters < 1:
            raise ValueError("max_clusters must be >= 1 or None.")
        self._lo = X.min(axis=0)
        self._span = X.max(axis=0) - self._lo
        Z = self._normalize(X)
        self.radii_ = radii
        d2 = _scaled_sq_distances(Z, Z, radii)  # (N, N), in units of r_a^2
        P = np.exp(-4.0 * d2).sum(axis=1)
        self.initial_potentials_ = P.copy()
        beta_over_alpha = 1.0 / self.squash**2  # exp(-4 d2 / squash^2) = exp(-beta ||.||^2)
        centers, pots = [], []
        max_c = N if self.max_clusters is None else min(self.max_clusters, N)
        P_first = None
        while len(centers) < max_c:
            k = int(P.argmax())
            Pk = P[k]
            if P_first is None:
                P_first = Pk
                accept = Pk > 0
            elif Pk > self.accept_ratio * P_first:
                accept = True
            elif Pk < self.reject_ratio * P_first:
                break
            else:
                d_min = np.sqrt(d2[k, centers].min())  # in units of r_a
                if d_min + Pk / P_first >= 1.0:
                    accept = True
                else:
                    P[k] = 0.0  # not a center; examine the next candidate
                    continue
            if not accept:
                break
            centers.append(k)
            pots.append(Pk / P_first)
            P = P - Pk * np.exp(-4.0 * beta_over_alpha * d2[:, k])
            np.maximum(P, 0.0, out=P)
            if P.max() <= 0:
                break
        self.center_indices_ = np.asarray(centers, dtype=int)
        self._centers_unit = Z[self.center_indices_]
        self.cluster_centers_ = X[self.center_indices_].copy()
        self.n_clusters_ = len(centers)
        self.potentials_ = np.asarray(pots)
        self.membership_ = self.predict_proba(X)
        self.labels_ = self.predict(X)
        return self


class MountainClustering(_DensityPeaks):
    r"""Mountain method (Yager & Filev, 1994): cluster centers from a mountain function on a grid.

    A grid is laid over the data space and at every grid node the *mountain
    function*---the sum of Gaussian influences of all data points---is
    evaluated. The node of the highest mountain is the first center; its
    mountain is subtracted from all nodes (the *destruction* of the
    mountain) and the next peak is selected, until the remaining peak is a
    small fraction of the first. Subtractive clustering (:class:`SubtractiveClustering`)
    replaces the grid by the data points themselves; the mountain method is
    kept for its didactic value and for low-dimensional problems, since the
    number of nodes grows as ``grid_size ** n_features``.

    Parameters
    ----------
    grid_size : int or None, default=None
        Number of grid nodes per feature (the grid spans the data range).
        ``None`` chooses the largest value whose total number of nodes
        ``grid_size ** n_features`` does not exceed ``max_nodes`` (at least
        two per feature).
    max_nodes : int, default=10000
        Upper bound on the total number of grid nodes; an explicit
        ``grid_size`` that exceeds it raises an error. The bound makes the
        exponential cost of the grid explicit.
    radii : float or array-like of shape (n_features,), default=0.5
        Width of the Gaussian influence as a fraction of the feature range;
        :math:`\alpha=4/r_a^2` in the unit hypercube.
    squash : float, default=1.25
        Ratio :math:`r_b/r_a` of the destruction radius to the influence
        radius, :math:`\beta=4/r_b^2`.
    stop_ratio : float, default=0.15
        The procedure stops when the highest remaining mountain falls below
        this fraction of the first peak :math:`M^*_1`.
    max_clusters : int or None, default=None
        Upper bound on the number of centers.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (``n_clusters_``, n_features)
        The selected grid nodes, in the original units.
    n_clusters_ : int
    grid_ : ndarray of shape (grid_size_ ** n_features, n_features)
        The grid nodes (original units).
    grid_size_ : int
        Number of nodes per feature actually used.
    mountain_ : ndarray of shape (grid_size_ ** n_features,)
        Mountain function on the grid before any destruction.
    potentials_ : ndarray of shape (``n_clusters_``,)
        Peak heights :math:`M^*_i/M^*_1` at selection.
    membership_, labels_, radii_
        As in :class:`SubtractiveClustering`.

    Notes
    -----
    With the data and the grid nodes :math:`\mathbf{N}_i` scaled to the unit
    hypercube,

    .. math::
        M(\mathbf{N}_i)=\sum_{k=1}^{N}\exp\bigl(-\alpha\|\mathbf{N}_i-\mathbf{x}_k\|^2\bigr),\qquad
        M(\mathbf{N}_i)\leftarrow
         M(\mathbf{N}_i)-M^*_j\exp\bigl(-\beta\|\mathbf{N}_i-\mathbf{N}^*_j\|^2\bigr),

    where Yager and Filev's constants :math:`\alpha` and :math:`\beta` are
    expressed through the radii as :math:`\alpha=4/r_a^2`, :math:`\beta=4/r_b^2`.

    References
    ----------
    .. [1] R. R. Yager, D. P. Filev, "Approximate clustering via the mountain
           method", IEEE Transactions on Systems, Man, and Cybernetics 24(8),
           1279--1284, 1994.

    Examples
    --------
    >>> from fuzzyclust.cluster import MountainClustering
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, random_state=0)
    >>> mc = MountainClustering(grid_size=15, radii=0.3).fit(X)
    >>> int(mc.n_clusters_), mc.grid_.shape
    (3, (225, 2))
    >>> MountainClustering(radii=0.3).fit(X).grid_size_      # automatic: 100 x 100 nodes
    100
    """

    def __init__(
        self,
        grid_size=None,
        max_nodes: int = 10000,
        radii=0.5,
        squash: float = 1.25,
        stop_ratio: float = 0.15,
        max_clusters=None,
    ):
        self.grid_size = grid_size
        self.max_nodes = max_nodes
        self.radii = radii
        self.squash = squash
        self.stop_ratio = stop_ratio
        self.max_clusters = max_clusters

    def fit(self, X, y=None):
        """Estimate the cluster centers on the grid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        N, n = X.shape
        radii = self._check_radii(n)
        if not isinstance(self.max_nodes, (int, np.integer)) or self.max_nodes < 2:
            raise ValueError("max_nodes must be an integer >= 2.")
        if self.grid_size is None:
            grid_size = max(2, int(np.floor(self.max_nodes ** (1.0 / n) + 1e-9)))
        else:
            if not isinstance(self.grid_size, (int, np.integer)) or self.grid_size < 2:
                raise ValueError("grid_size must be an integer >= 2 or None.")
            grid_size = int(self.grid_size)
            if grid_size**n > self.max_nodes:
                raise ValueError(
                    f"grid_size={grid_size} gives {grid_size}**{n} nodes > max_nodes={self.max_nodes}; "
                    "use subtractive clustering for higher dimensions or raise max_nodes."
                )
        self.grid_size_ = grid_size
        if not 0.0 <= self.stop_ratio <= 1.0:
            raise ValueError("stop_ratio must lie in [0, 1].")
        if self.max_clusters is not None and self.max_clusters < 1:
            raise ValueError("max_clusters must be >= 1 or None.")
        self._lo = X.min(axis=0)
        self._span = X.max(axis=0) - self._lo
        Z = self._normalize(X)
        self.radii_ = radii
        axis = np.linspace(0.0, 1.0, grid_size)
        G = np.asarray(list(itertools.product(axis, repeat=n)), dtype=float)
        d2 = _scaled_sq_distances(G, Z, radii)  # (grid, N)
        M = np.exp(-4.0 * d2).sum(axis=1)
        self.mountain_ = M.copy()
        beta_over_alpha = 1.0 / self.squash**2
        centers, pots = [], []
        max_c = G.shape[0] if self.max_clusters is None else min(self.max_clusters, G.shape[0])
        M_first = None
        while len(centers) < max_c:
            i = int(M.argmax())
            Mi = M[i]
            if M_first is None:
                if Mi <= 0:
                    break
                M_first = Mi
            elif Mi < self.stop_ratio * M_first or Mi <= 0:
                break
            centers.append(i)
            pots.append(Mi / M_first)
            # distances to the new center only: a full grid-by-grid matrix would need O(nodes^2) memory
            d2_center = _scaled_sq_distances(G, G[i : i + 1], radii)[:, 0]
            M = M - Mi * np.exp(-4.0 * beta_over_alpha * d2_center)
            np.maximum(M, 0.0, out=M)
        span = np.where(self._span > 0, self._span, 1.0)
        self.grid_ = G * span + self._lo
        self._centers_unit = G[centers]
        self.cluster_centers_ = self.grid_[centers].copy()
        self.n_clusters_ = len(centers)
        self.potentials_ = np.asarray(pots)
        self.membership_ = self.predict_proba(X)
        self.labels_ = self.predict(X)
        return self


__all__ = ["SubtractiveClustering", "MountainClustering"]
