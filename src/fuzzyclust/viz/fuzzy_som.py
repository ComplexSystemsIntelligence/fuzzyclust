# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy self-organizing map based on regularized fuzzy c-means (Abonyi, Migály & Szeifert)."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .._utils import EPS, sq_euclidean
from ..base import BaseFuzzyClusterer

__all__ = ["FuzzySOM", "grid_laplacian"]


def grid_laplacian(grid_shape, order: int = 2):
    r"""Regularization matrix :math:`\mathbf{L}=\mathbf{G}^T\mathbf{G}` of a rectangular grid.

    The prototypes :math:`\mathbf{v}_i` of a fuzzy SOM sit on the nodes of a
    regular grid in :math:`s` dimensions. Smoothness of the map is measured by
    finite differences of the prototypes along every grid axis: with
    ``order=2`` the rows of :math:`\mathbf{G}` are the second-order
    differences :math:`\mathbf{v}_{i-1}-2\mathbf{v}_i+\mathbf{v}_{i+1}` at
    every interior node (Abonyi, Migály & Szeifert), with ``order=1`` the
    first-order differences :math:`\mathbf{v}_{i+1}-\mathbf{v}_i` of every
    pair of adjacent nodes (the "smoothly distributed FCM" of Pascual-Marqui
    et al., 2001). The penalty
    :math:`\vartheta\,\mathrm{tr}(\mathbf{V}^T\mathbf{L}\mathbf{V})`
    is then :math:`\vartheta\sum\|\text{difference}\|^2`.

    Parameters
    ----------
    grid_shape : tuple of int
        Number of nodes along every grid axis, e.g. ``(5, 5)``.
    order : {1, 2}, default=2
        Order of the difference operator.

    Returns
    -------
    L : ndarray of shape (n_nodes, n_nodes)
        Symmetric positive semi-definite regularization matrix.
    positions : ndarray of shape (n_nodes, len(grid_shape))
        Integer grid coordinates of the nodes (C order, node ``i`` is
        ``positions[i]``).

    Examples
    --------
    >>> from fuzzyclust.viz import grid_laplacian
    >>> L, pos = grid_laplacian((5,), order=2)
    >>> L.astype(int)
    array([[ 1, -2,  1,  0,  0],
           [-2,  5, -4,  1,  0],
           [ 1, -4,  6, -4,  1],
           [ 0,  1, -4,  5, -2],
           [ 0,  0,  1, -2,  1]])
    """
    shape = tuple(int(s) for s in np.atleast_1d(grid_shape))
    if len(shape) == 0 or any(s < 1 for s in shape):
        raise ValueError("grid_shape must contain positive integers.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    c = int(np.prod(shape))
    index = np.arange(c).reshape(shape)
    positions = np.stack(np.unravel_index(np.arange(c), shape), axis=1)
    rows = []
    for axis, size in enumerate(shape):
        if order == 2 and size >= 3:
            prev = np.moveaxis(index, axis, 0)[:-2].ravel()
            mid = np.moveaxis(index, axis, 0)[1:-1].ravel()
            nxt = np.moveaxis(index, axis, 0)[2:].ravel()
            for p, q, r in zip(prev, mid, nxt, strict=True):
                g = np.zeros(c)
                g[p], g[q], g[r] = 1.0, -2.0, 1.0
                rows.append(g)
        elif order == 1 and size >= 2:
            a = np.moveaxis(index, axis, 0)[:-1].ravel()
            b = np.moveaxis(index, axis, 0)[1:].ravel()
            for p, q in zip(a, b, strict=True):
                g = np.zeros(c)
                g[p], g[q] = -1.0, 1.0
                rows.append(g)
    G = np.asarray(rows) if rows else np.zeros((0, c))
    return G.T @ G, positions


class FuzzySOM(BaseFuzzyClusterer):
    r"""Fuzzy self-organizing map: regularized fuzzy c-means on a grid.

    The cluster prototypes are the nodes of a low-dimensional (usually 2-D)
    grid. A smoothness penalty on the prototypes of neighboring nodes is
    added to the FCM functional, so that the fitted prototypes form an
    ordered, topology-preserving map of the data, as in Kohonen's SOM, while
    the partition stays fuzzy and the objective is minimized exactly by
    alternating optimization. The map (U-matrix, component planes, hit
    histogram) is a visualization of high-dimensional data in its own right.

    Parameters
    ----------
    grid_shape : tuple of int, default=(5, 5)
        Nodes per grid axis; the number of clusters is ``prod(grid_shape)``
        and the map dimension :math:`s` is ``len(grid_shape)`` (1, 2 or 3).
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    regularization : float, default=1.0
        Smoothness weight :math:`\vartheta\geq 0`. ``0`` recovers plain FCM
        with ``prod(grid_shape)`` clusters (unordered prototypes); larger
        values pull neighboring prototypes toward a smooth (locally linear)
        arrangement.
    difference_order : {1, 2}, default=2
        Finite-difference operator of the penalty, see :func:`grid_laplacian`.
        Second-order differences (default) penalize curvature of the map and
        leave a linear grid unpenalised; first-order differences penalize the
        length of grid edges and contract the map toward the data center.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"pca", "random", "kmeans++"} or ndarray, default="pca"
        ``"pca"`` places the grid regularly in the subspace of the first
        :math:`s` principal components (as in the paper), which already gives
        an ordered start. Other options as in
        :class:`~fuzzyclust.base.BaseFuzzyClusterer`.
    n_init : int, default=1
        Number of random restarts (only meaningful for ``init="random"`` or
        ``"kmeans++"``); the run with the lowest final objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_nodes, n_features)
        Prototypes :math:`\mathbf{v}_i` (node ``i`` has grid coordinates ``grid_positions_[i]``).
    membership_ : ndarray of shape (n_samples, n_nodes)
    labels_ : ndarray of shape (n_samples,)
        Best-matching node of every sample.
    distances_ : ndarray of shape (n_samples, n_nodes)
    grid_positions_ : ndarray of shape (n_nodes, s)
        Integer grid coordinates of the nodes.
    regularization_matrix_ : ndarray of shape (n_nodes, n_nodes)
        The matrix :math:`\mathbf{L}`.
    n_clusters_ : int
        Number of nodes.
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; the objective
        includes the penalty term.

    Notes
    -----
    The algorithm minimizes the regularized c-means functional

    .. math::
        J(\mathbf{X};\mathbf{U},\mathbf{V}) = \sum_{i=1}^{c}\sum_{k=1}^{N}
        \mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|^2
        + \vartheta\,\mathrm{tr}\!\left(\mathbf{V}^T\mathbf{L}\mathbf{V}\right),
        \qquad \mathbf{L}=\mathbf{G}^T\mathbf{G},

    where the rows of :math:`\mathbf{G}` are finite differences of the
    prototypes along the grid axes, so that
    :math:`\mathrm{tr}(\mathbf{V}^T\mathbf{L}\mathbf{V})\approx
    \sum_i\|\partial^2\mathbf{v}_i/\partial\mathbf{x}^2\|^2` approximates the
    integrated curvature of the map. The membership update is that of FCM.
    For fixed :math:`\mathbf{U}` the prototypes are coupled through
    :math:`\mathbf{L}`: setting the gradient to zero gives the linear system

    .. math::
        \mathbf{v}_i\sum_{k=1}^{N}\mu_{ik}^m + \vartheta\sum_{j=1}^{c}L_{ij}\mathbf{v}_j
        = \sum_{k=1}^{N}\mu_{ik}^m\mathbf{x}_k,\qquad i=1,\dots,c,

    i.e. :math:`(\mathrm{diag}(\textstyle\sum_k\mu_{ik}^m)+\vartheta\mathbf{L})\mathbf{V}
    = (\mathbf{U}^m)\mathbf{X}`, which is solved exactly in every iteration
    (the paper iterates its row-wise form, eq. (12), whose fixed point this is).
    With :math:`\vartheta=0` the system decouples into the FCM center formula.

    References
    ----------
    .. [1] J. Abonyi, S. Migály, F. Szeifert, "Fuzzy self-organizing map based
           on regularized fuzzy c-means clustering", in: Advances in Soft
           Computing - Engineering Design and Manufacturing, Springer, 2003.
    .. [2] T. Kohonen, "The self-organizing map", Proc. IEEE 78(9),
           1464-1480, 1990.
    .. [3] R. D. Pascual-Marqui, A. D. Pascual-Montano, K. Kochi, J. M. Carazo,
           "Smoothly distributed fuzzy c-means: a new self-organizing map",
           Pattern Recognition 34, 2395-2402, 2001.

    Examples
    --------
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.viz import FuzzySOM
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, n_features=4, random_state=0)
    >>> som = FuzzySOM(grid_shape=(4, 4), regularization=0.5, random_state=0).fit(X)
    >>> som.cluster_centers_.shape, som.u_matrix().shape
    ((16, 4), (4, 4))
    """

    def __init__(
        self,
        grid_shape=(5, 5),
        m: float = 2.0,
        regularization: float = 1.0,
        difference_order: int = 2,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="pca",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.grid_shape = grid_shape
        self.m = m
        self.regularization = regularization
        self.difference_order = difference_order
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ checks
    def _check_params(self, X):
        shape = tuple(int(s) for s in np.atleast_1d(self.grid_shape))
        if len(shape) not in (1, 2, 3) or any(s < 1 for s in shape):
            raise ValueError("grid_shape must be a tuple of 1-3 positive integers.")
        c = int(np.prod(shape))
        if c > X.shape[0]:
            raise ValueError(f"prod(grid_shape)={c} must be <= n_samples={X.shape[0]}.")
        if not self.m > 1.0:
            raise ValueError("FuzzySOM requires m > 1.")
        if self.regularization < 0:
            raise ValueError("regularization must be >= 0.")
        if self.difference_order not in (1, 2):
            raise ValueError("difference_order must be 1 or 2.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.grid_shape_ = shape
        self.n_clusters_ = c
        self.regularization_matrix_, self.grid_positions_ = grid_laplacian(shape, self.difference_order)

    # ------------------------------------------------------------------ init
    def _init_state(self, X, rng):
        c, n = self.n_clusters_, X.shape[1]
        init = self.init
        if isinstance(init, str) and init == "pca":
            return self._init_memberships(sq_euclidean(X, self._pca_grid(X, rng)))
        if isinstance(init, str):
            if init == "random":
                mean = X.mean(axis=0)
                half_range = np.abs(X - mean).max(axis=0)
                V = 2.0 * half_range * (rng.rand(c, n) - 0.5) + mean
            elif init == "kmeans++":
                from sklearn.cluster import kmeans_plusplus

                V, _ = kmeans_plusplus(X, n_clusters=c, random_state=rng)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'pca', 'random', 'kmeans++' or an array.")
            return self._init_memberships(sq_euclidean(X, V))
        init = np.asarray(init, dtype=float)
        if init.shape == (c, n):
            return self._init_memberships(sq_euclidean(X, init))
        if init.shape == (X.shape[0], c):
            return init / init.sum(axis=1, keepdims=True)
        raise ValueError(
            "init array must have shape (n_nodes, n_features) or (n_samples, n_nodes), "
            f"got {init.shape}."
        )

    def _pca_grid(self, X, rng):
        """Regular grid spanning the first ``s`` principal components (paper's initialization)."""
        shape = self.grid_shape_
        s, n = len(shape), X.shape[1]
        mean = X.mean(axis=0)
        Xc = X - mean
        q = min(s, n, X.shape[0])
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:q]
        scores = Xc @ comps.T
        lo, hi = np.percentile(scores, 2.5, axis=0), np.percentile(scores, 97.5, axis=0)
        coords = np.empty((self.n_clusters_, s))
        for axis, size in enumerate(shape):
            if axis < q:
                if size > 1:
                    ticks = np.linspace(lo[axis], hi[axis], size)
                else:
                    ticks = np.array([0.5 * (lo[axis] + hi[axis])])
                coords[:, axis] = ticks[self.grid_positions_[:, axis]]
            else:  # more grid axes than data dimensions: tiny jitter keeps prototypes distinct
                coords[:, axis] = 0.0
        V = mean + coords[:, :q] @ comps
        if s > q:
            V = V + 1e-3 * Xc.std(axis=0, keepdims=True) * rng.randn(*V.shape)
        return V

    # ------------------------------------------------------------------ AO hooks
    def _update_prototypes(self, X, Um):
        s = Um.sum(axis=0)
        A = np.diag(np.maximum(s, EPS)) + self.regularization * self.regularization_matrix_
        self.cluster_centers_ = np.linalg.solve(A, Um.T @ X)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _objective(self, Um, d2):
        V = self.cluster_centers_
        penalty = self.regularization * np.einsum("ij,ik,jk->", self.regularization_matrix_, V, V)
        return float(np.sum(Um * d2) + penalty)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_node_{i}" for i in range(self.n_clusters_)], dtype=object)

    # ------------------------------------------------------------------ map views
    def _neighbors(self):
        """List of (i, j) pairs of grid-adjacent nodes (4-neighborhood)."""
        pos = self.grid_positions_
        pairs = []
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                if np.abs(pos[i] - pos[j]).sum() == 1:
                    pairs.append((i, j))
        return pairs

    def u_matrix(self) -> np.ndarray:
        r"""Unified distance matrix: mean distance of every node's prototype to its grid neighbors.

        Large values mark "walls" between groups of similar prototypes, so
        cluster borders of the data appear as ridges on the map.

        Returns
        -------
        umat : ndarray of shape ``grid_shape``
        """
        check_is_fitted(self, "cluster_centers_")
        V = self.cluster_centers_
        acc = np.zeros(self.n_clusters_)
        cnt = np.zeros(self.n_clusters_)
        for i, j in self._neighbors():
            d = np.linalg.norm(V[i] - V[j])
            acc[i] += d
            acc[j] += d
            cnt[i] += 1
            cnt[j] += 1
        return (acc / np.maximum(cnt, 1)).reshape(self.grid_shape_)

    def component_planes(self) -> np.ndarray:
        """Prototype coordinates arranged on the grid, one plane per feature.

        Returns
        -------
        planes : ndarray of shape ``(n_features, *grid_shape)``
        """
        check_is_fitted(self, "cluster_centers_")
        return self.cluster_centers_.T.reshape((self.n_features_in_, *self.grid_shape_))

    def hits(self, X=None) -> np.ndarray:
        r"""Fuzzy hit histogram :math:`h_i=\sum_k\mu_{ik}` on the grid (training data or ``X``)."""
        check_is_fitted(self, "cluster_centers_")
        U = self.membership_ if X is None else self.predict_proba(X)
        return U.sum(axis=0).reshape(self.grid_shape_)

    def neighbor_distance(self) -> float:
        r"""Mean prototype distance over all grid edges (a scalar "roughness" of the map)."""
        check_is_fitted(self, "cluster_centers_")
        V = self.cluster_centers_
        pairs = self._neighbors()
        if not pairs:
            return 0.0
        return float(np.mean([np.linalg.norm(V[i] - V[j]) for i, j in pairs]))
