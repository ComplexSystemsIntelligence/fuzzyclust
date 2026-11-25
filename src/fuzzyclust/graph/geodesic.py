# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Geodesic-distance-based fuzzy clustering (Feil and Abonyi).

Two algorithms of the paper are implemented:

* :class:`GeodesicFCM` -- *Algorithm II*: fuzzy c-medoid clustering in the
  original feature space with geodesic (shortest-path) distances;
* :class:`IsomapFCM` -- *Algorithm I*: fuzzy c-means on the Isomap projection
  of the data, the centers being re-projected onto the closest samples.

Both rest on the same graph utilities, :func:`neighbors_graph` and
:func:`geodesic_distances`.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import validate_data

from .._utils import EPS, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer
from ..cluster import FCM

__all__ = ["GeodesicFCM", "IsomapFCM", "geodesic_distances", "isomap_embedding", "neighbors_graph"]


# --------------------------------------------------------------------------- graph helpers
def neighbors_graph(X, n_neighbors: int | None = 10, radius: float | None = None):
    r"""Symmetric :math:`k`-nearest-neighbor (or :math:`\varepsilon`-radius) graph with Euclidean weights.

    Two samples are linked when either one is among the :math:`k` nearest
    neighbors of the other (:math:`k`-Isomap) or when they are closer than
    ``radius`` (:math:`\varepsilon`-Isomap). A sample is never its own
    neighbor; coincident samples are linked by an edge of length ``EPS``
    so that a zero weight never means "no edge".

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_neighbors : int or None, default=10
        Number of neighbors :math:`k`. Ignored when ``radius`` is given.
    radius : float or None, default=None
        Connect all pairs closer than ``radius``.

    Returns
    -------
    G : scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Symmetric adjacency with the Euclidean distances as weights (a
        structurally missing entry means "no edge").

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.graph import neighbors_graph
    >>> X = np.c_[np.arange(5.0), np.zeros(5)]
    >>> G = neighbors_graph(X, n_neighbors=1)
    >>> G.nnz, float(G.max())
    (8, 1.0)
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if radius is not None:
        if radius <= 0:
            raise ValueError("radius must be positive.")
        nn = NearestNeighbors(radius=radius).fit(X)
        G = nn.radius_neighbors_graph(None, mode="distance")
    else:
        if n_neighbors is None or int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be a positive integer (or give radius).")
        k = min(int(n_neighbors), N - 1)
        if k < 1:
            raise ValueError("At least two samples are needed for a neighborhood graph.")
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        G = nn.kneighbors_graph(None, mode="distance")
    G = G.tocsr()
    G.data = np.maximum(G.data, EPS)  # coincident points stay connected
    G = G.maximum(G.T).tocsr()  # an edge exists if either point lists the other
    G.setdiag(0.0)
    G.eliminate_zeros()
    return G


def _connect_components(X, G, labels, penalty):
    """Join graph components through their closest pairs of points (greedy, by Euclidean distance).

    Every bridge gets the Euclidean length of the closest pair plus ``penalty``.
    Returns the connected graph and the array of bridge edges ``(n_bridges, 2)``.
    """
    G = G.tolil()
    D = np.sqrt(sq_euclidean(X, X))
    labels = labels.copy()
    bridges = []
    while True:
        ids = np.unique(labels)
        if len(ids) == 1:
            break
        best = None
        for ia_, a in enumerate(ids):
            ia = np.flatnonzero(labels == a)
            for b in ids[ia_ + 1 :]:
                ib = np.flatnonzero(labels == b)
                sub = D[np.ix_(ia, ib)]
                idx = np.unravel_index(sub.argmin(), sub.shape)
                if best is None or sub[idx] < best[0]:
                    best = (sub[idx], ia[idx[0]], ib[idx[1]], a, b)
        d, p, q, a, b = best
        G[p, q] = d + penalty
        G[q, p] = d + penalty
        bridges.append((p, q))
        labels[labels == b] = a
    return G.tocsr(), np.asarray(bridges, dtype=int).reshape(-1, 2)


def geodesic_distances(
    X,
    n_neighbors: int | None = 10,
    radius: float | None = None,
    connect_components: bool = True,
    bridge_penalty: float = 1.0,
    method: str = "auto",
):
    r"""Graph (geodesic) distances between all pairs of samples, Isomap style.

    The geodesic distance along the manifold the data lie on is
    approximated by the length of the shortest path in a neighborhood graph
    whose edges are weighted by Euclidean distances (Tenenbaum, de Silva and
    Langford, 2000); the approximation converges to the true geodesic as the
    sampling density grows (Bernstein et al., 2000).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_neighbors : int or None, default=10
    radius : float or None, default=None
        See :func:`neighbors_graph`.
    connect_components : bool, default=True
        If the graph falls apart into several components, link them through
        their closest pairs of points so that all distances are finite. With
        ``False`` the distances between components are ``inf``.
    bridge_penalty : float, default=1.0
        Length added to every bridge, as a multiple of the largest finite
        geodesic distance of the unconnected graph. With the default a path
        that crosses a bridge is always longer than any path inside a
        component, so that the components keep their identity in the
        clustering although all distances are finite; ``0`` gives the plain
        Euclidean bridge.
    method : str, default="auto"
        Shortest-path algorithm of :func:`scipy.sparse.csgraph.shortest_path`.

    Returns
    -------
    result : :class:`sklearn.utils.Bunch`
        ``distances`` -- geodesic distance matrix ``(n_samples, n_samples)``;
        ``graph`` -- the (possibly connected) neighborhood graph (CSR);
        ``n_components`` -- number of connected components of the raw graph;
        ``component_labels`` -- component of every sample in the raw graph;
        ``bridges`` -- the added bridge edges ``(n_bridges, 2)``.

    Examples
    --------
    >>> from fuzzyclust.datasets import make_rings
    >>> from fuzzyclust.graph import geodesic_distances
    >>> X, _ = make_rings(random_state=0)
    >>> geo = geodesic_distances(X, n_neighbors=6)
    >>> geo.n_components, geo.distances.shape, bool(np.isfinite(geo.distances).all())
    (2, (300, 300), True)
    """
    X = np.asarray(X, dtype=float)
    if bridge_penalty < 0:
        raise ValueError("bridge_penalty must be non-negative.")
    G = neighbors_graph(X, n_neighbors=n_neighbors, radius=radius)
    n_comp, labels = connected_components(G, directed=False)
    D = shortest_path(G, method=method, directed=False)
    bridges = np.empty((0, 2), dtype=int)
    if connect_components and n_comp > 1:
        finite = D[np.isfinite(D)]
        penalty = bridge_penalty * (finite.max() if finite.size else 0.0)
        G, bridges = _connect_components(X, G, labels, penalty)
        D = shortest_path(G, method=method, directed=False)
    return Bunch(distances=D, graph=G, n_components=int(n_comp), component_labels=labels, bridges=bridges)


def isomap_embedding(D, n_components: int = 2):
    r"""Classical multidimensional scaling of a (geodesic) distance matrix (the Isomap projection).

    The doubly centered matrix of squared distances
    :math:`\mathbf{B}=-\tfrac12\mathbf{H}\mathbf{D}^{(2)}\mathbf{H}`,
    :math:`\mathbf{H}=\mathbf{I}-\tfrac1N\mathbf{1}\mathbf{1}^T`, is
    decomposed and the coordinates are
    :math:`\mathbf{Y}=\mathbf{Q}_q\boldsymbol\Lambda_q^{1/2}` from its
    :math:`q` largest eigenpairs.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Finite, symmetric distance matrix (e.g. ``geodesic_distances(X).distances``).
    n_components : int, default=2

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
    eigenvalues : ndarray of shape (n_samples,)
        All eigenvalues of :math:`\mathbf{B}` in decreasing order; the
        residual variance of the embedding is the share of the discarded ones.
    """
    D = np.asarray(D, dtype=float)
    if not np.all(np.isfinite(D)):
        raise ValueError("isomap_embedding needs finite distances; connect the graph components first.")
    N = D.shape[0]
    D2 = D**2
    B = -0.5 * (D2 - D2.mean(axis=0) - D2.mean(axis=1)[:, None] + D2.mean())
    w, Q = np.linalg.eigh((B + B.T) / 2.0)
    order = np.argsort(w)[::-1]
    w, Q = w[order], Q[:, order]
    q = min(int(n_components), N)
    Y = Q[:, :q] * np.sqrt(np.maximum(w[:q], 0.0))
    return Y, w


# --------------------------------------------------------------------------- Algorithm II
class GeodesicFCM(BaseFuzzyClusterer):
    r"""Geodesic-distance-based fuzzy c-medoid clustering (Feil and Abonyi, Algorithm II).

    Point prototypes with the Euclidean distance assume convex, roughly
    spherical groups. When the data lie on a curved low-dimensional
    manifold (spiral arms, moons, rings), two points that are close in
    Euclidean distance may be far apart *along the manifold*. This
    algorithm measures the distance between samples and prototypes with the
    *geodesic* distance -- the shortest path in a :math:`k`-nearest-neighbor
    graph -- and restricts the prototypes to be data points (medoids), so that
    they lie on the manifold and the geodesic distance to them is defined.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    n_neighbors : int, default=10
        Number of neighbors :math:`k` of the graph. Too small a value
        disconnects the manifold, too large a value "short-circuits" it
        (edges jump between arms). The paper's spiral needs :math:`k\le4`
        to keep the two arms apart.
    radius : float or None, default=None
        Use an :math:`\varepsilon`-neighborhood graph instead of
        :math:`k` neighbors.
    connect_components : bool, default=True
        Join disconnected components through their closest points, so that
        every geodesic distance is finite (see :func:`geodesic_distances`).
        With ``False`` the samples of a component without medoid share
        their membership uniformly, and clusters never span components.
    bridge_penalty : float, default=1.0
        Extra length of the bridges, as a multiple of the largest finite
        geodesic distance; ``0`` gives plain Euclidean bridges. The penalty
        is right when the components *are* the clusters (rings, moons):
        it keeps them apart although all distances are finite. When
        ``n_neighbors`` is too small and the components are fragments of
        one manifold (``n_components_`` far above ``n_clusters``), a plain
        bridge re-joins the fragments at their natural junctions and works
        better.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=100
    init : {"kmeans++", "random"} or array-like of int, default="kmeans++"
        ``"kmeans++"`` is the D\ :sup:`2`-weighted seeding on geodesic
        distances, ``"random"`` picks :math:`c` random samples as initial
        medoids (the paper's "arbitrary" start); an integer array gives the
        initial medoid indices. The seeding is the default because a medoid
        can hardly migrate across a penalized bridge during the iterations:
        a random start that puts every medoid into one graph component
        leaves the other components without a prototype.
    n_init : int, default=1
        Random restarts; the run with the lowest :math:`J` is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    medoid_indices_ : ndarray of shape (n_clusters,)
        Row indices of the medoids in the training data.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        The medoids ``X[medoid_indices_]``.
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    geodesic_distances_ : ndarray of shape (n_samples, n_samples)
        Geodesic distance matrix of the training data (``inf`` between
        components when ``connect_components=False``).
    neighbors_graph_ : scipy.sparse.csr_matrix
        The neighborhood graph used (after connecting components).
    n_components_ : int
        Number of connected components of the raw neighborhood graph.
    component_labels_ : ndarray of shape (n_samples,)
    bridge_edges_ : ndarray of shape (n_bridges, 2)
        Edges added to connect the components.
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    The objective is the c-means functional with geodesic distances
    :math:`d_g`,

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\, d_g(\mathbf{x}_k,\mathbf{v}_i)^2,
        \qquad \mathbf{v}_i\in\{\mathbf{x}_1,\dots,\mathbf{x}_N\},

    alternating the FCM membership formula (with :math:`d_g`)

    .. math::
        \mu_{ik} = \frac{1}{\sum_{j=1}^{c}\big(d_g(\mathbf{x}_k,\mathbf{v}_i)/
        d_g(\mathbf{x}_k,\mathbf{v}_j)\big)^{2/(m-1)}}

    and the medoid update

    .. math::
        \mathbf{v}_i = \mathbf{x}_j,\qquad
        j = \argmin_{l}\sum_{k=1}^{N}\mu_{ik}^m\, d_g(\mathbf{x}_k,\mathbf{x}_l)^2 ,

    i.e. every sample is tried as a medoid, which costs :math:`O(cN^2)` per
    iteration (the paper mentions sampling schemes such as CLARA for large
    :math:`N`). The geodesic distance matrix is dense, :math:`O(N^2)`
    memory.

    **New samples.** The geodesic distance of a point outside the training
    set is not defined. :meth:`predict_proba` uses the nearest training
    sample as a proxy: the distance of a new point to a medoid is the
    Euclidean distance to its nearest training neighbor plus that
    neighbor's geodesic distance to the medoid. This is exact for training
    points and a reasonable approximation close to the manifold.

    References
    ----------
    .. [1] B. Feil, J. Abonyi, "Geodesic distance based fuzzy clustering",
           in: Soft Computing in Industrial Applications, Advances in Soft
           Computing 39, Springer, 50--59, 2007.
    .. [2] J. B. Tenenbaum, V. de Silva, J. C. Langford, "A global geometric
           framework for nonlinear dimensionality reduction", Science 290,
           2319--2323, 2000.
    .. [3] L. Kaufman, P. J. Rousseeuw, "Finding Groups in Data", Wiley, 1990.

    Examples
    --------
    >>> from fuzzyclust.graph import GeodesicFCM
    >>> from fuzzyclust.datasets import make_curved_manifold
    >>> X, y = make_curved_manifold(n_samples=300, random_state=0)
    >>> geo = GeodesicFCM(n_clusters=2, n_neighbors=5, random_state=0).fit(X)
    >>> geo.medoid_indices_.shape, geo.geodesic_distances_.shape
    ((2,), (300, 300))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_neighbors: int = 10,
        radius=None,
        connect_components: bool = True,
        bridge_penalty: float = 1.0,
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
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.connect_components = connect_components
        self.bridge_penalty = bridge_penalty

    # ------------------------------------------------------------------ graph
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("GeodesicFCM requires m > 1.")
        if self.radius is None and (
            not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer.")
        if self.radius is not None and self.radius <= 0:
            raise ValueError("radius must be positive.")
        if self.bridge_penalty < 0:
            raise ValueError("bridge_penalty must be non-negative.")
        self._prepare_graph(X)

    def _prepare_graph(self, X):
        """Geodesic distances of the training data (``X`` is the array the AO loop receives)."""
        geo = geodesic_distances(
            X,
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            connect_components=self.connect_components,
            bridge_penalty=self.bridge_penalty,
        )
        D = geo.distances
        self._X_fit = X
        self.geodesic_distances_ = D
        self.neighbors_graph_ = geo.graph
        self.n_components_ = geo.n_components
        self.component_labels_ = geo.component_labels
        self.bridge_edges_ = geo.bridges
        finite = np.isfinite(D)
        big = 10.0 * D[finite].max() if finite.any() else 1.0
        self._D2 = np.where(finite, D, big) ** 2

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        c, N = self.n_clusters, X.shape[0]
        init = self.init
        if isinstance(init, str):
            if init == "random":
                idx = rng.choice(N, size=c, replace=False)
            elif init == "kmeans++":
                idx = [int(rng.randint(N))]
                dmin = self._D2[:, idx[0]].copy()
                while len(idx) < c:
                    dmin[idx] = 0.0
                    probs = dmin / dmin.sum() if dmin.sum() > 0 else np.full(N, 1.0 / N)
                    j = int(rng.choice(N, p=probs))
                    idx.append(j)
                    dmin = np.minimum(dmin, self._D2[:, j])
                idx = np.asarray(idx)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an index array.")
        else:
            idx = np.asarray(init).ravel()
            if idx.shape != (c,) or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError("init array must hold n_clusters integer medoid indices.")
            if idx.min() < 0 or idx.max() >= N:
                raise ValueError("medoid indices out of range.")
        self.medoid_indices_ = idx
        self.cluster_centers_ = X[idx].copy()
        return self._memberships(self._D2[:, idx])

    def _update_prototypes(self, X, Um):
        # cost of every sample as medoid of every cluster: (N x c) = D2 (N x N) @ Um (N x c)
        cost = self._D2 @ Um
        idx = cost.argmin(axis=0)
        self.medoid_indices_ = idx
        self.cluster_centers_ = X[idx].copy()

    def _distances(self, X):
        if X is self._X_fit:
            return self._D2[:, self.medoid_indices_]
        # nearest training sample as graph entry point (cdist is exact for identical rows)
        d_near = cdist(X, self._X_fit)
        nearest = d_near.argmin(axis=1)
        hop = d_near[np.arange(X.shape[0]), nearest]
        geo = np.sqrt(self._D2[nearest][:, self.medoid_indices_])
        return (hop[:, None] + geo) ** 2

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)


# --------------------------------------------------------------------------- Algorithm I
class IsomapFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means on the Isomap projection (Feil and Abonyi, Algorithm I).

    The manifold exploration is left entirely to Isomap: geodesic distances
    on the neighborhood graph are embedded in :math:`q` dimensions by
    classical multidimensional scaling, fuzzy c-means is run on the
    projected data, and every center is "re-projected" into the feature
    space as the sample whose projection is closest to it. The paper's
    Algorithm II (:class:`GeodesicFCM`) avoids the projection step, which
    is the bottleneck of this approach: Isomap can only embed a connected
    graph, so disconnected components are bridged first
    (see :func:`geodesic_distances`), and a projection that fails to unfold
    the manifold spoils the clustering.

    Parameters
    ----------
    n_clusters : int, default=3
    n_components : int, default=2
        Dimension :math:`q` of the Isomap embedding.
    n_neighbors : int, default=10
    radius : float or None, default=None
    bridge_penalty : float, default=1.0
        Graph construction, see :class:`GeodesicFCM` (components are always
        connected here, because MDS needs finite distances).
    m, tol, max_iter, init, n_init, random_state, verbose
        Passed to the :class:`~fuzzyclust.cluster.FCM` run on the embedding.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Isomap coordinates :math:`\mathbf{y}_k` of the training data.
    embedded_centers_ : ndarray of shape (n_clusters, n_components)
        FCM centers in the embedding.
    medoid_indices_ : ndarray of shape (n_clusters,)
        Samples closest to the centers in the embedding.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        The re-projected centers ``X[medoid_indices_]``.
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        Of the FCM run in the embedding.
    geodesic_objective_ : float
        :math:`J` evaluated with geodesic distances to the re-projected
        centers -- the quantity the paper compares between the algorithms.
    residual_variance_ : float
        Share of the MDS eigenvalue mass not captured by the embedding.
    geodesic_distances_, neighbors_graph_, n_components_, component_labels_, bridge_edges_
        See :class:`GeodesicFCM`.

    Notes
    -----
    New samples are embedded through their nearest training sample
    (:meth:`predict_proba`, :meth:`transform`): the proxy keeps training
    points exact and is adequate close to the manifold.

    References
    ----------
    .. [1] B. Feil, J. Abonyi, "Geodesic distance based fuzzy clustering",
           in: Soft Computing in Industrial Applications, Advances in Soft
           Computing 39, Springer, 50--59, 2007.
    .. [2] J. B. Tenenbaum, V. de Silva, J. C. Langford, "A global geometric
           framework for nonlinear dimensionality reduction", Science 290,
           2319--2323, 2000.

    Examples
    --------
    >>> from fuzzyclust.graph import IsomapFCM
    >>> from fuzzyclust.datasets import make_curved_manifold
    >>> X, y = make_curved_manifold(n_samples=300, random_state=0)
    >>> iso = IsomapFCM(n_clusters=2, n_neighbors=5, random_state=0).fit(X)
    >>> iso.embedding_.shape, iso.cluster_centers_.shape
    ((300, 2), (2, 2))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_components: int = 2,
        n_neighbors: int = 10,
        radius=None,
        bridge_penalty: float = 1.0,
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
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.bridge_penalty = bridge_penalty

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("IsomapFCM requires m > 1.")
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if self.radius is None and (
            not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer.")
        if self.radius is not None and self.radius <= 0:
            raise ValueError("radius must be positive.")

    def fit(self, X, y=None):
        """Embed the data by Isomap, cluster the embedding with FCM, re-project the centers.

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
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        rng = check_random_state(self.random_state)
        geo = geodesic_distances(
            X, n_neighbors=self.n_neighbors, radius=self.radius, connect_components=True,
            bridge_penalty=self.bridge_penalty,
        )
        self._X_fit = X
        self.geodesic_distances_ = geo.distances
        self.neighbors_graph_ = geo.graph
        self.n_components_ = geo.n_components
        self.component_labels_ = geo.component_labels
        self.bridge_edges_ = geo.bridges
        Y, w = isomap_embedding(geo.distances, self.n_components)
        self.embedding_ = Y
        pos = np.maximum(w, 0.0)
        self.residual_variance_ = float(1.0 - pos[: Y.shape[1]].sum() / max(pos.sum(), EPS))

        fcm = FCM(
            n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
            init=self.init, n_init=self.n_init, random_state=rng, verbose=self.verbose,
        ).fit(Y)
        self.fcm_ = fcm
        self.embedded_centers_ = fcm.cluster_centers_
        self.medoid_indices_ = sq_euclidean(Y, fcm.cluster_centers_).argmin(axis=0)
        self.cluster_centers_ = X[self.medoid_indices_].copy()
        self.membership_ = fcm.membership_
        self.labels_ = fcm.labels_
        self.distances_ = fcm.distances_
        self.objective_history_ = fcm.objective_history_
        self.objective_ = fcm.objective_
        self.n_iter_ = fcm.n_iter_
        dg2 = geo.distances[:, self.medoid_indices_] ** 2
        self.geodesic_objective_ = float(np.sum(self.membership_**self.m * dg2))
        return self

    # ------------------------------------------------------------------ base API hooks
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _embed(self, X):
        if X is self._X_fit:
            return self.embedding_
        nearest = cdist(X, self._X_fit).argmin(axis=1)
        return self.embedding_[nearest]

    def _distances(self, X):
        return sq_euclidean(self._embed(X), self.embedded_centers_)

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)
