# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Hybrid clustering with a minimal spanning tree and Gath--Geva (Vathy-Fogarassy, Feil, and Abonyi)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.csgraph import minimum_spanning_tree as _csgraph_mst
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from .._utils import EPS, labels_to_membership, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer
from ..cluster import FCM, GG

__all__ = ["MSTClustering", "fuzzy_cluster_similarity", "minimum_spanning_tree"]


# --------------------------------------------------------------------------- tree helpers
def minimum_spanning_tree(X, backend: str = "scipy"):
    r"""Euclidean minimal spanning tree of the samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    backend : {"scipy", "networkx"}, default="scipy"
        ``"scipy"`` uses :func:`scipy.sparse.csgraph.minimum_spanning_tree`
        (Kruskal) on the dense distance matrix; ``"networkx"`` uses
        :func:`networkx.minimum_spanning_tree` and requires the optional
        dependency ``fuzzyclust[graph]``. Both return the same tree up to
        ties.

    Returns
    -------
    edges : ndarray of shape (n_samples - 1, 2)
        End points (row indices into ``X``) of the tree edges.
    weights : ndarray of shape (n_samples - 1,)
        Euclidean lengths of the edges.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if N < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)
    D = squareform(pdist(X))
    if backend == "scipy":
        # csgraph treats explicit zeros as missing edges: shift by EPS so duplicates stay connected
        T = _csgraph_mst(csr_matrix(D + EPS - np.diag(np.full(N, EPS)))).tocoo()
        edges = np.c_[T.row, T.col]
        weights = T.data - EPS
    elif backend == "networkx":
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "backend='networkx' requires networkx; install it with "
                "`pip install networkx` or `pip install fuzzyclust[graph]`."
            ) from exc
        G = nx.Graph()
        iu, ju = np.triu_indices(N, 1)
        G.add_weighted_edges_from(zip(iu.tolist(), ju.tolist(), D[iu, ju].tolist(), strict=True))
        T = nx.minimum_spanning_tree(G)
        edges = np.asarray(T.edges(), dtype=int).reshape(-1, 2)
        weights = D[edges[:, 0], edges[:, 1]]
    else:
        raise ValueError("backend must be 'scipy' or 'networkx'.")
    order = np.argsort(weights, kind="stable")
    return edges[order], np.maximum(weights[order], 0.0)


def _forest_components(N, edges, keep):
    """Connected components of the forest formed by the edges where ``keep`` is True."""
    e = edges[keep]
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    return connected_components(A, directed=False)[1]


def fuzzy_cluster_similarity(U):
    r"""Pairwise similarity of fuzzy clusters seen as fuzzy sets over the samples.

    .. math::
        S_{ij} = \frac{\sum_{k=1}^{N}\min(\mu_{ik},\mu_{jk})}{\sum_{k=1}^{N}\max(\mu_{ik},\mu_{jk})}
        = \frac{|A_i\cap A_j|}{|A_i\cup A_j|},

    the Jaccard-type ratio of the fuzzy intersection and union (Vathy-Fogarassy,
    Feil and Abonyi, 2005, eq. 11). Two clusters that share many samples with
    comparable memberships are similar and are candidates for merging; the
    matrix can be fed to a hierarchical clustering of the clusters
    (:func:`~fuzzyclust.graph.plot_similarity_dendrogram`).

    Parameters
    ----------
    U : array-like of shape (n_samples, n_clusters)
        Fuzzy partition (``membership_`` of a fitted estimator).

    Returns
    -------
    S : ndarray of shape (n_clusters, n_clusters)
        Symmetric, ones on the diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.graph import fuzzy_cluster_similarity
    >>> U = np.array([[0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [0.0, 0.0, 1.0]])
    >>> np.round(fuzzy_cluster_similarity(U), 3)
    array([[1.   , 0.333, 0.   ],
           [0.333, 1.   , 0.   ],
           [0.   , 0.   , 1.   ]])
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("U must be 2-D (n_samples, n_clusters).")
    inter = np.minimum(U[:, :, None], U[:, None, :]).sum(axis=0)
    union = np.maximum(U[:, :, None], U[:, None, :]).sum(axis=0)
    return inter / np.maximum(union, EPS)


class _SplitStats:
    """Subtree statistics of a rooted tree for evaluating all binary splits in O(N n^2)."""

    def __init__(self, X, edges, root):
        N = X.shape[0]
        A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()
        order, pred = breadth_first_order(A, root, directed=False, return_predecessors=True)
        self.order, self.pred = order, pred
        n = X.shape[1]
        cnt = np.ones(N)
        s1 = X.copy()
        s2 = np.einsum("ki,kj->kij", X, X)
        for node in order[::-1]:
            p = pred[node]
            if p >= 0:
                cnt[p] += cnt[node]
                s1[p] += s1[node]
                s2[p] += s2[node]
        self.cnt, self.s1, self.s2 = cnt, s1, s2
        self.total = (cnt[root], s1[root].copy(), s2[root].copy())
        self.n = n

    def split(self, child):
        """Statistics (count, sum, sum of outer products) of the two parts when the edge
        ``pred[child] -> child`` is removed."""
        a = (self.cnt[child], self.s1[child], self.s2[child])
        b = (self.total[0] - a[0], self.total[1] - a[1], self.total[2] - a[2])
        return a, b


def _hypervolume(stats):
    """Fuzzy hypervolume det(F)^{1/2} of a crisp group from its (count, sum, outer-sum)."""
    cnt, s1, s2 = stats
    if cnt < 1:
        return 0.0
    mean = s1 / cnt
    F = s2 / cnt - np.outer(mean, mean)
    F = (F + F.T) / 2.0 + EPS * np.eye(len(mean))
    sign, logdet = np.linalg.slogdet(F)
    return float(np.exp(0.5 * logdet)) if sign > 0 else 0.0


def _fukuyama_sugeno(stats, global_mean):
    """Fukuyama-Sugeno term of a crisp group: sum ||x - v||^2 - N ||v - vbar||^2."""
    cnt, s1, s2 = stats
    if cnt < 1:
        return 0.0
    mean = s1 / cnt
    within = float(np.trace(s2) - cnt * mean @ mean)
    return within - cnt * float((mean - global_mean) @ (mean - global_mean))


# --------------------------------------------------------------------------- estimator
class MSTClustering(BaseFuzzyClusterer):
    r"""Hybrid clustering with a minimal spanning tree and Gath--Geva models.

    A divisive graph-theoretic clusterer that combines the *minimal spanning
    tree* (MST) of the data with a *fuzzy hypervolume* splitting criterion
    and a final Gath--Geva (GG) refinement, following Vathy-Fogarassy, Feil
    and Abonyi (2005). The procedure has two phases (plus the refinement):

    1. **Classical MST cutting** (Step 0 of the paper). Edges are declared
       inconsistent and removed when they are longer than ``threshold``
       times the mean edge length of the tree (Zahn's global criterion,
       *Criterion 1*) or longer than ``inconsistency`` times the mean length
       -- or more than ``inconsistency_std`` standard deviations above it --
       of the edges within ``n_hops`` tree steps of their end points (Zahn's
       local criterion, *Criterion 2*). The connected components of the
       remaining forest are the coarse clusters: well separated groups and
       groups of clearly different density are found here, but a *chain* of
       points between two groups keeps them together, and a curved group
       stays one cluster.
    2. **Volume-based binary splitting** (Steps 1--3). The sub-cluster with
       the largest fuzzy hypervolume :math:`V_i=\sqrt{\det\mathbf{F}_i}`
       is split by cutting *every* edge of its sub-tree in turn; each cut
       yields a candidate partition whose total hypervolume
       :math:`FHV=\sum_j V_j` is evaluated, and the cut with the smallest
       :math:`FHV` is kept if it lowers the volume of the split cluster by
       at least the relative gain ``min_gain`` (*Criterion 3*). This
       repeats depth-first until no cluster can be improved (or until
       ``n_clusters`` / ``max_clusters`` is reached). Elongated Gaussian groups are **not**
       split (halving them *increases* the total volume), whereas chains and
       curved groups are, so that every final cluster is well approximated by
       one ellipsoid.
    3. **Refinement** (Step 4). The crisp tree partition initializes a
       fuzzy clusterer (``refine="gg"`` by default) with as many clusters as
       there are tree clusters. GG is very sensitive to its start and fails
       from an FCM start on such data; the tree partition is an excellent
       start and GG only polishes it, delivering memberships, covariance
       matrices and a parametric mixture-of-Gaussians description.

    Parameters
    ----------
    n_clusters : int or None, default=None
        Number of clusters. ``None`` lets the volume criterion decide. If
        given, the coarse phase may be undone edge by edge (shortest cut
        first) when it produces too many clusters, and the splitting phase
        continues (best available cut) until the requested number is
        reached.
    threshold : float or None, default=None
        Criterion 1: edges longer than ``threshold`` times the mean MST edge
        length are cut. ``None`` disables the criterion.
    inconsistency : float or None, default=2.5
        Criterion 2, ratio form: an edge is cut when it is at least
        ``inconsistency`` times longer than the mean of the other edges
        within ``n_hops`` steps of its end points (the factor :math:`f` of
        Zahn and of the paper). ``None`` disables this form.
    inconsistency_std : float or None, default=None
        Criterion 2, deviation form (Zahn): an edge is cut when it exceeds
        the mean of the nearby edges by more than ``inconsistency_std``
        standard deviations of their lengths. Either form flags an edge;
        ``None`` disables this form.
    n_hops : int, default=2
        Neighborhood depth (in tree steps) of Criterion 2. A deeper
        neighborhood averages over more edges and tolerates sparse spots
        inside a cluster; a shallow one reacts to every local gap.
    criterion : {"hypervolume", "fukuyama_sugeno"}, default="hypervolume"
        Validity measure of the splitting phase: the total fuzzy
        hypervolume of Gath and Geva, or the (crisp) Fukuyama--Sugeno
        measure used by Varma and Simon, both evaluated on the crisp tree
        partition.
    min_gain : float, default=0.15
        Minimum *relative* decrease of the volume criterion,
        :math:`(V_i - V_a - V_b)/V_i`, for a split of cluster :math:`i` into
        :math:`a` and :math:`b` to be accepted when ``n_clusters=None``.
        Cutting a few tail points off a compact cluster always lowers the
        volume a little; the threshold separates such cosmetic cuts from
        real structure (chains, bends, merged groups).
    min_cluster_size : int, default=5
        Cuts that would create a component with fewer samples are not
        applied (in either phase). Keeps single outliers from becoming
        clusters.
    max_clusters : int, default=20
        Upper bound on the number of clusters when ``n_clusters=None``:
        coarse cuts beyond it are undone (shortest first) and the splitting
        phase stops at it.
    refine : {"gg", "fcm"} or None, default="gg"
        Fuzzy clusterer initialized with the tree partition. ``None`` keeps
        the crisp partition; memberships are then FCM-type memberships to the
        cluster means.
    m : float, default=2.0
        Fuzziness exponent of the refinement.
    gamma : float, default=0.0
        Covariance regularization of the GG refinement, see
        :class:`~fuzzyclust.cluster.GG`.
    tol : float, default=1e-4
    max_iter : int, default=100
        Termination of the refinement.
    backend : {"scipy", "networkx"}, default="scipy"
        MST backend, see :func:`minimum_spanning_tree`.
    random_state : int, RandomState instance or None, default=None
        Only used to break ties in the refinement (the tree is deterministic).
    verbose : int, default=0

    Attributes
    ----------
    n_clusters_ : int
        Number of clusters found.
    labels_ : ndarray of shape (n_samples,)
        Final labels (``argmax`` of ``membership_``).
    tree_labels_ : ndarray of shape (n_samples,)
        Crisp labels of the tree partition before refinement.
    membership_ : ndarray of shape (n_samples, ``n_clusters_``)
    cluster_centers_ : ndarray of shape (``n_clusters_``, n_features)
    covariances_ : ndarray of shape (``n_clusters_``, n_features, n_features)
        Fuzzy covariance matrices of the refined clusters (GG) or of the
        crisp tree clusters.
    mst_edges_ : ndarray of shape (n_samples - 1, 2)
        Edges of the minimal spanning tree (sorted by length).
    mst_weights_ : ndarray of shape (n_samples - 1,)
    cut_edges_ : ndarray of shape (n_cuts, 2)
        End points of the removed edges.
    cut_phases_ : ndarray of shape (n_cuts,)
        ``0`` for cuts of the threshold/inconsistency phase, ``1`` for
        volume-based splits.
    cut_quality_ : ndarray of shape (n_cuts,)
        Adequacy :math:`CP(e_{ab})=\sum_{k\in\{a,b\}}|\mu_{ik}-\mu_{jk}|` of
        every cut in the final partition, where :math:`i,j` are the clusters
        of the end points :math:`a,b` (eq. 9 of the paper): values near
        :math:`2` mean that the refinement confirmed the separation, values
        near :math:`0` that the end points were re-united.
    hypervolumes_ : ndarray of shape (``n_clusters_``,)
        Fuzzy hypervolume of every tree cluster.
    cluster_similarity_ : ndarray of shape (``n_clusters_``, ``n_clusters_``)
        Fuzzy similarity :math:`S_{ij}` of the final clusters
        (:func:`fuzzy_cluster_similarity`), the basis of the paper's
        dendrogram of clusters.
    refiner_ : fitted :class:`~fuzzyclust.cluster.GG` or :class:`~fuzzyclust.cluster.FCM`, or None
    distances_, objective_, objective_history_, n_iter_
        Of the refinement (``None``/zeros when ``refine=None``).

    Notes
    -----
    The fuzzy hypervolume of cluster :math:`j` is

    .. math::
        V_j = \sqrt{\det\mathbf{F}_j},\qquad
        \mathbf{F}_j = \frac{\sum_k\mu_{jk}^m(\mathbf{x}_k-\mathbf{v}_j)
        (\mathbf{x}_k-\mathbf{v}_j)^T}{\sum_k\mu_{jk}^m},
        \qquad FHV=\sum_{j=1}^{c}V_j ,

    evaluated here with the crisp memberships of the tree partition; the
    volume of the subtree on either side of every edge is obtained from
    subtree sums of :math:`\mathbf{x}_k` and :math:`\mathbf{x}_k\mathbf{x}_k^T`,
    so that the statistics of all :math:`N_i-1` candidate cuts of a
    sub-tree cost :math:`O(N_i n^2)`, plus two :math:`n\times n`
    determinants (one for each side, :math:`O(n^3)` each) per cut. The MST
    itself is built from the dense distance matrix, :math:`O(N^2)` memory,
    which limits the method to a few thousand samples.

    References
    ----------
    .. [1] Á. Vathy-Fogarassy, B. Feil, J. Abonyi, "Minimal spanning tree
           based fuzzy clustering", International Journal of Computer and
           Information Engineering 1(8), 2505--2510, 2007 (re-issue of the
           2005 conference paper).
    .. [2] C. T. Zahn, "Graph-theoretical methods for detecting and
           describing gestalt clusters", IEEE Trans. Computers C-20(1),
           68--86, 1971.
    .. [3] I. Gath, A. B. Geva, "Unsupervised optimal fuzzy clustering",
           IEEE Trans. Pattern Anal. Mach. Intell. 11(7), 773--781, 1989.
    .. [4] S. Varma, R. Simon, "Iterative class discovery and feature
           selection using minimal spanning trees", BMC Bioinformatics 5,
           126, 2004.

    Examples
    --------
    >>> from fuzzyclust.graph import MSTClustering
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, y = make_fuzzy_blobs(n_clusters=3, random_state=0)
    >>> mst = MSTClustering(random_state=0).fit(X)
    >>> mst.n_clusters_, mst.mst_edges_.shape
    (3, (299, 2))
    """

    def __init__(
        self,
        n_clusters=None,
        threshold=None,
        inconsistency=2.5,
        inconsistency_std=None,
        n_hops: int = 2,
        criterion: str = "hypervolume",
        min_gain: float = 0.15,
        min_cluster_size: int = 5,
        max_clusters: int = 20,
        refine="gg",
        m: float = 2.0,
        gamma: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 100,
        backend: str = "scipy",
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init="random",
            random_state=random_state,
            verbose=verbose,
        )
        self.threshold = threshold
        self.inconsistency = inconsistency
        self.inconsistency_std = inconsistency_std
        self.n_hops = n_hops
        self.criterion = criterion
        self.min_gain = min_gain
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.refine = refine
        self.gamma = gamma
        self.backend = backend

    # ------------------------------------------------------------------ validation
    def _check_params(self, X):
        if self.n_clusters is not None:
            if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
                raise ValueError("n_clusters must be a positive integer or None.")
            if self.n_clusters > X.shape[0]:
                raise ValueError(f"n_clusters={self.n_clusters} must be <= n_samples={X.shape[0]}.")
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError("threshold must be positive or None.")
        if self.inconsistency is not None and self.inconsistency <= 0:
            raise ValueError("inconsistency must be positive or None.")
        if self.inconsistency_std is not None and self.inconsistency_std <= 0:
            raise ValueError("inconsistency_std must be positive or None.")
        if self.n_hops < 1:
            raise ValueError("n_hops must be >= 1.")
        if self.criterion not in ("hypervolume", "fukuyama_sugeno"):
            raise ValueError("criterion must be 'hypervolume' or 'fukuyama_sugeno'.")
        if self.min_gain < 0:
            raise ValueError("min_gain must be non-negative.")
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1.")
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be >= 1.")
        if self.refine not in ("gg", "fcm", None):
            raise ValueError("refine must be 'gg', 'fcm' or None.")
        if self.m <= 1.0:
            raise ValueError("m must be > 1.")

    # ------------------------------------------------------------------ phase 0
    def _inconsistent_edges(self, N, edges, weights):
        """Boolean mask of the edges flagged by Criterion 1 and/or Criterion 2."""
        flagged = np.zeros(len(edges), dtype=bool)
        if self.threshold is not None and len(weights):
            flagged |= weights > self.threshold * weights.mean()
        local = self.inconsistency is not None or self.inconsistency_std is not None
        if local and len(weights):
            for e, nearby in enumerate(self._nearby_edges(N, edges)):
                if not nearby:
                    continue
                w = weights[nearby]
                mean = w.mean()
                if self.inconsistency is not None and weights[e] >= self.inconsistency * mean:
                    flagged[e] = True
                if self.inconsistency_std is not None and len(w) > 1:
                    if weights[e] > mean + self.inconsistency_std * w.std():
                        flagged[e] = True
        return flagged

    def _nearby_edges(self, N, edges):
        """For every tree edge, the ids of the edges within ``n_hops`` steps of its end points."""
        adj = [[] for _ in range(N)]
        for e, (a, b) in enumerate(edges):
            adj[a].append((b, e))
            adj[b].append((a, e))
        result = []
        for e, (a, b) in enumerate(edges):
            nearby = set()
            for start in (a, b):
                frontier = [start]
                seen = {start}
                for _ in range(self.n_hops):
                    nxt = []
                    for u in frontier:
                        for v, eid in adj[u]:
                            if eid != e:
                                nearby.add(eid)
                            if v not in seen:
                                seen.add(v)
                                nxt.append(v)
                    frontier = nxt
            result.append(sorted(nearby))
        return result

    def _apply_cuts(self, N, edges, keep, candidates, min_size):
        """Cut candidate edges (longest first) when both resulting parts are large enough."""
        cut = []
        for e in sorted(candidates, key=lambda i: -self.mst_weights_[i]):
            trial = keep.copy()
            trial[e] = False
            comp = _forest_components(N, edges, trial)
            a, b = comp[edges[e, 0]], comp[edges[e, 1]]
            if np.sum(comp == a) >= min_size and np.sum(comp == b) >= min_size:
                keep = trial
                cut.append(e)
        return keep, cut

    # ------------------------------------------------------------------ phase 1
    def _volume(self, stats, global_mean):
        if self.criterion == "hypervolume":
            return _hypervolume(stats)
        return _fukuyama_sugeno(stats, global_mean)

    def _best_split(self, X, edges, keep, members, min_size, global_mean):
        """Best binary split of one tree component; returns (gain, edge id) or None."""
        if len(members) < 2 * min_size:
            return None
        local = np.zeros(X.shape[0], dtype=int) - 1
        local[members] = np.arange(len(members))
        in_comp = keep & (local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)
        eids = np.flatnonzero(in_comp)
        if len(eids) == 0:
            return None
        sub_edges = local[edges[eids]]
        Xs = X[members]
        stats = _SplitStats(Xs, sub_edges, root=0)
        whole = self._volume(stats.total, global_mean)
        best = None
        for e, (a, b) in zip(eids, sub_edges, strict=True):
            child = b if stats.pred[b] == a else a
            part_a, part_b = stats.split(child)
            if part_a[0] < min_size or part_b[0] < min_size:
                continue
            val = self._volume(part_a, global_mean) + self._volume(part_b, global_mean)
            gain = whole - val
            if best is None or gain > best[0]:
                best = (gain, e, whole)
        return best

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        """Build the tree, cut it, split by volume and refine with GG.

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
        N = X.shape[0]
        target = self.n_clusters
        min_size = self.min_cluster_size
        if target is not None:
            min_size = max(1, min(min_size, N // target))

        edges, weights = minimum_spanning_tree(X, backend=self.backend)
        self.mst_edges_, self.mst_weights_ = edges, weights
        keep = np.ones(len(edges), dtype=bool)
        cuts, phases = [], []

        # ---- phase 0: threshold / inconsistency criteria
        flagged = self._inconsistent_edges(N, edges, weights)
        keep, cut0 = self._apply_cuts(N, edges, keep, np.flatnonzero(flagged), min_size)
        limit = self.max_clusters if target is None else target
        if len(cut0) + 1 > limit:
            # too many coarse clusters: undo the shortest cuts first
            for e in sorted(cut0, key=lambda i: weights[i]):
                if len(cut0) + 1 <= limit:
                    break
                keep[e] = True
                cut0.remove(e)
        cuts += cut0
        phases += [0] * len(cut0)

        # ---- phase 1: volume-based binary splitting
        global_mean = X.mean(axis=0)
        while True:
            comp = _forest_components(N, edges, keep)
            n_comp = comp.max() + 1
            if n_comp >= limit:
                break
            # candidate components in decreasing order of volume
            vols = []
            for j in range(n_comp):
                members = np.flatnonzero(comp == j)
                Xs = X[members]
                stats = (len(members), Xs.sum(axis=0), Xs.T @ Xs)
                vols.append(self._volume(stats, global_mean))
            applied = False
            best_any = None
            for j in np.argsort(vols)[::-1]:
                members = np.flatnonzero(comp == j)
                best = self._best_split(X, edges, keep, members, min_size, global_mean)
                if best is None:
                    continue
                if best_any is None or best[0] > best_any[0]:
                    best_any = best
                if best[0] > self.min_gain * max(abs(best[2]), EPS):
                    keep[best[1]] = False
                    cuts.append(best[1])
                    phases.append(1)
                    applied = True
                    break
            if not applied:
                if target is not None and best_any is not None and n_comp < target:
                    keep[best_any[1]] = False  # forced split toward the requested number
                    cuts.append(best_any[1])
                    phases.append(1)
                    continue
                break

        comp = _forest_components(N, edges, keep)
        self.tree_labels_ = comp
        self.n_clusters_ = int(comp.max() + 1)
        self.cut_edges_ = edges[cuts] if cuts else np.empty((0, 2), dtype=int)
        self.cut_phases_ = np.asarray(phases, dtype=int)
        self.hypervolumes_ = np.array(
            [
                _hypervolume((np.sum(comp == j), X[comp == j].sum(axis=0), X[comp == j].T @ X[comp == j]))
                for j in range(self.n_clusters_)
            ]
        )

        # ---- phase 2: fuzzy refinement initialized with the tree partition
        U0 = labels_to_membership(comp, self.n_clusters_)
        if self.refine is None:
            self.refiner_ = None
            self.membership_ = U0
            self.cluster_centers_ = (U0.T @ X) / U0.sum(axis=0)[:, None]
            d2 = sq_euclidean(X, self.cluster_centers_)
            self.covariances_ = self._crisp_covariances(X, comp)
            self.distances_ = np.sqrt(d2)
            self.objective_history_ = np.array([float(np.sum(U0 * d2))])
            self.objective_ = self.objective_history_[-1]
            self.n_iter_ = 0
        else:
            cls = GG if self.refine == "gg" else FCM
            kwargs = dict(gamma=self.gamma) if self.refine == "gg" else {}
            if self.n_clusters_ == 1:
                # a single cluster: nothing to refine, keep the crisp description
                self.refiner_ = None
                self.membership_ = U0
                self.cluster_centers_ = X.mean(axis=0, keepdims=True)
                self.covariances_ = self._crisp_covariances(X, comp)
                d2 = sq_euclidean(X, self.cluster_centers_)
                self.distances_ = np.sqrt(d2)
                self.objective_history_ = np.array([float(d2.sum())])
                self.objective_ = self.objective_history_[-1]
                self.n_iter_ = 0
            else:
                ref = cls(
                    n_clusters=self.n_clusters_, m=self.m, tol=self.tol, max_iter=self.max_iter,
                    init=U0, random_state=rng, verbose=self.verbose, **kwargs,
                ).fit(X)
                self.refiner_ = ref
                self.membership_ = ref.membership_
                self.cluster_centers_ = ref.cluster_centers_
                self.covariances_ = ref.covariances_
                self.distances_ = ref.distances_
                self.objective_history_ = ref.objective_history_
                self.objective_ = ref.objective_
                self.n_iter_ = ref.n_iter_
        self.labels_ = self.membership_.argmax(axis=1)
        self.cluster_similarity_ = fuzzy_cluster_similarity(self.membership_)
        self.cut_quality_ = self._cut_quality(self.membership_, self.cut_edges_)
        return self

    @staticmethod
    def _cut_quality(U, cut_edges):
        """CP(e_ab) = sum over the end points a, b of |mu_i - mu_j|, i and j being their clusters."""
        if len(cut_edges) == 0:
            return np.empty(0)
        a, b = cut_edges[:, 0], cut_edges[:, 1]
        i, j = U[a].argmax(axis=1), U[b].argmax(axis=1)
        return np.abs(U[a, i] - U[a, j]) + np.abs(U[b, i] - U[b, j])

    @staticmethod
    def _crisp_covariances(X, labels):
        c = labels.max() + 1
        n = X.shape[1]
        F = np.empty((c, n, n))
        for j in range(c):
            Xj = X[labels == j]
            F[j] = np.cov(Xj, rowvar=False) if len(Xj) > 1 else np.eye(n) * EPS
        return F

    # ------------------------------------------------------------------ hooks for the base API
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _distances(self, X):
        if self.refiner_ is not None:
            return self.refiner_._distances(X)
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        if self.refiner_ is not None:
            return self.refiner_._memberships(d2)
        return memberships_from_distances(d2, self.m)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters_)], dtype=object)
