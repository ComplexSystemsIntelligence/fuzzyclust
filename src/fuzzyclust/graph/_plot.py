# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for trees, neighborhood graphs, geodesic paths and cluster dendrograms."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import issparse
from scipy.sparse.csgraph import shortest_path

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_mst", "plot_neighbors_graph", "plot_geodesic_path", "plot_similarity_dendrogram"]


def _edge_segments(X, edges):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    return np.stack([X[edges[:, 0]], X[edges[:, 1]]], axis=1)


def plot_mst(X, model, ax=None, labels=None, show_cuts=True, legend=True, **kwargs):
    """Draw the minimal spanning tree of a fitted :class:`~fuzzyclust.graph.MSTClustering`.

    Tree edges are gray; edges removed by the threshold/inconsistency phase
    are drawn in vermilion, edges removed by the volume-based splitting in
    orange (both dashed). Points are colored by ``labels`` (default: the
    tree partition ``model.tree_labels_``).

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    model : fitted :class:`~fuzzyclust.graph.MSTClustering`
    ax : matplotlib Axes, optional
    labels : array-like of shape (n_samples,), optional
    show_cuts : bool, default=True
    legend : bool, default=True
    **kwargs
        Passed to :func:`fuzzyclust.plotting.plot_clusters`.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    labels = model.tree_labels_ if labels is None else labels
    kwargs.setdefault("s", 12)
    ax = plot_clusters(X, labels, ax=ax, legend=False, **kwargs)
    ax.add_collection(
        LineCollection(
            _edge_segments(X, model.mst_edges_), colors="0.55", linewidths=0.8, zorder=1, label="MST edge"
        )
    )
    if show_cuts and len(model.cut_edges_):
        styles = ((0, "#D55E00", "cut: length criteria"), (1, "#E69F00", "cut: volume criterion"))
        for phase, color, name in styles:
            mask = model.cut_phases_ == phase
            if mask.any():
                ax.add_collection(
                    LineCollection(
                        _edge_segments(X, model.cut_edges_[mask]), colors=color, linewidths=2.2,
                        linestyles="--", zorder=3, label=name,
                    )
                )
    if legend:
        handles, names = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, names, frameon=False, fontsize=8, loc="best")
    return ax


def plot_neighbors_graph(X, graph, ax=None, labels=None, centers=None, show_bridges=True, **kwargs):
    """Draw a neighborhood graph (sparse adjacency) over the data.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    graph : scipy.sparse matrix or fitted :class:`~fuzzyclust.graph.GeodesicFCM` / :class:`IsomapFCM`
        Adjacency matrix, or a fitted model whose ``neighbors_graph_``,
        ``labels_``, ``cluster_centers_`` and ``bridge_edges_`` are used.
    labels, centers : optional
        Passed to :func:`fuzzyclust.plotting.plot_clusters`.
    show_bridges : bool, default=True
        Draw the edges that were added to connect the components (dashed,
        vermilion) when a fitted model is given.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    bridges = None
    if not issparse(graph):
        labels = graph.labels_ if labels is None else labels
        centers = graph.cluster_centers_ if centers is None else centers
        bridges = getattr(graph, "bridge_edges_", None)
        graph = graph.neighbors_graph_
    G = graph.tocoo()
    mask = G.row < G.col
    edges = np.c_[G.row[mask], G.col[mask]]
    kwargs.setdefault("s", 12)
    ax = plot_clusters(X, labels, centers=centers, ax=ax, legend=False, **kwargs)
    ax.add_collection(LineCollection(_edge_segments(X, edges), colors="0.6", linewidths=0.6, zorder=1))
    if show_bridges and bridges is not None and len(bridges):
        ax.add_collection(
            LineCollection(
                _edge_segments(X, bridges), colors="#D55E00", linewidths=1.8, linestyles="--", zorder=3,
                label="bridge",
            )
        )
    return ax


def plot_geodesic_path(X, graph, source, target, ax=None, color=None):
    """Highlight the shortest graph path between two samples and the Euclidean chord.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    graph : scipy.sparse matrix or fitted model
        Weighted adjacency (``neighbors_graph_`` of a fitted model is used
        when a model is given).
    source, target : int
        Row indices of the two samples.
    ax : matplotlib Axes, optional
    color : optional
        Color of the path.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    if not issparse(graph):
        graph = graph.neighbors_graph_
    color = PALETTE[2] if color is None else color
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    dist, pred = shortest_path(graph, directed=False, indices=[source], return_predecessors=True)
    path = [target]
    while path[-1] != source and pred[0, path[-1]] >= 0:
        path.append(pred[0, path[-1]])
    path = path[::-1]
    P = X[path]
    ax.plot(P[:, 0], P[:, 1], color=color, lw=2.4, zorder=4, label=f"geodesic path ({dist[0, target]:.2f})")
    chord = np.linalg.norm(X[source] - X[target])
    ax.plot(
        [X[source, 0], X[target, 0]], [X[source, 1], X[target, 1]], color="black", lw=1.6, ls=":",
        zorder=4, label=f"Euclidean chord ({chord:.2f})",
    )
    ends = X[[source, target]]
    ax.scatter(ends[:, 0], ends[:, 1], s=70, color="black", zorder=5, edgecolor="white")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def plot_similarity_dendrogram(S, ax=None, method="single", labels=None):
    """Dendrogram of fuzzy clusters built from their similarity matrix.

    The similarity :math:`S_{ij}` (:func:`~fuzzyclust.graph.fuzzy_cluster_similarity`)
    is turned into the distance :math:`1-S_{ij}` and clustered hierarchically
    (single linkage by default, as in the paper); the height of a merge is the
    similarity level at which two clusters join.

    Parameters
    ----------
    S : array-like of shape (n_clusters, n_clusters)
    ax : matplotlib Axes, optional
    method : str, default="single"
        Linkage method of :func:`scipy.cluster.hierarchy.linkage`.
    labels : sequence of str, optional
        Leaf labels (default ``1..c``).

    Returns
    -------
    ax : matplotlib Axes
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    S = np.asarray(S, dtype=float)
    c = S.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    if c < 2:
        ax.text(0.5, 0.5, "one cluster", ha="center", va="center", transform=ax.transAxes)
        return ax
    Dm = np.clip(1.0 - (S + S.T) / 2.0, 0.0, None)
    np.fill_diagonal(Dm, 0.0)
    Z = linkage(squareform(Dm, checks=False), method=method)
    labels = [str(i + 1) for i in range(c)] if labels is None else list(labels)
    dendrogram(Z, ax=ax, labels=labels, color_threshold=0, above_threshold_color="0.3")
    # GG memberships are nearly crisp, so all similarities are small and the merge
    # heights crowd just below one: zoom the axis to the range they occupy
    lo = float(Z[:, 2].min())
    ax.set_ylim(lo - 0.2 * (1.0 - lo) - 1e-9, 1.0 + 0.02 * (1.0 - lo) + 1e-9)
    ax.set_ylabel("$1 - S_{ij}$")
    ax.set_xlabel("cluster")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
