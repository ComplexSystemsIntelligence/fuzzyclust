# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for projections and fuzzy self-organizing maps (matplotlib only)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE, _ax, _grid, plot_clusters
from .quality import recomputed_membership

__all__ = ["plot_projection", "plot_map", "plot_som_lattice", "plot_stress"]


def plot_projection(
    Y,
    Z=None,
    labels=None,
    U=None,
    ax=None,
    contours: bool = True,
    m: float = 2.0,
    levels=8,
    misclassified=None,
    resolution: int = 150,
    legend: bool = True,
    label_prefix: str = "cluster",
    **kwargs,
):
    r"""Scatter plot of a 2-D projection with projected prototypes and membership contours.

    Reproduces the plots of the MATLAB ``projeval`` function: the points
    :math:`\mathbf{y}_k`, the projected centers :math:`\mathbf{z}_i` (black
    crosses) and the level curves of the recomputed maximal membership
    :math:`\max_i\mu^*_i(\mathbf{y})` computed from the Euclidean distances
    to the :math:`\mathbf{z}_i` (see :func:`~fuzzyclust.viz.recomputed_membership`).
    Because the contours are always circular-symmetric around the centers,
    their agreement with the point clouds shows how well the projection
    represents the clustering.

    Parameters
    ----------
    Y : array-like of shape (n_samples, 2)
        Projected data.
    Z : array-like of shape (n_clusters, 2), optional
        Projected prototypes.
    labels : array-like of shape (n_samples,), optional
        Color code (hard labels, cluster or class index).
    U : array-like of shape (n_samples, n_clusters), optional
        Original memberships; if given together with ``Z`` and ``labels`` is
        ``None``, the points are colored by ``U.argmax(axis=1)``.
    contours : bool, default=True
        Draw the recomputed membership contours (needs ``Z``).
    m : float, default=2.0
        Fuzziness exponent for the recomputed memberships.
    levels : int or sequence, default=8
    misclassified : array-like of bool of shape (n_samples,), optional
        Points to mark with an extra circle (e.g. cluster label differs from
        the known class, as in the toolbox case studies).
    resolution : int, default=150
        Grid resolution of the contour map.
    label_prefix : str, default="cluster"
        Legend text in front of the label number (e.g. ``"class"`` when the
        colors are known classes rather than clusters).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape[1] != 2:
        raise ValueError("plot_projection draws 2-D embeddings only.")
    ax = _ax(ax)
    if labels is None and U is not None:
        labels = np.asarray(U).argmax(axis=1)
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        if contours:
            gx, gy, G = _grid(np.vstack([Y, Z]), resolution)
            Ustar = recomputed_membership(G, Z, m).max(axis=1).reshape(gx.shape)
            cs = ax.contour(gx, gy, Ustar, levels=levels, cmap="Greys", linewidths=0.8, vmin=0.3, vmax=1.1)
            ax.clabel(cs, fmt="%.2f", fontsize=6)
    plot_clusters(Y, labels, Z, ax=ax, legend=False, **kwargs)
    if label_prefix != "cluster":
        for coll in ax.collections:
            lab = coll.get_label()
            if lab.startswith("cluster "):
                coll.set_label(label_prefix + lab[len("cluster"):])
    if misclassified is not None:
        mis = np.asarray(misclassified, dtype=bool)
        ax.scatter(Y[mis, 0], Y[mis, 1], s=60, facecolors="none", edgecolors="black", linewidths=0.8,
                   label="misclassified", zorder=4)
    if legend and labels is not None:
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    return ax


def plot_stress(histories, labels=None, ax=None, ylabel="stress"):
    """Stress versus iteration for one or several ``stress_history_`` arrays."""
    ax = _ax(ax)
    if isinstance(histories, np.ndarray) and histories.ndim == 1:
        histories = [histories]
    for j, h in enumerate(histories):
        h = np.asarray(h)
        name = labels[j] if labels else f"run {j + 1}"
        ax.plot(np.arange(1, h.size + 1), h, lw=1.8, color=PALETTE[j % len(PALETTE)], label=name)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_map(som, kind: str = "umatrix", feature=None, X=None, ax=None, cmap=None, annotate=False, **kwargs):
    r"""Draw a fuzzy SOM as a colored grid: U-matrix, a component plane or the hit histogram.

    Parameters
    ----------
    som : fitted :class:`~fuzzyclust.viz.FuzzySOM` with a 1-D or 2-D grid
    kind : {"umatrix", "component", "hits"}, default="umatrix"
        ``"umatrix"``: mean prototype distance to the grid neighbors (large =
        border between groups); ``"component"``: value of one feature of the
        prototypes over the grid (needs ``feature``); ``"hits"``: fuzzy hit
        histogram :math:`\sum_k\mu_{ik}` of the training data or of ``X``.
    feature : int or str, optional
        Feature index (or name from ``feature_names_in_``) for ``kind="component"``.
    X : array-like, optional
        Data for the hit histogram (default: training partition).
    annotate : bool, default=False
        Write the value into every cell.
    """
    from matplotlib.colors import Normalize

    if kind == "umatrix":
        M = som.u_matrix()
        title = "U-matrix"
        cmap = cmap or "Greys"
    elif kind == "component":
        if feature is None:
            raise ValueError("kind='component' needs the feature argument.")
        if isinstance(feature, str):
            names = list(getattr(som, "feature_names_in_", []))
            if feature not in names:
                raise ValueError(f"unknown feature name {feature!r}.")
            idx, title = names.index(feature), feature
        else:
            idx = int(feature)
            names = getattr(som, "feature_names_in_", None)
            title = names[idx] if names is not None else f"feature {idx}"
        M = som.component_planes()[idx]
        cmap = cmap or "viridis"
    elif kind == "hits":
        M = som.hits(X)
        title = "fuzzy hits"
        cmap = cmap or "Blues"
    else:
        raise ValueError("kind must be 'umatrix', 'component' or 'hits'.")
    M = np.atleast_2d(M)
    if M.ndim > 2:
        raise ValueError("plot_map draws 1-D and 2-D grids only.")
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(M.T, origin="lower", cmap=cmap, norm=Normalize(), **kwargs)
    ax.set_xticks(range(M.shape[0]))
    ax.set_yticks(range(M.shape[1]))
    ax.set_xlabel("grid axis 1")
    ax.set_ylabel("grid axis 2")
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(i, j, f"{M[i, j]:.2g}", ha="center", va="center", fontsize=7,
                        color="white" if (M[i, j] - M.min()) / max(np.ptp(M), 1e-12) > 0.6 else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_som_lattice(som, X=None, dims=(0, 1), ax=None, labels=None, s=10, **kwargs):
    r"""Prototypes of a fuzzy SOM in the data space, with grid neighbors connected.

    Shows how the map "folds" into the data: an ordered map has a regular,
    non-crossing lattice, an unordered one (``regularization=0``) a tangle.

    Parameters
    ----------
    som : fitted :class:`~fuzzyclust.viz.FuzzySOM`
    X : array-like of shape (n_samples, n_features), optional
        Data to draw in the background.
    dims : tuple of two int, default=(0, 1)
        Which two features to plot.
    labels : array-like of shape (n_samples,), optional
        Color code of the data points.
    """
    ax = _ax(ax)
    a, b = dims
    if X is not None:
        X = np.asarray(X)
        if labels is None:
            ax.scatter(X[:, a], X[:, b], s=s, color="0.75", alpha=0.7)
        else:
            labels = np.asarray(labels)
            for i in np.unique(labels):
                mask = labels == i
                ax.scatter(X[mask, a], X[mask, b], s=s, color=PALETTE[int(i) % len(PALETTE)], alpha=0.7)
    V = som.cluster_centers_
    for i, j in som._neighbors():
        ax.plot([V[i, a], V[j, a]], [V[i, b], V[j, b]], color="black", lw=1.0, **kwargs)
    ax.scatter(V[:, a], V[:, b], marker="o", s=38, color="black", edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xlabel(f"$x_{{{a + 1}}}$")
    ax.set_ylabel(f"$x_{{{b + 1}}}$")
    return ax
