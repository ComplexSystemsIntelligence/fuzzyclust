# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for the chunked algorithms: chunk timelines and prototype trajectories."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_chunk_timeline", "plot_center_trajectories"]


def _ax(ax, figsize=(6.0, 3.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def plot_chunk_timeline(model, feature: int = 0, ax=None, true_centers=None, weights=True):
    """Prototype coordinate against the chunk index for a chunked model.

    Draws, for every cluster, the coordinate ``feature`` of its prototype
    after each chunk (``chunk_centers_`` of :class:`~fuzzyclust.large.SinglePassFCM`
    or :class:`~fuzzyclust.large.OnlineFCM`); the marker area is
    proportional to the weight the prototype carries, so a growing marker
    shows the single-pass prototype accumulating observations. Optional
    true centers per chunk are drawn as dotted lines (drifting streams).

    Parameters
    ----------
    model : fitted :class:`~fuzzyclust.large.SinglePassFCM` or :class:`~fuzzyclust.large.OnlineFCM`
    feature : int, default=0
        Which coordinate to show.
    ax : matplotlib Axes, optional
    true_centers : array-like of shape (n_chunks, n_clusters, n_features), optional
    weights : bool, default=True
        Scale the markers by ``chunk_weights_``.

    Returns
    -------
    ax : matplotlib Axes
    """
    ax = _ax(ax)
    C = np.asarray(model.chunk_centers_)
    W = np.asarray(model.chunk_weights_, dtype=float)
    s, c = C.shape[:2]
    chunks = np.arange(1, s + 1)
    size = 12 + 120 * W / max(W.max(), 1e-12) if weights else np.full_like(W, 30.0)
    for i in range(c):
        col = PALETTE[i % len(PALETTE)]
        ax.plot(chunks, C[:, i, feature], color=col, lw=1.2, zorder=2)
        ax.scatter(chunks, C[:, i, feature], s=size[:, i], color=col, edgecolor="white", zorder=3,
                   label=f"cluster {i + 1}")
    if true_centers is not None:
        T = np.asarray(true_centers, dtype=float)
        for i in range(T.shape[1]):
            ax.plot(np.arange(1, T.shape[0] + 1), T[:, i, feature], color="0.3", ls=":", lw=1.0, zorder=1,
                    label="true centers" if i == 0 else None)
    ax.set_xlabel("chunk")
    ax.set_ylabel(f"prototype coordinate $v_{{i,{feature + 1}}}$")
    ax.set_title("Prototypes after every chunk" + (" (marker area $\\propto$ weight)" if weights else ""))
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def plot_center_trajectories(X, model, ax=None, max_points: int = 4000, random_state=0, labels=None):
    """Trajectories of the prototypes over the chunks on top of (a subsample of) the 2-D data.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    model : fitted :class:`~fuzzyclust.large.SinglePassFCM` or :class:`~fuzzyclust.large.OnlineFCM`
    ax : matplotlib Axes, optional
    max_points : int, default=4000
        At most this many samples are drawn (random subsample).
    random_state : int, default=0
    labels : array-like of shape (n_samples,), optional
        Hard labels for coloring the samples (default: ``model.labels_``
        when it covers ``X``, otherwise gray).

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    N = X.shape[0]
    sel = rng.choice(N, size=min(max_points, N), replace=False) if N > max_points else np.arange(N)
    if labels is None and hasattr(model, "labels_") and len(model.labels_) == N:
        labels = model.labels_
    lab = None if labels is None else np.asarray(labels)[sel]
    ax = plot_clusters(X[sel], lab, ax=ax, legend=False, s=6, alpha=0.35)
    C = np.asarray(model.chunk_centers_)
    for i in range(C.shape[1]):
        col = PALETTE[i % len(PALETTE)]
        ax.plot(C[:, i, 0], C[:, i, 1], color="black", lw=2.6, zorder=4)
        ax.plot(C[:, i, 0], C[:, i, 1], color=col, lw=1.4, zorder=5, marker="o", ms=3.5)
    V = np.asarray(model.cluster_centers_)
    ax.scatter(V[:, 0], V[:, 1], marker="X", s=160, color="black", edgecolor="white", linewidth=1.2, zorder=6,
               label="final prototypes")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax
