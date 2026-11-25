# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for relational data: ordered dissimilarity matrices, DTW alignments, modes."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE
from .dissimilarity import dtw_alignment

__all__ = ["plot_dissimilarity_matrix", "plot_dtw_alignment", "plot_modes", "plot_series_clusters"]


def plot_dissimilarity_matrix(R, labels=None, ax=None, cmap="viridis", colorbar=True, medoids=None):
    """Heat map of a dissimilarity matrix with the objects ordered by their labels.

    Objects of the same cluster are placed next to each other (clusters in
    increasing label order, within a cluster the original order), so that a
    good partition shows dark diagonal blocks; the block boundaries are
    drawn as white lines and the medoids, when given, as markers.

    Parameters
    ----------
    R : array-like of shape (n_samples, n_samples)
    labels : array-like of shape (n_samples,), optional
        Hard cluster labels; without them the original order is kept.
    ax : matplotlib Axes, optional
    cmap : str, default="viridis"
    colorbar : bool, default=True
    medoids : array-like of int, optional
        Indices of medoid objects to mark on the diagonal.

    Returns
    -------
    ax : matplotlib Axes
    """
    R = np.asarray(R, dtype=float)
    N = R.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.4))
    if labels is None:
        order = np.arange(N)
        bounds = []
    else:
        labels = np.asarray(labels)
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels[order] - labels.min())
        bounds = np.cumsum(counts)[:-1]
    im = ax.imshow(R[np.ix_(order, order)], cmap=cmap, interpolation="nearest")
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axvline(b - 0.5, color="white", lw=1.0)
    if medoids is not None:
        pos = np.empty(N, dtype=int)
        pos[order] = np.arange(N)
        for i, q in enumerate(np.asarray(medoids, dtype=int)):
            ax.plot(pos[q], pos[q], marker="X", ms=9, color=PALETTE[i % len(PALETTE)], mec="white", zorder=5)
    which = "original order" if labels is None else "ordered by label"
    ax.set_xlabel(f"object ({which})")
    ax.set_ylabel(f"object ({which})")
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dissimilarity")
    return ax


def plot_dtw_alignment(x, y, window=None, ax=None, offset=None, every=1, colors=None):
    """Two series with the dynamic-time-warping correspondences between them.

    The second series is drawn shifted downwards by ``offset`` and every
    ``every``-th pair of the optimal warping path is connected by a gray
    line, so that stretched and compressed stretches are visible.

    Parameters
    ----------
    x, y : array-like of shape (length,)
        Univariate series.
    window : int, float or None, default=None
        Sakoe--Chiba band passed to :func:`~fuzzyclust.relational.dtw_alignment`.
    ax : matplotlib Axes, optional
    offset : float, optional
        Vertical shift of the second series (default: 1.5 times the range).
    every : int, default=1
        Draw every ``every``-th correspondence.
    colors : pair of colors, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dist, path = dtw_alignment(x, y, window=window)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6))
    if offset is None:
        offset = 1.5 * max(np.ptp(x), np.ptp(y), 1e-12)
    c0, c1 = colors if colors is not None else (PALETTE[0], PALETTE[1])
    for t, s in path[::every]:
        ax.plot([t, s], [x[t], y[s] - offset], color="0.75", lw=0.6, zorder=1)
    ax.plot(x, color=c0, lw=1.8, label="series 1")
    ax.plot(y - offset, color=c1, lw=1.8, label="series 2 (shifted)")
    ax.set_xlabel("time index")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    band = "no band" if window is None else f"band {window}"
    ax.set_title(f"DTW alignment ({band}): distance {dist:.2f}, {len(path)} steps", fontsize=10)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    return ax


def plot_modes(model, X=None, ax=None, cmap="Blues", fontsize=8):
    """Table of the fitted modes, shaded by how much membership mass the mode carries.

    Every cell shows the modal category of one cluster (row) in one
    categorical attribute (column); its shade is the share of the cluster's
    membership mass :math:`\\sum_k\\mu_{ik}^m` that falls on this category,
    i.e. how unanimous the cluster is in that attribute (1 = all members
    agree). Requires the training data ``X`` for the shares; without it the
    cells are unshaded.

    Parameters
    ----------
    model : fitted FuzzyKModes or FuzzyKPrototypes
        The clusterer whose ``modes_`` are shown.
    X : array-like or DataFrame, optional
        The training records (to compute the shares).
    ax : matplotlib Axes, optional
    cmap : str, default="Blues"
    fontsize : int, default=8

    Returns
    -------
    ax : matplotlib Axes
    """
    import pandas as pd

    modes = model.modes_
    if isinstance(modes, pd.DataFrame):
        names = [str(c) for c in modes.columns]
        M = modes.to_numpy(dtype=object)
    else:
        names = [f"attribute {j + 1}" for j in range(modes.shape[1])]
        M = np.asarray(modes, dtype=object)
    c, n = M.shape
    share = np.full((c, n), np.nan)
    if X is not None and n:
        Z = model._validate_and_encode(X, reset=False)
        Um = model.membership_**model.m
        cat = model.categorical_features_
        for p, j in enumerate(cat):
            codes = Z[:, j].astype(int)
            total = Um.sum(axis=0)
            for i in range(c):
                share[i, p] = Um[codes == int(model.cluster_centers_[i, j]), i].sum() / max(total[i], 1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * n + 1.5, 0.5 * c + 1.0))
    shade = np.nan_to_num(share, nan=0.0)
    ax.imshow(shade, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(c):
        for p in range(n):
            txt = str(M[i, p])
            if np.isfinite(share[i, p]):
                txt += f"\n{share[i, p]:.0%}"
            color = "white" if shade[i, p] > 0.6 else "black"
            ax.text(p, i, txt, ha="center", va="center", fontsize=fontsize, color=color)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=fontsize)
    ax.set_yticks(range(c))
    ax.set_yticklabels([f"cluster {i + 1}" for i in range(c)], fontsize=fontsize)
    ax.set_title("modes (shade: share of membership mass on the mode)", fontsize=9)
    return ax


def plot_series_clusters(X, labels, medoids=None, ax=None, alpha=0.35, max_per_cluster=None):
    """Time series drawn in the color of their cluster, medoid series emphasized.

    Parameters
    ----------
    X : array-like of shape (n_series, length)
    labels : array-like of shape (n_series,)
    medoids : array-like of int, optional
        Indices of the medoid series (drawn in black on top).
    ax : matplotlib Axes or sequence of Axes, optional
        One axes for all clusters, or one axes per cluster.
    alpha : float, default=0.35
    max_per_cluster : int, optional
        Draw at most this many series per cluster.

    Returns
    -------
    ax : matplotlib Axes or ndarray of Axes
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    ids = np.unique(labels)
    multi = ax is not None and np.ndim(ax) > 0
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6))
    axes = np.asarray(ax).ravel() if multi else None
    for p, i in enumerate(ids):
        a = axes[p] if multi else ax
        members = np.flatnonzero(labels == i)
        if max_per_cluster is not None:
            members = members[:max_per_cluster]
        col = PALETTE[int(i) % len(PALETTE)]
        for k in members:
            a.plot(X[k], color=col, alpha=alpha, lw=0.9)
        if multi:
            a.set_title(f"cluster {int(i) + 1} ({(labels == i).sum()} series)", fontsize=10)
            a.set_xlabel("time index")
    if medoids is not None:
        for p, q in enumerate(np.asarray(medoids, dtype=int)):
            a = axes[p] if multi else ax
            a.plot(X[q], color="black", lw=2.2, label="medoid" if p == 0 or multi else None)
            if multi:
                a.legend(frameon=False, fontsize=8)
    if not multi:
        for i in ids:
            ax.plot([], [], color=PALETTE[int(i) % len(PALETTE)], label=f"cluster {int(i) + 1}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("time index")
    return ax if multi else ax
