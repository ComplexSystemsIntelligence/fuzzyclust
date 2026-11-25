# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Matplotlib helpers for segmentations, FNN tables, embedding scans and phase-space views."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..plotting import PALETTE, _ax

__all__ = [
    "plot_segmentation",
    "plot_fnn_table",
    "plot_embedding_scan",
    "plot_phase_space",
    "plot_structure_ranking",
]


def _col(i):
    return PALETTE[i % len(PALETTE)]


def plot_segmentation(
    model,
    X,
    t=None,
    features=None,
    feature_names=None,
    axes=None,
    boundaries=True,
    show="both",
):
    r"""Time plots of the variables with the fuzzy segments underneath.

    The upper panels show the selected variables with the crisp segment
    borders ``model.boundaries_`` as dashed lines and the samples colored by
    the hard segment label. The lower panel(s) show the normalized Gaussian
    memberships in time :math:`\beta_i(t_k)` (the fuzzy segments) and/or the
    posterior memberships :math:`p(\eta_i\mid\mathbf{z}_k)` that also depend
    on the observations.

    Parameters
    ----------
    model : fitted :class:`~fuzzyclust.timeseries.FuzzySegmentation`
    X : array-like of shape (n_samples, n_features)
    t : array-like of shape (n_samples,), optional
        Time axis (default ``model.time_``).
    features : sequence of int, optional
        Columns of ``X`` to draw (default: at most four).
    feature_names : sequence of str, optional
    axes : sequence of matplotlib Axes, optional
        ``len(features) + (1 or 2)`` axes sharing the x axis.
    boundaries : bool, default=True
        Draw the crisp borders.
    show : {"both", "time", "posterior"}, default="both"
        Which membership panels to draw.

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    X = np.asarray(X, dtype=float)
    t = model.time_ if t is None else np.asarray(t, dtype=float)
    n = X.shape[1]
    features = list(range(min(n, 4))) if features is None else list(features)
    if feature_names is None:
        feature_names = list(getattr(model, "feature_names_in_", [f"$x_{j + 1}$" for j in range(n)]))
    panels = {"both": 2, "time": 1, "posterior": 1}[show]
    n_axes = len(features) + panels
    if axes is None:
        _, axes = plt.subplots(n_axes, 1, figsize=(9, 1.5 * len(features) + 1.6 * panels), sharex=True)
    axes = np.atleast_1d(axes)
    labels = model.labels_
    c = model.n_segments_
    for a, j in enumerate(features):
        ax = _ax(axes[a])
        ax.plot(t, X[:, j], color="0.75", lw=0.8)
        for i in range(c):
            mask = labels == i
            ax.plot(t[mask], X[mask, j], ".", color=_col(i), ms=2.5)
        ax.set_ylabel(feature_names[j])
        if boundaries:
            for b in model.boundaries_:
                ax.axvline(t[b], color="k", ls="--", lw=0.8)
    pos = len(features)
    if show in ("both", "time"):
        ax = _ax(axes[pos])
        for i in range(c):
            ax.plot(t, model.time_membership_[:, i], color=_col(i), lw=1.6)
        ax.set_ylabel(r"$\beta_i(t_k)$")
        ax.set_ylim(-0.03, 1.05)
        if boundaries:
            for b in model.boundaries_:
                ax.axvline(t[b], color="k", ls="--", lw=0.8)
        pos += 1
    if show in ("both", "posterior"):
        ax = _ax(axes[pos])
        for i in range(c):
            ax.plot(t, model.membership_[:, i], color=_col(i), lw=0.9)
        ax.set_ylabel(r"$p(\eta_i\,|\,\mathbf{z}_k)$")
        ax.set_ylim(-0.03, 1.05)
    axes[-1].set_xlabel("time")
    return axes


def plot_fnn_table(table: pd.DataFrame, ax=None, selected=None, annotate=True, cmap="Blues"):
    r"""Heat map of the percentage of false nearest neighbors over the orders :math:`(n_a, n_b)`.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows :math:`n_a`, columns :math:`n_b` (output of
        :func:`~fuzzyclust.timeseries.fnn_model_order` or
        :func:`~fuzzyclust.timeseries.false_nearest_neighbors`). A table with
        a single column (time series) is drawn as a curve instead.
    ax : matplotlib Axes, optional
    selected : tuple (n_a, n_b), optional
        Marks the selected order.
    annotate : bool, default=True
        Print the percentages in the cells.

    Returns
    -------
    ax : matplotlib Axes
    """
    ax = _ax(ax)
    if table.shape[1] == 1:
        ax.plot(table.index, table.iloc[:, 0], "o-", color=PALETTE[0], lw=1.8)
        ax.set_xlabel("number of lagged outputs $n_a$ (embedding dimension)")
        ax.set_ylabel("false nearest neighbors [%]")
        ax.set_xticks(list(table.index))
        if selected is not None:
            ax.axvline(selected[0], color=PALETTE[1], ls="--", lw=1.2)
        ax.grid(True, color="0.9")
        return ax
    M = table.to_numpy(dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, origin="lower", aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(list(table.columns))
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(list(table.index))
    ax.set_xlabel("number of lagged inputs $n_b$")
    ax.set_ylabel("number of lagged outputs $n_a$")
    if annotate:
        for r in range(M.shape[0]):
            for q in range(M.shape[1]):
                ax.text(q, r, f"{M[r, q]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[r, q] > 55 else "black")
    if selected is not None:
        r = list(table.index).index(selected[0])
        q = list(table.columns).index(selected[1])
        ax.add_patch(plt.Rectangle((q - 0.5, r - 0.5), 1, 1, fill=False, ec=PALETTE[1], lw=2.5))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="false neighbors [%]")
    return ax


def plot_embedding_scan(table: pd.DataFrame, axes=None, selected=None):
    r"""One-step prediction error and local-dimension index versus the embedding dimension.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`~fuzzyclust.timeseries.embedding_scan`.
    axes : two matplotlib Axes, optional
    selected : :class:`sklearn.utils.Bunch`, optional
        Output of :func:`~fuzzyclust.timeseries.select_embedding_dimension`;
        the selected :math:`d_e` and :math:`d_l` are marked.

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    axes = np.atleast_1d(axes)
    d = table.index.to_numpy()
    ax = _ax(axes[0])
    ax.semilogy(d, table["mse_train"], "o--", color=PALETTE[0], lw=1.4, label="training")
    ax.semilogy(d, table["mse_test"], "s-", color=PALETTE[1], lw=1.8, label="test")
    ax.set_ylabel("one-step MSE")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9", which="both")
    ax2 = _ax(axes[1])
    ax2.semilogy(d, table["local_dimension_index"], "o-", color=PALETTE[2], lw=1.8)
    ax2.set_ylabel(r"$J^{d_e}=\sum_i p(\eta_i)\lambda_{i,\min}$")
    ax2.set_xlabel("embedding dimension $d_e$")
    ax2.set_xticks(list(d))
    ax2.grid(True, color="0.9", which="both")
    if selected is not None:
        ax.axvline(selected.embedding_dimension, color=PALETTE[1], ls=":", lw=1.2)
        ax.text(selected.embedding_dimension, ax.get_ylim()[1], f" $d_e={selected.embedding_dimension}$",
                va="top", fontsize=8, color=PALETTE[1])
        ax2.axvline(selected.local_dimension + 1, color=PALETTE[2], ls=":", lw=1.2)
        ax2.text(selected.local_dimension + 1, ax2.get_ylim()[1], f" $d_l={selected.local_dimension}$",
                 va="top", fontsize=8, color=PALETTE[2])
    return axes


def plot_phase_space(Y, labels=None, dims=(0, 1), centers=None, ax=None, s=3, lw=0.3, trajectory=True):
    r"""Two coordinates of a reconstructed (or true) state space, colored by cluster.

    Parameters
    ----------
    Y : array-like of shape (n_samples, dim)
        Delay vectors (or state trajectory).
    labels : array-like of shape (n_samples,), optional
        Hard cluster labels used for the colors.
    dims : tuple of two int, default=(0, 1)
        Coordinates to draw.
    centers : array-like of shape (n_clusters, dim), optional
        Cluster centers, drawn as large markers.
    ax : matplotlib Axes, optional
    s : float, default=3
        Marker size.
    lw : float, default=0.3
        Width of the gray trajectory line (``0`` or ``trajectory=False`` hides it).
    trajectory : bool, default=True

    Returns
    -------
    ax : matplotlib Axes
    """
    Y = np.asarray(Y, dtype=float)
    ax = _ax(ax)
    a, b = dims
    if trajectory and lw > 0:
        ax.plot(Y[:, a], Y[:, b], color="0.7", lw=lw, zorder=1)
    if labels is None:
        ax.scatter(Y[:, a], Y[:, b], s=s, color=PALETTE[0], zorder=2)
    else:
        labels = np.asarray(labels)
        for i in np.unique(labels):
            mask = labels == i
            ax.scatter(Y[mask, a], Y[mask, b], s=s, color=_col(int(i)), zorder=2, label=f"cluster {i + 1}")
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        for i in range(centers.shape[0]):
            ax.plot(centers[i, a], centers[i, b], "o", color=_col(i), mec="k", ms=9, zorder=3)

    def lag(j):
        return "$y_k$" if j == 0 else (r"$y_{k-\tau}$" if j == 1 else f"$y_{{k-{j}\\tau}}$")

    ax.set_xlabel(lag(a))
    ax.set_ylabel(lag(b))
    return ax


def plot_structure_ranking(ranking: pd.DataFrame, n_show: int = 30, ax=None, best=None, label_top: int = 5):
    r"""Variance--complexity trade-off :math:`VCT=\sigma_t n_i` along the ranking of structures.

    Reproduces the VCT curves of Girimonte, Babuška and Abonyi: structures
    on the x axis in the order of increasing :math:`\sigma_t`, their VCT on
    the y axis; the first clear dip marks the recommended structure.

    Parameters
    ----------
    ranking : pandas.DataFrame
        ``ranking_`` of :class:`~fuzzyclust.timeseries.MixedInputStructureSelection`.
    n_show : int, default=30
        Number of best structures drawn.
    ax : matplotlib Axes, optional
    best : tuple of str, optional
        Structure to mark (e.g. ``best_structure_``).
    label_top : int, default=5
        Annotate this many lowest-VCT structures with their inputs.

    Returns
    -------
    ax : matplotlib Axes
    """
    ax = _ax(ax)
    sub = ranking.iloc[:n_show]
    ax.plot(sub.index, sub["vct"], "o-", color=PALETTE[0], lw=1.4, ms=4)
    order = np.argsort(sub["vct"].to_numpy())[:label_top]
    # the lowest-VCT structures get a rank number next to the point and a key in the upper left
    key = []
    for rank, pos in enumerate(order, start=1):
        row = sub.iloc[pos]
        ax.annotate(str(rank), (sub.index[pos], row["vct"]), fontsize=7, color="0.25",
                    xytext=(0, -11 if rank % 2 else 6), textcoords="offset points", ha="center")
        key.append(f"{rank}: " + ", ".join(row["inputs"]))
    if key:
        ax.text(0.02, 0.97, "\n".join(key), transform=ax.transAxes, va="top", ha="left", fontsize=7,
                color="0.25", linespacing=1.4)
    if best is not None:
        hit = sub[sub["inputs"].apply(lambda s: tuple(s) == tuple(best))]
        if len(hit):
            ax.plot(hit.index, hit["vct"], "s", color=PALETTE[1], ms=9, mfc="none", mew=2)
    ax.set_xlabel(r"structure number (ranked by increasing $\sigma_t$)")
    ax.set_ylabel(r"variance–complexity trade-off $\sigma_t n_i$")
    ax.grid(True, color="0.9")
    return ax
