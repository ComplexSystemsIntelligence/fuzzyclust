# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for co-clustering results."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE

__all__ = ["plot_coclusters", "plot_item_memberships"]


def _order(labels, strength=None):
    labels = np.asarray(labels)
    if strength is None:
        return np.lexsort((np.arange(labels.shape[0]), labels))
    return np.lexsort((-np.asarray(strength), labels))


def plot_coclusters(R, model=None, row_labels=None, col_labels=None, ax=None, cmap="Greys",
                    show_blocks=True, cbar=True, **kwargs):
    """Heat map of an object--item matrix with rows and columns reordered by co-cluster.

    Rows are sorted by their (hard) co-cluster and, within a co-cluster, by
    decreasing membership; columns likewise by their item co-cluster. The
    boundaries of the co-clusters are drawn as colored lines, so that the
    blocks the algorithm found appear along the diagonal.

    Parameters
    ----------
    R : array-like of shape (n_objects, n_items)
    model : fitted :class:`~fuzzyclust.coclustering.FCCM` or :class:`~fuzzyclust.coclustering.FCoDoK`,
    optional
        Its ``labels_``, ``membership_``, ``item_labels_`` and
        ``item_membership_`` define the order; alternatively give
        ``row_labels`` and ``col_labels``.
    row_labels, col_labels : array-like, optional
        Explicit (hard) co-cluster labels of rows and columns (``-1`` is
        sorted last).
    ax : matplotlib Axes, optional
    cmap : str, default="Greys"
    show_blocks : bool, default=True
    cbar : bool, default=True
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    ax : matplotlib Axes
    """
    R = np.asarray(R, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    if model is not None:
        row_labels = model.labels_
        col_labels = model.item_labels_
        ro = _order(row_labels, model.membership_.max(axis=1))
        W = model.item_membership_ / np.maximum(model.item_membership_.max(axis=0, keepdims=True), 1e-12)
        co = _order(col_labels, W.max(axis=1))
    else:
        row_labels = np.arange(R.shape[0]) * 0 if row_labels is None else np.asarray(row_labels)
        col_labels = np.arange(R.shape[1]) * 0 if col_labels is None else np.asarray(col_labels)
        row_labels = np.where(row_labels < 0, row_labels.max() + 1, row_labels)
        col_labels = np.where(col_labels < 0, col_labels.max() + 1, col_labels)
        ro, co = _order(row_labels), _order(col_labels)
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("interpolation", "nearest")
    im = ax.imshow(R[ro][:, co], cmap=cmap, **kwargs)
    if show_blocks:
        rl, cl = np.asarray(row_labels)[ro], np.asarray(col_labels)[co]
        for i in np.unique(rl):
            r_idx = np.flatnonzero(rl == i)
            c_idx = np.flatnonzero(cl == i)
            if r_idx.size == 0 or c_idx.size == 0:
                continue
            col = PALETTE[int(i) % len(PALETTE)]
            x0, x1 = c_idx.min() - 0.5, c_idx.max() + 0.5
            y0, y1 = r_idx.min() - 0.5, r_idx.max() + 0.5
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=col, lw=1.8)
    ax.set_xlabel("items (reordered)")
    ax.set_ylabel("objects (reordered)")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return ax


def plot_item_memberships(model, ax=None, item_names=None, normalize=True, top=None):
    """Bar chart of the item memberships of every co-cluster.

    Parameters
    ----------
    model : fitted co-clustering estimator
    ax : matplotlib Axes, optional
    item_names : sequence of str, optional
    normalize : bool, default=True
        Divide every co-cluster's item memberships by their maximum, so
        that clusters with vocabularies of different size are comparable.
    top : int, optional
        Show only the ``top`` items of the largest membership (over all clusters).

    Returns
    -------
    ax : matplotlib Axes
    """
    W = np.asarray(model.item_membership_, dtype=float)  # (M, c)
    if normalize:
        W = W / np.maximum(W.max(axis=0, keepdims=True), 1e-12)
    M, c = W.shape
    order = np.lexsort((-W.max(axis=1), model.item_labels_))
    if top is not None:
        order = order[np.argsort(-W.max(axis=1)[order])[:top]]
        order = order[np.lexsort((-W.max(axis=1)[order], model.item_labels_[order]))]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(order.size)
    width = 0.8 / c
    for i in range(c):
        ax.bar(x + (i - (c - 1) / 2) * width, W[order, i], width=width, color=PALETTE[i % len(PALETTE)],
               label=f"co-cluster {i + 1}")
    names = [str(j) for j in order] if item_names is None else [str(item_names[j]) for j in order]
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("item membership" + (" (scaled)" if normalize else ""))
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
