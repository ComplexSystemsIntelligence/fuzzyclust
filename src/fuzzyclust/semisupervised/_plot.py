# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for partially supervised and constrained clustering."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_label_propagation", "plot_pairwise_constraints"]

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def plot_label_propagation(X, y_partial, model=None, labels=None, ax=None, legend=True, s=14):
    """Labeled seeds as large black-edged markers, the rest colored by the cluster they were pulled into.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    y_partial : array-like of shape (n_samples,)
        Class labels with ``-1`` for the unlabeled samples.
    model : fitted :class:`~fuzzyclust.semisupervised.SemiSupervisedFCM`, optional
        Its ``labels_`` and ``cluster_centers_`` are drawn; the marker of a
        labeled sample is that of its class, the color that of the
        cluster tied to the class (``label_mapping_``).
    labels : array-like of shape (n_samples,), optional
        Hard labels to color the samples when no model is given.
    ax : matplotlib Axes, optional
    legend : bool, default=True
    s : float, default=14
        Marker size of the unlabeled samples.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    y_partial = np.asarray(y_partial).ravel()
    if model is not None:
        labels = model.labels_
        centers = model.cluster_centers_
        mapping = getattr(model, "label_mapping_", None)
    else:
        centers, mapping = None, None
    labelled = y_partial != -1
    ax = plot_clusters(X[~labelled], None if labels is None else labels[~labelled], centers, ax=ax,
                       legend=False, s=s, alpha=0.6)
    classes = np.unique(y_partial[labelled])
    handles = []
    for j, cls in enumerate(classes):
        mask = y_partial == cls
        if mapping is not None and cls in mapping:
            col = PALETTE[int(np.flatnonzero(mapping == cls)[0]) % len(PALETTE)]
        else:
            col = PALETTE[j % len(PALETTE)]
        mk = _MARKERS[j % len(_MARKERS)]
        ax.scatter(X[mask, 0], X[mask, 1], s=5 * s, marker=mk, color=col, edgecolor="black",
                   linewidth=1.0, zorder=4)
        handles.append(Line2D([], [], ls="", marker=mk, color=col, markeredgecolor="black",
                              markersize=7, label=f"labeled: class {cls}"))
    if labels is not None:
        for i in np.unique(labels):
            handles.append(Line2D([], [], ls="", marker="o", color=PALETTE[int(i) % len(PALETTE)],
                                  markersize=5, label=f"cluster {int(i) + 1}"))
    if legend and handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    return ax


def plot_pairwise_constraints(X, must_link=None, cannot_link=None, labels=None, centers=None, ax=None,
                              legend=True, violated_only=False, **kwargs):
    """Overlay must-link (solid green) and cannot-link (dashed vermilion) pairs on a scatter plot.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    must_link, cannot_link : array-like of shape (n_pairs, 2), optional
    labels : array-like of shape (n_samples,), optional
        Hard labels for the colors; also used to mark violated constraints
        (drawn thicker) when given.
    centers : array-like of shape (n_clusters, 2), optional
    ax : matplotlib Axes, optional
    legend : bool, default=True
    violated_only : bool, default=False
        Draw only the constraints the labels violate.
    **kwargs
        Passed to :func:`fuzzyclust.plotting.plot_clusters`.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    kwargs.setdefault("s", 12)
    ax = plot_clusters(X, labels, centers, ax=ax, legend=False, **kwargs)
    labels = None if labels is None else np.asarray(labels)
    handles = []
    specs = (
        (must_link, "#009E73", "-", "must-link", lambda a, b: a != b),
        (cannot_link, "#D55E00", "--", "cannot-link", lambda a, b: a == b),
    )
    for pairs, color, ls, name, violates in specs:
        if pairs is None or len(pairs) == 0:
            continue
        P = np.asarray(pairs, dtype=int).reshape(-1, 2)
        if labels is not None:
            bad = violates(labels[P[:, 0]], labels[P[:, 1]])
        else:
            bad = np.zeros(P.shape[0], bool)
        groups = [(P[bad], 2.4, 1.0, f"{name} (violated)")]
        if not violated_only:
            groups.insert(0, (P[~bad], 1.0, 0.8, name))
        for Q, lw, alpha, label in groups:
            if Q.shape[0] == 0:
                continue
            segs = np.stack([X[Q[:, 0]], X[Q[:, 1]]], axis=1)
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, alpha=alpha,
                                             zorder=3))
            handles.append(Line2D([], [], color=color, ls=ls, lw=lw, label=label))
    if legend and handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    return ax
