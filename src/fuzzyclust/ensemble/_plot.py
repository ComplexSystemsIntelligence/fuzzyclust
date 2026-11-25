# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for cluster ensembles."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE

__all__ = ["plot_coassociation", "plot_lifetimes"]


def plot_coassociation(S, labels=None, ax=None, cmap="Greys", cbar=True, show_blocks=True):
    r"""Heat map of a co-association matrix with the samples reordered by (consensus) label.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples) or fitted ensemble
        Co-association matrix, or a :class:`~fuzzyclust.ensemble.FuzzyClusterEnsemble`.
        A fitted ensemble supplies ``coassociation_`` and ``labels_``.
    labels : array-like of shape (n_samples,), optional
        Order of the samples (grouped by label); default: the natural order.
    ax : matplotlib Axes, optional
    cmap : str, default="Greys"
    cbar : bool, default=True
    show_blocks : bool, default=True
        Outline the diagonal blocks of the clusters.

    Returns
    -------
    ax : matplotlib Axes
    """
    if not isinstance(S, np.ndarray):
        labels = S.labels_ if labels is None else labels
        S = S.coassociation_
    S = np.asarray(S, dtype=float)
    N = S.shape[0]
    order = np.arange(N) if labels is None else np.argsort(np.asarray(labels), kind="stable")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(S[order][:, order], cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    if show_blocks and labels is not None:
        lab = np.asarray(labels)[order]
        for i in np.unique(lab):
            idx = np.flatnonzero(lab == i)
            lo, hi = idx.min() - 0.5, idx.max() + 0.5
            ax.plot([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo], color=PALETTE[int(i) % len(PALETTE)], lw=1.6)
    ax.set_xlabel("samples (reordered)")
    ax.set_ylabel("samples (reordered)")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="co-association $s_{kl}$")
    return ax


def plot_lifetimes(model_or_lifetimes, ax=None, max_show=10):
    """Bar chart of the lifetime of every number of clusters in the co-association dendrogram.

    Parameters
    ----------
    model_or_lifetimes : fitted :class:`~fuzzyclust.ensemble.FuzzyClusterEnsemble` or array-like
        ``lifetimes_`` of the ensemble, or the array itself (``lifetimes[k - 1]`` for ``k`` clusters).
    ax : matplotlib Axes, optional
    max_show : int, default=10

    Returns
    -------
    ax : matplotlib Axes
    """
    life = np.asarray(getattr(model_or_lifetimes, "lifetimes_", model_or_lifetimes), dtype=float)
    k = np.arange(1, life.size + 1)[:max_show]
    life = life[:max_show]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.2))
    colors = ["0.6"] * life.size
    colors[int(np.argmax(life))] = PALETTE[1]
    ax.bar(k, life, color=colors)
    ax.set_xlabel("number of clusters $k$")
    ax.set_ylabel("lifetime")
    ax.set_xticks(k)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
