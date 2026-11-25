# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for collaborative clustering."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_site_partitions", "plot_collaboration_trajectory"]


def plot_site_partitions(Xs, model, axes=None, independent=False, **kwargs):
    """Scatter plot of every site's (two-dimensional) data colored by its own partition.

    Parameters
    ----------
    Xs : list of array-like of shape (n_samples, 2)
    model : fitted :class:`~fuzzyclust.collaborative.CollaborativeFCM`
    axes : sequence of matplotlib Axes, optional
        One per site.
    independent : bool, default=False
        Draw the partitions of the independent phase instead of the collaborative ones.
    **kwargs
        Passed to :func:`fuzzyclust.plotting.plot_clusters`.

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    P = len(Xs)
    if axes is None:
        _, axes = plt.subplots(1, P, figsize=(4.6 * P, 4.0))
    axes = np.atleast_1d(axes)
    Us = model.independent_memberships_ if independent else model.memberships_
    Vs = model.independent_centers_ if independent else model.cluster_centers_
    kwargs.setdefault("legend", False)
    for p in range(P):
        X = np.asarray(Xs[p])
        plot_clusters(X[:, :2], Us[p].argmax(axis=1), Vs[p][:, :2], ax=axes[p], **kwargs)
        axes[p].set_title(f"site {p + 1}" + (" (independent)" if independent else " (collaborative)"))
    return axes


def plot_collaboration_trajectory(alphas, models, ax=None, site_pairs=None, metric="fuzzy_rand"):
    r"""Agreement between the sites as a function of the collaboration weight :math:`\alpha`.

    Parameters
    ----------
    alphas : sequence of float
    models : sequence of fitted :class:`~fuzzyclust.collaborative.CollaborativeFCM`
        One per value of ``alphas``.
    ax : matplotlib Axes, optional
    site_pairs : sequence of (int, int), optional
        Pairs of sites to draw; default all pairs.
    metric : {"fuzzy_rand", "distance"}, default="fuzzy_rand"
        Fuzzy Rand index between the site partitions (``site_agreement``)
        or the squared partition distance (``partition_distances_``).

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.8))
    P = models[0].n_sites_
    if site_pairs is None:
        site_pairs = [(a, b) for a in range(P) for b in range(a + 1, P)]
    for j, (a, b) in enumerate(site_pairs):
        if metric == "fuzzy_rand":
            vals = [m.site_agreement()[a, b] for m in models]
            ylabel = "fuzzy Rand index between sites"
        elif metric == "distance":
            vals = [m.partition_distances_[a, b] for m in models]
            ylabel = r"partition distance $\frac{1}{N}\sum(\mu[ii]-\mu[jj])^2$"
        else:
            raise ValueError("metric must be 'fuzzy_rand' or 'distance'.")
        ax.plot(alphas, vals, "o-", color=PALETTE[j % len(PALETTE)], label=f"sites {a + 1} and {b + 1}")
    ax.set_xlabel(r"collaboration weight $\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.9")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(site_pairs) > 1:
        ax.legend(frameon=False, fontsize=8)
    return ax
