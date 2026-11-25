# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Matplotlib helpers for inspecting clustering results.

All functions accept an optional ``ax`` and return the :class:`~matplotlib.axes.Axes`
they drew on; they never call ``plt.show()``. Colors follow a fixed,
color-blind-safe categorical order (Okabe–Ito palette); cluster :math:`i` always
gets color ``PALETTE[i % len(PALETTE)]``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

__all__ = [
    "PALETTE",
    "plot_clusters",
    "plot_membership_contours",
    "plot_covariance_ellipses",
    "plot_validity_scan",
    "plot_objective",
    "plot_membership_profiles",
]

#: Okabe–Ito color-blind-safe palette, fixed assignment order.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def _ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _grid(X, resolution=200, pad=0.05):
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def plot_clusters(X, labels=None, centers=None, ax=None, alpha=0.8, s=14, legend=True, **kwargs):
    """Scatter plot of 2-D data colored by (hard) cluster label with prototypes.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    labels : array-like of shape (n_samples,), optional
        Hard labels; ``-1`` is drawn as gray "noise".
    centers : array-like of shape (n_clusters, 2), optional
    ax : matplotlib Axes, optional
    """
    X = np.asarray(X)
    ax = _ax(ax)
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=s, color="0.4", alpha=alpha, **kwargs)
    else:
        labels = np.asarray(labels)
        for i in np.unique(labels):
            mask = labels == i
            col = "0.6" if i < 0 else PALETTE[int(i) % len(PALETTE)]
            name = "noise" if i < 0 else f"cluster {int(i) + 1}"
            ax.scatter(X[mask, 0], X[mask, 1], s=s, color=col, alpha=alpha, label=name, **kwargs)
    if centers is not None:
        centers = np.asarray(centers)
        ax.scatter(
            centers[:, 0], centers[:, 1], marker="X", s=140, color="black",
            edgecolor="white", linewidth=1.2, zorder=5, label="prototypes",
        )
    if legend and labels is not None:
        ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def plot_membership_contours(estimator, X, ax=None, levels=8, resolution=200, cmap="Greys", labels=True):
    r"""Contour map of the maximal membership :math:`\max_i \mu_i(\mathbf{x})` over the plane.

    Reproduces the contour maps of the MATLAB ``clusteval`` function: the
    level curves show how each algorithm's distance norm shapes the clusters
    (circles for FCM, ellipses for GK, Gaussian-like bumps for GG).

    Parameters
    ----------
    estimator : fitted fuzzyclust estimator with ``predict_proba``
    X : array-like of shape (n_samples, 2)
        Data (used for the plotting range and the scatter).
    """
    X = np.asarray(X)
    ax = _ax(ax)
    gx, gy, G = _grid(X, resolution)
    U = estimator.predict_proba(G)
    Z = np.round(U.max(axis=1).reshape(gx.shape), 9)
    if np.isscalar(levels):
        # interior levels only: a level at the exact maximum would trace the
        # rounding noise of a plateau of membership one
        from matplotlib.ticker import MaxNLocator

        lv = MaxNLocator(int(levels) + 1).tick_values(Z.min(), Z.max())
        levels = lv[(lv > Z.min()) & (lv < Z.max() - 1e-6)]
    cs = ax.contour(gx, gy, Z, levels=levels, cmap=cmap, linewidths=0.9, vmin=0.3, vmax=1.1)
    if labels:
        ax.clabel(cs, fmt="%.2f", fontsize=7)
    hard = estimator.predict(X)
    plot_clusters(X, hard, getattr(estimator, "cluster_centers_", None), ax=ax, legend=False)
    return ax


def plot_covariance_ellipses(centers, covariances, ax=None, n_std=2.0, **kwargs):
    r"""Draw the :math:`n_\text{std}`-sigma ellipses of (fuzzy) covariance matrices.

    Parameters
    ----------
    centers : array-like of shape (n_clusters, 2)
    covariances : array-like of shape (n_clusters, 2, 2)
    """
    ax = _ax(ax)
    for i, (v, F) in enumerate(zip(np.asarray(centers), np.asarray(covariances), strict=False)):
        w, Q = np.linalg.eigh(F)
        angle = np.degrees(np.arctan2(Q[1, -1], Q[0, -1]))
        width, height = 2 * n_std * np.sqrt(np.maximum(w[::-1], 0))
        e = Ellipse(v, width, height, angle=angle, fill=False, lw=1.8,
                    edgecolor=PALETTE[i % len(PALETTE)], **kwargs)
        ax.add_patch(e)
    return ax


def plot_validity_scan(table, ax=None, normalize=True):
    """Plot validity indices versus the number of clusters.

    Takes the output of :func:`~fuzzyclust.validity.scan_n_clusters`.

    Parameters
    ----------
    table : pandas.DataFrame
    normalize : bool, default=True
        Min–max scale every index to [0, 1] so that they fit one axis; the
        legend states whether the optimum is a maximum or a minimum.
    """
    from .validity import INDEX_DIRECTION

    ax = _ax(ax)
    for j, col in enumerate(table.columns):
        y = table[col].to_numpy(dtype=float)
        if normalize:
            rng_ = np.nanmax(y) - np.nanmin(y)
            y = (y - np.nanmin(y)) / (rng_ if rng_ > 0 else 1.0)
        direction = INDEX_DIRECTION.get(col, "")
        ax.plot(table.index, y, marker="o", ms=5, lw=1.8, color=PALETTE[j % len(PALETTE)],
                label=f"{col} ({direction})")
    ax.set_xlabel("number of clusters $c$")
    ax.set_ylabel("normalized index value" if normalize else "index value")
    ax.set_xticks(list(table.index))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return ax


def plot_objective(estimators, labels=None, ax=None):
    """Objective :math:`J` versus iteration for one or several fitted estimators."""
    ax = _ax(ax)
    if not isinstance(estimators, (list, tuple)):
        estimators = [estimators]
    for j, est in enumerate(estimators):
        name = labels[j] if labels else type(est).__name__
        ax.plot(np.arange(1, est.n_iter_ + 1), est.objective_history_, lw=1.8,
                color=PALETTE[j % len(PALETTE)], label=name)
    ax.set_xlabel("iteration $l$")
    ax.set_ylabel("objective $J$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_membership_profiles(U, ax=None, x=None, stacked=False):
    r"""Membership degrees :math:`\mu_{ik}` of every cluster along the sample index.

    Useful for time-series / ordered data where the sample order is meaningful.

    Parameters
    ----------
    U : array-like of shape (n_samples, n_clusters)
    x : array-like of shape (n_samples,), optional
        Abscissa (e.g. time); defaults to the sample index.
    stacked : bool, default=False
        Stacked area plot instead of lines.
    """
    U = np.asarray(U)
    ax = _ax(ax)
    x = np.arange(U.shape[0]) if x is None else np.asarray(x)
    cols = [PALETTE[i % len(PALETTE)] for i in range(U.shape[1])]
    if stacked:
        ax.stackplot(x, U.T, colors=cols, labels=[f"cluster {i + 1}" for i in range(U.shape[1])], alpha=0.85)
    else:
        for i in range(U.shape[1]):
            ax.plot(x, U[:, i], color=cols[i], lw=1.6, label=f"cluster {i + 1}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(r"membership $\mu_{ik}$")
    ax.set_xlabel("sample $k$")
    ax.legend(frameon=False, fontsize=8, ncol=min(U.shape[1], 4))
    return ax
