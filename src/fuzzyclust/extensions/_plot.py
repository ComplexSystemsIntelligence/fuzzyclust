# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for shell and variety prototypes and for typicalities."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_shells", "plot_varieties", "plot_typicality"]


def plot_shells(X, model, ax=None, labels=None, **kwargs):
    """Scatter the data and draw the fitted circles of a :class:`FuzzyCShells` model.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    model : fitted :class:`~fuzzyclust.extensions.FuzzyCShells`
    ax : matplotlib Axes, optional
    labels : array-like, optional
        Labels used for colouring; defaults to ``model.labels_``.
    """
    X = np.asarray(X)
    labels = model.labels_ if labels is None else labels
    ax = plot_clusters(X, labels, ax=ax, legend=False, **kwargs)
    for i, (v, r) in enumerate(zip(model.cluster_centers_, model.radii_, strict=True)):
        ax.add_patch(Circle(v, r, fill=False, lw=2.0, edgecolor=PALETTE[i % len(PALETTE)]))
        ax.plot(v[0], v[1], marker="+", ms=10, mew=1.5, color=PALETTE[i % len(PALETTE)])
    ax.set_aspect("equal")
    return ax


def plot_varieties(X, model, ax=None, labels=None, length=None, **kwargs):
    """Scatter the data and draw the fitted lines of a :class:`FuzzyCVarieties` model (``n_dims=1``).

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    model : fitted :class:`~fuzzyclust.extensions.FuzzyCVarieties`
    ax : matplotlib Axes, optional
    labels : array-like, optional
    length : float, optional
        Half-length of the drawn line segments; defaults to the data range.
    """
    X = np.asarray(X)
    labels = model.labels_ if labels is None else labels
    ax = plot_clusters(X, labels, ax=ax, legend=False, **kwargs)
    if length is None:
        length = 0.6 * np.ptp(X, axis=0).max()
    for i, (v, S) in enumerate(zip(model.cluster_centers_, model.directions_, strict=True)):
        s = S[0]
        seg = np.vstack([v - length * s, v + length * s])
        ax.plot(seg[:, 0], seg[:, 1], lw=2.0, color=PALETTE[i % len(PALETTE)])
        ax.plot(v[0], v[1], marker="X", ms=9, color="black", mec="white")
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    return ax


def plot_typicality(X, T, ax=None, cmap="viridis", s=18, label=None, **kwargs):
    r"""Scatter plot colored by the maximal typicality :math:`\max_i t_{ik}` of every sample.

    Low values single out outliers: samples that are atypical of every
    cluster. Works for any matrix of absolute memberships (PCM
    ``membership_``, PFCM ``typicality_``, or ``1 - noise_membership_``).

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
    T : array-like of shape (n_samples, n_clusters) or (n_samples,)
    label : str, optional
        Color-bar label; defaults to :math:`\max_i t_{ik}`.
    """
    X = np.asarray(X)
    T = np.asarray(T)
    tmax = T.max(axis=1) if T.ndim == 2 else T
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=tmax, cmap=cmap, s=s, vmin=0.0, vmax=1.0, **kwargs)
    ax.figure.colorbar(sc, ax=ax, label=r"$\max_i t_{ik}$" if label is None else label, shrink=0.8)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax
