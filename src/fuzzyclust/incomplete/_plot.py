# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for clustering results on incomplete data."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE, plot_clusters

__all__ = ["plot_incomplete_clusters", "plot_imputation"]


def plot_incomplete_clusters(X_completed, labels, mask, centers=None, ax=None, legend=True, **kwargs):
    """Scatter plot of a clustering of incomplete 2-D data at the completed coordinates.

    Complete samples are drawn as filled dots, samples with a missing value
    as hollow markers at their completed (imputed) coordinates; a sample
    whose feature ``j`` was missing therefore sits at the imputed position
    along axis ``j``.

    Parameters
    ----------
    X_completed : array-like of shape (n_samples, 2)
        Completed data (e.g. ``model.X_completed_``).
    labels : array-like of shape (n_samples,)
        Hard labels.
    mask : array-like of bool of shape (n_samples, 2)
        ``True`` where a value was missing (``model.missing_mask_``).
    centers : array-like of shape (n_clusters, 2), optional
    ax : matplotlib Axes, optional
    legend : bool, default=True
    **kwargs
        Passed to :func:`fuzzyclust.plotting.plot_clusters` for the complete samples.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X_completed, dtype=float)
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    incomplete = mask.any(axis=1)
    kwargs.setdefault("s", 14)
    ax = plot_clusters(X[~incomplete], labels[~incomplete], centers=centers, ax=ax, legend=False, **kwargs)
    for i in np.unique(labels[incomplete]):
        sel = incomplete & (labels == i)
        ax.scatter(
            X[sel, 0], X[sel, 1], s=28, facecolors="none", edgecolors=PALETTE[int(i) % len(PALETTE)],
            linewidths=1.0, label=f"cluster {int(i) + 1}, incomplete",
        )
    if legend:
        handles, names = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, names, frameon=False, fontsize=7, loc="best")
    return ax


def plot_imputation(X_true, X_completed, mask, ax=None, feature_names=None):
    """Imputed against true values of the missing entries, one color per feature.

    Parameters
    ----------
    X_true : array-like of shape (n_samples, n_features)
        The complete data.
    X_completed : array-like of shape (n_samples, n_features)
        The completed data (``model.X_completed_``).
    mask : array-like of bool of shape (n_samples, n_features)
        ``True`` where a value was missing.
    ax : matplotlib Axes, optional
    feature_names : sequence of str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    X_true = np.asarray(X_true, dtype=float)
    X_completed = np.asarray(X_completed, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    n = X_true.shape[1]
    names = [f"feature {j + 1}" for j in range(n)] if feature_names is None else list(feature_names)
    lo, hi = np.inf, -np.inf
    for j in range(n):
        sel = mask[:, j]
        if not sel.any():
            continue
        t, e = X_true[sel, j], X_completed[sel, j]
        rmse = float(np.sqrt(np.mean((t - e) ** 2)))
        ax.scatter(
            t, e, s=14, color=PALETTE[j % len(PALETTE)], alpha=0.8, label=f"{names[j]} (RMSE {rmse:.2f})"
        )
        lo, hi = min(lo, t.min(), e.min()), max(hi, t.max(), e.max())
    if np.isfinite(lo):
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1.0, ls="--")
    ax.set_xlabel("true value")
    ax.set_ylabel("imputed value")
    ax.set_title("Imputation of the missing values")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, color="0.9", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
