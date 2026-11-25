# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Matplotlib helpers for rule bases, local models, fuzzy partitions and hinge trees."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..plotting import PALETTE, _ax, _grid
from .partition import membership_1d

__all__ = [
    "plot_membership_functions",
    "plot_local_models",
    "plot_fuzzy_partition",
    "plot_hinge_tree_surface",
    "plot_decision_regions",
]


def _antecedents(model):
    """(centers, sigmas, feature indices, weights) of any rule-based model of this module."""
    if hasattr(model, "sigmas_"):  # TakagiSugeno
        idx = getattr(model, "antecedent_features_", np.arange(model.centers_.shape[1]))
        return model.centers_, model.sigmas_, np.asarray(idx), None
    if hasattr(model, "input_variances_"):  # ModifiedGG, SupervisedFuzzyClassifier
        idx = getattr(model, "_ante_idx", np.arange(model.cluster_centers_.shape[1]))
        return model.cluster_centers_, np.sqrt(model.input_variances_), np.asarray(idx), None
    raise TypeError("model must be a fitted TakagiSugeno, ModifiedGG or SupervisedFuzzyClassifier.")


def plot_membership_functions(model, X=None, feature_names=None, axes=None, n_points: int = 300, ranges=None):
    r"""Gaussian antecedent membership functions :math:`A_{ij}(x_j)` of every rule, one panel per input.

    Parameters
    ----------
    model : fitted TakagiSugeno, ModifiedGG or SupervisedFuzzyClassifier
    X : array-like of shape (n_samples, n_features), optional
        Data used to set the plotting range of every feature (otherwise
        centers :math:`\pm 3\sigma`), and drawn as a rug.
    feature_names : sequence of str, optional
    axes : array of matplotlib Axes, optional
        One axis per antecedent feature.
    n_points : int, default=300
    ranges : sequence of (lo, hi), optional
        Explicit plotting ranges per antecedent feature.

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    V, S, idx, _ = _antecedents(model)
    c, na = V.shape
    n_in = getattr(model, "n_features_in_", na)
    if feature_names is None:
        feature_names = list(getattr(model, "feature_names_in_", [f"x{j + 1}" for j in range(n_in)]))
    if axes is None:
        _, axes = plt.subplots(na, 1, figsize=(6.5, 1.9 * na), squeeze=False)
        axes = axes[:, 0]
    axes = np.atleast_1d(axes)
    Xa = None if X is None else np.asarray(X, dtype=float)[:, idx]
    for a in range(na):
        ax = _ax(axes[a])
        if ranges is not None:
            lo, hi = ranges[a]
        elif Xa is not None:
            lo, hi = Xa[:, a].min(), Xa[:, a].max()
        else:
            lo, hi = (V[:, a] - 3 * S[:, a]).min(), (V[:, a] + 3 * S[:, a]).max()
        grid = np.linspace(lo, hi, n_points)
        A = membership_1d(grid, V[:, a], S[:, a], "gaussian")
        for i in range(c):
            ax.plot(grid, A[:, i], color=PALETTE[i % len(PALETTE)], lw=1.8, label=f"rule {i + 1}")
        if Xa is not None:
            ax.plot(Xa[:, a], np.full(Xa.shape[0], -0.04), "|", color="0.4", ms=6, alpha=0.5)
        ax.set_ylim(-0.08, 1.05)
        ax.set_xlabel(feature_names[idx[a]])
        ax.set_ylabel("membership")
    axes[0].legend(fontsize=8, ncol=min(c, 4), loc="upper right")
    return axes


def plot_local_models(model, X, y, ax=None, n_points: int = 300, show_firing: bool = True):
    r"""One-input Takagi–Sugeno model: data, local linear models, model output and firing strengths.

    Parameters
    ----------
    model : fitted TakagiSugeno or ModifiedGG with one input
    X : array-like of shape (n_samples, 1)
    y : array-like of shape (n_samples,)
    ax : matplotlib Axes, optional
    show_firing : bool, default=True
        Draw the normalized firing strengths on a twin axis.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 1:
        raise ValueError("plot_local_models is defined for models with one input.")
    ax = _ax(ax)
    grid = np.linspace(X.min(), X.max(), n_points)[:, None]
    local = model.local_predict(grid)
    beta = model.firing_strengths(grid)
    c = local.shape[1]
    ax.scatter(X[:, 0], y, s=10, color="0.45", alpha=0.6, label="data", zorder=2)
    for i in range(c):
        strong = beta[:, i] > 0.05
        ax.plot(grid[:, 0], local[:, i], color=PALETTE[i % len(PALETTE)], lw=1.0, alpha=0.35)
        ax.plot(
            grid[strong, 0],
            local[strong, i],
            color=PALETTE[i % len(PALETTE)],
            lw=2.0,
            label=f"local model {i + 1}",
        )
    ax.plot(grid[:, 0], model.predict(grid), color="black", lw=2.0, label="TS output", zorder=3)
    lo, hi = np.min(y), np.max(y)
    pad = 0.15 * (hi - lo + 1e-12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    if show_firing:
        ax2 = ax.twinx()
        for i in range(c):
            ax2.fill_between(grid[:, 0], 0, beta[:, i], color=PALETTE[i % len(PALETTE)], alpha=0.10, lw=0)
        ax2.set_ylim(0, 4.0)
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_ylabel(r"$\beta_i(x)$ (normalized)", fontsize=8)
    return ax


def plot_fuzzy_partition(
    discretizer, feature: int, X=None, y=None, ax=None, feature_name=None, n_points: int = 300
):
    r"""Fuzzy sets of one feature of a fitted :class:`~fuzzyclust.models.FuzzyPartitionDiscretizer`.

    If ``X`` and ``y`` are given, the class-conditional data are drawn as a
    rug below the sets so that the alignment of the cut points with the class
    structure can be judged.

    Returns
    -------
    ax : matplotlib Axes
    """
    ax = _ax(ax)
    p = discretizer.partitions_[feature]
    lo, hi = p["range"]
    if X is not None:
        col = np.asarray(X, dtype=float)[:, feature]
        lo, hi = min(lo, col.min()), max(hi, col.max())
    grid = np.linspace(lo, hi, n_points)
    A = membership_1d(grid, p["centers"], p["sigmas"], p["shape"])
    for s in range(A.shape[1]):
        ax.plot(grid, A[:, s], color=PALETTE[s % len(PALETTE)], lw=1.8, label=f"set {s + 1}")
    if X is not None and y is not None:
        y = np.asarray(y)
        for k, cls in enumerate(np.unique(y)):
            ax.plot(
                col[y == cls],
                np.full((y == cls).sum(), -0.05 - 0.05 * k),
                "|",
                color=PALETTE[k % len(PALETTE)],
                ms=6,
                alpha=0.6,
            )
        ax.set_ylim(-0.05 - 0.05 * (k + 1), 1.05)
    else:
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(feature_name if feature_name is not None else f"x{feature + 1}")
    ax.set_ylabel("membership")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_decision_regions(
    clf,
    X,
    y=None,
    ax=None,
    resolution: int = 200,
    show_rules: bool = True,
    proba_class=None,
    levels: int = 10,
):
    r"""Decision regions of a two-input classifier, optionally with the rule centers and antecedent ellipses.

    The background shows the predicted class (``proba_class=None``) or the
    posterior probability :math:`P(c_j\mid\mathbf{x})` of the class with index
    ``proba_class`` as filled contours. For a fitted
    :class:`~fuzzyclust.models.SupervisedFuzzyClassifier` the centers of the
    rules and their :math:`\pm\sigma_{ij}` extents are drawn as ellipses
    (``show_rules``), so that one sees which rule covers which part of a class.

    Parameters
    ----------
    clf : fitted classifier with ``predict`` (and ``predict_proba`` if ``proba_class`` is used)
    X : array-like of shape (n_samples, 2)
    y : array-like of shape (n_samples,), optional
        Class labels of the samples (colored scatter).
    ax : matplotlib Axes, optional
    resolution : int, default=200
    show_rules : bool, default=True
    proba_class : int, optional
        Index into ``clf.classes_`` of the class whose posterior is shown.
    levels : int, default=10

    Returns
    -------
    ax : matplotlib Axes
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse

    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError("plot_decision_regions needs two input features.")
    ax = _ax(ax)
    gx, gy, G = _grid(X, resolution=resolution)
    classes = np.asarray(getattr(clf, "classes_", np.unique(y) if y is not None else [0, 1]))
    if proba_class is None:
        Z = clf.predict(G)
        codes = np.searchsorted(classes, Z).reshape(gx.shape)
        cmap = ListedColormap([PALETTE[i % len(PALETTE)] for i in range(len(classes))])
        ax.contourf(gx, gy, codes, levels=np.arange(len(classes) + 1) - 0.5, cmap=cmap, alpha=0.18)
        ax.contour(gx, gy, codes, levels=np.arange(len(classes)) + 0.5, colors="black", linewidths=1.0)
    else:
        P = clf.predict_proba(G)[:, proba_class].reshape(gx.shape)
        cs = ax.contourf(gx, gy, P, levels=np.linspace(0, 1, levels + 1), cmap="RdBu_r", alpha=0.75)
        plt.colorbar(cs, ax=ax, label=f"$P({classes[proba_class]}\\mid x)$", fraction=0.046, pad=0.03)
        ax.contour(gx, gy, P, levels=[0.5], colors="black", linewidths=1.2)
    if y is not None:
        y = np.asarray(y)
        for k, cls in enumerate(classes):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=12, color=PALETTE[k % len(PALETTE)], edgecolor="white",
                       linewidth=0.3, label=f"class {cls}", zorder=3)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    if show_rules and hasattr(clf, "input_variances_"):
        V, S = clf.cluster_centers_, np.sqrt(clf.input_variances_)
        for i in range(V.shape[0]):
            ax.add_patch(Ellipse(V[i], 2 * S[i, 0], 2 * S[i, 1], fill=False, lw=1.6, ls="--", color="black",
                                 zorder=4))
            ax.annotate(f"$r_{i + 1}$", V[i], textcoords="offset points", xytext=(4, 4), fontsize=9,
                        zorder=5)
            ax.plot(V[i, 0], V[i, 1], "k+", ms=9, mew=1.8, zorder=5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    return ax


def plot_hinge_tree_surface(
    tree, X, y=None, ax=None, resolution: int = 120, levels: int = 14, show_hinges: bool = True
):
    r"""Prediction surface (filled contours) of a two-input hinging-hyperplane tree with its hinges.

    The hinge of every internal node, :math:`\mathbf{x}_e^T(\boldsymbol\theta_1-\boldsymbol\theta_2)=0`,
    is drawn as a line restricted to the region of the node, so the oblique
    recursive partition of the input space becomes visible.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError("plot_hinge_tree_surface needs two input features.")
    ax = _ax(ax)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    gx, gy = np.meshgrid(
        np.linspace(lo[0] - 0.03 * span[0], hi[0] + 0.03 * span[0], resolution),
        np.linspace(lo[1] - 0.03 * span[1], hi[1] + 0.03 * span[1], resolution),
    )
    G = np.c_[gx.ravel(), gy.ravel()]
    Z = tree.predict(G).reshape(gx.shape)
    cs = ax.contourf(gx, gy, Z, levels=levels, cmap="viridis", alpha=0.85)
    plt.colorbar(cs, ax=ax, label=r"$\hat y$", fraction=0.046, pad=0.03)
    if y is not None:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=8, edgecolor="white", linewidth=0.3)
    if show_hinges:
        Ge = np.c_[G, np.ones(G.shape[0])]

        def rec(node, idx):
            if node.kind == "linear":
                return
            h = Ge[idx] @ (node.theta[0] - node.theta[1])
            field = np.full(G.shape[0], np.nan)
            field[idx] = h
            lw = 2.2 if not node.is_leaf else 1.0
            ax.contour(
                gx,
                gy,
                field.reshape(gx.shape),
                levels=[0.0],
                colors=["white" if node.is_leaf else "black"],
                linewidths=lw,
                linestyles="-" if not node.is_leaf else "--",
            )
            if not node.is_leaf:
                active1 = h >= 0 if node.kind == "max" else h <= 0
                rec(node.left, idx[active1])
                rec(node.right, idx[~active1])

        rec(tree.tree_, np.arange(G.shape[0]))
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    return ax
