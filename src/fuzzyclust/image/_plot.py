# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Plot helpers for image segmentation: image panels, label images and membership maps."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ..plotting import PALETTE

__all__ = ["plot_image", "plot_label_image", "plot_membership_maps", "plot_segmentation"]


def _ax(ax, figsize=(3.6, 3.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_image(image, ax=None, title=None, cmap="gray", vmin=None, vmax=None):
    """Show a gray-level or RGB image without axes ticks.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, 3)
    ax : matplotlib Axes, optional
    title : str, optional
    cmap : str, default="gray"
    vmin, vmax : float, optional
        Gray-level range (default: the range of the image).

    Returns
    -------
    ax : matplotlib Axes
    """
    img = np.asarray(image, dtype=float)
    ax = _ax(ax)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 3:
        ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_label_image(labels, ax=None, title=None, n_clusters=None, order=None):
    """Show a label image with the fixed cluster palette (cluster *i* in color *i*).

    Parameters
    ----------
    labels : ndarray of shape (H, W)
    ax : matplotlib Axes, optional
    title : str, optional
    n_clusters : int, optional
        Number of labels (default: ``labels.max() + 1``).
    order : array-like of int, optional
        Permutation applied to the labels before drawing, e.g. the
        matching of :func:`~fuzzyclust.image.match_labels`, so that the
        same region has the same color in several panels.

    Returns
    -------
    ax : matplotlib Axes
    """
    lab = np.asarray(labels, dtype=int)
    if order is not None:
        lab = np.asarray(order, dtype=int)[lab]
    c = int(lab.max()) + 1 if n_clusters is None else int(n_clusters)
    cmap = ListedColormap([PALETTE[i % len(PALETTE)] for i in range(max(c, 1))])
    ax = _ax(ax)
    ax.imshow(lab, cmap=cmap, vmin=-0.5, vmax=c - 0.5, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_membership_maps(membership, axes=None, cmap="viridis", titles=True):
    """One panel per cluster with its membership image in ``[0, 1]``.

    Parameters
    ----------
    membership : ndarray of shape (H, W, n_clusters)
    axes : sequence of matplotlib Axes of length n_clusters, optional
    cmap : str, default="viridis"
    titles : bool, default=True

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    U = np.asarray(membership, dtype=float)
    c = U.shape[2]
    if axes is None:
        _, axes = plt.subplots(1, c, figsize=(2.6 * c, 2.8))
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for i in range(c):
        ax = _ax(axes[i])
        ax.imshow(U[..., i], cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        if titles:
            ax.set_title(rf"$\mu_{{{i + 1}}}$", fontsize=9)
    return axes


def plot_segmentation(model, image, clean=None, truth=None, title=None, membership=True):
    """Panels original / noisy / labels / membership maps of a fitted :class:`~fuzzyclust.image.SpatialFCM`.

    Parameters
    ----------
    model : fitted :class:`~fuzzyclust.image.SpatialFCM`
    image : ndarray
        The (noisy) image that was segmented.
    clean : ndarray, optional
        The noise-free image, shown first when given.
    truth : ndarray of shape (H, W), optional
        Ground-truth labels; when given, the predicted labels are
        permuted to match it, and the accuracy is printed in the title.
    title : str, optional
        Prefix of the label-panel title (default: the method name).
    membership : bool, default=True
        Append one membership panel per cluster.

    Returns
    -------
    axes : ndarray of matplotlib Axes
    """
    from ._metrics import match_labels, segmentation_accuracy

    labels = model.labels_
    c = model.membership_.shape[2]
    order = None
    acc = None
    if truth is not None:
        order = match_labels(truth, labels)
        acc = segmentation_accuracy(truth, labels)
    panels = (clean is not None) + 1 + (truth is not None) + 1 + (c if membership else 0)
    fig, axes = plt.subplots(1, panels, figsize=(2.5 * panels, 2.8))
    axes = np.atleast_1d(axes)
    j = 0
    vmin, vmax = float(np.min(image)), float(np.max(image))
    if clean is not None:
        plot_image(clean, ax=axes[j], title="original", vmin=vmin, vmax=vmax)
        j += 1
    plot_image(image, ax=axes[j], title="noisy", vmin=vmin, vmax=vmax)
    j += 1
    if truth is not None:
        plot_label_image(truth, ax=axes[j], title="ground truth", n_clusters=max(c, int(truth.max()) + 1))
        j += 1
    name = title if title is not None else model.method.upper().replace("_", "\\_")
    lab_title = f"{name}" + (f" (acc. {acc:.3f})" if acc is not None else "")
    plot_label_image(labels, ax=axes[j], title=lab_title, n_clusters=c, order=order)
    j += 1
    if membership:
        U = model.membership_
        if order is not None and (order >= 0).all() and order.max() < c and len(np.unique(order)) == c:
            inv = np.empty_like(order)
            inv[order] = np.arange(c)  # membership panel i shows the cluster matched to true label i
            U = U[..., inv]
        plot_membership_maps(U, axes=axes[j : j + c])
    fig.tight_layout()
    return axes
