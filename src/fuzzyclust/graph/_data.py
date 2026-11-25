# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the graph-based clustering algorithms."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_chained_blobs", "make_density_blobs", "make_s_curve"]


def make_chained_blobs(
    n_samples_per_blob: int = 120,
    n_chain: int = 25,
    spread: float = 0.5,
    distance: float = 7.0,
    chain_noise: float = 0.08,
    random_state=None,
):
    r"""Two Gaussian blobs connected by a sparse chain of points (the *chaining effect*).

    Single-linkage and plain MST cutting see one cluster, because every
    consecutive pair along the chain is close. The volume-based splitting
    of :class:`~fuzzyclust.graph.MSTClustering` separates the blobs (and
    the chain) because the union has a much larger fuzzy hypervolume than
    its parts.

    Parameters
    ----------
    n_samples_per_blob : int, default=120
    n_chain : int, default=25
        Number of chain points between the blobs.
    spread : float, default=0.5
        Standard deviation of the blobs.
    distance : float, default=7.0
        Distance between the blob centers.
    chain_noise : float, default=0.08
        Standard deviation of the chain points around the straight line.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (2 * n_samples_per_blob + n_chain, 2)
    y : ndarray of shape (n_samples,)
        ``0`` and ``1`` for the blobs, ``2`` for the chain.
    """
    rng = check_random_state(random_state)
    c0 = np.array([0.0, 0.0])
    c1 = np.array([distance, 0.0])
    A = rng.randn(n_samples_per_blob, 2) * spread + c0
    B = rng.randn(n_samples_per_blob, 2) * spread + c1
    t = np.linspace(0.15, 0.85, n_chain)
    chain = c0 + t[:, None] * (c1 - c0) + rng.randn(n_chain, 2) * chain_noise
    X = np.vstack([A, B, chain])
    y = np.concatenate(
        [np.zeros(n_samples_per_blob, int), np.ones(n_samples_per_blob, int), np.full(n_chain, 2)]
    )
    return X, y


def make_density_blobs(
    sizes=(200, 40, 40),
    spreads=(1.2, 0.25, 0.25),
    centers=((0.0, 0.0), (5.0, 0.9), (5.0, -0.9)),
    random_state=None,
):
    r"""A diffuse cluster next to two tight ones (very different densities).

    The longest MST edges inside the diffuse group are as long as the edge
    that separates the two tight groups from each other, so a *global*
    length threshold (Zahn's Criterion 1) either leaves the tight groups
    merged or cuts the tails off the diffuse one; the *local* inconsistency
    criterion (Criterion 2) compares each edge with its neighborhood and
    finds the three groups.

    Parameters
    ----------
    sizes : tuple of int, default=(200, 40, 40)
    spreads : tuple of float, default=(1.2, 0.25, 0.25)
        Standard deviations of the groups.
    centers : tuple of pairs, default=((0, 0), (5, 0.9), (5, -0.9))
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (sum(sizes), 2)
    y : ndarray
    """
    rng = check_random_state(random_state)
    centers = np.asarray(centers, dtype=float)
    if not len(sizes) == len(spreads) == len(centers):
        raise ValueError("sizes, spreads and centers must have the same length.")
    X = np.vstack([rng.randn(sizes[i], 2) * spreads[i] + centers[i] for i in range(len(sizes))])
    y = np.repeat(np.arange(len(sizes)), sizes)
    return X, y


def make_s_curve(n_samples: int = 600, noise: float = 0.0, random_state=None):
    r"""Points on the three-dimensional S-curve, a two-dimensional manifold (wraps scikit-learn).

    The benchmark of the Isomap paper and of the geodesic clustering paper:
    the sheet is bent into an S, so Euclidean neighbors across the fold are
    far apart along the manifold. The continuous position along the curve is
    returned as ``y`` for colouring.

    Parameters
    ----------
    n_samples : int, default=600
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the points.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
    t : ndarray of shape (n_samples,)
        Univariate position of every sample along the main dimension of the manifold.
    """
    from sklearn.datasets import make_s_curve as _make_s_curve

    return _make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
