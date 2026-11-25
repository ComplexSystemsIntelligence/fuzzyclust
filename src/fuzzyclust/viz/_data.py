# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Synthetic data for the visualization examples (same conventions as :mod:`fuzzyclust.datasets`)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_ellipsoids", "make_spiral_3d", "make_hemisphere"]


def make_ellipsoids(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_features: int = 6,
    elongation: float = 6.0,
    separation: float = 6.0,
    random_state=None,
):
    r"""Elongated Gaussian clusters with random orientations in :math:`n` dimensions.

    Every cluster is :math:`\mathcal{N}(\mathbf{v}_i,\mathbf{F}_i)` with
    :math:`\mathbf{F}_i=\mathbf{Q}_i\mathrm{diag}(\lambda_1,\dots,\lambda_n)\mathbf{Q}_i^T`,
    a random orthogonal :math:`\mathbf{Q}_i` and eigenvalues decreasing
    geometrically from ``elongation``\ :math:`^2` to 1, so that the clusters
    are ellipsoids whose longest axis is ``elongation`` times the shortest
    one. The centers are drawn at random with mutual distance at least
    ``separation``. The data set is the higher-dimensional counterpart of
    :func:`fuzzyclust.datasets.make_ellipses`: FCM confuses the clusters,
    GK recovers them, and the FUZZSAM projection of the GK result shows
    compact spherical groups.

    Returns
    -------
    X : ndarray of shape (n_clusters * n_samples_per_cluster, n_features)
    y : ndarray of shape (n_samples,)
    """
    rng = check_random_state(random_state)
    n = n_features
    centers = [rng.randn(n) * separation]
    while len(centers) < n_clusters:
        cand = rng.randn(n) * separation
        if all(np.linalg.norm(cand - v) >= separation for v in centers):
            centers.append(cand)
    lam = np.geomspace(elongation**2, 1.0, n)
    Xs, ys = [], []
    for i, v in enumerate(centers):
        Q, _ = np.linalg.qr(rng.randn(n, n))
        Z = rng.randn(n_samples_per_cluster, n) * np.sqrt(lam)
        Xs.append(Z @ Q.T + v)
        ys.append(np.full(n_samples_per_cluster, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_spiral_3d(n_samples: int = 300, turns: float = 1.0, noise: float = 0.2, random_state=None):
    r"""Noisy points along a part of a helix in 3-D (Migály et al., first example).

    :math:`\mathbf{x}(t)=[\cos t,\ \sin t,\ t/(2\pi)]^T+\boldsymbol{\varepsilon}`
    for :math:`t\in[0, 2\pi\cdot\text{turns}]`, with Gaussian noise of standard
    deviation ``noise``. A one-dimensional fuzzy SOM should line its
    prototypes up along the curve.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
    t : ndarray of shape (n_samples,)
        Curve parameter (ordering of the points along the spiral).
    """
    rng = check_random_state(random_state)
    t = np.sort(rng.rand(n_samples)) * 2 * np.pi * turns
    X = np.c_[np.cos(t), np.sin(t), t / (2 * np.pi)] + noise * rng.randn(n_samples, 3)
    return X, t


def make_hemisphere(n_samples: int = 900, noise: float = 0.1, random_state=None):
    r"""Noisy points on the upper half of the unit sphere (Migály et al., second example).

    Points are sampled uniformly on a square grid in :math:`(u,v)\in[-1,1]^2`
    and folded onto the hemisphere :math:`z=\sqrt{\max(0, 1-u^2-v^2)}`; the
    returned labels are the :math:`(u, v)` coordinates, i.e. the intrinsic
    two-dimensional parametrization that a 2-D fuzzy SOM should recover.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
    uv : ndarray of shape (n_samples, 2)
        Intrinsic coordinates of every point.
    """
    rng = check_random_state(random_state)
    side = int(np.ceil(np.sqrt(n_samples)))
    g = np.linspace(-0.95, 0.95, side)
    u, v = np.meshgrid(g, g)
    uv = np.c_[u.ravel(), v.ravel()][:n_samples]
    r2 = np.sum(uv**2, axis=1)
    uv = uv[r2 <= 1.0]
    z = np.sqrt(np.maximum(0.0, 1.0 - np.sum(uv**2, axis=1)))
    X = np.c_[uv, z] + noise * rng.randn(uv.shape[0], 3)
    return X, uv
