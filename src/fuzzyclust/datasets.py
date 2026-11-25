# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Synthetic data generators and small bundled data sets.

Every generator returns ``(X, y)`` with ``X`` of shape ``(n_samples, n_features)``
and ``y`` the generating component index, and accepts ``random_state`` for
reproducibility. The generators are designed to expose the strengths and
weaknesses of the individual clustering algorithms (spherical vs. elongated
vs. differently dense clusters, curved manifolds, lines, rings, noise).
"""

from __future__ import annotations

from importlib import resources

import numpy as np
from sklearn.utils import Bunch, check_random_state

__all__ = [
    "make_fuzzy_blobs",
    "make_ellipses",
    "make_different_densities",
    "make_rings",
    "make_lines",
    "make_noisy_blobs",
    "make_curved_manifold",
    "make_two_moons",
    "load_motorcycle",
    "load_iris",
    "load_wine",
    "load_breast_cancer",
]


def _separated_centers(rng, c, n, box, min_dist, max_tries=2000):
    centers = [rng.rand(n) * box]
    tries = 0
    while len(centers) < c and tries < max_tries:
        cand = rng.rand(n) * box
        tries += 1
        if all(np.linalg.norm(cand - v) >= min_dist for v in centers):
            centers.append(cand)
    while len(centers) < c:  # give up on separation if the box is too crowded
        centers.append(rng.rand(n) * box)
    return np.asarray(centers)


def make_fuzzy_blobs(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_features: int = 2,
    spread: float = 1.0 / 1.5,
    box: float = 10.0,
    random_state=None,
):
    r"""Isotropic Gaussian clusters with random centers (MATLAB ``nDexample``).

    Centers are drawn uniformly in :math:`[0, \text{box}]^n` (rejecting draws
    closer than :math:`4\sigma` to an existing center, so that clusters are
    well separated); every cluster
    has :math:`N_i` samples :math:`\mathbf{x}_k\sim\mathcal{N}(\mathbf{v}_i,\sigma^2\mathbf{I})`
    with :math:`\sigma=` ``spread``. The ideal data set for FCM.

    Parameters
    ----------
    n_clusters : int, default=3
    n_samples_per_cluster : int, default=100
    n_features : int, default=2
    spread : float, default=1/1.5
        Standard deviation of the clusters.
    box : float, default=10.0
        Side length of the cube the centers are drawn from.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_clusters * n_samples_per_cluster, n_features)
    y : ndarray of shape (n_samples,)
        Index of the generating cluster.
    """
    rng = check_random_state(random_state)
    centers = _separated_centers(rng, n_clusters, n_features, box, min_dist=4.0 * spread)
    X = np.vstack(
        [rng.randn(n_samples_per_cluster, n_features) * spread + centers[i] for i in range(n_clusters)]
    )
    y = np.repeat(np.arange(n_clusters), n_samples_per_cluster)
    return X, y


def make_ellipses(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    axes=(3.0, 0.4),
    separation: float = 2.5,
    random_state=None,
):
    r"""Elongated 2-D Gaussian clusters with different orientations.

    Each cluster is a Gaussian with principal standard deviations ``axes``
    rotated by a random angle; the centers lie on a circle such that the
    distance between neighboring centers is ``separation`` times the major
    standard deviation. With the default separation the clusters are close
    enough for FCM (spherical norm) to split or merge them, while GK recovers
    them thanks to the cluster-specific norm.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    """
    rng = check_random_state(random_state)
    major = float(max(axes))
    radius = 0.5 * separation * major / np.sin(np.pi / max(n_clusters, 2))
    phase = rng.rand() * 2 * np.pi
    Xs, ys = [], []
    for i in range(n_clusters):
        ang = phase + 2 * np.pi * i / n_clusters
        center = radius * np.array([np.cos(ang), np.sin(ang)]) + rng.randn(2) * 0.3 * major
        theta = rng.rand() * np.pi
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Z = rng.randn(n_samples_per_cluster, 2) * np.asarray(axes)
        Xs.append(Z @ R.T + center)
        ys.append(np.full(n_samples_per_cluster, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_different_densities(
    sizes=(300, 60, 60),
    spreads=(1.5, 0.3, 0.6),
    random_state=None,
):
    r"""Clusters of very different size (density) and spread.

    A large diffuse cluster next to small dense ones: FCM and GK (fixed cluster
    volumes) tend to steal points from the big cluster, the Gath–Geva
    algorithm (volume and prior adapt) recovers the structure.

    Parameters
    ----------
    sizes : tuple of int
        Number of samples per cluster.
    spreads : tuple of float
        Standard deviation per cluster.

    Returns
    -------
    X : ndarray of shape (sum(sizes), 2)
    y : ndarray
    """
    rng = check_random_state(random_state)
    c = len(sizes)
    centers = np.array([[0.0, 0.0], [4.0, 3.0], [4.5, -2.5], [-3.0, 4.0], [-4.0, -3.0]])[:c]
    X = np.vstack([rng.randn(sizes[i], 2) * spreads[i] + centers[i] for i in range(c)])
    y = np.repeat(np.arange(c), sizes)
    return X, y


def make_rings(
    n_samples_per_ring: int = 150,
    radii=(1.0, 3.0),
    noise: float = 0.1,
    random_state=None,
):
    r"""Concentric rings (hollow clusters) in 2-D.

    Prototype-based algorithms with point prototypes cannot separate them; the
    graph-based (MST, geodesic) and shell (fuzzy c-shells) algorithms can.

    Returns
    -------
    X : ndarray of shape (n_rings * n_samples_per_ring, 2)
    y : ndarray
    """
    rng = check_random_state(random_state)
    Xs, ys = [], []
    for i, r in enumerate(radii):
        phi = rng.rand(n_samples_per_ring) * 2 * np.pi
        rr = r + rng.randn(n_samples_per_ring) * noise
        Xs.append(np.c_[rr * np.cos(phi), rr * np.sin(phi)])
        ys.append(np.full(n_samples_per_ring, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_lines(
    n_lines: int = 3,
    n_samples_per_line: int = 80,
    noise: float = 0.15,
    random_state=None,
):
    r"""Noisy line segments in 2-D (linear sub-space prototypes).

    Suitable for fuzzy c-varieties / c-regression models and for GK, whose
    smallest covariance eigenvalue direction is the normal of the line.

    Returns
    -------
    X : ndarray of shape (n_lines * n_samples_per_line, 2)
    y : ndarray
    """
    rng = check_random_state(random_state)
    Xs, ys = [], []
    for i in range(n_lines):
        a = rng.uniform(-2.0, 2.0)
        b = rng.uniform(-5.0, 5.0)
        x = rng.uniform(-4.0, 4.0, n_samples_per_line)
        yv = a * x + b + rng.randn(n_samples_per_line) * noise
        Xs.append(np.c_[x, yv])
        ys.append(np.full(n_samples_per_line, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_noisy_blobs(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_outliers: int = 30,
    spread: float = 0.6,
    random_state=None,
):
    r"""Gaussian clusters plus uniformly distributed outliers.

    Outliers get label ``-1``. Illustrates the robustness of the
    possibilistic and noise-clustering variants compared with FCM.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray (``-1`` for outliers)
    """
    rng = check_random_state(random_state)
    X, y = make_fuzzy_blobs(n_clusters, n_samples_per_cluster, 2, spread, 8.0, rng)
    lo, hi = X.min(axis=0) - 3.0, X.max(axis=0) + 3.0
    O = rng.uniform(lo, hi, size=(n_outliers, 2))
    return np.vstack([X, O]), np.concatenate([y, np.full(n_outliers, -1)])


def make_curved_manifold(n_samples: int = 400, noise: float = 0.05, random_state=None):
    r"""Points along two interleaved spiral arms (nonconvex clusters).

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray
    """
    rng = check_random_state(random_state)
    n = n_samples // 2
    t = np.sqrt(rng.rand(n)) * 2.5 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)] + rng.randn(n, 2) * noise * t[:, None]
    t = np.sqrt(rng.rand(n_samples - n)) * 2.5 * np.pi
    b = np.c_[-t * np.cos(t), -t * np.sin(t)] + rng.randn(n_samples - n, 2) * noise * t[:, None]
    return np.vstack([a, b]), np.concatenate([np.zeros(n, int), np.ones(n_samples - n, int)])


def make_two_moons(n_samples: int = 300, noise: float = 0.08, random_state=None):
    """Two interleaving half circles (thin wrapper around scikit-learn)."""
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


# ---------------------------------------------------------------------- loaders
def load_motorcycle() -> Bunch:
    r"""Motorcycle impact data (Silverman, 1985) used throughout the MATLAB toolbox.

    133 measurements of head acceleration (g) of a crash-test dummy against
    time after impact (ms), plus a strain-gauge index in the third column.
    The time–acceleration pairs form a curved, non-uniformly dense cloud that
    is a classical benchmark for local models and time-aligned clustering.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        ``data`` (133, 2) with columns ``time_ms``, ``acceleration_g``;
        ``strain`` (133,), ``feature_names``, ``DESCR``.

    References
    ----------
    .. [1] B. W. Silverman, "Some aspects of the spline smoothing approach to
           non-parametric regression curve fitting", J. R. Stat. Soc. B, 1985.
    """
    with resources.files("fuzzyclust.data").joinpath("motorcycle.txt").open("r") as fh:
        raw = np.loadtxt(fh)
    return Bunch(
        data=raw[:, :2],
        strain=raw[:, 2].astype(int),
        feature_names=["time_ms", "acceleration_g"],
        DESCR=load_motorcycle.__doc__,
    )


def load_iris() -> Bunch:
    """Fisher's iris data (150 x 4, 3 classes) via scikit-learn."""
    from sklearn.datasets import load_iris as _load

    return _load()


def load_wine() -> Bunch:
    """Wine recognition data (178 x 13, 3 classes) via scikit-learn."""
    from sklearn.datasets import load_wine as _load

    return _load()


def load_breast_cancer() -> Bunch:
    """Wisconsin breast cancer data (569 x 30, 2 classes) via scikit-learn."""
    from sklearn.datasets import load_breast_cancer as _load

    return _load()
