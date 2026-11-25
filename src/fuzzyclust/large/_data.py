# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the very-large-data algorithms: big blob sets and drifting streams."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_large_blobs", "make_drifting_stream"]


def make_large_blobs(
    n_samples: int = 100_000,
    n_clusters: int = 5,
    n_features: int = 2,
    weights=None,
    spread: float = 1.0 / 1.5,
    box: float = 10.0,
    shuffle: bool = True,
    random_state=None,
):
    r"""Many samples from a few isotropic Gaussian clusters, in random order.

    The centers are drawn as in :func:`fuzzyclust.datasets.make_fuzzy_blobs`
    (uniform in the box, at least :math:`4\sigma` apart); the cluster sizes
    follow ``weights`` (equal by default). The samples are shuffled so
    that consecutive chunks of the array are random samples of the whole,
    which is what the single-pass and online algorithms assume.

    Parameters
    ----------
    n_samples : int, default=100000
    n_clusters : int, default=5
    n_features : int, default=2
    weights : array-like of shape (n_clusters,), optional
        Relative sizes of the clusters (normalized internally).
    spread : float, default=1/1.5
        Standard deviation of the clusters.
    box : float, default=10.0
    shuffle : bool, default=True
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.large import make_large_blobs
    >>> X, y = make_large_blobs(n_samples=1000, n_clusters=4, random_state=0)
    >>> X.shape, int(y.max()) + 1
    ((1000, 2), 4)
    """
    from ..datasets import _separated_centers

    rng = check_random_state(random_state)
    if weights is None:
        w = np.full(n_clusters, 1.0 / n_clusters)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_clusters,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be n_clusters non-negative numbers.")
        w = w / w.sum()
    sizes = np.floor(w * n_samples).astype(int)
    sizes[: n_samples - sizes.sum()] += 1
    centers = _separated_centers(rng, n_clusters, n_features, box, min_dist=4.0 * spread)
    y = np.repeat(np.arange(n_clusters), sizes)
    X = rng.randn(n_samples, n_features) * spread + centers[y]
    if shuffle:
        order = rng.permutation(n_samples)
        X, y = X[order], y[order]
    return X, y


def make_drifting_stream(
    n_chunks: int = 20,
    chunk_size: int = 2000,
    n_clusters: int = 3,
    n_features: int = 2,
    drift: float = 0.15,
    spread: float = 0.6,
    box: float = 8.0,
    random_state=None,
):
    r"""A stream of chunks whose cluster centers drift from chunk to chunk.

    Every chunk :math:`l` holds ``chunk_size`` samples of ``n_clusters``
    isotropic Gaussian clusters whose centers move by a constant drift
    vector per chunk (each cluster its own random direction, step length
    ``drift``). A generator is returned, so that chunks can be consumed
    one by one with :meth:`~fuzzyclust.large.SinglePassFCM.partial_fit`.

    Parameters
    ----------
    n_chunks : int, default=20
    chunk_size : int, default=2000
    n_clusters : int, default=3
    n_features : int, default=2
    drift : float, default=0.15
        Displacement of every center per chunk.
    spread : float, default=0.6
    box : float, default=8.0
    random_state : int, RandomState instance or None

    Yields
    ------
    X : ndarray of shape (chunk_size, n_features)
    y : ndarray of shape (chunk_size,)
    centers : ndarray of shape (n_clusters, n_features)
        The true centers of this chunk.

    Examples
    --------
    >>> from fuzzyclust.large import make_drifting_stream
    >>> chunks = list(make_drifting_stream(n_chunks=3, chunk_size=100, random_state=0))
    >>> len(chunks), chunks[0][0].shape, chunks[-1][2].shape
    (3, (100, 2), (3, 2))
    """
    from ..datasets import _separated_centers

    rng = check_random_state(random_state)
    centers = _separated_centers(rng, n_clusters, n_features, box, min_dist=4.0 * spread)
    direction = rng.randn(n_clusters, n_features)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    step = drift * direction
    for l in range(n_chunks):
        C = centers + l * step
        y = rng.randint(n_clusters, size=chunk_size)
        X = rng.randn(chunk_size, n_features) * spread + C[y]
        yield X, y, C
