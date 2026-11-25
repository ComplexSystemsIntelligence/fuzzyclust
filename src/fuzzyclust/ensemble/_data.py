# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the cluster ensemble examples."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_overlapping_blobs"]


def make_overlapping_blobs(
    n_clusters: int = 4,
    n_samples_per_cluster: int = 75,
    n_features: int = 2,
    spread: float = 1.0,
    separation: float = 2.5,
    box: float = 6.0,
    random_state=None,
):
    r"""Gaussian groups that overlap enough for single clustering runs to disagree.

    Centers are drawn in a cube of side ``box`` at least ``separation``
    standard deviations apart (:func:`~fuzzyclust.datasets.make_fuzzy_blobs`
    uses four), so neighboring groups overlap in their tails. Fuzzy c-means
    from different random starts, on different sub-samples or with
    different numbers of clusters then finds noticeably different
    partitions -- the situation an ensemble is meant for.

    Parameters
    ----------
    n_clusters : int, default=4
    n_samples_per_cluster : int, default=75
    n_features : int, default=2
    spread : float, default=1.0
        Standard deviation of the groups.
    separation : float, default=2.5
        Minimal distance between centers in units of ``spread``.
    box : float, default=6.0
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_clusters * n_samples_per_cluster, n_features)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.ensemble import make_overlapping_blobs
    >>> X, y = make_overlapping_blobs(random_state=0)
    >>> X.shape, int(y.max())
    ((300, 2), 3)
    """
    from ..datasets import _separated_centers

    rng = check_random_state(random_state)
    centers = _separated_centers(rng, n_clusters, n_features, box, min_dist=separation * spread)
    X = np.vstack(
        [rng.randn(n_samples_per_cluster, n_features) * spread + centers[i] for i in range(n_clusters)]
    )
    y = np.repeat(np.arange(n_clusters), n_samples_per_cluster)
    return X, y
