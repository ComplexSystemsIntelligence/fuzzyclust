# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the collaborative clustering examples."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_collaborative_sites"]


def make_collaborative_sites(
    n_clusters: int = 3,
    n_samples_per_cluster: int = 100,
    n_sites: int = 2,
    n_features_per_site=2,
    spreads=(0.6, 1.3),
    box: float = 8.0,
    swap_fraction: float = 0.0,
    random_state=None,
):
    r"""The same objects described at several data sites by different features.

    Every object belongs to one of ``n_clusters`` groups. Site :math:`p`
    observes its own ``n_features_per_site`` Gaussian features around
    site-specific group centers with a site-specific spread, so the same
    group structure is seen clearly at a site with a small spread and
    blurred at a site with a large one. With ``swap_fraction > 0`` a share
    of the objects is moved to another group at the *last* site only,
    which gives the sites a genuine disagreement about those objects.

    Parameters
    ----------
    n_clusters : int, default=3
    n_samples_per_cluster : int, default=100
    n_sites : int, default=2
    n_features_per_site : int or sequence of int, default=2
    spreads : sequence of float, default=(0.6, 1.3)
        Standard deviation of the groups at every site (cycled if shorter than ``n_sites``).
    box : float, default=8.0
        Side of the cube the group centers are drawn from (every site has its own centers).
    swap_fraction : float, default=0.0
        Share of the objects whose group is changed at the last site.
    random_state : int, RandomState instance or None

    Returns
    -------
    Xs : list of ndarray, each of shape (n_samples, n_features_p)
    y : ndarray of shape (n_samples,)
        Group of every object (as seen by all sites but the swapped objects of the last one).

    Examples
    --------
    >>> from fuzzyclust.collaborative import make_collaborative_sites
    >>> Xs, y = make_collaborative_sites(n_sites=3, n_features_per_site=(2, 3, 2), random_state=0)
    >>> [X.shape for X in Xs]
    [(300, 2), (300, 3), (300, 2)]
    """
    from ..datasets import _separated_centers

    rng = check_random_state(random_state)
    n_feat = np.broadcast_to(np.asarray(n_features_per_site, dtype=int), (n_sites,))
    N = n_clusters * n_samples_per_cluster
    y = np.repeat(np.arange(n_clusters), n_samples_per_cluster)
    Xs = []
    for p in range(n_sites):
        s = spreads[p % len(spreads)]
        centers = _separated_centers(rng, n_clusters, int(n_feat[p]), box, min_dist=4.0 * s)
        labels = y.copy()
        if p == n_sites - 1 and swap_fraction > 0:
            idx = rng.choice(N, size=int(round(swap_fraction * N)), replace=False)
            labels[idx] = (labels[idx] + rng.randint(1, n_clusters, size=idx.size)) % n_clusters
        Xs.append(centers[labels] + rng.randn(N, int(n_feat[p])) * s)
    return Xs, y
