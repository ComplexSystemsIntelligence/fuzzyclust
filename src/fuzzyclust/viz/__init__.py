# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
"""Visualization of high-dimensional fuzzy clustering results.

Projections
    :func:`pca_projection`   principal component projection of data and prototypes
    :class:`Sammon`          Sammon's nonlinear mapping (all pairwise distances)
    :class:`FuzzySammon`     fuzzy Sammon mapping (FUZZSAM): sample--prototype distances only
    :func:`sammon_stress`, :func:`recomputed_membership`, :func:`projection_quality`

Topology preserving clustering
    :class:`FuzzySOM`        fuzzy self-organising map based on regularized FCM
    :func:`grid_laplacian`

Plot helpers
    :func:`plot_projection`, :func:`plot_stress`, :func:`plot_map`, :func:`plot_som_lattice`

Data generators for the examples
    :func:`make_ellipsoids`, :func:`make_spiral_3d`, :func:`make_hemisphere`
"""

from ._data import make_ellipsoids, make_hemisphere, make_spiral_3d
from ._pca import pca_projection
from ._plot import plot_map, plot_projection, plot_som_lattice, plot_stress
from .fuzzy_sammon import FuzzySammon
from .fuzzy_som import FuzzySOM, grid_laplacian
from .quality import projection_quality, recomputed_membership
from .sammon import Sammon, sammon_stress

__all__ = [
    "pca_projection",
    "Sammon",
    "FuzzySammon",
    "FuzzySOM",
    "grid_laplacian",
    "sammon_stress",
    "recomputed_membership",
    "projection_quality",
    "plot_projection",
    "plot_stress",
    "plot_map",
    "plot_som_lattice",
    "make_ellipsoids",
    "make_spiral_3d",
    "make_hemisphere",
]
