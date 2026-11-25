# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Collaborative fuzzy clustering.

Several data sites describe the *same* objects with different features
and may not share their raw data. Collaborative clustering (Pedrycz,
2002) lets every site run its own fuzzy c-means and then exchange only
the partition matrices, so that the structures found at the sites are
reconciled without moving the data.

* :class:`CollaborativeFCM` -- horizontal collaboration between :math:`P`
  sites with the collaboration matrix :math:`\alpha[ii,jj]`;
* :func:`align_partitions` -- matching of cluster labels between
  partitions (the site partitions are compared with
  :func:`fuzzyclust.validity.fuzzy_rand_index`);
* :func:`make_collaborative_sites` -- data generator;
* :func:`plot_site_partitions`, :func:`plot_collaboration_trajectory` --
  matplotlib helpers.
"""

from ._data import make_collaborative_sites
from ._plot import plot_collaboration_trajectory, plot_site_partitions
from .cfcm import CollaborativeFCM, align_partitions

__all__ = [
    "CollaborativeFCM",
    "align_partitions",
    "make_collaborative_sites",
    "plot_site_partitions",
    "plot_collaboration_trajectory",
]
