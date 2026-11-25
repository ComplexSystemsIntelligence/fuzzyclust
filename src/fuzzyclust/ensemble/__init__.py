# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy cluster ensembles and consensus clustering.

Instead of trusting one run of an alternating-optimization clusterer, fit
a *library* of soft partitions (random starts, sub-samples, feature
subsets, several numbers of clusters) and combine them.

* :class:`FuzzyClusterEnsemble` -- library generation, soft
  co-association matrix and consensus by fuzzy c-means on the
  co-associations, spectral partitioning, evidence accumulation, the soft
  meta-clustering algorithm (sMCLA) or information-theoretic k-means
  (Punera and Ghosh, 2008; Fred and Jain, 2005);
* :func:`coassociation_matrix`, :func:`evidence_accumulation`,
  :func:`soft_mcla`, :func:`information_theoretic_kmeans` -- the building blocks;
* :func:`cspa`, :func:`mcla` -- the classical hard consensus functions of
  Strehl and Ghosh (2002);
* :func:`stability_curve` -- reproducibility of the consensus against the library size;
* :func:`make_overlapping_blobs` -- data generator;
* :func:`plot_coassociation`, :func:`plot_lifetimes` -- matplotlib helpers.
"""

from ._data import make_overlapping_blobs
from ._plot import plot_coassociation, plot_lifetimes
from .consensus import (
    FuzzyClusterEnsemble,
    coassociation_matrix,
    cspa,
    evidence_accumulation,
    information_theoretic_kmeans,
    mcla,
    soft_mcla,
    stability_curve,
)

__all__ = [
    "FuzzyClusterEnsemble",
    "coassociation_matrix",
    "evidence_accumulation",
    "soft_mcla",
    "information_theoretic_kmeans",
    "cspa",
    "mcla",
    "stability_curve",
    "make_overlapping_blobs",
    "plot_coassociation",
    "plot_lifetimes",
]
