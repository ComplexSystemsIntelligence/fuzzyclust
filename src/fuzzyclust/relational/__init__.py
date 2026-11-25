# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy clustering of relational, categorical and mixed data.

The prototype-based algorithms of :mod:`fuzzyclust.cluster` need feature
vectors: a mean and a Euclidean distance. This sub-package covers the
cases where the objects are *not* vectors or where the mean makes no
sense -- only pairwise dissimilarities are known (time series compared by
dynamic time warping, documents, graphs), the attributes are categorical,
or numeric and categorical attributes are mixed:

* :class:`RelationalFCM` -- relational fuzzy c-means, the exact dual of FCM
  on a matrix of squared Euclidean distances (Hathaway, Davenport, Bezdek
  1989);
* :class:`NERFCM` -- NERF c-means, relational c-means made robust to
  non-Euclidean matrices by the adaptive :math:`\beta`-spread (Hathaway,
  Bezdek 1994);
* :class:`FuzzyCMedoids` -- fuzzy c-medoids with object prototypes, the
  linear-complexity candidate search and the trimmed variant
  (Krishnapuram, Joshi, Nasraoui, Yi 2001);
* :class:`FuzzyKModes` -- fuzzy k-modes for categorical data with the
  matching dissimilarity and membership-mass modes (Huang, Ng 1999);
* :class:`FuzzyKPrototypes` -- fuzzy k-prototypes for mixed data with the
  weight :math:`\gamma` (Huang 1998);
* :func:`pairwise_dissimilarity`, :func:`dtw_distance_matrix`,
  :func:`dtw_alignment`, :func:`autocorrelation_distance_matrix`,
  :func:`autocorrelation_features`, :func:`gower_distance_matrix`,
  :func:`matching_dissimilarity`, :func:`relational_distances`,
  :func:`check_dissimilarity_matrix` -- dissimilarity helpers;
* :func:`plot_dissimilarity_matrix`, :func:`plot_dtw_alignment`,
  :func:`plot_modes`, :func:`plot_series_clusters` -- matplotlib helpers;
* :func:`make_categorical_blobs`, :func:`make_mixed_blobs`,
  :func:`make_warped_series` -- data generators.

The relational estimators follow scikit-learn's convention:
``metric="precomputed"`` makes ``fit`` take the square dissimilarity
matrix and ``predict_proba`` the rectangular matrix of new objects
against the training objects; any other metric name computes the matrix
from feature vectors.
"""

from ._data import make_categorical_blobs, make_mixed_blobs, make_warped_series
from ._plot import plot_dissimilarity_matrix, plot_dtw_alignment, plot_modes, plot_series_clusters
from .dissimilarity import (
    autocorrelation_distance_matrix,
    autocorrelation_features,
    check_dissimilarity_matrix,
    dtw_alignment,
    dtw_distance_matrix,
    gower_distance_matrix,
    pairwise_dissimilarity,
)
from .fcmdd import FuzzyCMedoids
from .kmodes import FuzzyKModes, FuzzyKPrototypes, matching_dissimilarity
from .rfcm import NERFCM, RelationalFCM, relational_distances

__all__ = [
    "RelationalFCM",
    "NERFCM",
    "FuzzyCMedoids",
    "FuzzyKModes",
    "FuzzyKPrototypes",
    "relational_distances",
    "matching_dissimilarity",
    "pairwise_dissimilarity",
    "check_dissimilarity_matrix",
    "dtw_distance_matrix",
    "dtw_alignment",
    "autocorrelation_features",
    "autocorrelation_distance_matrix",
    "gower_distance_matrix",
    "plot_dissimilarity_matrix",
    "plot_dtw_alignment",
    "plot_modes",
    "plot_series_clusters",
    "make_categorical_blobs",
    "make_mixed_blobs",
    "make_warped_series",
]
