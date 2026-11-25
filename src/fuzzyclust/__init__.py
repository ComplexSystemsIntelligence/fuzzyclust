# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
"""fuzzyclust — fuzzy clustering and data analysis toolbox.

A scikit-learn compatible Python successor of the *Fuzzy Clustering and Data
Analysis Toolbox for MATLAB* (Balasko, Abonyi, Feil) extended with the
clustering-based model identification, time-series, visualization and
graph-based methods of the Abonyi research group and with classical
possibilistic, kernel and shell variants.

Sub-packages
------------
cluster       KMeans, KMedoids, FCM, GK, GG; regularized fuzzifiers, subtractive clustering, merging
validity      validity indices, number-of-clusters scans
datasets      synthetic generators and bundled data
preprocessing scaling and regressor construction
plotting      matplotlib helpers
viz           PCA, Sammon, fuzzy Sammon mapping, fuzzy SOM
models        Takagi–Sugeno identification, fuzzy classifiers, regression trees
timeseries    segmentation, model-order selection, state-space reconstruction
graph         MST-based and geodesic-distance fuzzy clustering
extensions    possibilistic, noise, kernel, shell, volume-prototype and feature-weighting clustering
relational    relational FCM, NERF, fuzzy c-medoids, fuzzy k-modes/k-prototypes, DTW/Gower dissimilarities
incomplete    FCM for data with missing values (WDS, PDS, OCS, NPS)
large         weighted, random-sample, single-pass and online FCM for very large data and streams
image         spatially constrained FCM for image segmentation (FCM_S, EnFCM, FGFCM, FLICM)
semisupervised  partial supervision and pairwise constraints
coclustering  fuzzy co-clustering of objects and items (FCCM, Fuzzy CoDoK)
collaborative collaborative FCM across data sites
ensemble      fuzzy cluster ensembles and consensus
"""

from . import datasets, plotting, preprocessing, validity
from .cluster import FCM, GG, GK, KMeans, KMedoids

__version__ = "1.0"

__all__ = [
    "KMeans", "KMedoids", "FCM", "GK", "GG",
    "datasets", "validity", "preprocessing", "plotting",
    "__version__",
]
