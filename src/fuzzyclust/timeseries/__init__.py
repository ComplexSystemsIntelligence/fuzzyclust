# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Clustering of time series.

Time-ordered data ask for clusters with a structure that plain fuzzy
clustering does not know: segments must be contiguous in time, regressors of
a dynamic model must be chosen before the model is fitted, and the state of a
chaotic system has to be reconstructed from a scalar measurement. This
sub-package collects the clustering-based answers of the Abonyi group:

* :class:`FuzzySegmentation` -- fuzzy segmentation of multivariate series by
  modified Gath–Geva clustering with local PCA models and compatible-cluster
  merging; :func:`bottom_up_segmentation` is the classical PCA-based
  bottom-up baseline (:func:`pca_segment_cost` its cost);
* :func:`fnn_model_order` / :func:`select_model_order` -- model-order
  selection by false nearest neighbors with a cluster-based adaptive
  threshold (:func:`cluster_based_threshold`);
  :func:`false_nearest_neighbors` is the classical FNN of Kennel et al.;
* :class:`ChaoticPredictor` -- state-space reconstruction of a chaotic series
  by delay embedding (:func:`delay_embedding`, :func:`select_delay`) and
  clustering-based local linear models, with :func:`embedding_scan` /
  :func:`select_embedding_dimension` for the embedding and local dimension
  and :func:`correlation_dimension` as the fractal counterpart;
* :class:`MixedInputStructureSelection` / :func:`select_structure_mixed` --
  model-free regressor selection for models with discrete and continuous
  inputs by fuzzy variance decomposition (:func:`fuzzy_variance`);
* data generators ``make_*`` and the plot helpers ``plot_*``.
"""

from ._data import (
    make_lorenz,
    make_mackey_glass,
    make_mixed_input_system,
    make_nonlinear_io_system,
    make_piecewise_timeseries,
    make_rossler,
)
from ._plot import (
    plot_embedding_scan,
    plot_fnn_table,
    plot_phase_space,
    plot_segmentation,
    plot_structure_ranking,
)
from .bottom_up import bottom_up_segmentation, pca_segment_cost
from .chaotic import (
    ChaoticPredictor,
    correlation_dimension,
    delay_embedding,
    embedding_scan,
    select_delay,
    select_embedding_dimension,
)
from .fnn import cluster_based_threshold, false_nearest_neighbors, fnn_model_order, select_model_order
from .segmentation import FuzzySegmentation
from .structure import MixedInputStructureSelection, fuzzy_variance, select_structure_mixed

__all__ = [
    # estimators
    "FuzzySegmentation",
    "ChaoticPredictor",
    "MixedInputStructureSelection",
    # functions
    "bottom_up_segmentation",
    "pca_segment_cost",
    "false_nearest_neighbors",
    "fnn_model_order",
    "select_model_order",
    "cluster_based_threshold",
    "delay_embedding",
    "select_delay",
    "embedding_scan",
    "select_embedding_dimension",
    "correlation_dimension",
    "select_structure_mixed",
    "fuzzy_variance",
    # plotting
    "plot_segmentation",
    "plot_fnn_table",
    "plot_embedding_scan",
    "plot_phase_space",
    "plot_structure_ranking",
    # data
    "make_piecewise_timeseries",
    "make_mackey_glass",
    "make_lorenz",
    "make_rossler",
    "make_nonlinear_io_system",
    "make_mixed_input_system",
]
