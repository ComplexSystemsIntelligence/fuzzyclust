# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Clustering for model identification.

Fuzzy clusters in the product space of inputs and outputs are local models:
each cluster becomes a rule whose antecedent is the projection of the cluster
onto the inputs and whose consequent is the local linear model hidden in the
cluster. This sub-package collects the algorithms that exploit this view:

* :class:`ModifiedGG` -- modified Gath–Geva clustering with axis-parallel
  Gaussian antecedents and local linear (SISO or MIMO) consequents; the
  clustering engine behind the Takagi–Sugeno model;
* :class:`TakagiSugeno` -- the user-facing regressor: Takagi–Sugeno model
  identified by :class:`ModifiedGG` or by Gustafson–Kessel clustering in the
  product space with projected antecedents;
* :class:`FuzzyCRegression` -- fuzzy c-regression models (switching regression,
  Hathaway & Bezdek 1993), clusters are hyperplanes;
* :class:`SupervisedFuzzyClassifier` -- supervised fuzzy clustering whose rules
  carry class-probability consequents (Abonyi & Szeifert 2003);
* :func:`supervised_fuzzy_partition`, :class:`FuzzyPartitionDiscretizer` --
  per-feature fuzzy partitions derived from supervised clustering, for
  decision-tree induction (Abonyi 2010);
* :class:`HingingHyperplaneTree` -- binary regression tree with hinging
  hyperplane nodes identified by fuzzy c-regression (Kenesei & Abonyi 2013);
* :func:`select_features_by_separability` -- Fisher interclass-separability
  based backward feature elimination of the fuzzy classifier;
* :func:`simulate_free_run` -- free-run simulation of NARX models built with
  :func:`fuzzyclust.preprocessing.make_regressors`;
* data generators ``make_*`` and plot helpers ``plot_*``.
"""

from ._data import (
    make_banded_classes,
    make_mimo_plant,
    make_piecewise_linear,
    make_second_order_plant,
    make_sinc,
    make_switching_lines,
    make_two_class_overlap,
    piecewise_linear_function,
    second_order_plant_function,
)
from ._plot import (
    plot_decision_regions,
    plot_fuzzy_partition,
    plot_hinge_tree_surface,
    plot_local_models,
    plot_membership_functions,
)
from .fuzzy_c_regression import FuzzyCRegression
from .hinging_hyperplane import HingeNode, HingingHyperplaneTree
from .modified_gg import ModifiedGG, weighted_least_squares
from .partition import (
    FuzzyPartitionDiscretizer,
    membership_1d,
    ruspini_partition,
    set_similarity,
    supervised_fuzzy_partition,
)
from .supervised import SupervisedFuzzyClassifier, select_features_by_separability
from .takagi_sugeno import TakagiSugeno, simulate_free_run

__all__ = [
    # estimators
    "ModifiedGG",
    "TakagiSugeno",
    "FuzzyCRegression",
    "SupervisedFuzzyClassifier",
    "FuzzyPartitionDiscretizer",
    "HingingHyperplaneTree",
    "HingeNode",
    # functions
    "simulate_free_run",
    "weighted_least_squares",
    "select_features_by_separability",
    "supervised_fuzzy_partition",
    "ruspini_partition",
    "membership_1d",
    "set_similarity",
    # plotting
    "plot_membership_functions",
    "plot_local_models",
    "plot_fuzzy_partition",
    "plot_hinge_tree_surface",
    "plot_decision_regions",
    # data
    "make_sinc",
    "make_piecewise_linear",
    "piecewise_linear_function",
    "make_switching_lines",
    "make_second_order_plant",
    "second_order_plant_function",
    "make_mimo_plant",
    "make_two_class_overlap",
    "make_banded_classes",
]
