# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Clustering with partial supervision.

Unsupervised clustering finds the structure the *distance* suggests, which
is not always the structure the *user* means. This sub-package lets a
little knowledge steer the partition:

* :class:`SemiSupervisedFCM` -- fuzzy c-means with partial supervision
  (Pedrycz and Waletzky, 1997): a few labeled samples (``y == -1`` marks the
  unlabeled ones) pull the prototypes toward the classes;
* :class:`PairwiseConstrainedFCM` -- pairwise-constrained competitive
  agglomeration, PCCA (Grira, Crucianu and Boujemaa, 2006): must-link and
  cannot-link pairs are penalized when violated, optionally with the
  competitive agglomeration term that lets the number of clusters shrink;
* :func:`constraint_satisfaction`, :func:`check_pairwise_constraints` --
  utilities for pairwise constraints;
* :func:`make_partially_labeled_pair`, :func:`make_partial_labels`,
  :func:`make_pairwise_constraints` -- data generators;
* :func:`plot_label_propagation`, :func:`plot_pairwise_constraints` --
  matplotlib helpers.
"""

from ._data import make_pairwise_constraints, make_partial_labels, make_partially_labeled_pair
from ._plot import plot_label_propagation, plot_pairwise_constraints
from .pcca import PairwiseConstrainedFCM, check_pairwise_constraints, constraint_satisfaction
from .ssfcm import SemiSupervisedFCM

__all__ = [
    "SemiSupervisedFCM",
    "PairwiseConstrainedFCM",
    "check_pairwise_constraints",
    "constraint_satisfaction",
    "make_partially_labeled_pair",
    "make_partial_labels",
    "make_pairwise_constraints",
    "plot_label_propagation",
    "plot_pairwise_constraints",
]
