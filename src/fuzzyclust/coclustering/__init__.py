# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy co-clustering of objects and items.

A non-negative object--item matrix (documents and terms, users and
products, cooccurrence counts) has two sets of entities that are
clustered *together*: a co-cluster is a group of objects and the group of
items that characterizes it.

* :class:`FCCM` -- fuzzy clustering for categorical multivariate data
  (Oh, Honda and Ichihashi, 2001): aggregation degree with entropy
  regularization of both membership sets;
* :class:`FCoDoK` -- fuzzy co-clustering of documents and keywords
  (Kummamuru, Dhawale and Krishnapuram, 2003): the same objective with
  quadratic regularizers and linear updates;
* :func:`make_block_matrix` -- block-structured count matrix with known co-clusters;
* :func:`plot_coclusters`, :func:`plot_item_memberships` -- matplotlib helpers.
"""

from ._data import make_block_matrix
from ._plot import plot_coclusters, plot_item_memberships
from .fccm import FCCM, FCoDoK

__all__ = ["FCCM", "FCoDoK", "make_block_matrix", "plot_coclusters", "plot_item_memberships"]
