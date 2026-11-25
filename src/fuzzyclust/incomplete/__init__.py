# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy clustering of incomplete data.

Real data sets have holes: sensors fail, questionnaires are left blank,
records are merged from sources that do not measure the same things. This
sub-package implements the four strategies of Hathaway and Bezdek (2001)
for running fuzzy c-means on data with missing values (``NaN``):

* :class:`IncompleteFCM` -- ``strategy="wds"`` (whole-data: drop
  incomplete samples), ``"pds"`` (partial distances), ``"ocs"`` (optimal
  completion: missing values become optimization variables), ``"nps"``
  (nearest prototype);
* :func:`partial_distances` -- the partial distance of Dixon (1979) behind
  PDS and the classification of incomplete samples;
* :func:`make_mcar`, :func:`make_incomplete_blobs` -- plant values missing
  completely at random;
* :func:`plot_incomplete_clusters`, :func:`plot_imputation` -- matplotlib helpers.
"""

from ._data import make_incomplete_blobs, make_mcar
from ._plot import plot_imputation, plot_incomplete_clusters
from .incomplete_fcm import STRATEGIES, IncompleteFCM, partial_distances

__all__ = [
    "IncompleteFCM",
    "STRATEGIES",
    "partial_distances",
    "make_mcar",
    "make_incomplete_blobs",
    "plot_incomplete_clusters",
    "plot_imputation",
]
