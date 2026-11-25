# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-means for image segmentation with spatial constraints.

An image is a data set with a twist: the samples (pixels) have neighbors,
and neighboring pixels almost always belong to the same region. Pixel-wise
fuzzy c-means ignores this and labels every noisy pixel on its own. This
sub-package collects the classical ways of putting the neighborhood into
the c-means functional:

* :class:`SpatialFCM` -- one estimator, variants chosen by ``method``:
  ``"fcm_s"`` (Ahmed et al. 2002), ``"fcm_s1"``/``"fcm_s2"`` (Chen and
  Zhang 2004), ``"enfcm"`` (Szilágyi et al. 2003), ``"fgfcm"`` (Cai, Chen
  and Zhang 2007), ``"flicm"`` (Krinidis and Chatzis 2010) and plain
  ``"fcm"`` as the reference;
* :func:`neighborhood_sum`, :func:`weighted_sum_image`,
  :func:`local_similarity_image` -- the neighborhood operations behind them
  (built on :mod:`scipy.ndimage`, no OpenCV or scikit-image needed);
* :func:`make_synthetic_image`, :func:`add_image_noise` -- test images with
  known labels; :func:`segmentation_accuracy`, :func:`match_labels` --
  evaluation;
* :func:`plot_segmentation`, :func:`plot_image`, :func:`plot_label_image`,
  :func:`plot_membership_maps` -- matplotlib helpers.
"""

from ._data import add_image_noise, make_synthetic_image
from ._metrics import match_labels, segmentation_accuracy
from ._plot import plot_image, plot_label_image, plot_membership_maps, plot_segmentation
from .spatial import METHODS, SpatialFCM, local_similarity_image, neighborhood_sum, weighted_sum_image

__all__ = [
    "SpatialFCM",
    "METHODS",
    "neighborhood_sum",
    "weighted_sum_image",
    "local_similarity_image",
    "make_synthetic_image",
    "add_image_noise",
    "segmentation_accuracy",
    "match_labels",
    "plot_segmentation",
    "plot_image",
    "plot_label_image",
    "plot_membership_maps",
]
