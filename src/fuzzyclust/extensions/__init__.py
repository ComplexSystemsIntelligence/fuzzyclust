# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Robust and shape-oriented variants of fuzzy c-means.

FCM assumes compact, roughly spherical clusters of similar size and is
pulled away by outliers. The estimators of this sub-package relax one of
these assumptions at a time:

* :class:`PCM` -- possibilistic c-means: absolute (unnormalised) typicalities,
  robust to outliers, independent mode seeking per cluster;
* :class:`PFCM` -- possibilistic fuzzy c-means: memberships *and* typicalities;
* :class:`NoiseClustering` -- FCM with a virtual noise cluster at constant distance;
* :class:`KernelFCM` -- kernel-induced distance (input-space or feature-space prototypes);
* :class:`FuzzyCVarieties` -- lines / linear varieties as prototypes (fuzzy c-lines, c-elliptotypes);
* :class:`FuzzyCShells` -- circles / hyperspheres as prototypes;
* :class:`VolumePrototypeFCM` -- hyperellipsoidal cores of zero distance, with adaptive cluster merging;
* :class:`SCAD`, :class:`FeatureWeightedFCM` -- cluster-specific and global attribute weights learned with the
  partition;
* :func:`plot_shells`, :func:`plot_varieties`, :func:`plot_typicality` -- matplotlib helpers;
* :func:`make_circles`, :func:`make_close_blobs`, :func:`make_crossing_lines`,
  :func:`make_collinear_segments` -- data generators.
"""

from ._data import make_circles, make_close_blobs, make_collinear_segments, make_crossing_lines
from ._plot import plot_shells, plot_typicality, plot_varieties
from .kernel import KERNELS, KernelFCM, rbf_kernel
from .noise import NoiseClustering
from .pcm import PCM
from .pfcm import PFCM
from .shells import FuzzyCShells
from .varieties import FuzzyCVarieties
from .volume import VolumePrototypeFCM
from .weighted import SCAD, FeatureWeightedFCM

__all__ = [
    "PCM",
    "PFCM",
    "NoiseClustering",
    "KernelFCM",
    "rbf_kernel",
    "KERNELS",
    "FuzzyCVarieties",
    "FuzzyCShells",
    "VolumePrototypeFCM",
    "SCAD",
    "FeatureWeightedFCM",
    "plot_shells",
    "plot_varieties",
    "plot_typicality",
    "make_circles",
    "make_close_blobs",
    "make_crossing_lines",
    "make_collinear_segments",
]
