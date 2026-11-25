# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-means for very large and streaming data.

Fuzzy c-means needs the whole data matrix and the :math:`N\times c`
partition in memory at every iteration. When :math:`N` is in the
millions, or when the data arrive as a stream, three schemes of Havens,
Bezdek, Leckie, Hall and Palaniswami (2012) make FCM feasible while
approximating its result:

* :class:`WeightedFCM` -- weighted fuzzy c-means (wFCM), FCM with a weight
  per sample; the building block of the other two and of the histogram
  based image segmentation algorithms;
* :class:`RandomSampleFCM` -- random sample and extend (rseFCM): FCM on a
  random sample, one membership step for the rest;
* :class:`SinglePassFCM` -- single-pass FCM (spFCM): chunks are processed
  in sequence, the weighted prototypes of a chunk are carried into the
  next one; :meth:`~SinglePassFCM.partial_fit` for streams;
* :class:`OnlineFCM` -- online FCM (oFCM): every chunk is clustered on its
  own, the weighted prototypes of all chunks are clustered at the end;
* :func:`extend_partition` -- the extension step shared by all three;
* :func:`make_large_blobs`, :func:`make_drifting_stream` -- data generators;
* :func:`plot_chunk_timeline`, :func:`plot_center_trajectories` -- matplotlib helpers.
"""

from ._data import make_drifting_stream, make_large_blobs
from ._plot import plot_center_trajectories, plot_chunk_timeline
from .online import OnlineFCM
from .sample import RandomSampleFCM
from .single_pass import SinglePassFCM
from .weighted import WeightedFCM, extend_partition

__all__ = [
    "WeightedFCM",
    "RandomSampleFCM",
    "SinglePassFCM",
    "OnlineFCM",
    "extend_partition",
    "make_large_blobs",
    "make_drifting_stream",
    "plot_chunk_timeline",
    "plot_center_trajectories",
]
