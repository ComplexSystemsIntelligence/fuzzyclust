# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Evaluation of segmentations against a ground truth."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["segmentation_accuracy", "match_labels"]


def match_labels(y_true, y_pred):
    r"""Best one-to-one mapping of predicted labels onto true labels (Hungarian algorithm).

    Parameters
    ----------
    y_true, y_pred : array-like
        Label arrays of the same shape (label images are flattened).

    Returns
    -------
    mapping : ndarray of shape (n_pred_labels,)
        ``mapping[p]`` is the true label assigned to predicted label ``p``
        (``-1`` for a predicted label without partner when there are more
        predicted than true labels).
    """
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same number of elements.")
    nt, npred = int(t.max()) + 1, int(p.max()) + 1
    C = np.zeros((npred, nt), dtype=np.int64)
    np.add.at(C, (p, t), 1)
    rows, cols = linear_sum_assignment(C, maximize=True)
    mapping = np.full(npred, -1, dtype=int)
    mapping[rows] = cols
    return mapping


def segmentation_accuracy(y_true, y_pred):
    r"""Fraction of correctly labeled pixels after the best matching of the labels.

    Cluster labels are arbitrary, so the predicted labels are first mapped
    one-to-one onto the true labels by the assignment that maximizes the
    number of agreeing pixels (:func:`match_labels`); the accuracy is the
    share of pixels whose mapped label equals the truth. It is the
    "segmentation accuracy" used by Cai et al. (2007) and Krinidis and
    Chatzis (2010) to compare segmentation algorithms.

    Parameters
    ----------
    y_true : array-like of shape (H, W) or (n_pixels,)
    y_pred : array-like, same shape

    Returns
    -------
    accuracy : float
        In ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.image import segmentation_accuracy
    >>> truth = np.array([[0, 0, 1], [0, 1, 1]])
    >>> segmentation_accuracy(truth, 1 - truth)
    1.0
    >>> segmentation_accuracy(truth, np.array([[0, 0, 1], [0, 1, 0]]))
    0.8333333333333334
    """
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    mapping = match_labels(t, p)
    return float(np.mean(mapping[p] == t))
