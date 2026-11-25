# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the robust and shape-oriented clustering variants.

Same conventions as :mod:`fuzzyclust.datasets`: every generator returns
``(X, y)`` and accepts ``random_state``.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_circles", "make_close_blobs", "make_crossing_lines", "make_collinear_segments"]


def make_circles(
    centers=((0.0, 0.0), (3.0, 1.0), (1.0, 3.5)),
    radii=(2.0, 1.2, 1.0),
    n_samples_per_circle: int = 120,
    arc: float = 1.0,
    noise: float = 0.06,
    n_outliers: int = 0,
    random_state=None,
):
    r"""Intersecting circles (shell clusters) with optional arcs and outliers.

    Unlike :func:`fuzzyclust.datasets.make_rings` the circles are not
    concentric, so they intersect and partly cover each other -- the
    situation the fuzzy c-(spherical) shells algorithms were designed for.

    Parameters
    ----------
    centers : sequence of (x, y)
        Centers of the circles.
    radii : sequence of float
        Radii, one per center.
    n_samples_per_circle : int, default=120
    arc : float, default=1.0
        Fraction of the full circle that is sampled (``1.0`` = closed circle,
        ``0.5`` = half circle starting at a random angle).
    noise : float, default=0.06
        Radial standard deviation of the points around the circle.
    n_outliers : int, default=0
        Uniformly distributed outliers (label ``-1``).
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
        Index of the generating circle, ``-1`` for outliers.
    """
    rng = check_random_state(random_state)
    Xs, ys = [], []
    for i, (center, r) in enumerate(zip(centers, radii, strict=True)):
        start = rng.rand() * 2 * np.pi
        phi = start + rng.rand(n_samples_per_circle) * 2 * np.pi * arc
        rr = r + rng.randn(n_samples_per_circle) * noise
        Xs.append(np.c_[rr * np.cos(phi), rr * np.sin(phi)] + np.asarray(center, dtype=float))
        ys.append(np.full(n_samples_per_circle, i))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    if n_outliers > 0:
        lo, hi = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
        O = rng.uniform(lo, hi, size=(n_outliers, 2))
        X = np.vstack([X, O])
        y = np.concatenate([y, np.full(n_outliers, -1)])
    return X, y


def make_close_blobs(
    sizes=(200, 40),
    spreads=(1.0, 0.4),
    distance: float = 2.5,
    random_state=None,
):
    r"""Two Gaussian clusters of different size and density close to each other.

    The classical test case for the *coincident clusters* problem of the
    possibilistic c-means: because every PCM cluster seeks a dense region
    independently of the others, both prototypes may end up in the larger
    (or denser) group when the groups are close.

    Parameters
    ----------
    sizes : tuple of int, default=(200, 40)
        Samples per cluster.
    spreads : tuple of float, default=(1.0, 0.4)
        Standard deviation per cluster.
    distance : float, default=2.5
        Distance between the two centers (along the first axis).
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (sum(sizes), 2)
    y : ndarray of shape (sum(sizes),)
    """
    rng = check_random_state(random_state)
    centers = np.array([[0.0, 0.0], [distance, 0.0]])
    X = np.vstack([rng.randn(sizes[i], 2) * spreads[i] + centers[i] for i in range(2)])
    y = np.repeat(np.arange(2), sizes)
    return X, y


def make_crossing_lines(
    n_lines: int = 3,
    n_samples_per_line: int = 80,
    length: float = 8.0,
    noise: float = 0.1,
    angles=None,
    center=(0.0, 0.0),
    random_state=None,
):
    r"""Line segments that cross at a common point.

    Crossing lines are the classical test case of the fuzzy c-lines
    algorithm: near the crossing the points of all lines are close to each
    other, so point and ellipsoid prototypes (FCM, GK) cut the lines into
    pieces around the crossing, whereas a line prototype measures only the
    orthogonal distance and assigns each point to the line it lies on.

    Parameters
    ----------
    n_lines : int, default=3
        Number of lines (ignored when ``angles`` is given).
    n_samples_per_line : int, default=80
    length : float, default=8.0
        Length of every segment, centered on the crossing.
    noise : float, default=0.1
        Standard deviation of the orthogonal scatter around the lines.
    angles : sequence of float, optional
        Orientation of the lines in degrees; default equally spaced in
        :math:`[0^\circ, 180^\circ)`.
    center : (x, y), default=(0, 0)
        The crossing point.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_lines * n_samples_per_line, 2)
    y : ndarray of shape (n_lines * n_samples_per_line,)
    """
    rng = check_random_state(random_state)
    if angles is None:
        angles = np.arange(n_lines) * 180.0 / n_lines
    angles = np.deg2rad(np.asarray(angles, dtype=float))
    Xs, ys = [], []
    for i, phi in enumerate(angles):
        t = rng.uniform(-length / 2, length / 2, n_samples_per_line)
        e = rng.randn(n_samples_per_line) * noise
        d = np.array([np.cos(phi), np.sin(phi)])
        nrm = np.array([-np.sin(phi), np.cos(phi)])
        Xs.append(t[:, None] * d + e[:, None] * nrm + np.asarray(center, dtype=float))
        ys.append(np.full(n_samples_per_line, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_collinear_segments(
    n_samples_per_segment: int = 60,
    length: float = 3.0,
    gap: float = 3.0,
    noise: float = 0.1,
    extra_line: bool = True,
    random_state=None,
):
    r"""Two collinear segments separated by a gap, optionally with a third, oblique segment.

    A line prototype has no extent, so pure fuzzy c-lines cannot tell two
    collinear segments apart: both lie on the same line. The fuzzy
    c-elliptotypes compromise, which adds a share of the point distance,
    separates them by their position. The optional third segment makes
    the problem a genuine three-cluster one.

    Parameters
    ----------
    n_samples_per_segment : int, default=60
    length : float, default=3.0
        Length of each collinear segment (on the x-axis).
    gap : float, default=3.0
        Gap between the two collinear segments.
    noise : float, default=0.1
        Orthogonal scatter around the segments.
    extra_line : bool, default=True
        Add an oblique segment above the gap (label 2).
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    """
    rng = check_random_state(random_state)
    half = gap / 2
    Xs, ys = [], []
    for i, (lo, hi) in enumerate(((-half - length, -half), (half, half + length))):
        x = rng.uniform(lo, hi, n_samples_per_segment)
        Xs.append(np.c_[x, rng.randn(n_samples_per_segment) * noise])
        ys.append(np.full(n_samples_per_segment, i))
    if extra_line:
        t = rng.uniform(-length / 2, length / 2, n_samples_per_segment)
        e = rng.randn(n_samples_per_segment) * noise
        phi = np.deg2rad(60.0)
        d, nrm = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        Xs.append(t[:, None] * d + e[:, None] * nrm + np.array([0.0, 2.5]))
        ys.append(np.full(n_samples_per_segment, 2))
    return np.vstack(Xs), np.concatenate(ys)
