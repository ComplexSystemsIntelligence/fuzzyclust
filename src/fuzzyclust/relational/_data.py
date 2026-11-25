# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for relational, categorical, mixed and time-series clustering."""

from __future__ import annotations

import string

import numpy as np
from sklearn.utils import Bunch, check_random_state

__all__ = ["make_categorical_blobs", "make_mixed_blobs", "make_warped_series"]


def _category_labels(n_categories: int) -> np.ndarray:
    letters = string.ascii_uppercase
    if n_categories <= len(letters):
        return np.array(list(letters[:n_categories]), dtype=object)
    return np.array([f"c{q}" for q in range(n_categories)], dtype=object)


def make_categorical_blobs(
    n_samples: int = 300,
    n_features: int = 6,
    n_categories: int = 4,
    n_clusters: int = 3,
    noise: float = 0.25,
    random_state=None,
    as_frame: bool = True,
):
    r"""Categorical records scattered around planted modes.

    Every cluster has a mode (one category per attribute); a record of the
    cluster copies the mode and then, independently per attribute with
    probability ``noise``, replaces the category by a uniformly random one.
    Distinct clusters get distinct modes that differ in every attribute, so
    the matching dissimilarity between modes is ``n_features``. The data
    mimic a survey in which groups of respondents share typical answers
    with individual deviations.

    Parameters
    ----------
    n_samples : int, default=300
    n_features : int, default=6
    n_categories : int, default=4
        Categories per attribute (at least ``n_clusters``).
    n_clusters : int, default=3
    noise : float, default=0.25
        Probability that an entry is replaced by a uniformly random category.
        The random category can coincide with the mode, so an entry differs
        from the mode of its cluster with probability
        ``noise * (n_categories - 1) / n_categories``.
    random_state : int, RandomState instance or None
    as_frame : bool, default=True
        Return a :class:`pandas.DataFrame` with string categories
        (``"A"``, ``"B"``, ...) and columns ``q1, q2, ...``; with
        ``False`` an object ndarray.

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        ``data`` -- the records; ``target`` -- cluster of every record;
        ``modes`` -- the planted modes in the same format as ``data``.

    Examples
    --------
    >>> from fuzzyclust.relational import make_categorical_blobs
    >>> d = make_categorical_blobs(n_samples=50, n_features=4, random_state=0)
    >>> d.data.shape, d.modes.shape, sorted(d.data.iloc[:, 0].unique())[:2]
    ((50, 4), (3, 4), ['A', 'B'])
    """
    rng = check_random_state(random_state)
    if n_categories < n_clusters:
        raise ValueError("n_categories must be at least n_clusters so that the modes can differ.")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must lie in [0, 1].")
    labels = _category_labels(n_categories)
    # modes that differ in every attribute: a random permutation per attribute
    modes = np.stack([rng.permutation(n_categories)[:n_clusters] for _ in range(n_features)], axis=1)
    y = np.repeat(np.arange(n_clusters), int(np.ceil(n_samples / n_clusters)))[:n_samples]
    rng.shuffle(y)
    codes = modes[y].copy()
    flip = rng.rand(n_samples, n_features) < noise
    codes[flip] = rng.randint(0, n_categories, size=int(flip.sum()))
    X = labels[codes]
    M = labels[modes]
    if as_frame:
        import pandas as pd

        cols = [f"q{j + 1}" for j in range(n_features)]
        X = pd.DataFrame(X, columns=cols)
        M = pd.DataFrame(M, columns=cols)
    return Bunch(data=X, target=y, modes=M)


def make_mixed_blobs(
    n_samples: int = 300,
    n_numeric: int = 3,
    n_categorical: int = 3,
    n_categories: int = 3,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    center_distance: float = 3.0,
    noise: float = 0.25,
    random_state=None,
):
    r"""Records with Gaussian numeric attributes and categorical attributes around modes.

    Every cluster has a numeric center (Gaussian with ``cluster_std``) and
    a categorical mode; the categorical part is generated as in
    :func:`make_categorical_blobs`. Both parts carry the cluster structure,
    so a method that uses only one of them sees it less clearly than one
    that combines them with a sensible weight.

    Parameters
    ----------
    n_samples : int, default=300
    n_numeric : int, default=3
    n_categorical : int, default=3
    n_categories : int, default=3
    n_clusters : int, default=3
    cluster_std : float, default=1.0
    center_distance : float, default=3.0
        Distance between neighboring numeric centers: the centers are
        placed equally spaced on a circle in the first two numeric
        dimensions (on a line when there is one numeric attribute).
    noise : float, default=0.25
        Probability that a categorical entry is replaced by a uniformly
        random category (which can coincide with the mode), as in
        :func:`make_categorical_blobs`.
    random_state : int, RandomState instance or None

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        ``data`` -- :class:`pandas.DataFrame` with numeric columns
        ``x1, x2, ...`` and categorical columns ``c1, c2, ...``;
        ``target``; ``centers`` ``(n_clusters, n_numeric)``;
        ``modes`` -- DataFrame of the planted modes.

    Examples
    --------
    >>> from fuzzyclust.relational import make_mixed_blobs
    >>> d = make_mixed_blobs(n_samples=60, random_state=0)
    >>> d.data.shape, d.data.select_dtypes(include="number").shape[1], list(d.data.columns)
    ((60, 6), 3, ['x1', 'x2', 'x3', 'c1', 'c2', 'c3'])
    """
    import pandas as pd

    rng = check_random_state(random_state)
    if n_numeric < 1 and n_categorical < 1:
        raise ValueError("At least one numeric or one categorical attribute is needed.")
    y = np.repeat(np.arange(n_clusters), int(np.ceil(n_samples / n_clusters)))[:n_samples]
    rng.shuffle(y)
    frame = {}
    centers = np.zeros((n_clusters, n_numeric))
    if n_numeric:
        if n_numeric == 1 or n_clusters < 3:
            centers[:, 0] = np.arange(n_clusters) * center_distance
        else:
            theta = 2.0 * np.pi * np.arange(n_clusters) / n_clusters
            radius = center_distance / (2.0 * np.sin(np.pi / n_clusters))
            centers[:, 0] = radius * np.cos(theta)
            centers[:, 1] = radius * np.sin(theta)
        Xn = centers[y] + rng.randn(n_samples, n_numeric) * cluster_std
        for j in range(n_numeric):
            frame[f"x{j + 1}"] = Xn[:, j]
    modes = pd.DataFrame(index=range(n_clusters))
    if n_categorical:
        cat = make_categorical_blobs(
            n_samples=n_samples,
            n_features=n_categorical,
            n_categories=n_categories,
            n_clusters=n_clusters,
            noise=noise,
            random_state=rng,
            as_frame=False,
        )
        # realign the categorical records with our labels: the generator shuffles its own
        order = np.empty(n_samples, dtype=int)
        for i in range(n_clusters):
            order[y == i] = np.flatnonzero(cat.target == i)[: int((y == i).sum())]
        for j in range(n_categorical):
            frame[f"c{j + 1}"] = cat.data[order, j]
        modes = pd.DataFrame(cat.modes, columns=[f"c{j + 1}" for j in range(n_categorical)])
    return Bunch(data=pd.DataFrame(frame), target=y, centers=centers, modes=modes)


def make_warped_series(
    n_series_per_class: int = 20,
    length: int = 60,
    warp: float = 0.25,
    noise: float = 0.1,
    shift: float = 0.1,
    random_state=None,
):
    r"""Time series of three shape classes with random time warps.

    The classes are a *bell* (one Gaussian bump), a *step* (a sigmoid rise
    from 0 to 1) and a *double bump* (two narrower Gaussian bumps). Each
    series is the class shape evaluated on a randomly *warped* time axis --
    a monotone distortion :math:`\tau(t)=t+w\,\sin(\pi t+\phi)` whose
    strength :math:`w` is drawn up to ``warp`` (monotone as long as
    :math:`w<1/\pi`) -- shifted in time by up to
    ``shift`` and corrupted by Gaussian noise. Because the warps move the
    peaks and edges to different time indices, the Euclidean distance
    between two series of the same class can exceed the distance between
    classes, whereas dynamic time warping re-aligns them.

    Parameters
    ----------
    n_series_per_class : int, default=20
    length : int, default=60
    warp : float, default=0.25
        Maximal warp amplitude (fraction of the unit time axis), below
        :math:`1/\pi` so that every warp is monotone.
    noise : float, default=0.1
        Standard deviation of the additive noise.
    shift : float, default=0.1
        Maximal time shift (fraction of the unit time axis).
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (3 * n_series_per_class, length)
    y : ndarray of shape (3 * n_series_per_class,)
        Class of every series (0: bell, 1: step, 2: double bump).

    Examples
    --------
    >>> from fuzzyclust.relational import make_warped_series
    >>> X, y = make_warped_series(n_series_per_class=5, length=40, random_state=0)
    >>> X.shape, sorted(set(y.tolist()))
    ((15, 40), [0, 1, 2])
    """
    rng = check_random_state(random_state)
    if not 0.0 <= warp < 1.0 / np.pi:
        raise ValueError("warp must lie in [0, 1/pi) for a monotone time warp.")
    t = np.linspace(0.0, 1.0, length)
    shapes = (
        lambda u: np.exp(-0.5 * ((u - 0.5) / 0.08) ** 2),
        lambda u: 1.0 / (1.0 + np.exp(-(u - 0.5) / 0.02)),
        lambda u: np.exp(-0.5 * ((u - 0.3) / 0.05) ** 2) + np.exp(-0.5 * ((u - 0.7) / 0.05) ** 2),
    )
    X = np.empty((3 * n_series_per_class, length))
    y = np.repeat(np.arange(3), n_series_per_class)
    for k in range(X.shape[0]):
        w = rng.uniform(-warp, warp)
        phi = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(-shift, shift)
        tau = t + w * np.sin(np.pi * t + phi) + s
        X[k] = shapes[y[k]](np.clip(tau, 0.0, 1.0)) + noise * rng.randn(length)
    return X, y
