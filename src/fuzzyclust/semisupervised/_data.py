# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the semi-supervised clustering examples."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_partially_labeled_pair", "make_partial_labels", "make_pairwise_constraints"]


def make_partial_labels(y, fraction: float = 0.1, n_labeled: int | None = None, random_state=None):
    """Hide most labels of ``y``: keep a random subset, set the rest to ``-1``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
    fraction : float, default=0.1
        Share of the samples that keep their label (ignored if ``n_labeled`` is given).
    n_labeled : int, optional
        Exact number of labeled samples. At least one sample of every class
        keeps its label whenever the budget allows it.
    random_state : int, RandomState instance or None

    Returns
    -------
    y_partial : ndarray of shape (n_samples,)
        ``y`` with ``-1`` at the hidden positions.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.semisupervised import make_partial_labels
    >>> yp = make_partial_labels(np.repeat([0, 1, 2], 50), fraction=0.1, random_state=0)
    >>> int((yp >= 0).sum())
    15
    """
    rng = check_random_state(random_state)
    y = np.asarray(y)
    N = y.shape[0]
    n = int(round(fraction * N)) if n_labeled is None else int(n_labeled)
    n = max(0, min(n, N))
    classes = np.unique(y)
    chosen = []
    if n >= classes.size:
        for cls in classes:  # one representative per class first
            chosen.append(rng.choice(np.flatnonzero(y == cls)))
    rest = np.setdiff1d(np.arange(N), chosen)
    rng.shuffle(rest)
    chosen = np.r_[chosen, rest[: n - len(chosen)]].astype(int)
    y_partial = np.full(N, -1, dtype=int)
    y_partial[chosen] = y[chosen]
    return y_partial


def make_partially_labeled_pair(
    n_samples: int = 300,
    length: float = 8.0,
    gap: float = 3.0,
    width: float = 0.3,
    label_fraction: float = 0.1,
    random_state=None,
):
    r"""Two long parallel groups whose *natural* split disagrees with the classes.

    Each class is a Gaussian stretched along the first axis (standard
    deviation ``length``/4) and narrow along the second (``width``); the two
    classes are ``gap`` apart along the second axis. With two clusters and
    :math:`m=2`, FCM does not separate the two bars: the variance along the
    bars is so dominant that the optimum is a tilted boundary that cuts both
    bars, a compromise between a left/right and a top/bottom split. Labeled
    samples rotate the partition toward the classes -- the more labels, the
    further -- which makes the data set a sensitive probe of partial
    supervision.

    Parameters
    ----------
    n_samples : int, default=300
    length : float, default=8.0
        Extent of the groups along the first axis (four standard deviations).
    gap : float, default=3.0
        Distance between the group centers along the second axis.
    width : float, default=0.3
        Standard deviation across the groups.
    label_fraction : float, default=0.1
        Share of samples that keep their label in ``y_partial``.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
        True class (0 or 1).
    y_partial : ndarray of shape (n_samples,)
        Partial labels, ``-1`` for the unlabeled samples.
    """
    rng = check_random_state(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    sx = length / 4.0
    X0 = np.c_[rng.randn(n0) * sx, rng.randn(n0) * width - gap / 2.0]
    X1 = np.c_[rng.randn(n1) * sx, rng.randn(n1) * width + gap / 2.0]
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    y_partial = make_partial_labels(y, fraction=label_fraction, random_state=rng)
    return X, y, y_partial


def make_pairwise_constraints(y, n_must_link: int = 20, n_cannot_link: int = 20, random_state=None):
    """Draw random must-link and cannot-link pairs consistent with the labels ``y``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
    n_must_link : int, default=20
        Number of pairs of samples from the same class.
    n_cannot_link : int, default=20
        Number of pairs of samples from different classes.
    random_state : int, RandomState instance or None

    Returns
    -------
    must_link : ndarray of shape (n_must_link, 2)
    cannot_link : ndarray of shape (n_cannot_link, 2)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.semisupervised import make_pairwise_constraints
    >>> y = np.repeat([0, 1], 20)
    >>> ml, cl = make_pairwise_constraints(y, 5, 5, random_state=0)
    >>> bool((y[ml[:, 0]] == y[ml[:, 1]]).all()), bool((y[cl[:, 0]] != y[cl[:, 1]]).all())
    (True, True)
    """
    rng = check_random_state(random_state)
    y = np.asarray(y)
    N = y.shape[0]
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    diff = ~(y[:, None] == y[None, :])
    iu = np.triu_indices(N, k=1)
    same_pairs = np.c_[iu[0], iu[1]][same[iu]]
    diff_pairs = np.c_[iu[0], iu[1]][diff[iu]]
    if n_must_link > same_pairs.shape[0] or n_cannot_link > diff_pairs.shape[0]:
        raise ValueError("Not enough pairs of the requested kind in y.")
    ml = same_pairs[rng.choice(same_pairs.shape[0], size=n_must_link, replace=False)]
    cl = diff_pairs[rng.choice(diff_pairs.shape[0], size=n_cannot_link, replace=False)]
    return ml, cl
