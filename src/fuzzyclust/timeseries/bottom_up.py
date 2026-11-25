# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Bottom-up PCA-based time-series segmentation (baseline of the segmentation paper)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array

__all__ = ["bottom_up_segmentation", "pca_segment_cost"]


def pca_segment_cost(X, n_components: int, cost: str = "q", center: bool = True) -> float:
    r"""Cost of approximating a segment by one PCA model with ``n_components`` components.

    With the eigenvalues :math:`\lambda_1\ge\dots\ge\lambda_n` of the
    covariance matrix of the segment (``center=True``) or of its
    second-moment matrix :math:`\frac{1}{N}\sum_k\mathbf{x}_k\mathbf{x}_k^T`
    (``center=False``, the uncentred form printed in the paper):

    * ``"q"`` -- mean reconstruction error, eq. (30) of Abonyi et al. (2005):
      :math:`\frac1N\sum_k Q_k =
      \frac1N\sum_k\mathbf{x}_k^T(\mathbf{I}-\mathbf{U}_p\mathbf{U}_p^T)\mathbf{x}_k=\sum_{j>p}\lambda_j`;
    * ``"t2"`` -- mean Hotelling :math:`T^2`, eq. (31):
      :math:`\frac1N\sum_k\mathbf{y}_k^T\mathbf{y}_k` with
      :math:`\mathbf{y}_k=\mathbf{W}^T\mathbf{x}_k`,
      :math:`\mathbf{W}=\mathbf{U}_p\boldsymbol{\Lambda}_p^{1/2}`,
      which equals :math:`\sum_{j\le p}\lambda_j^2`.
    """
    X = np.asarray(X, dtype=float)
    N, n = X.shape
    if center:
        S = np.cov(X, rowvar=False).reshape(n, n) if N > 1 else np.zeros((n, n))
    else:
        S = X.T @ X / N
    lam = np.sort(np.maximum(np.linalg.eigvalsh(S), 0.0))[::-1]
    p = min(n_components, n)
    if cost == "q":
        return float(lam[p:].sum())
    if cost == "t2":
        return float((lam[:p] ** 2).sum())
    raise ValueError("cost must be 'q' or 't2'.")


def bottom_up_segmentation(
    X,
    n_components: int = 2,
    max_error: float | None = None,
    n_segments: int | None = None,
    cost: str = "q",
    init_length: int = 10,
    center: bool = True,
):
    r"""Bottom-up segmentation with PCA models (Keogh et al. style, multivariate extension).

    The series is first cut into pieces of ``init_length`` samples. The cost
    of merging every pair of adjacent segments -- the PCA cost
    (:func:`pca_segment_cost`) of the union -- is computed and the cheapest
    pair is merged, until the cheapest merge would cost more than
    ``max_error`` or until ``n_segments`` segments are left. The cost is the
    level of the model error of the merged segment (not its increase), so
    ``max_error`` is the natural stopping rule; ``n_segments`` is offered
    for comparisons with a known number of segments. This is the
    multivariate counterpart of the piecewise-linear-approximation bottom-up
    algorithm (Appendix A of Abonyi et al., 2005) used there as the
    baseline: with the :math:`Q` cost it detects changes of the correlation
    structure, with the :math:`T^2` cost shifts of the operating point.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_components : int, default=2
        Number of principal components :math:`p` of the segment models.
    max_error : float, optional
        Stop when the cheapest merge costs more than this (ignored if
        ``n_segments`` is given).
    n_segments : int, optional
        Stop when this many segments are left.
    cost : {"q", "t2"}, default="q"
    init_length : int, default=10
        Length of the initial fine segments.
    center : bool, default=True
        Use the covariance of each segment (centered data). ``False`` uses the
        uncentred second-moment matrix as printed in eqs. (30)-(31) of the
        paper; the cost is then dominated by the level of the variables and
        the greedy merging becomes unreliable unless the data are centered
        beforehand.

    Returns
    -------
    result : :class:`sklearn.utils.Bunch`
        ``labels`` (segment index per sample), ``boundaries`` (start indices
        of the segments except the first), ``segments`` (list of
        ``(start, stop)`` pairs), ``merge_costs`` (cost of every merge made,
        in order) and ``n_segments``.

    References
    ----------
    .. [1] E. Keogh, S. Chu, D. Hart, M. Pazzani, "An online algorithm for
           segmenting time series", Proc. IEEE ICDM, 2001.
    .. [2] J. Abonyi, B. Feil, S. Németh, P. Árva, "Modified Gath–Geva
           clustering for fuzzy segmentation of multivariate time-series",
           Fuzzy Sets and Systems 149 (2005) 39–56, Appendix A.

    Examples
    --------
    >>> from fuzzyclust.timeseries import bottom_up_segmentation, make_piecewise_timeseries
    >>> X, _ = make_piecewise_timeseries(n_samples=400, change_points=(200,),
    ...                                  change_kinds=("correlation",), random_state=0)
    >>> res = bottom_up_segmentation(X, n_components=2, n_segments=2)
    >>> res.n_segments, abs(int(res.boundaries[0]) - 200) <= 10
    (2, True)
    """
    X = check_array(X, dtype=float)
    N = X.shape[0]
    if n_segments is None and max_error is None:
        raise ValueError("Give max_error or n_segments as stopping criterion.")
    if init_length < 1:
        raise ValueError("init_length must be >= 1.")
    bounds = list(range(0, N, init_length)) + [N]
    segs = list(zip(bounds[:-1], bounds[1:], strict=True))

    def merge_cost(i):
        return pca_segment_cost(X[segs[i][0] : segs[i + 1][1]], n_components, cost, center)

    costs = [merge_cost(i) for i in range(len(segs) - 1)]
    made = []
    while len(segs) > 1:
        if n_segments is not None and len(segs) <= n_segments:
            break
        i = int(np.argmin(costs))
        if n_segments is None and costs[i] > max_error:
            break
        made.append(costs[i])
        segs[i] = (segs[i][0], segs[i + 1][1])
        del segs[i + 1]
        del costs[i]
        if i < len(segs) - 1:
            costs[i] = merge_cost(i)
        if i > 0:
            costs[i - 1] = merge_cost(i - 1)
    labels = np.empty(N, dtype=int)
    for s, (a, b) in enumerate(segs):
        labels[a:b] = s
    boundaries = np.array([a for a, _ in segs[1:]], dtype=int)
    return Bunch(
        labels=labels,
        boundaries=boundaries,
        segments=segs,
        merge_costs=np.asarray(made),
        n_segments=len(segs),
    )
