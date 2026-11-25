# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Model-order selection by false nearest neighbors (classical and clustering-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import Bunch, check_random_state

from .._utils import EPS
from ..cluster import FCM, GG
from ..preprocessing import make_regressors

__all__ = ["false_nearest_neighbors", "fnn_model_order", "select_model_order", "cluster_based_threshold"]


def _regression_pairs(y, u, n_a, n_b, delay=1):
    """Regressors and targets for order (n_a, n_b) with unit input delay; None for (0, 0).

    For a scalar series (``u is None``) ``delay`` is the lag of the delay embedding:
    the regressors are ``y[k-delay], ..., y[k-delay*n_a]`` and the target is ``y[k]``.
    """
    if n_a == 0 and n_b == 0:
        return None, None
    if u is None and n_b > 0:
        raise ValueError("n_b > 0 requires an input sequence u.")
    if u is None and delay > 1:
        span = delay * n_a
        Phi = np.column_stack([y[span - j * delay : y.shape[0] - j * delay] for j in range(1, n_a + 1)])
        return Phi, y[span:]
    Phi, target, _ = make_regressors(y, u, n_a=n_a, n_b=n_b, n_k=1)
    return Phi, target


def _check_delay(delay, u):
    if delay < 1:
        raise ValueError("delay must be a positive integer.")
    if u is not None and delay > 1:
        raise ValueError("delay > 1 is only defined for a scalar time series (u=None).")
    return int(delay)


def _scale(Phi, target):
    sx = Phi.std(axis=0)
    sx[sx == 0] = 1.0
    sy = target.std() or 1.0
    return (Phi - Phi.mean(axis=0)) / sx, (target - target.mean()) / sy


def _nearest(Phi):
    nn = NearestNeighbors(n_neighbors=2).fit(Phi)
    dist, idx = nn.kneighbors(Phi)
    return dist[:, 1], idx[:, 1]


def _orders(max_na, max_nb, u):
    nas = range(0, max_na + 1)
    nbs = range(0, (max_nb if u is not None else 0) + 1)
    return list(nas), list(nbs)


def false_nearest_neighbors(
    y,
    u=None,
    max_na: int = 4,
    max_nb: int = 0,
    R_tol: float = 15.0,
    A_tol: float = 2.0,
    scale: bool = True,
    delay: int = 1,
) -> pd.DataFrame:
    r"""Classical false-nearest-neighbor percentages over the model orders :math:`(n_a, n_b)`.

    For every regression vector
    :math:`\mathbf{x}_k=[y_{k-1},\dots,y_{k-n_a},u_{k-1},\dots,u_{k-n_b}]^T`
    the nearest neighbor :math:`\mathbf{x}_j` in the regression space is
    found. If the regressors carry enough information the outputs of
    neighbors are close too; following Kennel, Brown & Abarbanel the
    neighbor is declared *false* when

    .. math::
        \frac{|y_k-y_j|}{\|\mathbf{x}_k-\mathbf{x}_j\|} > R_{tol}
        \qquad\text{or}\qquad
        \frac{\sqrt{\|\mathbf{x}_k-\mathbf{x}_j\|^2+(y_k-y_j)^2}}{R_A} > A_{tol},

    where :math:`R_A` is the standard deviation of the output (the size of the
    attractor). The percentage of false neighbors is tabulated over the
    orders; it drops to (nearly) zero once the regression vector is complete.
    With ``u=None`` this is the original embedding-dimension test for a
    scalar time series (the new coordinate being :math:`y_k`), with inputs it
    is the input–output extension of Rhodes & Morari.

    Parameters
    ----------
    y : array-like of shape (T,)
    u : array-like of shape (T,), optional
    max_na, max_nb : int
        Largest numbers of lagged outputs and inputs considered.
    R_tol : float, default=15.0
        Threshold of the slope criterion (Kennel: 10–50).
    A_tol : float, default=2.0
        Threshold of the attractor-size criterion.
    scale : bool, default=True
        Standardize regressors and output before measuring distances
        (strongly recommended for input–output data of different units).
    delay : int, default=1
        Lag :math:`\tau` of the delay embedding for a scalar time series
        (``u=None``): the regressors are :math:`y_{k-\tau},\dots,y_{k-n_a\tau}`.

    Returns
    -------
    table : pandas.DataFrame
        Percentage of false nearest neighbors; rows :math:`n_a`, columns
        :math:`n_b` (a single column ``0`` for a time series). The entry
        :math:`(0,0)` is 100 by definition.

    References
    ----------
    .. [1] M. B. Kennel, R. Brown, H. D. I. Abarbanel, "Determining embedding
           dimension for phase-space reconstruction using a geometrical
           construction", Physical Review A 45 (1992) 3403–3411.
    .. [2] C. Rhodes, M. Morari, "Determining the model order of nonlinear
           input/output systems", AIChE Journal 44 (1998) 151–163.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.timeseries import false_nearest_neighbors
    >>> rng = np.random.RandomState(0); y = np.zeros(400)
    >>> for k in range(2, 400): y[k] = 0.6 * y[k-1] - 0.3 * y[k-2] + 0.1 * rng.randn()
    >>> table = false_nearest_neighbors(y, max_na=4)
    >>> bool(table.loc[2, 0] < table.loc[1, 0])
    True
    """
    y = np.asarray(y, dtype=float).ravel()
    u = None if u is None else np.asarray(u, dtype=float).ravel()
    delay = _check_delay(delay, u)
    nas, nbs = _orders(max_na, max_nb, u)
    table = pd.DataFrame(index=pd.Index(nas, name="n_a"), columns=pd.Index(nbs, name="n_b"), dtype=float)
    for na in nas:
        for nb in nbs:
            Phi, target = _regression_pairs(y, u, na, nb, delay)
            if Phi is None:
                table.loc[na, nb] = 100.0
                continue
            if scale:
                Phi, target = _scale(Phi, target)
            R_A = target.std() or 1.0
            dist, idx = _nearest(Phi)
            dy = np.abs(target - target[idx])
            crit1 = dy / np.maximum(dist, EPS) > R_tol
            crit2 = np.sqrt(dist**2 + dy**2) / R_A > A_tol
            table.loc[na, nb] = 100.0 * np.mean(crit1 | crit2)
    return table


def cluster_based_threshold(Phi, target, n_clusters: int = 6, m: float = 2.0, random_state=None):
    r"""Point-wise FNN thresholds :math:`R_k` from Gath–Geva clusters in the regression space.

    The joint data :math:`\mathbf{z}_k=[\mathbf{x}_k^T\,y_k]^T` are clustered
    with the Gath–Geva algorithm (initialized by fuzzy c-means). Every
    cluster is a local linear model: the eigenvector :math:`\mathbf{t}_i`
    of the smallest eigenvalue of the fuzzy covariance matrix
    :math:`\mathbf{F}_i` is the normal of the hyperplane
    :math:`y=\mathbf{a}_i^T\mathbf{x}+b_i` with
    :math:`\mathbf{a}_i=-\mathbf{t}^{x}_i/t^{y}_i`. The gradient of the
    unknown regression function at :math:`\mathbf{x}_k` is approximated by
    the membership-weighted average of the local slopes, which gives the
    adaptive threshold of Feil, Abonyi & Szeifert:

    .. math::
        R_k = \Big\|\sum_{i=1}^{c}\mu_{ik}\,\frac{-1}{t^{y}_i}\,\mathbf{t}^{x}_i\Big\|_2 .

    Returns
    -------
    R : ndarray of shape (n_samples,)
    gg : fitted :class:`~fuzzyclust.cluster.GG`
    """
    rng = check_random_state(random_state)
    Z = np.column_stack([Phi, target])
    c = min(n_clusters, Z.shape[0])
    fcm = FCM(n_clusters=c, m=m, random_state=rng).fit(Z)
    gg = GG(n_clusters=c, m=m, init=fcm.membership_, beta=1e8).fit(Z)
    d = Phi.shape[1]
    slopes = np.empty((c, d))
    for i in range(c):
        t_i = gg.eigenvectors_[i][:, 0]  # smallest eigenvalue first (ascending order)
        ty = t_i[-1]
        ty = np.sign(ty) * max(abs(ty), 1e-6) if ty != 0 else 1e-6
        slopes[i] = -t_i[:-1] / ty
    R = np.linalg.norm(gg.membership_ @ slopes, axis=1)
    return R, gg


def fnn_model_order(
    y,
    u=None,
    max_na: int = 4,
    max_nb: int = 0,
    n_clusters: int = 6,
    threshold: float | None = None,
    fnn_tol: float = 5.0,
    m: float = 2.0,
    scale: bool = True,
    delay: int = 1,
    random_state=None,
) -> Bunch:
    r"""Model order of a NARX model by false nearest neighbors with a cluster-based threshold.

    The false-nearest-neighbor test of :func:`false_nearest_neighbors`
    needs a threshold :math:`R` for the ratio :math:`|y_k-y_j|/\|\mathbf{x}_k-\mathbf{x}_j\|`.
    By the mean-value theorem the ratio is bounded by the norm of the gradient
    of the regression function, :math:`\|\mathrm{d}f(\mathbf{x}_k)\|`, which
    varies over the operating range of a nonlinear system; a single constant
    (Rhodes & Morari: :math:`R=\max_k\|\mathrm{d}f\|` from a nonlinear model,
    or heuristics :math:`10\le R\le 50`) is therefore either too loose or too
    strict. Feil, Abonyi & Szeifert [1]_ estimate the local gradient from a
    Gath–Geva clustering of the regression data
    (:func:`cluster_based_threshold`) and use the point-wise threshold
    :math:`R_k`; the neighbor :math:`j` of :math:`k` is false when
    :math:`|y_k-y_j|/\|\mathbf{x}_k-\mathbf{x}_j\| > R_k`.
    The percentage :math:`J(n_a,n_b)` of false neighbors is tabulated for all
    orders and the smallest order in the corner where :math:`J` drops to
    (nearly) zero is selected.

    Parameters
    ----------
    y, u : array-like of shape (T,)
        Output and (optional) input sequences.
    max_na, max_nb : int
        Largest orders considered.
    n_clusters : int, default=6
        Number of Gath–Geva clusters :math:`c` per order (the paper used 6).
    threshold : float, optional
        If given, a constant :math:`R` is used instead of the cluster-based
        :math:`R_k` (reference behavior, Table 2 of the paper).
    fnn_tol : float, default=5.0
        The selected order is the smallest :math:`n_a+n_b` whose percentage
        of false neighbors is at most ``fnn_tol`` percent (ties: lower
        percentage, then smaller :math:`n_a`).
    m : float, default=2.0
        Fuzziness exponent of the clustering.
    scale : bool, default=True
        Standardize regressors and output first (the threshold and the
        distances are then dimensionless).
    delay : int, default=1
        Lag of the delay embedding for a scalar time series (``u=None``);
        ``n_a`` is then the embedding dimension at lag ``delay``.
    random_state : int, RandomState instance or None

    Returns
    -------
    result : :class:`sklearn.utils.Bunch`
        ``table`` (DataFrame, rows :math:`n_a`, columns :math:`n_b`, in per
        cent), ``n_a``, ``n_b`` (selected orders), ``thresholds`` (dict
        mapping ``(n_a, n_b)`` to the array :math:`R_k`).

    References
    ----------
    .. [1] B. Feil, J. Abonyi, F. Szeifert, "Determining the model order of
           nonlinear input-output systems by fuzzy clustering", in: Advances in
           Soft Computing -- Engineering Design and Manufacturing, Springer,
           2003.
    .. [2] C. Rhodes, M. Morari, "Determining the model order of nonlinear
           input/output systems", AIChE Journal 44 (1998) 151–163.
    .. [3] J. D. Bomberger, D. E. Seborg, "Determination of model order for
           NARX models directly from input-output data", J. Process Control 8
           (1998) 459–468.

    Examples
    --------
    >>> from fuzzyclust.timeseries import fnn_model_order, make_nonlinear_io_system
    >>> u, y = make_nonlinear_io_system(n_samples=600, random_state=0)
    >>> res = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0)
    >>> res.table.shape, bool(res.table.loc[1, 2] < res.table.loc[0, 0])
    ((3, 3), True)
    """
    y = np.asarray(y, dtype=float).ravel()
    u = None if u is None else np.asarray(u, dtype=float).ravel()
    rng = check_random_state(random_state)
    delay = _check_delay(delay, u)
    nas, nbs = _orders(max_na, max_nb, u)
    table = pd.DataFrame(index=pd.Index(nas, name="n_a"), columns=pd.Index(nbs, name="n_b"), dtype=float)
    thresholds = {}
    for na in nas:
        for nb in nbs:
            Phi, target = _regression_pairs(y, u, na, nb, delay)
            if Phi is None:
                table.loc[na, nb] = 100.0
                continue
            if scale:
                Phi, target = _scale(Phi, target)
            if threshold is None:
                R, _ = cluster_based_threshold(Phi, target, n_clusters, m, rng)
            else:
                R = np.full(Phi.shape[0], float(threshold))
            thresholds[(na, nb)] = R
            dist, idx = _nearest(Phi)
            ratio = np.abs(target - target[idx]) / np.maximum(dist, EPS)
            table.loc[na, nb] = 100.0 * np.mean(ratio > R)
    candidates = [
        (na + nb, table.loc[na, nb], na, nb) for na in nas for nb in nbs if table.loc[na, nb] <= fnn_tol
    ]
    if candidates:
        _, _, na_sel, nb_sel = min(candidates)
    else:  # nothing below the tolerance: take the overall minimum
        idx = np.unravel_index(np.argmin(table.to_numpy()), table.shape)
        na_sel, nb_sel = nas[idx[0]], nbs[idx[1]]
    best = (na_sel, nb_sel)
    return Bunch(table=table, n_a=int(best[0]), n_b=int(best[1]), thresholds=thresholds)


def select_model_order(y, u=None, max_na: int = 4, max_nb: int = 0, **kwargs) -> tuple[int, int]:
    r"""Convenience wrapper of :func:`fnn_model_order` returning only the selected orders.

    Parameters
    ----------
    y, u : array-like of shape (T,)
        Output and (optional) input sequences.
    max_na, max_nb : int
        Largest orders considered.
    **kwargs
        Further keyword arguments of :func:`fnn_model_order` (``n_clusters``,
        ``threshold``, ``fnn_tol``, ``m``, ``scale``, ``random_state``).

    Returns
    -------
    n_a, n_b : int
        Selected numbers of lagged outputs and inputs. For a scalar time
        series (``u=None``) ``n_a`` is the embedding dimension and ``n_b``
        is ``0``.

    Examples
    --------
    >>> from fuzzyclust.timeseries import select_model_order, make_nonlinear_io_system
    >>> u, y = make_nonlinear_io_system(n_samples=600, random_state=0)
    >>> select_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0)
    (1, 2)
    """
    res = fnn_model_order(y, u, max_na=max_na, max_nb=max_nb, **kwargs)
    return res.n_a, res.n_b
