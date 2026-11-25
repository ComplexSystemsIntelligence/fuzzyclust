# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Model-free structure (regressor) selection for models with mixed discrete and continuous inputs."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ..cluster import FCM
from ..validity import xie_beni

__all__ = ["MixedInputStructureSelection", "select_structure_mixed", "fuzzy_variance"]


def _crisp_partition(x):
    values, inverse = np.unique(x, return_inverse=True)
    U = np.zeros((x.shape[0], values.shape[0]))
    U[np.arange(x.shape[0]), inverse] = 1.0
    return U, values


def _fcm_partition(x, n_clusters, m, rng, candidates=(2, 3, 4, 5, 6)):
    X = x[:, None]
    if n_clusters == "xie-beni":
        best = None
        for c in candidates:
            if c >= X.shape[0]:
                break
            fcm = FCM(n_clusters=c, m=m, random_state=rng).fit(X)
            xb = xie_beni(X, fcm.membership_, fcm.cluster_centers_, m)
            if best is None or xb < best[0]:
                best = (xb, fcm)
        fcm = best[1]
    else:
        fcm = FCM(n_clusters=int(n_clusters), m=m, random_state=rng).fit(X)
    order = np.argsort(fcm.cluster_centers_[:, 0])
    return fcm.membership_[:, order], fcm.cluster_centers_[order, 0]


def fuzzy_variance(memberships, y):
    r"""Total fuzzy variance :math:`\sigma_t` of ``y`` over the fuzzy segments spanned by several inputs.

    The segments are the Cartesian product of the fuzzy sets of the chosen
    inputs; with the product conjunction the degree of a sample in segment
    :math:`i=(i_1,\dots,i_{n_i})` is :math:`\beta_{ik}=\prod_j\mu^{j}_{i_j k}`
    (already normalized, because the memberships of every input sum to one).
    Each segment is represented by the weighted mean
    :math:`b_i=\sum_k\beta_{ik}y_k/\sum_k\beta_{ik}` and

    .. math::
        \sigma_t=\sum_{k=1}^{N}\Big(y_k-\sum_{i}\beta_{ik}b_i\Big)^2,

    the squared error of the singleton fuzzy model defined by the segments.

    Parameters
    ----------
    memberships : sequence of ndarrays of shape (n_samples, c_j)
        One membership matrix per selected input.
    y : ndarray of shape (n_samples,)

    Returns
    -------
    sigma_t : float
    """
    B = np.ones((y.shape[0], 1))
    for U in memberships:
        B = (B[:, :, None] * U[:, None, :]).reshape(y.shape[0], -1)
    w = B.sum(axis=0)
    b = (B.T @ y) / np.where(w > 0, w, 1.0)
    return float(np.sum((y - B @ b) ** 2))


def select_structure_mixed(
    X,
    y,
    discrete=None,
    n_clusters=3,
    max_inputs: int | None = None,
    m: float = 2.0,
    vct_tolerance: float = 0.2,
    feature_names=None,
    random_state=None,
) -> pd.DataFrame:
    r"""Rank all candidate input structures by their fuzzy variance (Girimonte, Babuška & Abonyi).

    Every continuous input is partitioned into fuzzy sets by fuzzy c-means
    (the number of sets fixed or chosen by the Xie–Beni index), every
    discrete input into crisp sets, one per value. For every subset of at
    most ``max_inputs`` inputs the total fuzzy variance :math:`\sigma_t`
    (:func:`fuzzy_variance`) of the output over the product-space segments is
    computed; structures are ranked by increasing :math:`\sigma_t` and the
    variance–complexity trade-off :math:`VCT=\sigma_t\,n_i` is reported to
    distinguish structures of similar :math:`\sigma_t` with different numbers
    of inputs :math:`n_i` (an irrelevant input lowers :math:`\sigma_t` a
    little, because it splits the data into more segments, but it raises the
    VCT). No model has to be fitted, and discrete and continuous inputs are
    handled alike. The final choice is left to the user, who typically
    compares a few top-ranked structures by fitting models; the column
    ``vct`` and :attr:`MixedInputStructureSelection.best_structure_` only
    point at the first clear dip of the VCT curve.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_inputs)
    y : array-like of shape (n_samples,)
    discrete : array-like of bool or of int, optional
        Mask or indices of the discrete (categorical) inputs.
    n_clusters : int, "xie-beni" or sequence, default=3
        Number of fuzzy sets per continuous input; a sequence gives one value
        per input (entries of discrete inputs are ignored).
    max_inputs : int, optional
        Largest cardinality of the structures searched (the search is
        exhaustive, :math:`\sum_{k\le\max}\binom{n}{k}` structures). Default:
        ``min(n_inputs, 6)``.
    m : float, default=2.0
    vct_tolerance : float, default=0.2
        Structures whose :math:`\sigma_t` is within this relative tolerance
        of the smallest one are considered of *similar* variance; among
        them the one with the smallest VCT is the recommended structure.
    feature_names : sequence of str, optional
    random_state : int, RandomState instance or None

    Returns
    -------
    ranking : pandas.DataFrame
        One row per structure sorted by ``sigma_t``, with columns
        ``inputs`` (tuple of names), ``n_inputs``, ``sigma_t``, ``vct``;
        the index is the rank (1 = best).

    References
    ----------
    .. [1] D. Girimonte, R. Babuška, J. Abonyi, "Fuzzy clustering for
           selecting structure of nonlinear models with mixed discrete and
           continuous inputs", Proc. IEEE Int. Conf. Fuzzy Systems, 2004.

    Examples
    --------
    >>> from fuzzyclust.timeseries import select_structure_mixed, make_mixed_input_system
    >>> X, y, disc = make_mixed_input_system(n_samples=600, random_state=0)
    >>> ranking = select_structure_mixed(X, y, discrete=disc, n_clusters=3, random_state=0)
    >>> set(ranking.iloc[0]["inputs"]) >= {"x1", "x2", "x3", "x4"}
    True
    """
    sel = MixedInputStructureSelection(
        n_clusters=n_clusters,
        max_inputs=max_inputs,
        m=m,
        vct_tolerance=vct_tolerance,
        random_state=random_state,
    ).fit(X, y, discrete=discrete, feature_names=feature_names)
    return sel.ranking_


class MixedInputStructureSelection(BaseEstimator):
    r"""Estimator form of :func:`select_structure_mixed`.

    Parameters
    ----------
    n_clusters : int, "xie-beni" or sequence, default=3
    max_inputs : int, optional
    m : float, default=2.0
    vct_tolerance : float, default=0.2
    random_state : int, RandomState instance or None

    Attributes
    ----------
    ranking_ : pandas.DataFrame
        All evaluated structures sorted by :math:`\sigma_t` (see
        :func:`select_structure_mixed`).
    memberships_ : list of ndarrays
        Fuzzy (or crisp) partition of every input, shape ``(n_samples, c_j)``.
    centers_ : list of ndarrays
        Cluster centers (continuous inputs) or the distinct values (discrete inputs).
    best_structure_ : tuple of str
        Recommended structure: among the structures whose :math:`\sigma_t`
        lies within ``vct_tolerance`` of the smallest one (structures of
        "similar variance", the only ones the paper compares by VCT), the one
        with the smallest :math:`VCT`. This is the first dip of the VCT
        curve in the paper's figures.
    feature_names_ : list of str
    discrete_ : ndarray of bool
    n_features_in_ : int

    See Also
    --------
    select_structure_mixed : functional interface with the full description.
    """

    def __init__(
        self,
        n_clusters=3,
        max_inputs: int | None = None,
        m: float = 2.0,
        vct_tolerance: float = 0.2,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_inputs = max_inputs
        self.m = m
        self.vct_tolerance = vct_tolerance
        self.random_state = random_state

    def fit(self, X, y, discrete=None, feature_names=None):
        """Partition every input and rank all structures of at most ``max_inputs`` inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_inputs)
        y : array-like of shape (n_samples,)
        discrete : array-like of bool or int, optional
            Mask or indices of the discrete inputs.
        feature_names : sequence of str, optional
            Defaults to the DataFrame columns or ``x1, x2, ...``.
        """
        X, y = validate_data(self, X, y, dtype=float, y_numeric=True)
        y = np.asarray(y, dtype=float).ravel()
        if self.m <= 1.0:
            raise ValueError("m must be > 1.")
        if self.vct_tolerance < 0:
            raise ValueError("vct_tolerance must be >= 0.")
        N, n = X.shape
        rng = check_random_state(self.random_state)
        disc = np.zeros(n, dtype=bool)
        if discrete is not None:
            discrete = np.asarray(discrete)
            if discrete.dtype == bool:
                if discrete.shape != (n,):
                    raise ValueError("A boolean discrete mask must have one entry per input.")
                disc = discrete
            else:
                disc[discrete.astype(int)] = True
        if feature_names is None:
            feature_names = getattr(self, "feature_names_in_", [f"x{j + 1}" for j in range(n)])
        names = [str(v) for v in feature_names]
        if len(names) != n:
            raise ValueError("feature_names must have one entry per input.")
        nc = self.n_clusters
        if isinstance(nc, str) or np.isscalar(nc):
            nc = [nc] * n
        if len(nc) != n:
            raise ValueError(
                "n_clusters must be a scalar, 'xie-beni' or a sequence with one entry per input."
            )
        memberships, centers = [], []
        for j in range(n):
            if disc[j]:
                U, cen = _crisp_partition(X[:, j])
            else:
                U, cen = _fcm_partition(X[:, j], nc[j], self.m, rng)
            memberships.append(U)
            centers.append(cen)
        max_inputs = min(n, 6) if self.max_inputs is None else int(self.max_inputs)
        if max_inputs < 1:
            raise ValueError("max_inputs must be >= 1.")
        rows = []
        for size in range(1, max_inputs + 1):
            for subset in combinations(range(n), size):
                st = fuzzy_variance([memberships[j] for j in subset], y)
                inputs = tuple(names[j] for j in subset)
                rows.append({"inputs": inputs, "n_inputs": size, "sigma_t": st, "vct": st * size})
        ranking = pd.DataFrame(rows).sort_values("sigma_t", kind="stable").reset_index(drop=True)
        ranking.index = ranking.index + 1
        ranking.index.name = "rank"
        sigma = ranking["sigma_t"].to_numpy()
        vct = ranking["vct"].to_numpy()
        similar = sigma <= (1.0 + self.vct_tolerance) * sigma[0]
        best = int(np.flatnonzero(similar)[np.argmin(vct[similar])])
        self.feature_names_ = names
        self.discrete_ = disc
        self.memberships_ = memberships
        self.centers_ = centers
        self.ranking_ = ranking
        self.best_structure_ = tuple(ranking.iloc[best]["inputs"])
        return self

    def transform_segments(self, inputs):
        """Product-space segment memberships ``(n_samples, n_segments)`` of a structure (input names)."""
        check_is_fitted(self, "ranking_")
        idx = [self.feature_names_.index(name) for name in inputs]
        B = np.ones((self.memberships_[0].shape[0], 1))
        for j in idx:
            U = self.memberships_[j]
            B = (B[:, :, None] * U[:, None, :]).reshape(B.shape[0], -1)
        return B
