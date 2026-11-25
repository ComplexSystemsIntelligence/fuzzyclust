# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-means for incomplete data (Hathaway and Bezdek, 2001).

An *incomplete* data set has samples with one or more missing feature
values. Hathaway and Bezdek compared four strategies of running fuzzy
c-means on such data; all four are implemented in :class:`IncompleteFCM`
and selected by ``strategy``. Missing values are marked by ``NaN``.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer

__all__ = ["IncompleteFCM", "partial_distances", "STRATEGIES"]

#: The four strategies of Hathaway and Bezdek (2001).
STRATEGIES = ("wds", "pds", "ocs", "nps")


def partial_distances(X, V):
    r"""Squared *partial distances* of (incomplete) samples to prototypes.

    For a sample :math:`\mathbf{x}_k` with the indicator :math:`I_{kj}=1`
    when feature :math:`j` is observed and :math:`I_k=\sum_j I_{kj}`
    observed features, the partial distance of Dixon (1979) is the squared
    Euclidean distance over the observed features, rescaled to the full
    dimension :math:`n`,

    .. math::
        D^2_{ik} = \frac{n}{I_k}\sum_{j=1}^{n} I_{kj}\,(x_{kj}-v_{ij})^2 .

    For a complete sample it is the ordinary squared Euclidean distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples; missing values are ``NaN``.
    V : array-like of shape (n_clusters, n_features)
        Prototypes (complete).

    Returns
    -------
    d2 : ndarray of shape (n_samples, n_clusters)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.incomplete import partial_distances
    >>> X = np.array([[0.0, 0.0], [1.0, np.nan]])
    >>> partial_distances(X, np.array([[1.0, 1.0]]))
    array([[2.],
           [0.]])
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    mask = np.isnan(X)
    I = (~mask).astype(float)
    X0 = np.where(mask, 0.0, X)
    d2 = (X0**2).sum(axis=1)[:, None] + I @ (V**2).T - 2.0 * X0 @ V.T
    np.maximum(d2, 0.0, out=d2)
    n_obs = I.sum(axis=1)
    if np.any(n_obs == 0):
        raise ValueError("Every sample must have at least one observed feature.")
    return d2 * (X.shape[1] / n_obs)[:, None]


class IncompleteFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means clustering of incomplete data (Hathaway and Bezdek, 2001).

    Missing feature values are marked by ``NaN``. Four strategies of running
    fuzzy c-means on such data are available (``strategy``):

    ``"wds"``
        *Whole-data strategy*: samples with missing values are dropped and
        FCM is run on the complete samples only. The incomplete samples
        then receive memberships from their partial distances to the
        prototypes (the paper's own suggestion for classifying them).
        Adequate when few samples are incomplete.
    ``"pds"``
        *Partial-distance strategy*: every sample takes part, the distance
        to a prototype is the partial distance :func:`partial_distances`
        over the observed features (rescaled to the full dimension), and
        every prototype coordinate is the membership-weighted mean of the
        samples that observe it.
    ``"ocs"``
        *Optimal-completion strategy*: the missing values are additional
        variables of the c-means functional. Each is updated to the value
        that minimizes :math:`J` for the current partition, the
        membership-weighted mean of the prototype coordinates; FCM then
        runs on the completed data. OCS imputes the data as a by-product.
    ``"nps"``
        *Nearest-prototype strategy*: like OCS, but a missing value is
        replaced by the coordinate of the *nearest* prototype of its
        sample (a hard version of OCS, cheaper and a little less accurate).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    strategy : {"wds", "pds", "ocs", "nps"}, default="pds"
        How the missing values are handled (see above).
    tol : float, default=1e-4
        Termination tolerance on :math:`\max_{ik}|\Delta\mu_{ik}|`.
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization, see :class:`~fuzzyclust.base.BaseFuzzyClusterer`.
        It is computed on the data with the missing values provisionally
        filled by the feature means of the observed values.
    n_init : int, default=1
        Random restarts; the run with the lowest final :math:`J` is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes :math:`\mathbf{v}_i`.
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition :math:`\mathbf{U}^T` of *all* samples (for
        ``"wds"`` the incomplete samples are classified by their partial
        distances after the fit).
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
        Distances at convergence: Euclidean on the completed data for
        ``"ocs"``/``"nps"``, partial distances otherwise.
    missing_mask_ : ndarray of bool of shape (n_samples, n_features)
        ``True`` where a value was missing.
    n_missing_ : int
        Number of missing values.
    X_completed_ : ndarray of shape (n_samples, n_features)
        The data with the missing values filled in. For ``"ocs"`` and
        ``"nps"`` these are the values the algorithm optimized; for
        ``"pds"`` and ``"wds"`` they are a post-hoc estimate by the OCS
        formula (membership-weighted prototype coordinates) from the final
        partition, not used by the algorithm.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariances of the final partition on the completed data.
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    Let :math:`I_{kj}=1` when :math:`x_{kj}` is observed (``0`` when it is
    missing), :math:`I_k=\sum_j I_{kj}`, :math:`X_M` the set of missing
    values and :math:`X_P` the observed ones. The strategies optimize the
    c-means functional :math:`J=\sum_i\sum_k\mu_{ik}^m D^2_{ik}` as follows.

    *PDS* -- the partial distance and the prototype update

    .. math::
        D^2_{ik}=\frac{n}{I_k}\sum_{j=1}^{n}I_{kj}(x_{kj}-v_{ij})^2,\qquad
        v_{ij}=\frac{\sum_{k=1}^{N}\mu_{ik}^m I_{kj}\,x_{kj}}{\sum_{k=1}^{N}\mu_{ik}^m I_{kj}},

    with the usual membership formula.

    *OCS* -- :math:`J` is minimized over :math:`(\mathbf{U},\mathbf{V},X_M)`;
    the necessary condition for a missing value is

    .. math::
        x_{kj}=\frac{\sum_{i=1}^{c}\mu_{ik}^m v_{ij}}{\sum_{i=1}^{c}\mu_{ik}^m},
        \qquad x_{kj}\in X_M ,

    and the memberships and prototypes are those of FCM on the completed
    data. The three-block alternating optimization decreases :math:`J`
    monotonically.

    *NPS* -- the missing value is set to the coordinate of the prototype
    nearest to the sample, :math:`x_{kj}=v_{ij}` with
    :math:`i=\arg\min_l D^2_{lk}`, the nearest prototype being judged by
    the partial distance over the observed features.

    *WDS* -- plain FCM on the complete samples; the incomplete ones are
    classified afterwards with partial distances.

    Hathaway and Bezdek ran the strategies on IRIS with 0--75 % of the
    values missing completely at random and compared the prototypes with
    those of FCM on the complete data: PDS and OCS were the most accurate,
    NPS close behind, and WDS degraded once a large share of the samples
    was affected; OCS has the extra merit of imputing the data. OCS and NPS
    change the data they cluster and therefore have more local optima than
    PDS -- use ``n_init`` or start them from the PDS prototypes
    (``init=pds.cluster_centers_``) when the result matters.

    References
    ----------
    .. [1] R. J. Hathaway, J. C. Bezdek, "Fuzzy c-means clustering of
           incomplete data", IEEE Transactions on Systems, Man, and
           Cybernetics, Part B 31(5), 735--744, 2001.
    .. [2] J. K. Dixon, "Pattern recognition with partly missing data",
           IEEE Transactions on Systems, Man, and Cybernetics 9(10),
           617--621, 1979.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.incomplete import IncompleteFCM, make_mcar
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> Xm = make_mcar(X, missing_rate=0.2, random_state=0)
    >>> model = IncompleteFCM(n_clusters=3, strategy="ocs", random_state=0).fit(Xm)
    >>> model.membership_.shape, bool(np.isnan(model.X_completed_).any())
    ((300, 3), False)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        strategy: str = "pds",
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.strategy = strategy

    # ------------------------------------------------------------------ validation
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("IncompleteFCM requires m > 1.")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}.")

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Cluster incomplete data (``NaN`` marks a missing value).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with ``NaN`` for missing values. Every sample must observe
            at least one feature and every feature must be observed at
            least once.
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan", ensure_min_samples=2)
        self._check_params(X)
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        mask = np.isnan(X)
        if mask.all(axis=1).any():
            raise ValueError("Every sample must have at least one observed feature.")
        if mask.all(axis=0).any():
            raise ValueError("Every feature must be observed in at least one sample.")
        self.missing_mask_ = mask
        self.n_missing_ = int(mask.sum())
        complete = ~mask.any(axis=1)
        means = np.nanmean(X, axis=0)
        self._X_init = np.where(mask, means, X)
        if self.strategy == "wds":
            if complete.sum() < self.n_clusters:
                raise ValueError(
                    f"strategy='wds' needs at least n_clusters={self.n_clusters} complete samples, "
                    f"found {int(complete.sum())}."
                )
            X_run = X[complete]
            self._mask = np.zeros_like(X_run, dtype=bool)
        else:
            X_run = X
            self._mask = mask
        self._X_fit = X_run
        self._X0 = np.where(self._mask, 0.0, X_run)
        self._I = (~self._mask).astype(float)
        self._X_init_run = self._X_init[complete] if self.strategy == "wds" else self._X_init

        rng = check_random_state(self.random_state)
        n_init = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for run in range(n_init):
            result = self._fit_single(X_run, rng)
            if best is None or result["history"][-1] < best["history"][-1]:
                best = result
            if self.verbose and n_init > 1:
                print(f"restart {run + 1}/{n_init}: J = {result['history'][-1]:.6g}")
        self.__dict__.update(best["state"])
        self.objective_history_ = np.asarray(best["history"])
        self.objective_ = best["history"][-1]
        self.n_iter_ = best["n_iter"]
        V = self.cluster_centers_

        # memberships of all samples (identical to the run's partition where the run saw them)
        if self.strategy == "wds":
            d2 = partial_distances(X, V)
            U = memberships_from_distances(d2, self.m)
        else:
            U, d2 = best["U"], best["d2"]
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.distances_ = np.sqrt(d2)
        Um = U**self.m
        if self.strategy in ("ocs", "nps"):
            Xc = self._Xc
        else:  # post-hoc imputation by the OCS formula
            est = (Um @ V) / np.maximum(Um.sum(axis=1, keepdims=True), EPS)
            Xc = np.where(mask, est, X)
        self.X_completed_ = Xc
        self.covariances_ = fuzzy_covariances(Xc, Um, V)
        return self

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        # The initial prototypes are seeded on the mean-filled data, but the
        # initial partition is computed from the *partial* distances, so that
        # the provisional fill-in values (column means) do not bias it; OCS and
        # NPS then start from the imputation of this partition instead of the
        # column means (a column mean can lie far from every cluster in its
        # coordinate and lure a sample into the wrong cluster).
        Xi = self._X_init_run
        c, n = self.n_clusters, Xi.shape[1]
        init = self.init
        if isinstance(init, str):
            if init == "random":
                mean = Xi.mean(axis=0)
                half_range = np.abs(Xi - mean).max(axis=0)
                V0 = 2.0 * half_range * (rng.rand(c, n) - 0.5) + mean
            elif init == "kmeans++":
                from sklearn.cluster import kmeans_plusplus

                V0, _ = kmeans_plusplus(Xi, n_clusters=c, random_state=rng)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
            U0 = memberships_from_distances(partial_distances(X, V0), self.m)
        elif np.asarray(init).shape == (c, n):
            V0 = np.asarray(init, dtype=float)
            U0 = memberships_from_distances(partial_distances(X, V0), self.m)
        else:
            U0 = super()._init_state(Xi, rng)
            V0 = fuzzy_centers(Xi, U0**self.m)
        self._Xc = Xi.copy()
        if self.strategy == "ocs":
            Um0 = U0**self.m
            est = (Um0 @ V0) / np.maximum(Um0.sum(axis=1, keepdims=True), EPS)
            self._Xc = np.where(self._mask, est, Xi)
        elif self.strategy == "nps":
            self._Xc = np.where(self._mask, V0[partial_distances(X, V0).argmin(axis=1)], Xi)
        return U0

    def _update_prototypes(self, X, Um):
        s = self.strategy
        if s == "wds":
            self.cluster_centers_ = fuzzy_centers(X, Um)
        elif s == "pds":
            num = Um.T @ self._X0
            den = Um.T @ self._I
            self.cluster_centers_ = num / np.maximum(den, EPS)
        else:
            V = fuzzy_centers(self._Xc, Um)
            self.cluster_centers_ = V
            if s == "ocs":
                est = (Um @ V) / np.maximum(Um.sum(axis=1, keepdims=True), EPS)
            else:  # nps: nearest prototype judged on the observed features
                nearest = partial_distances(X, V).argmin(axis=1)
                est = V[nearest]
            self._Xc = np.where(self._mask, est, self._Xc)

    def _distances(self, X):
        if X is self._X_fit:
            if self.strategy in ("ocs", "nps"):
                return sq_euclidean(self._Xc, self.cluster_centers_)
            if self.strategy == "wds":
                return sq_euclidean(X, self.cluster_centers_)
        return partial_distances(X, self.cluster_centers_)

    # ------------------------------------------------------------------ inference
    def predict_proba(self, X):
        """Memberships of (possibly incomplete) new samples from their partial distances.

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan", reset=False)
        return memberships_from_distances(partial_distances(X, self.cluster_centers_), self.m)

    def transform(self, X):
        """Partial distances :math:`D_{ik}` of (possibly incomplete) samples to the prototypes."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan", reset=False)
        return np.sqrt(partial_distances(X, self.cluster_centers_))

    def score(self, X, y=None):
        """Negative c-means functional with partial distances on ``X`` (higher is better)."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan", reset=False)
        d2 = partial_distances(X, self.cluster_centers_)
        U = memberships_from_distances(d2, self.m)
        return -float(np.sum(U**self.m * d2))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
