# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy segmentation of multivariate time series by modified Gath–Geva clustering."""

from __future__ import annotations

import numpy as np
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, regularize_covariance
from ..base import BaseFuzzyClusterer

_LOG_CAP = 700.0
_LOG2PI = np.log(2.0 * np.pi)


class FuzzySegmentation(BaseFuzzyClusterer):
    r"""Fuzzy segmentation of a multivariate time series (modified Gath–Geva clustering).

    Segmentation is clustering with a time-ordered structure: the samples of a
    segment must be consecutive in time. The algorithm of Abonyi, Feil, Németh
    and Árva [1]_ clusters the extended samples
    :math:`\mathbf{z}_k=[t_k,\ \mathbf{x}_k^T]^T` with a prototype that has two
    parts: a Gaussian membership function in time (center :math:`v^t_i`,
    variance :math:`\sigma^2_{i,t}`) that makes the clusters contiguous, and a
    local probabilistic PCA (PPCA) model of the observations (center
    :math:`\mathbf{v}^x_i`, loading matrix :math:`\mathbf{W}_i` with
    :math:`q` columns and noise variance :math:`\sigma^2_{i,x}`) that measures
    the homogeneity of the segment. Changes of the mean, of the variance and of
    the correlation structure of the variables are all detected, because the
    hyperplane of the local PCA model reacts to each of them. The number of
    segments can be reduced automatically by merging compatible adjacent
    clusters (a fuzzy decision based on the similarity of the PCA models and
    the distance of the centers).

    Parameters
    ----------
    n_segments : int, default=5
        Initial number of segments :math:`c`. With ``merge=True`` this is an
        upper bound; the paper starts from ten segments.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    n_components : int, float or None, default=None
        Number of principal components :math:`q` of the local PCA models. A
        float in ``(0, 1)`` is the required accuracy (fraction of variance
        explained) of eq. (29) of [1]_: :math:`q` is the smallest number of
        components that explains at least this fraction in every segment of
        the initial uniform segmentation. ``None`` uses an accuracy of 0.95.
        ``0`` gives spherical (isotropic) Gaussian segments.
    distance : {"ppca", "q", "t2"}, default="ppca"
        Observation part of the distance. ``"ppca"`` is the journal version:
        the Gaussian with covariance :math:`\mathbf{A}_i=\sigma^2_{i,x}\mathbf{I}
        +\mathbf{W}_i\mathbf{W}_i^T`. ``"q"`` and ``"t2"`` are the two
        variants of the conference version [2]_: a Gaussian restricted to the
        residual subspace (reconstruction error :math:`Q`, sensitive to changes
        of the correlation structure) or to the principal subspace (Hotelling
        :math:`T^2`, sensitive to shifts of the operating point).
    pca_update : {"em", "eig"}, default="em"
        How the PPCA parameters are updated in every iteration: by the EM
        step of eqs. (17)–(18) of [1]_ (as published) or by the closed-form
        eigen-solution of eqs. (11)–(12) (same fixed point, faster
        convergence).
    merge : bool, default=False
        Merge compatible adjacent segments during fitting (bottom-up
        reduction of the number of segments, Section 2.5 of [1]_).
    merge_threshold : float, default=0.75
        Threshold :math:`\gamma` on the compatibility :math:`O_{i,i+1}`; the
        paper recommends values between 0.3 and 0.75. Note that two segments
        with identical PCA models but distant centers have
        :math:`O=1/\sqrt2\approx0.71`, so :math:`\gamma` must exceed this
        value if shifts of the mean are to be preserved as separate segments.
    merge_every : int, default=100
        Merging is evaluated after this many iterations and at convergence.
    min_segment_weight : float, default=0.02
        Safeguard (not in the paper): during merging, a segment whose prior
        :math:`\alpha_i` is below this fraction of the data is absorbed by
        its more compatible neighbor regardless of :math:`\gamma`. Such
        tiny "transition" clusters appear when the initial number of
        segments is much larger than the true one. ``0`` disables it.
    gamma : float, default=0.0
        Shrinkage of the fuzzy covariance matrices toward a scaled identity
        (see :class:`~fuzzyclust.cluster.GK`).
    beta : float, default=1e15
        Condition-number bound of the fuzzy covariance matrices.
    tol : float, default=1e-4
    max_iter : int, default=300
        Maximum number of iterations per clustering stage (the counter is
        reset after every merge; ``n_iter_`` is the total).
    init : {"uniform", "random", "kmeans++"} or ndarray, default="uniform"
        ``"uniform"`` splits the series into ``n_segments`` equal intervals
        (the initialization of the paper); the other options are those of
        :class:`~fuzzyclust.base.BaseFuzzyClusterer` applied to
        :math:`\mathbf{z}_k`.
    n_init : int, default=1
        Number of restarts for the random initializations (``"random"``,
        ``"kmeans++"``); the run with the lowest final objective is kept.
        Ignored for the deterministic ``"uniform"`` and array initializations.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    membership_ : ndarray of shape (n_samples, ``n_segments_``)
        Fuzzy partition :math:`p(\eta_i|\mathbf{z}_k)`; segments are ordered
        in time.
    labels_ : ndarray of shape (n_samples,)
    boundaries_ : ndarray of int
        Crisp change points: indices :math:`k` where the segment of maximal
        time membership :math:`\beta_i(t_k)` changes (the fuzzy segments in
        time are the Gaussians; ``labels_`` is the argmax of the posterior
        ``membership_``, which also depends on the observations and may
        flicker around a boundary).
    n_segments_ : int
        Number of segments after merging.
    n_components_ : int
        Number of principal components :math:`q` used.
    segment_centers_ : ndarray of shape (``n_segments_``,)
        Time centers :math:`v^t_i`.
    segment_variances_ : ndarray of shape (``n_segments_``,)
        Time variances :math:`\sigma^2_{i,t}`.
    time_membership_ : ndarray of shape (n_samples, ``n_segments_``)
        Normalized Gaussian memberships in time, :math:`\beta_i(t_k)` of eq. (4).
    cluster_centers_ : ndarray of shape (``n_segments_``, n_features)
        Observation centers :math:`\mathbf{v}^x_i`.
    covariances_ : ndarray of shape (``n_segments_``, n_features, n_features)
        Fuzzy covariance matrices :math:`\mathbf{F}_i`.
    eigenvalues_, eigenvectors_ : ndarrays
        Eigen-decomposition of :math:`\mathbf{F}_i` in **descending** order
        (shapes ``(c, n)`` and ``(c, n, n)``, eigenvectors in columns).
    pca_loadings_ : ndarray of shape (``n_segments_``, n_features, ``n_components_``)
        PPCA loading matrices :math:`\mathbf{W}_i`.
    noise_variances_ : ndarray of shape (``n_segments_``,)
        PPCA noise variances :math:`\sigma^2_{i,x}`.
    norm_matrices_ : ndarray of shape (``n_segments_``, n_features, n_features)
        :math:`\mathbf{A}_i` of the observation distance.
    local_pca_ : list of :class:`sklearn.utils.Bunch`
        Per segment: ``mean``, ``components`` (``(q, n)``, principal axes),
        ``explained_variance`` (``(q,)``), ``noise_variance``.
    priors_ : ndarray of shape (``n_segments_``,)
        :math:`\alpha_i`.
    time_ : ndarray of shape (n_samples,)
        Time index used in ``fit``.
    objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    The objective is :math:`J=\sum_i\sum_k\mu_{ik}^m D^2(\mathbf{z}_k,\eta_i)`
    with the distance defined through a probability (eq. (8) of [1]_):

    .. math::
        \frac{1}{D^2(\mathbf{z}_k,\eta_i)} = \alpha_i\;
        \frac{1}{\sqrt{2\pi\sigma^2_{i,t}}}\exp\!\Big(-\tfrac12\frac{(t_k-v^t_i)^2}{\sigma^2_{i,t}}\Big)\;
        \frac{1}{(2\pi)^{r/2}\sqrt{\det\mathbf{A}_i}}
        \exp\!\Big(-\tfrac12(\mathbf{x}_k-\mathbf{v}^x_i)^T\mathbf{A}_i^{-1}(\mathbf{x}_k-\mathbf{v}^x_i)\Big),

    where :math:`r` is the rank of :math:`\mathbf{A}_i` (:math:`n` for the
    PPCA model, :math:`q` and :math:`n-q` for the ``"t2"`` and ``"q"``
    variants, with pseudo-determinant and pseudo-inverse). The alternating
    optimization updates

    .. math::
        \alpha_i=\frac1N\sum_k\mu_{ik},\qquad
        v^t_i=\frac{\sum_k\mu_{ik}^m t_k}{\sum_k\mu_{ik}^m},\qquad
        \sigma^2_{i,t}=\frac{\sum_k\mu_{ik}^m (t_k-v^t_i)^2}{\sum_k\mu_{ik}^m},

    the centers
    :math:`\mathbf{v}^x_i=\sum_k\mu_{ik}^m(\mathbf{x}_k-\mathbf{W}_i\langle\mathbf{y}_{ik}\rangle)/\sum_k\mu_{ik}^m`
    with the latent expectations
    :math:`\langle\mathbf{y}_{ik}\rangle=\mathbf{M}_i^{-1}\mathbf{W}_i^T(\mathbf{x}_k-\mathbf{v}^x_i)`,
    :math:`\mathbf{M}_i=\sigma^2_{i,x}\mathbf{I}+\mathbf{W}_i^T\mathbf{W}_i`,
    the fuzzy covariance :math:`\mathbf{F}_i`, and the PPCA EM step

    .. math::
        \tilde{\mathbf{W}}_i=\mathbf{F}_i\mathbf{W}_i\big(\sigma^2_{i,x}\mathbf{I}+\mathbf{M}_i^{-1}\mathbf{W}_i^T\mathbf{F}_i\mathbf{W}_i\big)^{-1},\qquad
        \sigma^2_{i,x}=\frac1n\tr\big(\mathbf{F}_i-\mathbf{F}_i\mathbf{W}_i\mathbf{M}_i^{-1}\tilde{\mathbf{W}}_i^T\big),

    (Tipping & Bishop; eq. (18) of [1]_ prints the factor as :math:`1/q`),
    followed by :math:`\mathbf{A}_i=\sigma^2_{i,x}\mathbf{I}+\tilde{\mathbf{W}}_i\tilde{\mathbf{W}}_i^T`
    and the usual membership update
    :math:`\mu_{ik}=1/\sum_j (D_{ik}/D_{jk})^{2/(m-1)}`.

    *Compatible cluster merging.* For every pair of segments the PCA
    similarity factor of Krzanowski
    :math:`S^{ij}_{PCA}=\frac1q\tr(\mathbf{U}_{i,q}^T\mathbf{U}_{j,q}\mathbf{U}_{j,q}^T\mathbf{U}_{i,q})`
    and the distance of the centers :math:`\|\mathbf{v}^x_i-\mathbf{v}^x_j\|`
    are evaluated, turned into degrees of *parallelism* and *closeness* by
    triangular membership functions whose knots are the averages
    :math:`\nu^1,\nu^2` over all pairs, and aggregated to the compatibility
    :math:`O_{ij}=\sqrt{(\mu^1_{ij})^2+(\mu^2_{ij})^2)/2}`. The most
    compatible adjacent pair is merged (Kelly's formulas for the merged
    mean and covariance) whenever :math:`O_{i,i+1}>\gamma`, the PPCA model
    of the new segment is computed from the eigen-decomposition of its
    covariance, and clustering continues.

    ``predict``/``predict_proba``/``transform`` evaluate the fitted prototypes
    for given observations **and time stamps**; without ``t`` they assume the
    rows are the training series (or a series starting at time 0), because the
    time part of the prototype has no meaning for unordered new data.

    References
    ----------
    .. [1] J. Abonyi, B. Feil, S. Németh, P. Árva, "Modified Gath–Geva
           clustering for fuzzy segmentation of multivariate time-series",
           Fuzzy Sets and Systems 149 (2005) 39–56.
    .. [2] J. Abonyi, B. Feil, S. Németh, P. Árva, "Fuzzy clustering based
           segmentation of time-series", IDA 2003, LNCS 2810, 275–285.
    .. [3] M. E. Tipping, C. M. Bishop, "Mixtures of probabilistic principal
           component analyzers", Neural Computation 11 (1999) 443–482.
    .. [4] W. J. Krzanowski, "Between-groups comparison of principal
           components", J. Amer. Statist. Assoc. 74 (1979) 703–707.

    Examples
    --------
    >>> from fuzzyclust.timeseries import FuzzySegmentation, make_piecewise_timeseries
    >>> X, seg = make_piecewise_timeseries(n_samples=600, change_points=(200, 400),
    ...                                    change_kinds=("mean", "mean"), random_state=0)
    >>> fs = FuzzySegmentation(n_segments=3, n_components=2).fit(X)
    >>> fs.membership_.shape, fs.boundaries_
    ((600, 3), array([200, 400]))
    """

    def __init__(
        self,
        n_segments: int = 5,
        m: float = 2.0,
        n_components=None,
        distance: str = "ppca",
        pca_update: str = "em",
        merge: bool = False,
        merge_threshold: float = 0.75,
        merge_every: int = 100,
        min_segment_weight: float = 0.02,
        gamma: float = 0.0,
        beta: float = 1e15,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="uniform",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_segments = n_segments
        self.m = m
        self.n_components = n_components
        self.distance = distance
        self.pca_update = pca_update
        self.merge = merge
        self.merge_threshold = merge_threshold
        self.merge_every = merge_every
        self.min_segment_weight = min_segment_weight
        self.gamma = gamma
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # the base class speaks of n_clusters; segments are clusters here
    @property
    def n_clusters(self):
        return self.n_segments

    @n_clusters.setter
    def n_clusters(self, value):
        self.n_segments = value

    # ------------------------------------------------------------------ validation
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("FuzzySegmentation requires m > 1.")
        if self.distance not in ("ppca", "q", "t2"):
            raise ValueError("distance must be 'ppca', 'q' or 't2'.")
        if self.pca_update not in ("em", "eig"):
            raise ValueError("pca_update must be 'em' or 'eig'.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if self.merge_every < 1:
            raise ValueError("merge_every must be >= 1.")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")

    def _check_time(self, X, t):
        if t is None:
            return np.arange(X.shape[0], dtype=float)
        t = check_array(t, ensure_2d=False, dtype=float)
        if t.ndim != 1 or t.shape[0] != X.shape[0]:
            raise ValueError("t must be a 1-D array with one entry per row of X.")
        if np.any(np.diff(t) < 0):
            raise ValueError("t must be non-decreasing.")
        return t

    def _select_n_components(self, X, t):
        n = X.shape[1]
        nc = self.n_components
        if nc is None:
            nc = 0.95
        if isinstance(nc, (int, np.integer)) and not isinstance(nc, bool):
            q = int(nc)
            if q < 0 or q > n - 1:
                raise ValueError(
                    f"n_components={q} must lie in [0, n_features - 1] = [0, {n - 1}] (n_features = {n})."
                )
            return q
        if not (0.0 < float(nc) < 1.0):
            raise ValueError("n_components must be an int or a float accuracy in (0, 1).")
        acc = float(nc)
        q = 0
        for seg in np.array_split(np.arange(X.shape[0]), self.n_segments):
            if len(seg) < 2:
                continue
            lam = np.sort(np.linalg.eigvalsh(np.cov(X[seg], rowvar=False).reshape(n, n)))[::-1]
            lam = np.maximum(lam, 0)
            if lam.sum() <= 0:
                continue
            cum = np.cumsum(lam) / lam.sum()
            q = max(q, int(np.searchsorted(cum, acc - 1e-12) + 1))
        return int(min(max(q, 0), n - 1))

    # ------------------------------------------------------------------ initialization
    def _init_state(self, Z, rng):
        if isinstance(self.init, str) and self.init == "uniform":
            N, c = Z.shape[0], self.n_segments
            labels = (np.arange(N) * c) // N
            U = np.full((N, c), 0.0)
            U[np.arange(N), labels] = 1.0
            return U
        return super()._init_state(Z, rng)

    # ------------------------------------------------------------------ prototypes
    def _pca_from_covariance(self, F, q):
        """Closed-form PPCA (eqs. (11)-(12)): loadings W and noise variance from F."""
        n = F.shape[0]
        lam, Q = np.linalg.eigh(F)
        lam, Q = lam[::-1], Q[:, ::-1]
        lam = np.maximum(lam, 0.0)
        s2 = lam[q:].mean() if q < n else EPS
        s2 = max(s2, EPS)
        W = Q[:, :q] * np.sqrt(np.maximum(lam[:q] - s2, 0.0))
        return W, s2, lam, Q

    def _update_prototypes(self, Z, Um):
        t, X = Z[:, 0], Z[:, 1:]
        N, n = X.shape
        c = Um.shape[1]
        q = self.n_components_
        s = np.maximum(Um.sum(axis=0), EPS)
        U = Um ** (1.0 / self.m)
        self.priors_ = np.maximum(U.sum(axis=0) / N, EPS)
        # time part (eq. 20)
        vt = (Um.T @ t) / s
        st2 = np.einsum("ki,ki->i", Um, (t[:, None] - vt) ** 2) / s
        self.segment_centers_ = vt
        self.segment_variances_ = np.maximum(st2, self._min_time_var)
        # observation part
        have_model = hasattr(self, "pca_loadings_") and self.pca_loadings_.shape[0] == c
        vx_mean = fuzzy_centers(X, Um)
        vx = np.empty((c, n))
        for i in range(c):
            if have_model and q > 0:
                W, s2, v_old = self.pca_loadings_[i], self.noise_variances_[i], self.cluster_centers_[i]
                M = s2 * np.eye(q) + W.T @ W
                Y = (X - v_old) @ W @ np.linalg.inv(M).T  # <y_ik>, eq. (16)
                vx[i] = Um[:, i] @ (X - Y @ W.T) / s[i]
            else:
                vx[i] = vx_mean[i]
        F = fuzzy_covariances(X, Um, vx)
        W_new = np.empty((c, n, q))
        s2_new = np.empty(c)
        A = np.empty((c, n, n))
        lam_all = np.empty((c, n))
        Q_all = np.empty((c, n, n))
        for i in range(c):
            Fi = regularize_covariance(F[i], self._F0, self.gamma, self.beta)
            F[i] = Fi
            W_eig, s2_eig, lam, Q = self._pca_from_covariance(Fi, q)
            lam_all[i], Q_all[i] = lam, Q
            if self.pca_update == "em" and have_model and q > 0:
                W, s2 = self.pca_loadings_[i], self.noise_variances_[i]
                Minv = np.linalg.inv(s2 * np.eye(q) + W.T @ W)
                FW = Fi @ W
                W_t = FW @ np.linalg.inv(s2 * np.eye(q) + Minv @ W.T @ FW)  # eq. (17)
                s2_t = np.trace(Fi - FW @ Minv @ W_t.T) / n  # eq. (18), Tipping-Bishop
                W_new[i], s2_new[i] = W_t, max(s2_t, EPS)
            else:
                W_new[i], s2_new[i] = W_eig, s2_eig
            A[i] = s2_new[i] * np.eye(n) + W_new[i] @ W_new[i].T  # eq. (19)
        self.cluster_centers_ = vx
        self.covariances_ = F
        self.pca_loadings_ = W_new
        self.noise_variances_ = s2_new
        self.norm_matrices_ = A
        self.eigenvalues_ = lam_all
        self.eigenvectors_ = Q_all
        self._prepare_distance()

    def _prepare_distance(self):
        """Pseudo-inverse and log pseudo-determinant of A_i for the chosen distance."""
        c, n = self.cluster_centers_.shape
        q = self.n_components_
        inv = np.empty((c, n, n))
        logdet = np.empty(c)
        if self.distance == "ppca":
            rank = n
            for i in range(c):
                inv[i] = np.linalg.inv(self.norm_matrices_[i])
                sign, ld = np.linalg.slogdet(self.norm_matrices_[i])
                logdet[i] = ld if sign > 0 else np.log(EPS)
        else:
            sl = slice(0, q) if self.distance == "t2" else slice(q, n)
            rank = q if self.distance == "t2" else n - q
            if rank == 0:
                raise ValueError(
                    f"distance='{self.distance}' needs a non-empty subspace (check n_components)."
                )
            for i in range(c):
                lam = np.maximum(self.eigenvalues_[i, sl], EPS)
                Q = self.eigenvectors_[i][:, sl]
                inv[i] = (Q / lam) @ Q.T
                logdet[i] = np.log(lam).sum()
        self._rank = rank
        self._inv_A = inv
        self._logdet_A = logdet

    def _log_distances(self, Z):
        t, X = Z[:, 0], Z[:, 1:]
        c = self.cluster_centers_.shape[0]
        logd2 = np.empty((X.shape[0], c))
        for i in range(c):
            dt = t - self.segment_centers_[i]
            st2 = self.segment_variances_[i]
            log_time = 0.5 * (_LOG2PI + np.log(st2)) + 0.5 * dt**2 / st2
            xv = X - self.cluster_centers_[i]
            maha = np.einsum("kj,kj->k", xv @ self._inv_A[i], xv)
            log_obs = 0.5 * self._rank * _LOG2PI + 0.5 * self._logdet_A[i] + 0.5 * maha
            logd2[:, i] = -np.log(self.priors_[i]) + log_time + log_obs
        return logd2

    def _distances(self, Z):
        return np.exp(np.minimum(self._log_distances(Z), _LOG_CAP))

    def _memberships(self, d2):
        logits = -np.log(d2 + EPS) / (self.m - 1.0)
        logits -= logits.max(axis=1, keepdims=True)
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ merging
    def compatibility_matrix(self):
        r"""Compatibility :math:`O_{ij}` of all pairs of segments (eqs. (22)–(26)).

        Returns
        -------
        O : ndarray of shape (c, c)
        S_pca : ndarray of shape (c, c)
            Krzanowski PCA similarity factors.
        D : ndarray of shape (c, c)
            Euclidean distances between the observation centers.
        """
        check_is_fitted(self, "cluster_centers_")
        return self._compatibility()

    def _compatibility(self):
        V = self.cluster_centers_
        c = V.shape[0]
        q = max(self.n_components_, 1)
        Uq = self.eigenvectors_[:, :, :q]
        S = np.empty((c, c))
        for i in range(c):
            for j in range(c):
                C = Uq[i].T @ Uq[j]
                S[i, j] = np.sum(C * C) / q
        D = np.sqrt(np.maximum(((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=2), 0.0))
        off = ~np.eye(c, dtype=bool)
        if c < 2:
            return np.ones((c, c)), S, D
        v1 = S[off].mean()
        v2 = D[off].mean()
        mu1 = np.clip((S - v1) / max(1.0 - v1, EPS), 0.0, 1.0)  # parallelism
        mu2 = np.clip(1.0 - D / max(v2, EPS), 0.0, 1.0)  # closeness
        O = np.sqrt((mu1**2 + mu2**2) / 2.0)
        return O, S, D

    def _order_in_time(self, U):
        order = np.argsort(self.segment_centers_)
        if np.all(order == np.arange(order.size)):
            return U
        for name in (
            "priors_", "segment_centers_", "segment_variances_", "cluster_centers_", "covariances_",
            "pca_loadings_", "noise_variances_", "norm_matrices_", "eigenvalues_", "eigenvectors_",
            "_inv_A", "_logdet_A",
        ):
            setattr(self, name, getattr(self, name)[order])
        return U[:, order]

    def _try_merge(self, Z, U, threshold):
        """Merge the most compatible adjacent pair if its compatibility exceeds the threshold."""
        c = U.shape[1]
        if c < 2:
            return False, U
        U = self._order_in_time(U)
        O, _, _ = self._compatibility()
        adj = np.array([O[i, i + 1] for i in range(c - 1)])
        weights = U.sum(axis=0) / U.shape[0]
        if self.min_segment_weight > 0 and weights.min() < self.min_segment_weight:
            k = int(weights.argmin())  # absorb the tiny segment into its more compatible neighbor
            cand = [idx for idx in (k - 1, k) if 0 <= idx < c - 1]
            i = max(cand, key=lambda idx: adj[idx])
        else:
            i = int(adj.argmax())
            if adj[i] <= threshold:
                return False, U
        j = i + 1
        Ni, Nj = U[:, i].sum(), U[:, j].sum()
        Ns = Ni + Nj
        V, F = self.cluster_centers_, self.covariances_
        v_new = (Ni * V[i] + Nj * V[j]) / Ns  # eq. (27)
        dv = V[i] - V[j]
        F_new = (
            (Ni - 1) / (Ns - 1) * F[i] + (Nj - 1) / (Ns - 1) * F[j]
            + Ni * Nj / (Ns * (Ns - 1)) * np.outer(dv, dv)
        )  # eq. (28)
        vt, st2 = self.segment_centers_, self.segment_variances_
        vt_new = (Ni * vt[i] + Nj * vt[j]) / Ns
        st2_new = (
            (Ni - 1) / (Ns - 1) * st2[i] + (Nj - 1) / (Ns - 1) * st2[j]
            + Ni * Nj / (Ns * (Ns - 1)) * (vt[i] - vt[j]) ** 2
        )
        keep = [k for k in range(c) if k != j]
        q = self.n_components_
        W_new, s2_new, lam, Q = self._pca_from_covariance(F_new, q)
        self.cluster_centers_ = V[keep]
        self.cluster_centers_[i] = v_new
        self.covariances_ = F[keep]
        self.covariances_[i] = F_new
        self.segment_centers_ = vt[keep]
        self.segment_centers_[i] = vt_new
        self.segment_variances_ = st2[keep]
        self.segment_variances_[i] = max(st2_new, self._min_time_var)
        alpha_new = self.priors_[i] + self.priors_[j]  # p(eta_i*) = p(eta_i) + p(eta_i+1)
        self.priors_ = self.priors_[keep]
        self.priors_[i] = alpha_new
        self.pca_loadings_ = self.pca_loadings_[keep]
        self.pca_loadings_[i] = W_new
        self.noise_variances_ = self.noise_variances_[keep]
        self.noise_variances_[i] = s2_new
        self.norm_matrices_ = self.norm_matrices_[keep]
        self.norm_matrices_[i] = s2_new * np.eye(F_new.shape[0]) + W_new @ W_new.T
        self.eigenvalues_ = self.eigenvalues_[keep]
        self.eigenvalues_[i] = lam
        self.eigenvectors_ = self.eigenvectors_[keep]
        self.eigenvectors_[i] = Q
        self._prepare_distance()
        U_new = self._memberships(self._distances(Z))
        if self.verbose:
            print(f"merged segments {i} and {j} (O = {adj[i]:.3f}), {c - 1} segments left")
        return True, U_new

    # ------------------------------------------------------------------ fitting
    def _ao(self, Z, U, allow_merge, threshold, max_iter):
        """Alternating optimization with optional merging; ``max_iter`` applies per merging stage."""
        history = []
        total, stage = 0, 0
        while True:
            total += 1
            stage += 1
            Um = U**self.m
            self._update_prototypes(Z, Um)
            d2 = self._distances(Z)
            U_new = self._memberships(d2)
            history.append(self._objective(Um, d2))
            if self.verbose:
                print(f"iter {total:4d}  J = {history[-1]:.6g}  c = {U.shape[1]}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            converged = delta < self.tol
            stage_done = converged or stage >= max_iter
            if allow_merge and (stage_done or stage % self.merge_every == 0):
                merged, U = self._try_merge(Z, U, threshold)
                if merged:
                    stage = 0
                    continue
            if stage_done:
                break
        return U, history, total

    def fit(self, X, y=None, t=None):
        """Segment the series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations in time order.
        y : Ignored
        t : array-like of shape (n_samples,), optional
            Time stamps (non-decreasing). Defaults to ``0, 1, ..., n_samples-1``.
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        t = self._check_time(X, t)
        self._check_params(X)
        rng = check_random_state(self.random_state)
        n = X.shape[1]
        self.time_ = t
        self.n_components_ = self._select_n_components(X, t)
        self._F0 = np.cov(X, rowvar=False).reshape(n, n) if X.shape[0] > 1 else np.eye(n)
        dt = np.diff(np.unique(t))
        self._min_time_var = (0.5 * np.median(dt)) ** 2 if dt.size else 1.0
        for name in ("pca_loadings_", "noise_variances_", "cluster_centers_"):
            if hasattr(self, name):
                delattr(self, name)
        Z = np.column_stack([t, X])
        self._Z = Z
        n_init = self.n_init if isinstance(self.init, str) and self.init != "uniform" else 1
        best = None
        for run in range(n_init):
            for name in ("pca_loadings_", "noise_variances_", "cluster_centers_"):
                if hasattr(self, name):
                    delattr(self, name)
            U = self._init_state(Z, rng)
            U, history, it = self._ao(Z, U, self.merge, self.merge_threshold, self.max_iter)
            keep = ("_inv_A", "_logdet_A", "_rank")
            state = {k: v for k, v in self.__dict__.items() if k.endswith("_") or k in keep}
            if self.verbose and n_init > 1:
                print(f"restart {run + 1}/{n_init}: J = {history[-1]:.6g}")
            if best is None or history[-1] < best[1][-1]:
                best = (U, history, it, state)
        U, history, it, state = best
        self.__dict__.update(state)
        self._store(Z, U, history, it)
        return self

    def _store(self, Z, U, history, it):
        U = self._order_in_time(U)
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.distances_ = np.sqrt(self._distances(Z))
        self.objective_history_ = np.asarray(history)
        self.objective_ = history[-1]
        self.n_iter_ = it
        self.n_segments_ = U.shape[1]
        t = Z[:, 0]
        logA = -0.5 * (t[:, None] - self.segment_centers_) ** 2 / self.segment_variances_
        A = np.exp(logA - logA.max(axis=1, keepdims=True))
        self.time_membership_ = A / A.sum(axis=1, keepdims=True)
        self.boundaries_ = np.flatnonzero(np.diff(self.time_membership_.argmax(axis=1)) != 0) + 1
        q = self.n_components_
        self.local_pca_ = [
            Bunch(
                mean=self.cluster_centers_[i],
                components=self.eigenvectors_[i][:, :q].T,
                explained_variance=self.eigenvalues_[i, :q],
                noise_variance=self.noise_variances_[i],
            )
            for i in range(self.n_segments_)
        ]

    def merge_compatible_segments(self, threshold=None, max_merges=None):
        r"""Merge compatible adjacent segments of a fitted model and re-cluster.

        Repeats *merge the most compatible adjacent pair, run the alternating
        optimization to convergence* until no adjacent pair has a
        compatibility :math:`O_{i,i+1}` above ``threshold`` (default:
        ``merge_threshold``) or ``max_merges`` merges were made.

        Returns
        -------
        self
        """
        check_is_fitted(self, "cluster_centers_")
        threshold = self.merge_threshold if threshold is None else float(threshold)
        Z, U = self._Z, self.membership_
        history = list(self.objective_history_)
        n_merges, total_it = 0, self.n_iter_
        while max_merges is None or n_merges < max_merges:
            merged, U = self._try_merge(Z, U, threshold)
            if not merged:
                break
            n_merges += 1
            U, h, it = self._ao(Z, U, False, threshold, self.max_iter)
            history += h
            total_it += it
        self._store(Z, U, history, total_it)
        return self

    # ------------------------------------------------------------------ inference
    def _check_X_t(self, X, t):
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        if t is None:
            if X.shape[0] == self.time_.shape[0]:
                t = self.time_
            else:
                t = np.arange(X.shape[0], dtype=float)
        else:
            t = self._check_time(X, t)
        return np.column_stack([t, X])

    def predict_proba(self, X, t=None):
        """Memberships of observations ``X`` at times ``t`` (default: the training time axis)."""
        return self._memberships(self._distances(self._check_X_t(X, t)))

    def predict(self, X, t=None):
        """Hard segment labels of observations ``X`` at times ``t``."""
        return self.predict_proba(X, t).argmax(axis=1)

    def transform(self, X, t=None):
        """Distances :math:`D(\\mathbf{z}_k,\\eta_i)` of observations at times ``t`` to the segments."""
        return np.sqrt(self._distances(self._check_X_t(X, t)))

    def score(self, X, y=None, t=None):
        Z = self._check_X_t(X, t)
        d2 = self._distances(Z)
        U = self._memberships(d2)
        return -self._objective(U**self.m, d2)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_segment_{i}" for i in range(self.n_segments_)], dtype=object)

    def hotelling_t2(self, X):
        r"""Hotelling :math:`T^2` of every observation with respect to every local PCA model.

        .. math::
            T^2_{ik} = \mathbf{y}_{ik}^T\boldsymbol{\Lambda}_{i,q}^{-1}\mathbf{y}_{ik},\qquad
            \mathbf{y}_{ik} = \mathbf{U}_{i,q}^T(\mathbf{x}_k-\mathbf{v}^x_i),

        the squared Mahalanobis distance of the projection inside the
        principal subspace (drift of the operating point).

        Returns
        -------
        T2 : ndarray of shape (n_samples, ``n_segments_``)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        q = self.n_components_
        out = np.empty((X.shape[0], self.n_segments_))
        for i in range(self.n_segments_):
            Y = (X - self.cluster_centers_[i]) @ self.eigenvectors_[i][:, :q]
            out[:, i] = (Y**2 / np.maximum(self.eigenvalues_[i, :q], EPS)).sum(axis=1)
        return out

    def q_reconstruction_error(self, X):
        r"""Reconstruction error :math:`Q` of every observation for every local PCA model.

        .. math::
            Q_{ik} = \|(\mathbf{I}-\mathbf{U}_{i,q}\mathbf{U}_{i,q}^T)(\mathbf{x}_k-\mathbf{v}^x_i)\|^2,

        the squared distance from the hyperplane of the segment (change of
        the correlation structure).

        Returns
        -------
        Q : ndarray of shape (n_samples, ``n_segments_``)
        """
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        q = self.n_components_
        out = np.empty((X.shape[0], self.n_segments_))
        for i in range(self.n_segments_):
            xv = X - self.cluster_centers_[i]
            P = self.eigenvectors_[i][:, :q]
            resid = xv - (xv @ P) @ P.T
            out[:, i] = (resid**2).sum(axis=1)
        return out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


__all__ = ["FuzzySegmentation"]
