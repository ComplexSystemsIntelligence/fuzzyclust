# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Generalized fuzzifiers: regularized fuzzy c-means, deterministic annealing and the polynomial fuzzifier.

Fuzzy c-means makes a partition fuzzy by raising the memberships to the power
``m``. Two other mechanisms do the same job with different properties: a
*regularization term* added to the hard c-means functional (entropy or
quadratic) and a *polynomial fuzzifier* that replaces :math:`\mu^m`. The
entropy term gives Gaussian-shaped memberships and is the free energy of
deterministic annealing; the quadratic term and the polynomial fuzzifier give
memberships that are exactly zero for distant clusters.
"""

from __future__ import annotations

import numpy as np

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, sq_euclidean
from ..base import BaseFuzzyClusterer

_REGULARIZATIONS = ("entropy", "quadratic")


def entropy_memberships(d2: np.ndarray, lam: float) -> np.ndarray:
    r"""Memberships of the entropy-regularized c-means, :math:`\mu_{ik}\propto\exp(-D^2_{ik}/\lambda)`.

    Evaluated as a softmax over the clusters after subtracting the largest
    logit of every sample, so that it does not underflow for small
    :math:`\lambda`.

    Parameters
    ----------
    d2 : ndarray of shape (n_samples, n_clusters)
        Squared distances.
    lam : float
        Regularization weight (temperature) :math:`\lambda>0`.
    """
    logits = -np.asarray(d2, dtype=float) / lam
    logits -= logits.max(axis=1, keepdims=True)
    U = np.exp(logits)
    return U / U.sum(axis=1, keepdims=True)


def quadratic_memberships(d2: np.ndarray, lam: float) -> np.ndarray:
    r"""Memberships of the quadratically regularized c-means (Miyamoto & Umayahara).

    Minimizes :math:`\sum_i\mu_i D^2_i+\lambda\sum_i\mu_i^2` over the
    simplex. For the set :math:`A_k` of clusters with positive membership,

    .. math::
        \mu_{ik}=\frac{1}{|A_k|}\Bigl(1+\frac{1}{2\lambda}\sum_{j\in A_k}(D^2_{jk}-D^2_{ik})\Bigr),

    and :math:`\mu_{ik}=0` otherwise; :math:`A_k` consists of the
    :math:`\hat c_k` nearest prototypes, where :math:`\hat c_k` is the largest
    number for which the formula is positive. This is the Euclidean
    projection of :math:`-\mathbf{D}^2_{\cdot k}/(2\lambda)` onto the simplex.

    Parameters
    ----------
    d2 : ndarray of shape (n_samples, n_clusters)
    lam : float
        Regularization weight :math:`\lambda>0`.
    """
    d2 = np.asarray(d2, dtype=float)
    N, c = d2.shape
    z = -d2 / (2.0 * lam)  # memberships are the projection of z onto the simplex
    zs = -np.sort(-z, axis=1)  # descending
    cum = np.cumsum(zs, axis=1)
    counts = np.arange(1, c + 1)[None, :]
    nu = (cum - 1.0) / counts  # candidate multiplier for every support size
    active = zs - nu > 0  # the size k is feasible if the k-th largest entry stays positive
    k = active.sum(axis=1)  # feasible sizes are the leading ones; k = number of positive memberships
    theta = nu[np.arange(N), k - 1]
    return np.maximum(z - theta[:, None], 0.0)


def polynomial_fuzzifier(U: np.ndarray, beta: float) -> np.ndarray:
    r"""Polynomial fuzzifier :math:`g(\mu)=\frac{1-\beta}{1+\beta}\mu^2+\frac{2\beta}{1+\beta}\mu`
    of Klawonn and Höppner."""
    return (1.0 - beta) / (1.0 + beta) * U**2 + 2.0 * beta / (1.0 + beta) * U


def polynomial_memberships(d2: np.ndarray, beta: float) -> np.ndarray:
    r"""Memberships of fuzzy c-means with the polynomial fuzzifier (Klawonn & Höppner).

    For the set :math:`A_k` of the :math:`\hat c_k` nearest prototypes,

    .. math::
        \mu_{ik}=\frac{1}{1-\beta}\left(\frac{1+(\hat c_k-1)\beta}
        {\sum_{j\in A_k}D^2_{ik}/D^2_{jk}}-\beta\right),

    and :math:`\mu_{ik}=0` for the remaining clusters; :math:`\hat c_k` is the
    largest number of nearest prototypes for which the formula is positive
    for all of them. ``beta=0`` gives the FCM memberships with :math:`m=2`.

    Parameters
    ----------
    d2 : ndarray of shape (n_samples, n_clusters)
    beta : float
        Fuzzifier parameter :math:`0\le\beta<1`.
    """
    d2 = np.asarray(d2, dtype=float) + EPS
    N, c = d2.shape
    if beta <= 0.0:
        inv = 1.0 / d2
        return inv / inv.sum(axis=1, keepdims=True)
    order = np.argsort(d2, axis=1)
    ds = np.take_along_axis(d2, order, axis=1)  # ascending distances
    cum_inv = np.cumsum(1.0 / ds, axis=1)
    counts = np.arange(1, c + 1)[None, :]
    # membership of the chat-th nearest prototype if exactly chat prototypes are active
    mu_last = (1.0 + (counts - 1) * beta) / (ds * cum_inv) - beta
    chat = (mu_last > 0).sum(axis=1)  # feasible sizes are the leading ones
    idx = np.arange(N)
    nu = (1.0 + (chat - 1) * beta) / cum_inv[idx, chat - 1]  # multiplier of the active set
    Us = np.maximum(nu[:, None] / ds - beta, 0.0) / (1.0 - beta)
    Us[np.arange(c)[None, :] >= chat[:, None]] = 0.0
    U = np.empty_like(Us)
    np.put_along_axis(U, order, Us, axis=1)
    return U


class RegularizedFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means by entropy or quadratic regularization, with optional deterministic annealing.

    Instead of fuzzifying the hard c-means functional with the exponent
    :math:`m`, a regularization term weighted by :math:`\lambda` is added to
    it. The *entropy* term (Li & Mukaidono; Miyamoto & Mukaidono) yields
    memberships proportional to :math:`\exp(-D^2_{ik}/\lambda)`, so a
    membership function is a Gaussian bump around every prototype and
    :math:`\lambda` plays the role of a temperature: :math:`\lambda\to0`
    gives k-means, :math:`\lambda\to\infty` gives uniform memberships. The
    same functional is the free energy of *deterministic annealing* (Rose),
    which tracks the minimum while the temperature is lowered from a value
    at which all prototypes coincide: the prototypes split at critical
    temperatures (phase transitions), so that the number of distinct
    clusters grows with decreasing :math:`\lambda`. The *quadratic* term
    (Miyamoto & Umayahara) yields piecewise linear memberships that are
    exactly zero for prototypes much farther away than the nearest one, so
    distant clusters receive no membership at all.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c` (with annealing: the maximal number of
        distinct prototypes; coinciding prototypes are counted once in
        ``n_effective_clusters_``).
    regularization : {"entropy", "quadratic"}, default="entropy"
        The regularization term.
    lam : float, default=1.0
        Regularization weight :math:`\lambda>0` in the units of the squared
        distances (the final temperature when annealing). Scale it with the
        squared cluster radius.
    annealing : bool, default=False
        Run deterministic annealing: start at ``lam_start`` with all
        prototypes at the grand mean, converge, lower :math:`\lambda`
        geometrically by ``anneal_rate`` and continue from the previous
        solution until ``lam`` is reached. Only for ``"entropy"``.
    lam_start : float or None, default=None
        Initial temperature. ``None`` uses :math:`2.2\lambda_{\max}` of the
        data covariance, just above the critical temperature of the first
        split :math:`2\lambda_{\max}(\mathbf{F}_0)` for the Euclidean distance.
    anneal_rate : float, default=0.8
        Geometric cooling factor :math:`\alpha\in(0,1)`,
        :math:`\lambda\leftarrow\alpha\lambda`.
    perturbation : float, default=1e-3
        Relative size of the random perturbation of the prototypes after
        every cooling step (needed to break the symmetry of coinciding
        prototypes at a phase transition).
    tol : float, default=1e-4
    max_iter : int, default=300
        Maximum number of iterations (per temperature when annealing).
    init : {"random", "kmeans++"} or ndarray, default="random"
        Ignored when annealing (all prototypes start at the grand mean).
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition; for ``"quadratic"`` it contains exact zeros.
    labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; with annealing
        ``objective_history_`` holds the converged objective at every
        temperature and ``n_iter_`` the total number of iterations.
    lam_ : float
        The regularization weight actually used (the final temperature).
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices of the final partition (weights :math:`\mu_{ik}`).
    cluster_history_ : pandas.DataFrame
        Only with annealing: one row per temperature with columns ``lam``
        (temperature), ``n_effective`` (number of distinct prototypes) and
        ``objective``; a jump of ``n_effective`` marks a phase transition.
    n_effective_clusters_ : int
        Only with annealing: number of distinct prototypes at the end.
    critical_temperatures_ : ndarray of shape (n_clusters,)
        Only with annealing: :math:`2\lambda_{\max}(\mathbf{F}_i)` of every
        final cluster, the temperature below which it would split further.

    Notes
    -----
    The entropy-regularized functional is

    .. math::
        J_E=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}D^2_{ik}+\lambda\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}\log\mu_{ik},

    whose necessary conditions are the Gibbs memberships and the
    membership-weighted means,

    .. math::
        \mu_{ik}=\frac{\exp(-D^2_{ik}/\lambda)}{\sum_{j=1}^{c}\exp(-D^2_{jk}/\lambda)},\qquad
        \mathbf{v}_i=\frac{\sum_k\mu_{ik}\mathbf{x}_k}{\sum_k\mu_{ik}} .

    The quadratic functional :math:`J_Q=\sum_{ik}\mu_{ik}D^2_{ik}+\lambda\sum_{ik}\mu_{ik}^2`
    has the closed-form solution of :func:`quadratic_memberships`. In both
    cases the memberships enter the prototypes linearly (no exponent).
    Deterministic annealing minimizes the free energy
    :math:`F=-\lambda\sum_k\log\sum_i\exp(-D^2_{ik}/\lambda)`, whose
    stationary points are exactly the equations above; a cluster with
    covariance :math:`\mathbf{F}_i` splits when :math:`\lambda` falls below
    :math:`2\lambda_{\max}(\mathbf{F}_i)`.

    References
    ----------
    .. [1] S. Miyamoto, M. Mukaidono, "Fuzzy c-means as a regularization and
           maximum entropy approach", Proc. 7th IFSA World Congress, Prague,
           vol. II, 86--92, 1997.
    .. [2] R.-P. Li, M. Mukaidono, "A maximum-entropy approach to fuzzy
           clustering", Proc. FUZZ-IEEE/IFES, Yokohama, 2227--2232, 1995.
    .. [3] S. Miyamoto, K. Umayahara, "Fuzzy clustering by quadratic
           regularization", Proc. FUZZ-IEEE, Anchorage, 1394--1399, 1998.
    .. [4] K. Rose, "Deterministic annealing for clustering, compression,
           classification, regression, and related optimization problems",
           Proc. IEEE 86(11), 2210--2239, 1998.

    Examples
    --------
    >>> from fuzzyclust.cluster import RegularizedFCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> efcm = RegularizedFCM(n_clusters=3, regularization="entropy", lam=0.5, random_state=0).fit(X)
    >>> efcm.membership_.shape
    (300, 3)
    >>> qfcm = RegularizedFCM(n_clusters=3, regularization="quadratic", lam=2.0, random_state=0).fit(X)
    >>> bool((qfcm.membership_ == 0).any())
    True
    >>> da = RegularizedFCM(n_clusters=3, lam=0.5, annealing=True, random_state=0).fit(X)
    >>> int(da.n_effective_clusters_)
    3
    """

    def __init__(
        self,
        n_clusters: int = 3,
        regularization: str = "entropy",
        lam: float = 1.0,
        annealing: bool = False,
        lam_start=None,
        anneal_rate: float = 0.8,
        perturbation: float = 1e-3,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=1.0,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.regularization = regularization
        self.lam = lam
        self.annealing = annealing
        self.lam_start = lam_start
        self.anneal_rate = anneal_rate
        self.perturbation = perturbation

    # ------------------------------------------------------------------ checks
    def _check_params(self, X):
        super()._check_params(X)
        if self.regularization not in _REGULARIZATIONS:
            raise ValueError(f"regularization must be one of {_REGULARIZATIONS}.")
        if not self.lam > 0:
            raise ValueError("lam must be positive.")
        if self.annealing:
            if self.regularization != "entropy":
                raise ValueError("annealing is only defined for regularization='entropy'.")
            if not 0.0 < self.anneal_rate < 1.0:
                raise ValueError("anneal_rate must lie in (0, 1).")
            if self.lam_start is not None and self.lam_start <= self.lam:
                raise ValueError("lam_start must be larger than lam.")
            if self.perturbation < 0:
                raise ValueError("perturbation must be non-negative.")
        self.lam_ = float(self.lam)

    # ------------------------------------------------------------------ hooks
    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, Um)

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        if self.regularization == "quadratic":
            return quadratic_memberships(d2, self.lam_)
        return entropy_memberships(d2, self.lam_)

    def _objective(self, Um, d2):
        U = Um  # m = 1
        if self.regularization == "quadratic":
            reg = self.lam_ * np.sum(U**2)
        else:
            reg = self.lam_ * np.sum(U * np.log(U + EPS))
        return float(np.sum(U * d2) + reg)

    def _finalize(self, X, Um):
        self.covariances_ = fuzzy_covariances(X, Um, self.cluster_centers_)
        if self.annealing:
            w = np.linalg.eigvalsh(self.covariances_)
            self.critical_temperatures_ = 2.0 * w[:, -1]

    # ------------------------------------------------------------------ annealing
    def _n_effective(self, V, scale):
        """Number of distinct prototypes (closer than ``0.02 * scale`` counts as one)."""
        d = np.sqrt(sq_euclidean(V, V))
        distinct = np.ones(len(V), dtype=bool)
        for i in range(1, len(V)):
            if np.any(d[i, :i][distinct[:i]] < 0.02 * scale):
                distinct[i] = False
        return int(distinct.sum())

    def _fit_single(self, X, rng):
        if not self.annealing:
            return super()._fit_single(X, rng)
        import pandas as pd

        c = self.n_clusters
        mean = X.mean(axis=0)
        F0 = np.atleast_2d(np.cov(X, rowvar=False))
        lam_max0 = float(np.linalg.eigvalsh(F0)[-1])
        scale = float(np.sqrt(max(lam_max0, EPS)))
        lam = float(self.lam_start) if self.lam_start is not None else max(2.2 * lam_max0, 1.01 * self.lam)
        lam_final = float(self.lam)
        self.cluster_centers_ = mean + self.perturbation * scale * rng.randn(c, X.shape[1])
        U = entropy_memberships(self._distances(X), lam)
        rows, history, total_iter = [], [], 0
        while True:
            self.lam_ = lam
            for _ in range(self.max_iter):
                total_iter += 1
                self._update_prototypes(X, U)
                d2 = self._distances(X)
                U_new = self._memberships(d2)
                delta = np.max(np.abs(U_new - U))
                U = U_new
                if delta < self.tol:
                    break
            J = self._objective(U, d2)
            n_eff = self._n_effective(self.cluster_centers_, scale)
            rows.append({"lam": lam, "n_effective": n_eff, "objective": J})
            history.append(J)
            if self.verbose:
                print(f"lam = {lam:.4g}  distinct prototypes = {n_eff}  J = {J:.6g}")
            if lam <= lam_final:
                break
            lam = max(lam * self.anneal_rate, lam_final)
            # break the symmetry of coinciding prototypes so that they can split
            jitter = self.perturbation * scale * rng.randn(c, X.shape[1])
            self.cluster_centers_ = self.cluster_centers_ + jitter
            U = entropy_memberships(self._distances(X), lam)
        self.lam_ = lam_final
        self.cluster_history_ = pd.DataFrame(rows)
        self.n_effective_clusters_ = int(rows[-1]["n_effective"])
        state = {k: v for k, v in self.__dict__.items() if k.endswith("_") and not k.startswith("_")}
        return {"U": U, "d2": d2, "history": history, "n_iter": total_iter, "state": state}


class PolynomialFuzzifierFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means with the polynomial fuzzifier of Klawonn and Höppner.

    The fuzzifier :math:`\mu^m` of FCM has a derivative of zero at
    :math:`\mu=0`, which forces every sample to have a positive membership
    in every cluster---even a cluster on the other side of the data set.
    Klawonn and Höppner replace :math:`\mu^m` by the quadratic polynomial

    .. math::
        g(\mu)=\frac{1-\beta}{1+\beta}\,\mu^2+\frac{2\beta}{1+\beta}\,\mu,\qquad 0\le\beta<1,

    whose slope at zero is :math:`2\beta/(1+\beta)>0`. A cluster whose
    distance exceeds that of the nearest prototype by a factor set by
    :math:`\beta` then receives an *exact zero* membership, while the
    memberships between nearby prototypes stay fuzzy. ``beta=0`` is FCM with
    :math:`m=2`; as :math:`\beta\to1` the partition becomes crisp (k-means).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    beta : float, default=0.5
        Fuzzifier parameter :math:`\beta\in[0,1)`. A sample at distance
        :math:`D_1` from its nearest prototype has zero membership in every
        cluster farther than :math:`D_1/\sqrt{\beta}`; :math:`\beta=0.5` is
        the value recommended by the authors.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition with exact zeros.
    n_active_ : ndarray of shape (n_samples,)
        Number of clusters with positive membership per training sample.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices of the final partition (weights :math:`g(\mu_{ik})`).
    labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    The functional :math:`J=\sum_i\sum_k g(\mu_{ik})D^2_{ik}` is minimized
    under :math:`\sum_i\mu_{ik}=1`, :math:`\mu_{ik}\ge0`. With the
    :math:`\hat c_k` nearest prototypes forming the active set :math:`A_k`
    the memberships are

    .. math::
        \mu_{ik}=\frac{1}{1-\beta}\left(\frac{1+(\hat c_k-1)\beta}
        {\sum_{j\in A_k}D^2_{ik}/D^2_{jk}}-\beta\right)\ (i\in A_k),\qquad
        \mu_{ik}=0\ (i\notin A_k),

    where :math:`\hat c_k` is the largest size for which all these values
    are positive; the prototypes are the :math:`g(\mu_{ik})`-weighted means.
    The weighting exponent is fixed at ``m=1`` internally so that the base
    class passes the memberships themselves to the prototype update, where
    :math:`g` is applied.

    References
    ----------
    .. [1] F. Klawonn, F. Höppner, "What is fuzzy about fuzzy clustering?
           Understanding and improving the concept of the fuzzifier",
           Advances in Intelligent Data Analysis V (IDA 2003), LNCS 2810,
           254--264, Springer, 2003.

    Examples
    --------
    >>> from fuzzyclust.cluster import PolynomialFuzzifierFCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> pf = PolynomialFuzzifierFCM(n_clusters=3, beta=0.5, random_state=0).fit(X)
    >>> bool((pf.membership_ == 0).mean() > 0.3)
    True
    >>> float(pf.membership_.sum(axis=1).round(6).min())
    1.0
    """

    def __init__(
        self,
        n_clusters: int = 3,
        beta: float = 0.5,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=1.0,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.beta = beta

    def _check_params(self, X):
        super()._check_params(X)
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must lie in [0, 1).")

    def _update_prototypes(self, X, Um):
        self.cluster_centers_ = fuzzy_centers(X, polynomial_fuzzifier(Um, self.beta))

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        return polynomial_memberships(d2, self.beta)

    def _objective(self, Um, d2):
        return float(np.sum(polynomial_fuzzifier(Um, self.beta) * d2))

    def _finalize(self, X, Um):
        G = polynomial_fuzzifier(Um, self.beta)
        self.covariances_ = fuzzy_covariances(X, G, self.cluster_centers_)
        self.n_active_ = (self.membership_ > 0).sum(axis=1)


__all__ = [
    "RegularizedFCM",
    "PolynomialFuzzifierFCM",
    "entropy_memberships",
    "quadratic_memberships",
    "polynomial_memberships",
    "polynomial_fuzzifier",
]
