# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Sammon's nonlinear mapping (Sammon, 1969)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS

__all__ = ["Sammon", "sammon_stress"]


def sammon_stress(X, Y) -> float:
    r"""Sammon's stress of an embedding ``Y`` of the data ``X``.

    .. math::
        E = \frac{1}{\lambda}\sum_{k<l}\frac{\left(d_{kl}-d^*_{kl}\right)^2}{d_{kl}},
        \qquad \lambda = \sum_{k<l} d_{kl},

    where :math:`d_{kl}=\|\mathbf{x}_k-\mathbf{x}_l\|` are the Euclidean
    distances in the original space and :math:`d^*_{kl}=\|\mathbf{y}_k-\mathbf{y}_l\|`
    those in the projection. Pairs with :math:`d_{kl}=0` (duplicate samples)
    are skipped. :math:`E=0` means that all inter-point distances are
    preserved; the normalization by :math:`\lambda` makes the value
    independent of the scale of the data (MATLAB toolbox ``samstr``).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_components)

    Returns
    -------
    E : float

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.viz import sammon_stress
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> sammon_stress(X, X)        # the identity preserves every distance
    0.0
    """
    X = check_array(X, dtype=float)
    Y = check_array(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    d = pdist(X)
    ds = pdist(Y)
    mask = d > 0
    if not mask.any():
        return 0.0
    lam = d[mask].sum()
    return float(np.sum((d[mask] - ds[mask]) ** 2 / d[mask]) / lam)


def _clip_displacements(step: np.ndarray, factor: float = 30.0) -> np.ndarray:
    """Limit the length of every row of ``step`` to ``factor`` times the median row length."""
    length = np.linalg.norm(step, axis=1)
    cap = factor * np.median(length)
    if cap <= 0:
        return step
    shrink = np.minimum(1.0, cap / np.maximum(length, EPS))
    return step * shrink[:, None]


class Sammon(TransformerMixin, BaseEstimator):
    r"""Sammon's nonlinear mapping to a low-dimensional space.

    Sammon mapping is a multidimensional-scaling method: it looks for
    :math:`N` points :math:`\mathbf{y}_k\in\mathbb{R}^q` whose pairwise Euclidean
    distances :math:`d^*_{kl}` reproduce the pairwise distances
    :math:`d_{kl}=\|\mathbf{x}_k-\mathbf{x}_l\|` of the original
    :math:`n`-dimensional data as closely as possible. In contrast to PCA, which
    preserves variance, the mapping preserves *distances*, which is the
    quantity clustering algorithms work with.

    Parameters
    ----------
    n_components : int, default=2
        Dimension :math:`q` of the embedding.
    max_iter : int, default=100
        Maximum number of iterations of the pseudo-Newton descent.
    alpha : float, default=0.3
        Step size ("magic factor") of the update, Sammon recommends 0.3-0.4.
        Larger values converge faster but may oscillate.
    init : {"pca", "random"} or ndarray of shape (n_samples, n_components), default="pca"
        Initial configuration. ``"pca"`` starts from the principal component
        scores (deterministic and usually close to the optimum), ``"random"``
        from uniform noise in :math:`[-0.5, 0.5]^q` (MATLAB toolbox behavior).
    tol : float, default=1e-6
        Stop when the relative decrease of the stress between two iterations
        falls below ``tol``.
    backtracking : bool, default=True
        Safeguard the update: displacements longer than 30 times the median
        displacement are clipped, and the step is halved (at most
        ``max_backtracks`` times) whenever it would increase the stress.
        Sammon's diagonal Newton step divides by the curvature, which can be
        arbitrarily small for single points and then throws them far away;
        the safeguard makes the stress history (almost always) monotone at the
        cost of one extra distance evaluation per halving. With ``False`` the
        raw update of the original algorithm is used.
    max_backtracks : int, default=10
    random_state : int, RandomState instance or None, default=None
        Seed for ``init="random"``.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The projected points :math:`\mathbf{y}_k` (centered at the origin).
    stress_ : float
        Final Sammon stress :math:`E`, see :func:`sammon_stress`.
    stress_history_ : ndarray of shape (``n_iter_``,)
        :math:`E` after every iteration.
    n_iter_ : int
    n_features_in_ : int

    Notes
    -----
    The stress

    .. math::
        E = \frac{1}{\lambda}\sum_{k<l}\frac{(d_{kl}-d^*_{kl})^2}{d_{kl}},\qquad
        \lambda=\sum_{k<l}d_{kl},

    is minimized in the :math:`Nq` coordinates :math:`y_{kl}` by the diagonal
    Newton step proposed by Sammon,

    .. math::
        y_{kl}(t+1) = y_{kl}(t) - \alpha\,
        \frac{\partial E/\partial y_{kl}}{\left|\partial^2 E/\partial y_{kl}^2\right|},

    with

    .. math::
        \frac{\partial E}{\partial y_{kl}} &= -\frac{2}{\lambda}\sum_{j\neq k}
            \frac{d_{kj}-d^*_{kj}}{d_{kj}d^*_{kj}}\,(y_{kl}-y_{jl}),\\
        \frac{\partial^2 E}{\partial y_{kl}^2} &= -\frac{2}{\lambda}\sum_{j\neq k}
            \frac{1}{d_{kj}d^*_{kj}}\left[(d_{kj}-d^*_{kj})
            - \frac{(y_{kl}-y_{jl})^2}{d^*_{kj}}\left(1+\frac{d_{kj}-d^*_{kj}}{d^*_{kj}}\right)\right].

    All points are updated simultaneously and the configuration is re-centered
    after every step; by default the step is halved when it would increase
    the stress (``backtracking``). Every iteration needs the :math:`N(N-1)/2` pairwise
    distances of the current configuration, which makes the method expensive
    for large :math:`N`; this motivates the fuzzy Sammon mapping
    (:class:`~fuzzyclust.viz.FuzzySammon`). The stress surface has local
    minima, so different initializations may give different maps.

    Projecting *new* samples is not defined for Sammon mapping (the whole
    optimization would have to be repeated), hence ``transform`` raises
    ``NotImplementedError``; use :meth:`fit_transform`.

    References
    ----------
    .. [1] J. W. Sammon Jr., "A nonlinear mapping for data structure analysis",
           IEEE Transactions on Computers C-18(5), 401-409, 1969.
    .. [2] J. Mao, A. K. Jain, "Artificial neural networks for feature
           extraction and multivariate data projection", IEEE Transactions on
           Neural Networks 6(2), 296-317, 1995.

    Examples
    --------
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.viz import Sammon
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, n_features=5, random_state=0)
    >>> sam = Sammon(n_components=2, max_iter=50, random_state=0)
    >>> Y = sam.fit_transform(X)
    >>> Y.shape, sam.stress_ < 0.1
    ((300, 2), True)
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        alpha: float = 0.3,
        init="pca",
        tol: float = 1e-6,
        backtracking: bool = True,
        max_backtracks: int = 10,
        random_state=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.init = init
        self.tol = tol
        self.backtracking = backtracking
        self.max_backtracks = max_backtracks
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers
    def _check_params(self, X):
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not self.alpha > 0:
            raise ValueError("alpha must be positive.")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be >= 0.")

    def _initial_embedding(self, X, rng):
        N, q = X.shape[0], self.n_components
        if isinstance(self.init, str):
            if self.init == "pca":
                from ._pca import pca_projection

                Y, _, _ = pca_projection(X, min(q, min(X.shape)))
                if Y.shape[1] < q:  # pad with zeros if q > n_features
                    Y = np.hstack([Y, np.zeros((N, q - Y.shape[1]))])
                return Y
            if self.init == "random":
                return rng.rand(N, q) - 0.5
            raise ValueError("init must be 'pca', 'random' or an array of shape (n_samples, n_components).")
        Y0 = np.array(self.init, dtype=float, copy=True)
        if Y0.shape != (N, q):
            raise ValueError(f"init array must have shape ({N}, {q}), got {Y0.shape}.")
        return Y0

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Compute the embedding of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        self._check_params(X)
        rng = check_random_state(self.random_state)
        N = X.shape[0]

        D = squareform(pdist(X))  # original distances d_kj
        valid = D > 0
        np.fill_diagonal(valid, False)
        lam = D[np.triu(valid, 1)].sum()
        if lam <= 0:  # all samples identical: nothing to map
            self.embedding_ = np.zeros((N, self.n_components))
            self.stress_history_ = np.zeros(1)
            self.stress_ = 0.0
            self.n_iter_ = 1
            return self

        upper = np.triu(valid, 1)

        def _stress(Ds):
            return float(np.sum((D[upper] - Ds[upper]) ** 2 / D[upper]) / lam)

        Y = self._initial_embedding(X, rng)
        Y = Y - Y.mean(axis=0)
        Ds = squareform(pdist(Y))
        prev = _stress(Ds)
        history = []
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007
            ok = valid & (Ds > 0)
            inv_dd = np.where(ok, 1.0 / np.where(ok, D * Ds, 1.0), 0.0)  # 1 / (d d*)
            inv_ds = np.where(ok, 1.0 / np.where(ok, Ds, 1.0), 0.0)  # 1 / d*
            dq = np.where(ok, D - Ds, 0.0)  # d - d*
            w = dq * inv_dd  # (d - d*) / (d d*)
            common = w.sum(axis=1)  # first part of the second derivative
            grad = np.empty_like(Y)
            hess = np.empty_like(Y)
            for l in range(self.n_components):
                u = Y[:, l][:, None] - Y[:, l][None, :]  # y_kl - y_jl
                grad[:, l] = -2.0 * (w * u).sum(axis=1)
                curv = (inv_dd * u**2 * inv_ds * (1.0 + dq * inv_ds)).sum(axis=1)
                hess[:, l] = -2.0 * (common - curv)
            hess_abs = np.abs(hess)
            step = np.divide(grad, hess_abs, out=np.zeros_like(grad), where=hess_abs > EPS)
            if self.backtracking:
                # safeguards against the near-zero curvatures that make the diagonal Newton
                # step explode for single points: clip outlying displacements at 30 times
                # their median, then halve the step while the stress would increase
                step = _clip_displacements(step)
            scale = self.alpha
            for _ in range(self.max_backtracks + 1):
                Y_new = Y - scale * step
                Y_new = Y_new - Y_new.mean(axis=0)
                Ds_new = squareform(pdist(Y_new))
                stress = _stress(Ds_new)
                if stress <= prev or not self.backtracking:
                    break
                scale *= 0.5
            Y, Ds = Y_new, Ds_new
            history.append(stress)
            if abs(prev - stress) < self.tol * max(prev, EPS):
                break
            prev = stress

        self.embedding_ = Y
        self.stress_history_ = np.asarray(history)
        self.stress_ = history[-1]
        self.n_iter_ = it
        return self

    def fit_transform(self, X, y=None):
        """Fit the mapping and return the embedding ``(n_samples, n_components)``."""
        return self.fit(X).embedding_

    def transform(self, X):
        """Not available: Sammon mapping has no out-of-sample extension (raises ``NotImplementedError``)."""
        check_is_fitted(self, "embedding_")
        validate_data(self, X, dtype=float, reset=False)
        raise NotImplementedError(
            "Sammon mapping defines no projection of new samples; the whole configuration "
            "has to be re-optimised. Use fit_transform on the complete data set, or "
            "FuzzySammon, whose projected prototypes allow an approximate placement."
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
