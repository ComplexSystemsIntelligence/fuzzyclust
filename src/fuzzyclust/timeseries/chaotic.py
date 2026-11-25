# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""State-space reconstruction and prediction of chaotic time series by fuzzy clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .._utils import EPS, fuzzy_centers, fuzzy_covariances, regularize_covariance
from ..cluster import FCM

__all__ = [
    "delay_embedding",
    "select_delay",
    "ChaoticPredictor",
    "embedding_scan",
    "select_embedding_dimension",
    "correlation_dimension",
]

_LOG_CAP = 700.0
_LOG2PI = np.log(2.0 * np.pi)


def delay_embedding(y, dim: int, delay: int = 1) -> np.ndarray:
    r"""Delay vectors :math:`\mathbf{y}_k=[y_k, y_{k-\tau},\dots,y_{k-\tau(d_e-1)}]^T` of a scalar series.

    Parameters
    ----------
    y : array-like of shape (T,)
    dim : int
        Embedding dimension :math:`d_e`.
    delay : int, default=1
        Lag :math:`\tau` (in samples), usually the first zero of the
        autocorrelation or the first minimum of the mutual information.

    Returns
    -------
    Y : ndarray of shape (T - delay * (dim - 1), dim)
        Row ``i`` is the delay vector at time ``k = i + delay * (dim - 1)``.

    Examples
    --------
    >>> from fuzzyclust.timeseries import delay_embedding
    >>> delay_embedding([0., 1., 2., 3., 4.], dim=3, delay=1)
    array([[2., 1., 0.],
           [3., 2., 1.],
           [4., 3., 2.]])
    """
    y = np.asarray(y, dtype=float).ravel()
    if dim < 1 or delay < 1:
        raise ValueError("dim and delay must be positive integers.")
    span = delay * (dim - 1)
    if y.shape[0] <= span:
        raise ValueError("The series is too short for this embedding.")
    T = y.shape[0]
    return np.column_stack([y[span - j * delay : T - j * delay] for j in range(dim)])


class ChaoticPredictor(BaseEstimator):
    r"""Operating-regime model of a chaotic series in the reconstructed state space.

    A scalar measurement :math:`y_k` of a deterministic system is embedded
    into delay vectors :math:`\mathbf{y}_k=[y_k,y_{k-\tau},\dots,y_{k-\tau(d_e-1)}]^T`
    (Takens). Abonyi, Feil, Németh, Árva and Babuška [1]_ then identify the
    map :math:`\mathbf{y}_{k+1}=\mathbf{f}_r(\mathbf{y}_k)` as an
    operating-regime based model

    .. math::
        \hat{\mathbf{y}}_{k+1}=\sum_{i=1}^{c}\beta_i(\mathbf{y}_k)\,(\mathbf{A}_i\mathbf{y}_k+\mathbf{b}_i),
        \qquad
        \beta_i(\mathbf{y})=\frac{\exp\big(-\frac12(\mathbf{y}-\mathbf{v}_i)^T\mathbf{F}_i^{-1}(\mathbf{y}-\mathbf{v}_i)\big)}
                                 {\sum_j\exp\big(-\frac12(\mathbf{y}-\mathbf{v}_j)^T\mathbf{F}_j^{-1}(\mathbf{y}-\mathbf{v}_j)\big)},

    whose Gaussian operating regimes and local linear models are obtained
    *simultaneously* by a modified Gath–Geva clustering of the pairs
    :math:`(\mathbf{y}_k,\mathbf{y}_{k+1})`. Three questions are answered by
    the same clustering: the embedding dimension (from the one-step
    prediction error as :math:`d_e` grows), the local (intrinsic) dimension
    (from the smallest eigenvalues of the cluster covariances) and a model
    for prediction.

    Parameters
    ----------
    dim : int, default=3
        Embedding dimension :math:`d_e`.
    delay : int, default=1
        Lag :math:`\tau` in samples.
    n_clusters : int, default=10
        Number of local linear models :math:`c` (the paper used 20).
    m : float, default=2.0
        Fuzziness exponent.
    target : {"scalar", "full"}, default="scalar"
        ``"scalar"`` identifies the map to the next sample :math:`y_{k+1}`
        only (the other components of :math:`\mathbf{y}_{k+1}` are past
        samples and are copied exactly, which is what a free-run simulation
        needs). ``"full"`` identifies the MIMO map
        :math:`\mathbf{y}_k\mapsto\mathbf{y}_{k+1}` as written in the paper;
        with ``delay=1`` the copied components make the error covariance
        singular, so the regularization ``beta`` is then essential.
    gamma : float, default=0.0
        Shrinkage of the fuzzy covariance matrices (see :class:`~fuzzyclust.cluster.GK`).
    beta : float, default=1e8
        Condition-number bound of :math:`\mathbf{F}_i` and :math:`\mathbf{P}_i`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"fcm", "random"} or ndarray, default="fcm"
        Initial partition: from a fuzzy c-means run on the delay vectors, a
        random partition, or a given membership matrix.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    embedding_ : ndarray of shape (N, dim)
        Delay vectors used for identification.
    targets_ : ndarray of shape (N,) or (N, dim)
    membership_ : ndarray of shape (N, n_clusters)
    cluster_centers_ : ndarray of shape (n_clusters, dim)
        Centers :math:`\mathbf{v}_i` of the operating regimes.
    covariances_ : ndarray of shape (n_clusters, dim, dim)
        Fuzzy covariance matrices :math:`\mathbf{F}_i`.
    local_models_ : ndarray of shape (n_clusters, dim + 1, n_outputs)
        Parameters :math:`\boldsymbol{\theta}_i=[\mathbf{A}_i\ \mathbf{b}_i]^T`
        (last row is the offset).
    error_covariances_ : ndarray of shape (n_clusters, n_outputs, n_outputs)
        Covariances :math:`\mathbf{P}_i` of the local prediction errors.
    priors_ : ndarray of shape (n_clusters,)
    eigenvalues_ : ndarray of shape (n_clusters, dim)
        Eigenvalues of :math:`\mathbf{F}_i` in ascending order.
    local_dimension_index_ : float
        :math:`J^{d_e}=\sum_i p(\eta_i)\lambda_{i,\min}`, the prior-weighted
        smallest eigenvalue (eq. (17) of [1]_); it drops sharply once
        :math:`d_e` exceeds the local dimension of the attractor.
    objective_, objective_history_, n_iter_

    Notes
    -----
    The distance of the modified Gath–Geva clustering is the reciprocal of a
    probability with three factors (eq. (10) of [1]_):

    .. math::
        \frac{1}{D^2_{ik}} = p(\eta_i)\,
        \frac{\exp\big(-\frac12(\mathbf{y}_k-\mathbf{v}_i)^T\mathbf{F}_i^{-1}(\mathbf{y}_k-\mathbf{v}_i)\big)}
             {(2\pi)^{d_e/2}\sqrt{\det\mathbf{F}_i}}\,
        \frac{\exp\big(-\frac12\mathbf{e}_{ik}^T\mathbf{P}_i^{-1}\mathbf{e}_{ik}\big)}
             {(2\pi)^{p/2}\sqrt{\det\mathbf{P}_i}},
        \qquad \mathbf{e}_{ik}=\mathbf{y}_{k+1}-f_i(\mathbf{y}_k,\boldsymbol{\theta}_i),

    so a sample belongs to a regime if it lies in its Gaussian *and* is
    predicted well by its local model. The alternating optimization updates
    :math:`\mathbf{v}_i` and :math:`\mathbf{F}_i` as fuzzy mean and
    covariance of the delay vectors, the local models by weighted least
    squares
    :math:`\boldsymbol{\theta}_i=(\mathbf{Y}^T\mathbf{B}_i\mathbf{Y})^{-1}\mathbf{Y}^T\mathbf{B}_i\mathbf{Y}_{(+1)}`
    with :math:`\mathbf{B}_i=\mathrm{diag}(\mu_{i1},\dots,\mu_{iN})`,
    the error covariances
    :math:`\mathbf{P}_i=\sum_k\mu_{ik}^m\mathbf{e}_{ik}\mathbf{e}_{ik}^T/\sum_k\mu_{ik}^m`,
    the priors :math:`p(\eta_i)=\frac1N\sum_k\mu_{ik}`, and the partition by
    :math:`\mu_{ik}=1/\sum_j(D_{ik}/D_{jk})^{2/(m-1)}`.

    References
    ----------
    .. [1] J. Abonyi, B. Feil, S. Németh, P. Árva, R. Babuška, "State-space
           reconstruction and prediction of chaotic time series based on fuzzy
           clustering", Proc. IEEE Int. Conf. Systems, Man and Cybernetics,
           2004, 2374–2379.
    .. [2] F. Takens, "Detecting strange attractors in turbulence", in:
           Dynamical Systems and Turbulence, Lecture Notes in Mathematics 898,
           Springer, 1981, 366–381.

    Examples
    --------
    >>> from fuzzyclust.timeseries import ChaoticPredictor, make_mackey_glass
    >>> x = make_mackey_glass(n_samples=800)
    >>> model = ChaoticPredictor(dim=4, delay=6, n_clusters=5, random_state=0).fit(x[:600])
    >>> err = model.one_step_error(x[600:])
    >>> bool(err < 1e-3)
    True
    """

    def __init__(
        self,
        dim: int = 3,
        delay: int = 1,
        n_clusters: int = 10,
        m: float = 2.0,
        target: str = "scalar",
        gamma: float = 0.0,
        beta: float = 1e8,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="fcm",
        random_state=None,
        verbose: int = 0,
    ):
        self.dim = dim
        self.delay = delay
        self.n_clusters = n_clusters
        self.m = m
        self.target = target
        self.gamma = gamma
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ data
    def _pairs(self, y):
        Y = delay_embedding(y, self.dim, self.delay)
        if self.target == "scalar":
            return Y[:-1], y[self.delay * (self.dim - 1) + 1 :]
        if self.target == "full":
            return Y[:-1], Y[1:]
        raise ValueError("target must be 'scalar' or 'full'.")

    # ------------------------------------------------------------------ fitting
    def fit(self, y, X=None):
        """Identify the model from a scalar series ``y`` (``X`` is ignored)."""
        y = check_array(y, ensure_2d=False, dtype=float).ravel()
        if self.m <= 1.0:
            raise ValueError("ChaoticPredictor requires m > 1.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        rng = check_random_state(self.random_state)
        Y, T = self._pairs(y)
        T2 = T[:, None] if T.ndim == 1 else T
        N, d = Y.shape
        c = self.n_clusters
        if N <= d + 1 or N < c:
            raise ValueError("Too few samples for this embedding and number of clusters.")
        self._F0 = np.cov(Y, rowvar=False).reshape(d, d)
        self._P0 = np.cov(T2, rowvar=False).reshape(T2.shape[1], T2.shape[1])
        if isinstance(self.init, str):
            if self.init == "fcm":
                U = FCM(n_clusters=c, m=self.m, random_state=rng).fit(Y).membership_
            elif self.init == "random":
                U = rng.rand(N, c)
                U /= U.sum(axis=1, keepdims=True)
            else:
                raise ValueError("init must be 'fcm', 'random' or a membership matrix.")
        else:
            U = np.asarray(self.init, dtype=float)
            if U.shape != (N, c):
                raise ValueError(f"init membership must have shape {(N, c)}.")
            U = U / U.sum(axis=1, keepdims=True)
        Phi = np.column_stack([Y, np.ones(N)])
        history = []
        for it in range(1, self.max_iter + 1):
            Um = U**self.m
            self._update(Y, Phi, T2, U, Um)
            d2 = self._distances(Y, Phi, T2)
            U_new = self._memberships(d2)
            history.append(float(np.sum(Um * d2)))
            if self.verbose:
                print(f"iter {it:4d}  J = {history[-1]:.6g}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol:
                break
        self.embedding_ = Y
        self.targets_ = T
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.objective_history_ = np.asarray(history)
        self.objective_ = history[-1]
        self.n_iter_ = it
        self.eigenvalues_ = np.linalg.eigvalsh(self.covariances_)
        self.local_dimension_index_ = float(np.sum(self.priors_ * self.eigenvalues_[:, 0]))
        return self

    def _update(self, Y, Phi, T2, U, Um):
        N, d = Y.shape
        c = Um.shape[1]
        p = T2.shape[1]
        V = fuzzy_centers(Y, Um)
        F = fuzzy_covariances(Y, Um, V)
        theta = np.empty((c, d + 1, p))
        P = np.empty((c, p, p))
        invF, logdetF = np.empty_like(F), np.empty(c)
        invP, logdetP = np.empty_like(P), np.empty(c)
        for i in range(c):
            Fi = regularize_covariance(F[i], self._F0, self.gamma, self.beta)
            F[i] = Fi
            invF[i] = np.linalg.inv(Fi)
            logdetF[i] = np.linalg.slogdet(Fi)[1]
            # weighted least squares with B_i = diag(mu_ik), eq. (7)
            w = U[:, i]
            A = Phi.T @ (Phi * w[:, None]) + 1e-10 * np.eye(d + 1)
            theta[i] = np.linalg.solve(A, Phi.T @ (T2 * w[:, None]))
            E = T2 - Phi @ theta[i]
            Pi = (E * Um[:, i][:, None]).T @ E / max(Um[:, i].sum(), EPS)
            Pi = regularize_covariance(Pi + EPS * np.eye(p), self._P0, self.gamma, self.beta)
            P[i] = Pi
            invP[i] = np.linalg.inv(Pi)
            logdetP[i] = np.linalg.slogdet(Pi)[1]
        self.cluster_centers_ = V
        self.covariances_ = F
        self.local_models_ = theta
        self.error_covariances_ = P
        self.priors_ = np.maximum(U.sum(axis=0) / N, EPS)
        self._invF, self._logdetF, self._invP, self._logdetP = invF, logdetF, invP, logdetP

    def _log_gauss(self, Y):
        """log of the Gaussian factor in y (without priors), shape (N, c)."""
        c, d = self.cluster_centers_.shape
        out = np.empty((Y.shape[0], c))
        for i in range(c):
            yv = Y - self.cluster_centers_[i]
            maha = np.einsum("kj,kj->k", yv @ self._invF[i], yv)
            out[:, i] = -0.5 * (d * _LOG2PI + self._logdetF[i] + maha)
        return out

    def _distances(self, Y, Phi, T2):
        c = self.cluster_centers_.shape[0]
        p = T2.shape[1]
        logp = self._log_gauss(Y) + np.log(self.priors_)
        for i in range(c):
            E = T2 - Phi @ self.local_models_[i]
            maha = np.einsum("kj,kj->k", E @ self._invP[i], E)
            logp[:, i] += -0.5 * (p * _LOG2PI + self._logdetP[i] + maha)
        return np.exp(np.minimum(-logp, _LOG_CAP))

    def _memberships(self, d2):
        logits = -np.log(d2 + EPS) / (self.m - 1.0)
        logits -= logits.max(axis=1, keepdims=True)
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ prediction
    def regime_membership(self, Y):
        r"""Normalized Gaussian degrees of fulfilment :math:`\beta_i(\mathbf{y})` (eq. (8)).

        Parameters
        ----------
        Y : array-like of shape (n_samples, dim)

        Returns
        -------
        beta : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        Y = check_array(Y, dtype=float)
        c, d = self.cluster_centers_.shape
        logb = np.empty((Y.shape[0], c))
        for i in range(c):
            yv = Y - self.cluster_centers_[i]
            logb[:, i] = -0.5 * np.einsum("kj,kj->k", yv @ self._invF[i], yv)
        logb -= logb.max(axis=1, keepdims=True)
        B = np.exp(logb)
        return B / B.sum(axis=1, keepdims=True)

    def predict_proba(self, Y):
        """Alias of :meth:`regime_membership` (the name :class:`~fuzzyclust.viz.FuzzySammon` expects)."""
        return self.regime_membership(Y)

    def transform(self, Y):
        r"""Mahalanobis distances of delay vectors to the operating regimes.

        .. math::
            d_{ik}=\sqrt{(\mathbf{y}_k-\mathbf{v}_i)^T\mathbf{F}_i^{-1}(\mathbf{y}_k-\mathbf{v}_i)}.

        These are the distances of the Gaussian factor of the prototype; they
        are what :class:`~fuzzyclust.viz.FuzzySammon` preserves when the
        clusters of the reconstructed state space are visualized.

        Returns
        -------
        D : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        Y = check_array(Y, dtype=float)
        c = self.cluster_centers_.shape[0]
        d2 = np.empty((Y.shape[0], c))
        for i in range(c):
            yv = Y - self.cluster_centers_[i]
            d2[:, i] = np.einsum("kj,kj->k", yv @ self._invF[i], yv)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, Y):
        r"""One-step prediction for delay vectors ``Y``.

        .. math::
            \hat{\mathbf{y}}_{k+1}=\sum_{i=1}^{c}\beta_i(\mathbf{y}_k)\,f_i(\mathbf{y}_k).

        Returns
        -------
        ndarray of shape (n_samples,) for ``target="scalar"`` or (n_samples, dim) for ``"full"``.
        """
        Y = check_array(Y, dtype=float)
        B = self.regime_membership(Y)
        Phi = np.column_stack([Y, np.ones(Y.shape[0])])
        out = np.einsum("ki,kj,ijp->kp", B, Phi, self.local_models_)
        return out[:, 0] if self.target == "scalar" else out

    def predict_series(self, y):
        """One-step predictions along a series.

        Returns
        -------
        y_hat, y_true : ndarrays of shape (T - delay * (dim - 1) - 1,)
            Predicted and true next samples, aligned.
        """
        y = check_array(y, ensure_2d=False, dtype=float).ravel()
        Y, T = self._pairs(y)
        pred = self.predict(Y)
        return (pred, T) if self.target == "scalar" else (pred[:, 0], T[:, 0])

    def one_step_error(self, y) -> float:
        """Mean squared one-step prediction error of the next sample along ``y``."""
        y_hat, y_true = self.predict_series(y)
        return float(np.mean((y_hat - y_true) ** 2))

    def simulate(self, y_history, n_steps: int):
        r"""Free-run simulation: predict ``n_steps`` samples beyond the given history.

        Parameters
        ----------
        y_history : array-like of shape (T,)
            At least ``delay * (dim - 1) + 1`` past samples.
        n_steps : int

        Returns
        -------
        ndarray of shape (n_steps,)
        """
        y_history = check_array(y_history, ensure_2d=False, dtype=float).ravel()
        span = self.delay * (self.dim - 1)
        if y_history.shape[0] < span + 1:
            raise ValueError(f"y_history needs at least {span + 1} samples.")
        buf = list(y_history[-(span + 1) :])
        out = np.empty(n_steps)
        for s in range(n_steps):
            Yk = np.array([buf[-1 - j * self.delay] for j in range(self.dim)])[None, :]
            pred = self.predict(Yk)
            nxt = float(pred[0] if self.target == "scalar" else pred[0, 0])
            out[s] = nxt
            buf.append(nxt)
            buf.pop(0)
        return out

    def multi_step_error(self, y, horizons=(1, 2, 5, 10)) -> pd.Series:
        r"""Mean squared free-run prediction error as a function of the horizon.

        From every possible starting point of ``y`` the model is iterated
        :math:`h` steps and compared with the true sample :math:`h` steps
        ahead (all starting points are simulated in parallel).

        Returns
        -------
        pandas.Series indexed by the horizon.
        """
        y = check_array(y, ensure_2d=False, dtype=float).ravel()
        H = int(max(horizons))
        span = self.delay * (self.dim - 1)
        T = y.shape[0]
        n_start = T - span - H
        if n_start < 1:
            raise ValueError("Series too short for the requested horizons.")
        # buffers: rows = starting points, columns = the last span+1 samples (oldest first)
        idx = np.arange(span + 1)[None, :] + np.arange(n_start)[:, None]
        buf = y[idx].copy()
        errors = {}
        for h in range(1, H + 1):
            Yk = np.column_stack([buf[:, -1 - j * self.delay] for j in range(self.dim)])
            pred = self.predict(Yk)
            nxt = pred if self.target == "scalar" else pred[:, 0]
            true = y[span + h : span + h + n_start]
            if h in horizons:
                errors[h] = float(np.mean((nxt - true) ** 2))
            buf = np.column_stack([buf[:, 1:], nxt])
        return pd.Series(errors, name="mse").rename_axis("horizon")


def embedding_scan(
    y,
    dims=(1, 2, 3, 4, 5, 6),
    delay: int = 1,
    n_clusters: int = 10,
    train_fraction: float = 0.7,
    random_state=None,
    **kwargs,
) -> pd.DataFrame:
    r"""Fit :class:`ChaoticPredictor` for several embedding dimensions and tabulate the criteria.

    The correct embedding dimension is the smallest :math:`d_e` after which
    the one-step prediction error stops decreasing (the ratio of consecutive
    errors approaches one), the local dimension the one where the
    prior-weighted smallest eigenvalue :math:`J^{d_e}` collapses.

    Parameters
    ----------
    y : array-like of shape (T,)
    dims : sequence of int
    delay : int, default=1
    n_clusters : int, default=10
    train_fraction : float, default=0.7
        The first part of the series is used for identification, the rest
        for the test error.
    random_state : int, RandomState instance or None
    **kwargs
        Further parameters of :class:`ChaoticPredictor`.

    Returns
    -------
    table : pandas.DataFrame indexed by ``dim`` with columns
        ``mse_train``, ``mse_test``, ``mse_ratio`` (test error of the
        previous dimension divided by the current one, the criterion of the
        paper: a ratio close to one means that the added coordinate does not
        help) and ``local_dimension_index`` (:math:`J^{d_e}`).

    Examples
    --------
    >>> from fuzzyclust.timeseries import embedding_scan, make_mackey_glass
    >>> x = make_mackey_glass(n_samples=800)
    >>> table = embedding_scan(x, dims=(2, 3, 4), delay=6, n_clusters=4, random_state=0)
    >>> list(table.columns)
    ['mse_train', 'mse_test', 'mse_ratio', 'local_dimension_index']
    """
    y = np.asarray(y, dtype=float).ravel()
    n_train = int(train_fraction * y.shape[0])
    rows = []
    for d in dims:
        model = ChaoticPredictor(
            dim=d, delay=delay, n_clusters=n_clusters, random_state=random_state, **kwargs
        )
        model.fit(y[:n_train])
        rows.append(
            {
                "dim": d,
                "mse_train": model.one_step_error(y[:n_train]),
                "mse_test": model.one_step_error(y[n_train:]),
                "local_dimension_index": model.local_dimension_index_,
            }
        )
    table = pd.DataFrame(rows).set_index("dim")
    mse = table["mse_test"].to_numpy()
    ratio = np.full(mse.shape[0], np.nan)
    ratio[1:] = mse[:-1] / np.maximum(mse[1:], EPS)
    table.insert(2, "mse_ratio", ratio)
    return table


def select_embedding_dimension(table: pd.DataFrame, ratio_tol: float = 1.2) -> Bunch:
    r"""Read the embedding and the local dimension off an :func:`embedding_scan` table.

    The embedding dimension :math:`d_e` is the smallest dimension after
    which a further coordinate reduces the one-step test error by less than
    the factor ``ratio_tol`` (the paper compares the ratio of neighboring
    MSE values, e.g. 1.231 against 1.005). The local dimension
    :math:`d_l` is one less than the dimension at which the prior-weighted
    smallest eigenvalue :math:`J^{d_e}` drops most sharply (largest ratio
    :math:`J^{d-1}/J^{d}`), because the clusters become flat once the
    embedding dimension exceeds the dimension of the manifold they
    approximate. The second criterion is reliable when the attractor lies
    close to a smooth low-dimensional manifold and only indicative for
    strongly fractal ones (then :func:`correlation_dimension` is the
    complement).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`embedding_scan` (dimensions must be consecutive).
    ratio_tol : float, default=1.2

    Returns
    -------
    result : :class:`sklearn.utils.Bunch`
        ``embedding_dimension``, ``local_dimension`` and the two criteria
        ``mse_ratio`` and ``eigenvalue_ratio`` as Series.
    """
    dims = np.asarray(table.index, dtype=int)
    mse_ratio = table["mse_ratio"].to_numpy()
    J = table["local_dimension_index"].to_numpy()
    eig_ratio = np.full(J.shape[0], np.nan)
    eig_ratio[1:] = J[:-1] / np.maximum(J[1:], EPS)
    # d_e: first dimension whose successor improves the error by less than ratio_tol
    d_e = int(dims[-1])
    for pos in range(len(dims) - 1):
        if mse_ratio[pos + 1] < ratio_tol:
            d_e = int(dims[pos])
            break
    d_l = int(dims[int(np.nanargmax(eig_ratio))] - 1) if len(dims) > 1 else int(dims[0])
    return Bunch(
        embedding_dimension=d_e,
        local_dimension=max(d_l, 1),
        mse_ratio=pd.Series(mse_ratio, index=table.index, name="mse_ratio"),
        eigenvalue_ratio=pd.Series(eig_ratio, index=table.index, name="eigenvalue_ratio"),
    )


def select_delay(y, max_lag: int = 100, method: str = "mutual_information", n_bins: int = 16) -> int:
    r"""Lag :math:`\tau` of the delay embedding from the series itself.

    Two classical rules are offered: the first minimum of the average mutual
    information :math:`I(\tau)=\sum p(y_k,y_{k-\tau})\log\frac{p(y_k,y_{k-\tau})}{p(y_k)p(y_{k-\tau})}`
    (Fraser & Swinney, estimated on a ``n_bins``:math:`\times`\ ``n_bins``
    histogram) and the first zero crossing of the autocorrelation function.
    If no minimum or zero is found within ``max_lag``, ``max_lag`` is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.timeseries import select_delay
    >>> y = np.sin(2 * np.pi * np.arange(1000) / 40.0)
    >>> select_delay(y, max_lag=30, method="autocorrelation")
    10
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.shape[0]
    max_lag = int(min(max_lag, T - 2))
    if max_lag < 1:
        raise ValueError("The series is too short for the requested max_lag.")
    if method == "autocorrelation":
        yc = y - y.mean()
        var = yc @ yc
        prev = 1.0
        for lag in range(1, max_lag + 1):
            cur = (yc[lag:] @ yc[:-lag]) / var
            if cur <= 0.0:  # zero crossing between lag-1 and lag: round the interpolated position
                return int(round(lag - 1 + prev / (prev - cur))) if lag > 1 else 1
            prev = cur
        return max_lag
    if method != "mutual_information":
        raise ValueError("method must be 'mutual_information' or 'autocorrelation'.")
    edges = np.linspace(y.min(), y.max() + 1e-12, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, n_bins - 1)

    def mutual_information(lag):
        a, b = idx[lag:], idx[:-lag]
        joint = np.zeros((n_bins, n_bins))
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        pa, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        return float(np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])))

    prev = mutual_information(1)
    for lag in range(2, max_lag + 1):
        cur = mutual_information(lag)
        if cur > prev:
            return lag - 1
        prev = cur
    return max_lag


def correlation_dimension(
    Y, radii=None, n_radii: int = 10, max_pairs: int = 200_000, random_state=None
) -> Bunch:
    r"""Grassberger–Procaccia correlation dimension of a point cloud (e.g. delay vectors).

    The correlation sum :math:`C(r)=\frac{2}{N(N-1)}\sum_{k<l}\Theta(r-\|\mathbf{y}_k-\mathbf{y}_l\|)`
    scales as :math:`r^{D_2}` for small :math:`r`; :math:`D_2` is the slope
    of :math:`\log C(r)` against :math:`\log r`, fitted here by least
    squares over the radii where :math:`0<C(r)<1`. Pairwise distances are
    computed for a random subset of at most ``max_pairs`` pairs. The
    estimate is a fractal (non-integer) counterpart of the integer local
    dimension of :func:`select_embedding_dimension`; the fuzzy Sammon paper
    uses it to confirm the clustering-based estimate.

    Parameters
    ----------
    Y : array-like of shape (n_samples, dim)
    radii : array-like, optional
        Radii :math:`r`; default: ``n_radii`` values log-spaced between the
        0.5th and the 10th percentile of the sampled distances (the scaling
        region must be small compared with the extent of the set).
    n_radii : int, default=10
    max_pairs : int, default=200000
    random_state : int, RandomState instance or None

    Returns
    -------
    result : :class:`sklearn.utils.Bunch`
        ``dimension`` (the slope :math:`D_2`), ``radii``, ``correlation_sum``.

    References
    ----------
    .. [1] P. Grassberger, I. Procaccia, "Measuring the strangeness of
           strange attractors", Physica D 9 (1983) 189–208.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.timeseries import correlation_dimension
    >>> rng = np.random.RandomState(0)
    >>> plane = np.c_[rng.rand(2000, 2), np.zeros(2000)]  # a flat 2-D set in 3-D
    >>> bool(1.8 < correlation_dimension(plane, random_state=0).dimension < 2.2)
    True
    """
    Y = check_array(Y, dtype=float)
    rng = check_random_state(random_state)
    N = Y.shape[0]
    n_pairs = N * (N - 1) // 2
    if n_pairs <= max_pairs:
        iu = np.triu_indices(N, k=1)
        d = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(axis=1))
    else:
        k = rng.randint(0, N, size=max_pairs)
        l = rng.randint(0, N - 1, size=max_pairs)
        l = l + (l >= k)
        d = np.sqrt(((Y[k] - Y[l]) ** 2).sum(axis=1))
    d = d[d > 0]
    if radii is None:
        lo, hi = np.percentile(d, [0.5, 10])
        radii = np.logspace(np.log10(lo), np.log10(hi), n_radii)
    radii = np.asarray(radii, dtype=float)
    C = np.array([np.mean(d < r) for r in radii])
    ok = (C > 0) & (C < 1)
    if ok.sum() < 2:
        raise ValueError("Not enough radii with 0 < C(r) < 1; choose other radii.")
    slope = np.polyfit(np.log(radii[ok]), np.log(C[ok]), 1)[0]
    return Bunch(dimension=float(slope), radii=radii, correlation_sum=C)
