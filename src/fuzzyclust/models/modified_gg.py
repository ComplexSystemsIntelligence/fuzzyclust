# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Modified Gath–Geva clustering for the identification of Takagi–Sugeno fuzzy models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS
from ..base import BaseFuzzyClusterer

_LOG_CAP = 700.0  # exp(700) is the largest finite double
_LOG_2PI = np.log(2.0 * np.pi)


def weighted_least_squares(Xe: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    r"""Weighted least-squares estimate of local model parameters.

    .. math::
        \boldsymbol\theta = (\mathbf{X}_e^T\boldsymbol\Phi\mathbf{X}_e)^{-1}
        \mathbf{X}_e^T\boldsymbol\Phi\mathbf{Y}

    Parameters
    ----------
    Xe : ndarray of shape (n_samples, n_params)
        Extended regressor matrix (last column of ones for the intercept).
    Y : ndarray of shape (n_samples, n_outputs)
    w : ndarray of shape (n_samples,)
        Non-negative weights (the diagonal of :math:`\boldsymbol\Phi`).

    Returns
    -------
    theta : ndarray of shape (n_params, n_outputs)
    """
    Xw = Xe * w[:, None]
    G = Xe.T @ Xw
    G.flat[:: G.shape[0] + 1] += EPS * max(np.trace(G) / G.shape[0], 1.0)
    return np.linalg.solve(G, Xw.T @ Y)


def _extend(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _alternating_optimisation(est, X: np.ndarray, rng) -> None:
    """One run of the alternating-optimization loop of :class:`~fuzzyclust.base.BaseFuzzyClusterer`."""
    U = est._init_state(X, rng)
    history = []
    for it in range(1, est.max_iter + 1):
        Um = U**est.m
        est._update_prototypes(X, Um)
        d2 = est._distances(X)
        U_new = est._memberships(d2)
        history.append(est._objective(Um, d2))
        if est.verbose:
            print(f"iter {it:4d}  J = {history[-1]:.6g}")
        delta = np.max(np.abs(U_new - U))
        U = U_new
        if delta < est.tol:
            break
    est.membership_ = U
    est.labels_ = U.argmax(axis=1)
    est.distances_ = np.sqrt(d2)
    est.objective_history_ = np.asarray(history)
    est.objective_ = history[-1]
    est.n_iter_ = it
    est._finalize(X, U**est.m)


def _require_y(name: str, y) -> None:
    """Raise the scikit-learn style error for a supervised estimator called without ``y``."""
    if y is None:
        raise ValueError(f"{name} requires y to be passed, but the target y is None.")


def _fit_with_restarts(est, X: np.ndarray, log_likelihood) -> None:
    r"""Run the alternating optimization ``est.n_init`` times and keep the most likely solution.

    The c-means functional :math:`J` is not comparable between runs of the
    exponential-distance algorithms (a cluster collapsing onto a few samples
    lowers :math:`J`), so the runs are ranked by the log-likelihood
    :math:`\sum_k \log\sum_i 1/D^2_{ik}` of the data under the mixture of
    local models, which ``log_likelihood(est)`` must return.
    """
    est._check_params(X)
    if not isinstance(est.n_init, (int, np.integer)) or est.n_init < 1:
        raise ValueError("n_init must be a positive integer.")
    rng = check_random_state(est.random_state)
    n_init = 1 if isinstance(est.init, np.ndarray) else int(est.n_init)
    best, best_state, scores = -np.inf, None, []
    for run in range(n_init):
        run_rng = np.random.RandomState(rng.randint(np.iinfo(np.int32).max))
        _alternating_optimisation(est, X, run_rng)
        score = float(log_likelihood(est))
        scores.append(score)
        if est.verbose and n_init > 1:
            print(f"restart {run + 1}/{n_init}: criterion = {score:.6g}")
        if score > best or best_state is None:
            best = score
            best_state = {k: v for k, v in vars(est).items() if k.endswith("_") or k.startswith("_")}
    vars(est).update(best_state)
    est.restart_scores_ = np.asarray(scores)
    est.log_likelihood_ = best


class ModifiedGG(BaseFuzzyClusterer):
    r"""Modified Gath–Geva clustering: clusters are interpretable Takagi–Sugeno rules.

    Each cluster is described by (a) Gaussian membership functions on the
    *individual* input variables (an axis-parallel Gaussian, so the cluster is
    directly a rule antecedent and no projection error arises), (b) a local
    linear model :math:`\hat y = \mathbf{a}_i^T\mathbf{x} + b_i` with the
    variance of its error, and (c) a prior probability :math:`\alpha_i`. The
    distance of a sample to a cluster is the reciprocal of the probability that
    the cluster generated the input *and* the output, so the alternating
    optimization is the EM identification of a mixture of local linear
    regression models, written as a Gath–Geva type fuzzy clustering [1]_.
    Vector-valued outputs (MIMO systems) are handled with a full error
    covariance matrix per rule [2]_.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters = rules :math:`c`.
    m : float, default=2.0
        Fuzziness exponent of the membership update. ``m = 2`` reproduces the
        EM algorithm (memberships are the posterior cluster probabilities).
    antecedent_features : array-like of int, optional
        Column indices of ``X`` used in the rule antecedents (scheduling
        variables). The local linear models always use all columns. ``None``
        uses every input.
    gamma : float, default=0.0
        Variance regularization :math:`\gamma\in[0,1]`: the membership-function
        variances are shrunk toward the feature variances,
        :math:`\sigma^2_{ij}\leftarrow(1-\gamma)\sigma^2_{ij}+\gamma\,\mathrm{var}(x_j)`,
        the diagonal analog of the covariance regularization of
        :class:`~fuzzyclust.cluster.GG`; it prevents rules from collapsing onto
        a few samples.
    var_floor : float, default=1e-6
        Lower bound of the membership-function variances relative to the
        variance of the corresponding feature (avoids collapsing clusters).
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="kmeans++"
        As in :class:`~fuzzyclust.base.BaseFuzzyClusterer`; passing the
        membership matrix of a fitted FCM/GK model is also a good choice
        because the exponential distance is sensitive to the initial partition.
    n_init : int, default=3
        Number of random restarts (ignored when ``init`` is an array). The
        c-means functional :math:`J` is not comparable between runs of an
        exponential-distance algorithm, so the kept run is chosen by
        ``restart_criterion``.
    restart_criterion : {"likelihood", "error"}, default="likelihood"
        ``"likelihood"`` keeps the run with the largest log-likelihood
        :math:`\sum_k\log\sum_i 1/D^2_{ik}` of the data under the mixture of
        local models (the quantity the EM algorithm maximizes);
        ``"error"`` keeps the run whose Takagi–Sugeno output has the smallest
        sum of squared errors on the training data. The two can disagree: a
        rule with a very small error variance on part of the data raises the
        likelihood without improving the global fit.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_antecedent)
        Centers :math:`v_{ij}` of the Gaussian membership functions.
    input_variances_ : ndarray of shape (n_clusters, n_antecedent)
        Variances :math:`\sigma^2_{ij}` of the membership functions.
    local_models_ : ndarray of shape (n_clusters, n_features + 1, n_outputs)
        Parameters :math:`\boldsymbol\theta_i=[\mathbf{a}_i^T\, b_i]^T` of the
        local linear models (intercept in the last row).
    coef_ : ndarray of shape (n_clusters, n_features) or (n_clusters, n_outputs, n_features)
    intercept_ : ndarray of shape (n_clusters,) or (n_clusters, n_outputs)
    error_covariances_ : ndarray of shape (n_clusters, n_outputs, n_outputs)
        Weighted covariance of the local model errors :math:`\mathbf{F}^{yy}_i`.
    error_variances_ : ndarray of shape (n_clusters,) or (n_clusters, n_outputs)
        Diagonal of ``error_covariances_`` (:math:`\sigma_i^2` for one output).
    priors_ : ndarray of shape (n_clusters,)
        Prior probabilities :math:`\alpha_i`.
    rule_weights_ : ndarray of shape (n_clusters,)
        :math:`w_i=\alpha_i\prod_j (2\pi\sigma^2_{ij})^{-1/2}`.
    log_likelihood_ : float
        :math:`\sum_k\log\sum_i 1/D^2_{ik}` of the kept run.
    restart_scores_ : ndarray of shape (n_init,)
        Value of the restart criterion of every run (higher is better).
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.
    n_outputs_ : int

    Notes
    -----
    With :math:`\mathbf{z}_k=[\mathbf{x}_k^T\,y_k]^T` the distance is

    .. math::
        \frac{1}{D^2_{ik}} = \alpha_i \prod_{j=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2_{ij}}}
        \exp\!\Big(-\tfrac12\frac{(x_{jk}-v_{ij})^2}{\sigma^2_{ij}}\Big)\cdot
        \frac{1}{\sqrt{2\pi\sigma_i^2}}
        \exp\!\Big(-\frac{(y_k-\mathbf{a}_i^T\mathbf{x}_k-b_i)^2}{2\sigma_i^2}\Big),

    and the parameters are updated with the memberships as weights:

    .. math::
        \mathbf{v}_i=\frac{\sum_k\mu_{ik}\mathbf{x}_k}{\sum_k\mu_{ik}},\quad
        \sigma^2_{ij}=\frac{\sum_k\mu_{ik}(x_{jk}-v_{ij})^2}{\sum_k\mu_{ik}},\quad
        \boldsymbol\theta_i=(\mathbf{X}_e^T\boldsymbol\Phi_i\mathbf{X}_e)^{-1}\mathbf{X}_e^T\boldsymbol\Phi_i\mathbf{y},\quad
        \alpha_i=\frac1N\sum_k\mu_{ik},

    where :math:`\boldsymbol\Phi_i=\operatorname{diag}(\mu_{i1},\dots,\mu_{iN})`
    and :math:`\sigma_i^2` is the :math:`\mu`-weighted mean squared residual
    (for several outputs the weighted error covariance
    :math:`\mathbf{F}^{yy}_i` and the multivariate Gaussian density replace
    :math:`\sigma_i^2`). The partition is updated with the usual
    :math:`\mu_{ik}=1/\sum_j (D_{ik}/D_{jk})^{2/(m-1)}`; for :math:`m=2` the
    memberships equal the posterior probabilities of the EM algorithm.
    All densities are evaluated in the log domain.

    Unlike the clusterers of :mod:`fuzzyclust.cluster`, :meth:`predict`
    returns the **Takagi–Sugeno model output**, not a hard label: the rules
    are aggregated with the normalized firing strengths
    :math:`\beta_i(\mathbf{x})=w_i\prod_j A_{ij}(x_j)` of the Gaussian
    antecedents (see :meth:`firing_strengths`). Hard labels of training
    samples remain available as ``labels_``.

    References
    ----------
    .. [1] J. Abonyi, R. Babuška, F. Szeifert, "Modified Gath–Geva fuzzy
           clustering for identification of Takagi–Sugeno fuzzy models",
           IEEE Trans. Systems, Man and Cybernetics B, 32(5), 612–621, 2002.
    .. [2] J. Abonyi, F. Szeifert, "Fuzzy clustering for the identification of
           Takagi–Sugeno fuzzy models of MIMO dynamical systems", 2002.

    Examples
    --------
    >>> from fuzzyclust.models import ModifiedGG, make_sinc
    >>> X, y = make_sinc(n_samples=200, random_state=0)
    >>> mgg = ModifiedGG(n_clusters=4, random_state=0).fit(X, y)
    >>> mgg.predict(X).shape
    (200,)
    >>> list(mgg.describe_rules().columns)
    ['rule', 'weight', 'prior', 'x1 center', 'x1 sigma', 'coef x1', 'intercept', 'error sigma']
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        antecedent_features=None,
        gamma: float = 0.0,
        var_floor: float = 1e-6,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="kmeans++",
        n_init: int = 3,
        restart_criterion: str = "likelihood",
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            random_state=random_state,
            verbose=verbose,
        )
        self.antecedent_features = antecedent_features
        self.gamma = gamma
        self.var_floor = var_floor
        self.n_init = n_init
        self.restart_criterion = restart_criterion

    # ------------------------------------------------------------------ helpers
    def _antecedent_index(self, n_features: int) -> np.ndarray:
        if self.antecedent_features is None:
            return np.arange(n_features)
        idx = np.asarray(self.antecedent_features, dtype=int).ravel()
        if idx.size == 0 or np.any(idx < 0) or np.any(idx >= n_features):
            raise ValueError("antecedent_features must be valid column indices of X.")
        return idx

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("ModifiedGG requires m > 1.")
        if self.var_floor < 0:
            raise ValueError("var_floor must be non-negative.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if self.restart_criterion not in ("likelihood", "error"):
            raise ValueError("restart_criterion must be 'likelihood' or 'error'.")
        self._ante_idx = self._antecedent_index(X.shape[1])
        Xa = X[:, self._ante_idx]
        self._feature_var = np.maximum(Xa.var(axis=0), EPS)
        self._var_floor_abs = self.var_floor * self._feature_var + EPS

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Cluster the input–output data and identify the Takagi–Sugeno rules.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Regressors (inputs).
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Output(s); required (the default ``None`` only exists for API
            compatibility and raises a ``ValueError``).

        Returns
        -------
        self : object
        """
        _require_y("ModifiedGG", y)
        X, y = validate_data(self, X, y, dtype=float, multi_output=True, y_numeric=True, ensure_min_samples=2)
        self._y_is_1d = y.ndim == 1
        self._Y = y[:, None] if y.ndim == 1 else y
        self.n_outputs_ = self._Y.shape[1]
        try:
            _fit_with_restarts(self, X, lambda est: est._restart_score(X, est._Y))
        finally:
            del self._Y
        self.log_likelihood_ = self._log_likelihood(X, y[:, None] if y.ndim == 1 else y)
        return self

    def _log_likelihood(self, X, Y):
        return float(np.sum(logsumexp(-self._log_distances(X, Y), axis=1)))

    def _restart_score(self, X, Y):
        """Criterion of a run (higher is better): log-likelihood or minus the TS output error."""
        if self.restart_criterion == "likelihood":
            return self._log_likelihood(X, Y)
        B = self._normalized_firing(X)
        local = np.stack([_extend(X) @ th for th in self.local_models_], axis=1)
        return -float(np.sum((Y - np.einsum("ki,kio->ko", B, local)) ** 2))

    def _update_prototypes(self, X, Um):
        N, n = X.shape
        c = self.n_clusters
        U = Um ** (1.0 / self.m)  # EM weights = posterior memberships (paper eqs. 38-41)
        s = np.maximum(U.sum(axis=0), EPS)
        Xa = X[:, self._ante_idx]
        V = (U.T @ Xa) / s[:, None]
        S2 = np.empty_like(V)
        Xe = _extend(X)
        Y = self._Y
        no = Y.shape[1]
        theta = np.empty((c, n + 1, no))
        Fyy = np.empty((c, no, no))
        for i in range(c):
            d = Xa - V[i]
            S2[i] = (U[:, i] @ d**2) / s[i]
            theta[i] = weighted_least_squares(Xe, Y, U[:, i])
            r = Y - Xe @ theta[i]
            Fyy[i] = (r * U[:, i][:, None]).T @ r / s[i]
        if self.gamma > 0:
            S2 = (1.0 - self.gamma) * S2 + self.gamma * self._feature_var
        S2 = np.maximum(S2, self._var_floor_abs)
        for i in range(c):
            Fyy[i].flat[:: no + 1] += EPS
        self.cluster_centers_ = V
        self.input_variances_ = S2
        self.local_models_ = theta
        self.error_covariances_ = Fyy
        self.priors_ = np.maximum(U.sum(axis=0) / N, EPS)
        self._set_derived()

    def _set_derived(self):
        inv = np.empty_like(self.error_covariances_)
        logdet = np.empty(self.n_clusters)
        for i in range(self.n_clusters):
            sign, ld = np.linalg.slogdet(self.error_covariances_[i])
            logdet[i] = ld if sign > 0 else np.log(EPS)
            inv[i] = np.linalg.pinv(self.error_covariances_[i])
        self._inv_err_cov = inv
        self._logdet_err = logdet
        # log w_i = log alpha_i - 1/2 sum_j log(2 pi sigma_ij^2)
        self._log_weights = np.log(self.priors_) - 0.5 * np.sum(
            _LOG_2PI + np.log(self.input_variances_), axis=1
        )
        self.rule_weights_ = np.exp(self._log_weights)
        theta = self.local_models_
        if self._y_is_1d:
            self.coef_ = theta[:, :-1, 0]
            self.intercept_ = theta[:, -1, 0]
            self.error_variances_ = self.error_covariances_[:, 0, 0]
        else:
            self.coef_ = np.transpose(theta[:, :-1, :], (0, 2, 1))
            self.intercept_ = theta[:, -1, :]
            self.error_variances_ = np.array([np.diag(F) for F in self.error_covariances_])

    # ------------------------------------------------------------------ distances
    def _log_firing(self, X):
        """Unnormalized log firing strengths log(w_i prod_j A_ij(x_j)), shape (N, c)."""
        Xa = X[:, self._ante_idx]
        c = self.n_clusters
        out = np.empty((X.shape[0], c))
        for i in range(c):
            d2 = (Xa - self.cluster_centers_[i]) ** 2 / self.input_variances_[i]
            out[:, i] = self._log_weights[i] - 0.5 * d2.sum(axis=1)
        return out

    def _log_output_density(self, X, Y):
        """log p(y | x, eta_i), shape (N, c)."""
        Xe = _extend(X)
        no = Y.shape[1]
        out = np.empty((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            r = Y - Xe @ self.local_models_[i]
            maha = np.einsum("kj,jl,kl->k", r, self._inv_err_cov[i], r)
            out[:, i] = -0.5 * (no * _LOG_2PI + self._logdet_err[i] + maha)
        return out

    def _log_distances(self, X, Y=None):
        logd2 = -self._log_firing(X)
        if Y is not None:
            logd2 -= self._log_output_density(X, Y)
        return logd2

    def _distances(self, X):
        Y = getattr(self, "_Y", None)
        return np.exp(np.minimum(self._log_distances(X, Y), _LOG_CAP))

    def _memberships(self, d2):
        logits = -np.log(d2 + EPS) / (self.m - 1.0)
        logits -= logits.max(axis=1, keepdims=True)
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ inference
    def _prepare(self, X, y=None):
        check_is_fitted(self, "cluster_centers_")
        if y is None:
            X = validate_data(self, X, dtype=float, reset=False)
            return X, None
        X, y = validate_data(self, X, y, dtype=float, multi_output=True, y_numeric=True, reset=False)
        Y = y[:, None] if y.ndim == 1 else y
        if Y.shape[1] != self.n_outputs_:
            raise ValueError(f"y has {Y.shape[1]} outputs, the model was fitted with {self.n_outputs_}.")
        return X, Y

    def _normalized_firing(self, X):
        logb = self._log_firing(X)
        logb -= logb.max(axis=1, keepdims=True)
        B = np.exp(logb)
        return B / B.sum(axis=1, keepdims=True)

    def firing_strengths(self, X, normalize: bool = True):
        r"""Rule firing strengths :math:`\beta_i(\mathbf{x})=w_i\prod_j A_{ij}(x_j)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        normalize : bool, default=True
            Return :math:`\beta_i/\sum_j\beta_j` (rows sum to one). These are
            the posterior cluster probabilities given the input only.

        Returns
        -------
        beta : ndarray of shape (n_samples, n_clusters)
        """
        X, _ = self._prepare(X)
        logb = self._log_firing(X)
        if not normalize:
            return np.exp(np.minimum(logb, _LOG_CAP))
        logb -= logb.max(axis=1, keepdims=True)
        B = np.exp(logb)
        return B / B.sum(axis=1, keepdims=True)

    def local_predict(self, X):
        """Outputs of the individual local linear models, shape ``(n_samples, n_clusters[, n_outputs])``."""
        X, _ = self._prepare(X)
        Xe = _extend(X)
        out = np.stack([Xe @ th for th in self.local_models_], axis=1)  # (N, c, no)
        return out[:, :, 0] if self._y_is_1d else out

    def predict(self, X):
        r"""Takagi–Sugeno model output.

        .. math::
            \hat y=\frac{\sum_i\beta_i(\mathbf{x})(\mathbf{a}_i^T\mathbf{x}+b_i)}{\sum_i\beta_i(\mathbf{x})}

        Returns
        -------
        y_hat : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        B = self.firing_strengths(X)
        X, _ = self._prepare(X)
        Xe = _extend(X)
        local = np.stack([Xe @ th for th in self.local_models_], axis=1)  # (N, c, no)
        yhat = np.einsum("ki,kio->ko", B, local)
        return yhat[:, 0] if self._y_is_1d else yhat

    def predict_proba(self, X, y=None):
        r"""Cluster memberships of (new) samples.

        Without ``y`` the memberships are the normalized firing strengths of
        the antecedents, :math:`p(\eta_i\mid\mathbf{x})`. With ``y`` the
        output-error term enters as during fitting and the memberships are
        :math:`p(\eta_i\mid\mathbf{x},y)`, i.e. the partition the clustering
        would produce for these input–output pairs.

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
        """
        X, Y = self._prepare(X, y)
        return self._memberships(np.exp(np.minimum(self._log_distances(X, Y), _LOG_CAP)))

    def transform(self, X, y=None):
        r"""Distances :math:`D_{ik}` of samples to the rules.

        Without ``y`` only the antecedent term enters (the distance a new
        input has to the rule); with ``y`` the output-error term is included
        as during fitting.
        """
        X, Y = self._prepare(X, y)
        return np.sqrt(np.exp(np.minimum(self._log_distances(X, Y), _LOG_CAP)))

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return the antecedent distances ``transform(X)`` of the training inputs."""
        return self.fit(X, y).transform(X)

    def fit_predict(self, X, y=None, **kwargs):
        """Fit and return the hard cluster labels of the training samples."""
        return self.fit(X, y).labels_

    def score(self, X, y):
        """Negative clustering objective :math:`-J` of the pairs ``(X, y)`` (higher is better)."""
        X, Y = self._prepare(X, y)
        d2 = np.exp(np.minimum(self._log_distances(X, Y), _LOG_CAP))
        U = self._memberships(d2)
        return -self._objective(U**self.m, d2)

    def describe_rules(self, feature_names=None, output_names=None) -> pd.DataFrame:
        """Readable rule base: one row per rule (and per output for MIMO models).

        Parameters
        ----------
        feature_names : sequence of str, optional
            Defaults to ``feature_names_in_`` or ``x1, x2, ...``.
        output_names : sequence of str, optional

        Returns
        -------
        pandas.DataFrame
            Columns ``rule``, ``weight``, ``prior``, the center and standard
            deviation of every antecedent set (``<name> center``, ``<name> sigma``),
            the consequent coefficients (``coef <name>``) and ``intercept``.
        """
        check_is_fitted(self, "cluster_centers_")
        n = self.n_features_in_
        if feature_names is None:
            feature_names = list(getattr(self, "feature_names_in_", [f"x{j + 1}" for j in range(n)]))
        feature_names = list(feature_names)
        if len(feature_names) != n:
            raise ValueError("feature_names must have n_features_in_ entries.")
        no = self.n_outputs_
        if output_names is None:
            output_names = ["y"] if no == 1 else [f"y{o + 1}" for o in range(no)]
        rows = []
        for i in range(self.n_clusters):
            for o in range(no):
                row = {"rule": i + 1}
                if no > 1:
                    row["output"] = output_names[o]
                row["weight"] = self.rule_weights_[i]
                row["prior"] = self.priors_[i]
                for a, j in enumerate(self._ante_idx):
                    row[f"{feature_names[j]} center"] = self.cluster_centers_[i, a]
                    row[f"{feature_names[j]} sigma"] = np.sqrt(self.input_variances_[i, a])
                for j in range(n):
                    row[f"coef {feature_names[j]}"] = self.local_models_[i, j, o]
                row["intercept"] = self.local_models_[i, -1, o]
                row["error sigma"] = np.sqrt(self.error_covariances_[i, o, o])
                rows.append(row)
        return pd.DataFrame(rows)

    rules_ = property(
        lambda self: self.describe_rules(), doc="Rule base as a DataFrame (see :meth:`describe_rules`)."
    )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_rule_{i}" for i in range(self.n_clusters)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags


__all__ = ["ModifiedGG", "weighted_least_squares"]
