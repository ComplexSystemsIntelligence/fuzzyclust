# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Takagi–Sugeno fuzzy model identified by product-space fuzzy clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS
from ..cluster.gg import GG
from ..cluster.gk import GK
from .modified_gg import _LOG_2PI, _LOG_CAP, ModifiedGG, _extend, _require_y, weighted_least_squares

_ANTECEDENTS = ("gaussian_axis_parallel", "gg_projected", "gk_projected")
_CONSEQUENTS = ("local", "global", "tls")


class TakagiSugeno(RegressorMixin, BaseEstimator):
    r"""Takagi–Sugeno (TS) fuzzy model with rules extracted from fuzzy clusters.

    The model consists of :math:`c` rules

    .. math::
        R_i:\ \text{If } x_1 \text{ is } A_{i1}(x_1) \text{ and } \dots \text{ and }
        x_n \text{ is } A_{in}(x_n) \text{ then } \hat y_i = \mathbf{a}_i^T\mathbf{x} + b_i,\ [w_i]

    with Gaussian antecedent sets :math:`A_{ij}(x_j)=\exp(-\tfrac12 (x_j-v_{ij})^2/\sigma_{ij}^2)`,
    aggregated by the fuzzy-mean formula with the firing strengths
    :math:`\beta_i(\mathbf{x})=w_i\prod_j A_{ij}(x_j)`:

    .. math::
        \hat y(\mathbf{x}) = \frac{\sum_{i=1}^{c}\beta_i(\mathbf{x})\,(\mathbf{a}_i^T\mathbf{x}+b_i)}
                                  {\sum_{i=1}^{c}\beta_i(\mathbf{x})}.

    Three ways of obtaining the antecedents from clusters in the product space
    :math:`\mathbf{z}=[\mathbf{x}^T\,y]^T` are offered:

    * ``"gaussian_axis_parallel"``: the modified Gath–Geva clustering of [1]_
      (:class:`ModifiedGG`). The clusters are axis-parallel Gaussians in the
      input space, so the antecedent sets *are* the clusters and no projection
      error arises; the rule weights :math:`w_i=\alpha_i\prod_j(2\pi\sigma^2_{ij})^{-1/2}`
      come out of the clustering.
    * ``"gg_projected"``: Gath–Geva clustering of :math:`\mathbf{z}` followed
      by the axis-orthogonal projection of the (oblique) Gaussian clusters,
      :math:`\sigma^2_{ij}=[\mathbf{F}^{xx}_i]_{jj}`, with the rule weights
      :math:`w_i=\alpha_i/\big((2\pi)^{n/2}\sqrt{\det\mathbf{F}^{xx}_i}\big)` of
      [1]_ (eq. 17). This is the classical GG route to a TS model; the
      projection of an oblique ellipsoid introduces a decomposition error.
    * ``"gk_projected"``: Gustafson–Kessel clustering of :math:`\mathbf{z}`
      and the same projection [3]_; GK finds clusters of equal volume and has
      no priors, so all rule weights are one.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of rules :math:`c`.
    m : float, default=2.0
        Fuzziness exponent of the clustering.
    antecedent : {"gaussian_axis_parallel", "gg_projected", "gk_projected"}, default="gaussian_axis_parallel"
        Clustering engine and antecedent extraction (see above).
    consequent : {"local", "global", "tls"}, default="local"
        Estimation of the local linear models. ``"local"``: weighted least
        squares per rule with the cluster memberships as weights (the local
        models are local linearizations, interpretable, [1]_ eq. 22).
        ``"global"``: one least-squares problem for all consequents with the
        normalized firing strengths as regressor weights (best global
        prediction, local interpretation lost). ``"tls"``: total least
        squares from the eigenvector of the smallest eigenvalue of the
        product-space cluster covariance ([1]_ eq. 26, [3]_), the hyperplane
        the cluster lies on.
    antecedent_features : array-like of int, optional
        Columns of ``X`` used in the antecedents (scheduling variables); the
        consequents always use all columns. ``None``: all inputs.
    gamma : float, default=0.0
    beta : float, default=1e15
        Covariance (variance) regularization of the clustering engine, see
        :class:`~fuzzyclust.cluster.GK`, :class:`~fuzzyclust.cluster.GG` and
        :class:`ModifiedGG` (``beta`` is ignored by the latter).
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="kmeans++"
        Initialization of the clustering engine (an initial partition of
        shape ``(n_samples, n_clusters)`` or initial centers are accepted; for
        the projected variants initial centers live in the product space).
    n_init : int, default=1
        Number of restarts of the clustering engine.
    restart_criterion : {"likelihood", "error"}, default="likelihood"
        How :class:`ModifiedGG` ranks its restarts (ignored by the projected
        variants, whose engines keep the run with the lowest :math:`J`).
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    clusterer_ : fitted ModifiedGG, GG or GK
        The clustering engine.
    centers_ : ndarray of shape (n_clusters, n_antecedent)
        Centers :math:`v_{ij}` of the antecedent sets.
    sigmas_ : ndarray of shape (n_clusters, n_antecedent)
        Standard deviations :math:`\sigma_{ij}` of the antecedent sets.
    rule_weights_ : ndarray of shape (n_clusters,)
        :math:`w_i` (from the cluster priors and variances for the modified
        GG engine, all one for the GK engine).
    local_models_ : ndarray of shape (n_clusters, n_features + 1, n_outputs)
        :math:`\boldsymbol\theta_i=[\mathbf{a}_i^T\,b_i]^T`.
    coef_ : ndarray of shape (n_clusters, n_features) or (n_clusters, n_outputs, n_features)
    intercept_ : ndarray of shape (n_clusters,) or (n_clusters, n_outputs)
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy partition of the training data in the product space.
    n_outputs_ : int
    antecedent_features_ : ndarray of int
    n_iter_ : int
        Iterations of the clustering engine.

    References
    ----------
    .. [1] J. Abonyi, R. Babuška, F. Szeifert, "Modified Gath–Geva fuzzy
           clustering for identification of Takagi–Sugeno fuzzy models",
           IEEE Trans. Systems, Man and Cybernetics B, 32(5), 612–621, 2002.
    .. [2] T. Takagi, M. Sugeno, "Fuzzy identification of systems and its
           applications to modeling and control", IEEE Trans. Systems, Man
           and Cybernetics, 15(1), 116–132, 1985.
    .. [3] R. Babuška, "Fuzzy Modeling for Control", Kluwer, Boston, 1998.

    Examples
    --------
    >>> from fuzzyclust.models import TakagiSugeno, make_sinc
    >>> X, y = make_sinc(n_samples=200, random_state=0)
    >>> ts = TakagiSugeno(n_clusters=4, random_state=0).fit(X, y)
    >>> round(ts.score(X, y), 2) > 0.9
    True
    >>> ts.rule_base().shape[0]
    4
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        antecedent: str = "gaussian_axis_parallel",
        consequent: str = "local",
        antecedent_features=None,
        gamma: float = 0.0,
        beta: float = 1e15,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="kmeans++",
        n_init: int = 1,
        restart_criterion: str = "likelihood",
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.antecedent = antecedent
        self.consequent = consequent
        self.antecedent_features = antecedent_features
        self.gamma = gamma
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.restart_criterion = restart_criterion
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Identify the rule base from input–output data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Required (``None`` raises a ``ValueError``).

        Returns
        -------
        self : object
        """
        _require_y("TakagiSugeno", y)
        X, y = validate_data(self, X, y, dtype=float, multi_output=True, y_numeric=True, ensure_min_samples=2)
        if self.antecedent not in _ANTECEDENTS:
            raise ValueError(f"antecedent must be one of {_ANTECEDENTS}.")
        if self.consequent not in _CONSEQUENTS:
            raise ValueError(f"consequent must be one of {_CONSEQUENTS}.")
        self._y_is_1d = y.ndim == 1
        Y = y[:, None] if y.ndim == 1 else y
        N, n = X.shape
        no = Y.shape[1]
        self.n_outputs_ = no
        if self.antecedent_features is None:
            idx = np.arange(n)
        else:
            idx = np.asarray(self.antecedent_features, dtype=int).ravel()
            if idx.size == 0 or np.any(idx < 0) or np.any(idx >= n):
                raise ValueError("antecedent_features must be valid column indices of X.")
        self.antecedent_features_ = idx
        common = dict(
            n_clusters=self.n_clusters,
            m=self.m,
            tol=self.tol,
            max_iter=self.max_iter,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        if self.antecedent == "gaussian_axis_parallel":
            eng = ModifiedGG(
                antecedent_features=self.antecedent_features,
                gamma=self.gamma,
                restart_criterion=self.restart_criterion,
                **common,
            ).fit(X, Y if no > 1 else y)
            self.centers_ = eng.cluster_centers_.copy()
            self.sigmas_ = np.sqrt(eng.input_variances_)
            self.rule_weights_ = eng.rule_weights_.copy()
            theta = eng.local_models_.copy()
        else:
            # product-space clustering followed by axis-orthogonal projection of F_i^xx
            Z = np.hstack([X, Y])
            engine = GG if self.antecedent == "gg_projected" else GK
            eng = engine(gamma=self.gamma, beta=self.beta, **common).fit(Z)
            self.centers_ = eng.cluster_centers_[:, idx].copy()
            Fxx = eng.covariances_[:, :n, :n]
            var = np.array([np.diag(F)[idx] for F in Fxx])
            self.sigmas_ = np.sqrt(np.maximum(var, EPS))
            if self.antecedent == "gg_projected":
                # eq. (17) of Abonyi, Babuska & Szeifert: w_i = alpha_i / ((2 pi)^(n/2) sqrt(det F_i^xx))
                Fa = Fxx[:, idx][:, :, idx]
                logdet = np.array([np.linalg.slogdet(F)[1] for F in Fa])
                self.rule_weights_ = np.exp(np.log(eng.priors_) - 0.5 * (idx.size * _LOG_2PI + logdet))
            else:
                self.rule_weights_ = np.ones(self.n_clusters)
            theta = None
        self.clusterer_ = eng
        self.n_iter_ = eng.n_iter_
        U = eng.membership_
        self.membership_ = U
        Xe = _extend(X)
        if self.consequent == "local" and theta is None:
            theta = np.stack([weighted_least_squares(Xe, Y, U[:, i]) for i in range(self.n_clusters)])
        elif self.consequent == "global":
            theta = self._global_least_squares(X, Y)
        elif self.consequent == "tls":
            theta = self._total_least_squares(X, Y, U)
        self.local_models_ = theta
        if self._y_is_1d:
            self.coef_ = theta[:, :-1, 0]
            self.intercept_ = theta[:, -1, 0]
        else:
            self.coef_ = np.transpose(theta[:, :-1, :], (0, 2, 1))
            self.intercept_ = theta[:, -1, :]
        return self

    def _global_least_squares(self, X, Y):
        B = self._firing(X, normalize=True)  # (N, c)
        Xe = _extend(X)
        n1 = Xe.shape[1]
        Phi = np.hstack([Xe * B[:, i][:, None] for i in range(self.n_clusters)])  # (N, c*(n+1))
        G = Phi.T @ Phi
        G.flat[:: G.shape[0] + 1] += EPS * max(np.trace(G) / G.shape[0], 1.0)
        theta_all = np.linalg.solve(G, Phi.T @ Y)  # (c*(n+1), no)
        return theta_all.reshape(self.n_clusters, n1, Y.shape[1])

    def _total_least_squares(self, X, Y, U):
        if Y.shape[1] != 1:
            raise ValueError("consequent='tls' is defined for a single output.")
        Z = np.hstack([X, Y])
        Um = U**self.m
        n = X.shape[1]
        theta = np.empty((self.n_clusters, n + 1, 1))
        for i in range(self.n_clusters):
            s = max(Um[:, i].sum(), EPS)
            v = (Um[:, i] @ Z) / s
            d = Z - v
            F = (d * Um[:, i][:, None]).T @ d / s
            w, Q = np.linalg.eigh(F)
            t = Q[:, 0]  # normal of the hyperplane the cluster lies on
            ty = t[-1] if abs(t[-1]) > EPS else EPS
            a = -t[:-1] / ty
            b = (t @ v) / ty
            theta[i, :-1, 0] = a
            theta[i, -1, 0] = b
        return theta

    # ------------------------------------------------------------------ inference
    def _check_X(self, X):
        check_is_fitted(self, "local_models_")
        return validate_data(self, X, dtype=float, reset=False)

    def antecedent_membership(self, X):
        r"""Univariate antecedent memberships :math:`A_{ij}(x_j)`.

        Returns an array of shape ``(n_samples, n_clusters, n_antecedent)``.
        """
        X = self._check_X(X)
        Xa = X[:, self.antecedent_features_]
        d = (Xa[:, None, :] - self.centers_[None, :, :]) / self.sigmas_[None, :, :]
        return np.exp(-0.5 * d**2)

    def _firing(self, X, normalize=True):
        Xa = X[:, self.antecedent_features_]
        logb = np.empty((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            d = (Xa - self.centers_[i]) / self.sigmas_[i]
            logb[:, i] = np.log(self.rule_weights_[i] + EPS) - 0.5 * np.sum(d**2, axis=1)
        if not normalize:
            return np.exp(np.minimum(logb, _LOG_CAP))
        logb -= logb.max(axis=1, keepdims=True)
        B = np.exp(logb)
        return B / B.sum(axis=1, keepdims=True)

    def firing_strengths(self, X, normalize: bool = True):
        r"""Firing strengths :math:`\beta_i(\mathbf{x})=w_i\prod_j A_{ij}(x_j)` (normalized by default)."""
        return self._firing(self._check_X(X), normalize=normalize)

    def local_predict(self, X):
        """Outputs of the local linear models, shape ``(n_samples, n_clusters[, n_outputs])``."""
        X = self._check_X(X)
        Xe = _extend(X)
        out = np.stack([Xe @ th for th in self.local_models_], axis=1)
        return out[:, :, 0] if self._y_is_1d else out

    def predict(self, X):
        """Model output (fuzzy-mean aggregation of the local models), shape ``(n_samples[, n_outputs])``."""
        X = self._check_X(X)
        B = self._firing(X, normalize=True)
        Xe = _extend(X)
        local = np.stack([Xe @ th for th in self.local_models_], axis=1)
        yhat = np.einsum("ki,kio->ko", B, local)
        return yhat[:, 0] if self._y_is_1d else yhat

    def rule_base(self, feature_names=None, output_names=None) -> pd.DataFrame:
        """The rule base as a table: antecedent sets (center, sigma), rule weight and consequents."""
        check_is_fitted(self, "local_models_")
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
                for a, j in enumerate(self.antecedent_features_):
                    row[f"{feature_names[j]} center"] = self.centers_[i, a]
                    row[f"{feature_names[j]} sigma"] = self.sigmas_[i, a]
                for j in range(n):
                    row[f"coef {feature_names[j]}"] = self.local_models_[i, j, o]
                row["intercept"] = self.local_models_[i, -1, o]
                rows.append(row)
        return pd.DataFrame(rows)

    def describe(self, feature_names=None, digits: int = 3) -> str:
        """Linguistic listing of the rules (``If x1 is N(v, sigma) and ... then y = ...``)."""
        check_is_fitted(self, "local_models_")
        n = self.n_features_in_
        if feature_names is None:
            feature_names = list(getattr(self, "feature_names_in_", [f"x{j + 1}" for j in range(n)]))
        lines = []
        f = f"{{:.{digits}g}}"
        for i in range(self.n_clusters):
            ante = " and ".join(
                f"{feature_names[j]} is N({f.format(self.centers_[i, a])}, {f.format(self.sigmas_[i, a])})"
                for a, j in enumerate(self.antecedent_features_)
            )
            for o in range(self.n_outputs_):
                th = self.local_models_[i, :, o]
                terms = " ".join(f"{th[j]:+.{digits}g}*{feature_names[j]}" for j in range(n))
                yname = "y" if self.n_outputs_ == 1 else f"y{o + 1}"
                lines.append(f"R{i + 1}: If {ante} then {yname} = {terms} {th[-1]:+.{digits}g}")
        return "\n".join(lines)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags


def simulate_free_run(model, u, y_init, n_a: int, n_b: int = 0, n_k: int = 1):
    r"""Free-run (parallel) simulation of a NARX model.

    The regressors follow the layout of :func:`~fuzzyclust.preprocessing.make_regressors`.
    The predicted outputs are fed back as regressors instead of the measured
    ones, :math:`\hat y_k = f(\hat y_{k-1},\dots,\hat y_{k-n_a}, u_{k-n_k},\dots)`,
    which is the demanding test of a dynamic model (errors can accumulate).

    Parameters
    ----------
    model : fitted regressor with ``predict``
        Takes rows ordered as the regressors of ``make_regressors``: lagged
        outputs first (lag 1 for all outputs, then lag 2, ...), then lagged inputs.
    u : array-like of shape (T,) or (T, n_inputs) or None
        Input sequence (``None`` for an autonomous model).
    y_init : array-like of shape (max_lag,) or (max_lag, n_outputs)
        Initial outputs (the first ``max_lag`` measured values).
    n_a, n_b, n_k : int
        Model orders and dead time, as used for ``make_regressors``.

    Returns
    -------
    y_sim : ndarray of shape (T,) or (T, n_outputs)
        Simulated output; the first ``max_lag`` entries are ``y_init``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.preprocessing import make_regressors
    >>> from fuzzyclust.models import TakagiSugeno, simulate_free_run, make_second_order_plant
    >>> u, y = make_second_order_plant(n_samples=150, random_state=0)
    >>> Phi, target, _ = make_regressors(y, u, n_a=2, n_b=1, n_k=0)
    >>> ts = TakagiSugeno(n_clusters=4, random_state=0).fit(Phi, target)
    >>> y_sim = simulate_free_run(ts, u, y[:2], n_a=2, n_b=1, n_k=0)
    >>> y_sim.shape
    (150,)
    """
    y_init = np.asarray(y_init, dtype=float)
    one_d = y_init.ndim == 1
    Y0 = y_init[:, None] if one_d else y_init
    if u is None:
        U = None
        n_b = 0
    else:
        U = np.asarray(u, dtype=float)
        U = U[:, None] if U.ndim == 1 else U
    max_lag = max(n_a, n_b + n_k - 1 if n_b > 0 else 0)
    if Y0.shape[0] < max_lag:
        raise ValueError(f"y_init must provide at least max_lag={max_lag} samples.")
    T = U.shape[0] if U is not None else Y0.shape[0]
    ny = Y0.shape[1]
    Y = np.zeros((T, ny))
    Y[:max_lag] = Y0[:max_lag]
    for k in range(max_lag, T):
        reg = [Y[k - d] for d in range(1, n_a + 1)]
        if U is not None:
            reg += [U[k - d] for d in range(n_k, n_k + n_b)]
        phi = np.concatenate(reg)[None, :]
        yk = np.asarray(model.predict(phi), dtype=float).reshape(1, -1)
        Y[k] = yk[0]
    return Y[:, 0] if one_d else Y


__all__ = ["TakagiSugeno", "simulate_free_run"]
