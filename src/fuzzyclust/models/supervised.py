# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Supervised fuzzy clustering for the identification of fuzzy classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS
from ..base import BaseFuzzyClusterer
from .modified_gg import _LOG_2PI, _LOG_CAP, _fit_with_restarts, _require_y


class SupervisedFuzzyClassifier(ClassifierMixin, BaseFuzzyClusterer):
    r"""Fuzzy rule-based classifier identified by supervised Gath–Geva type clustering.

    Every cluster is a rule with axis-parallel Gaussian antecedents and a
    *class-probability vector* as consequent,

    .. math::
        r_i:\ \text{If } x_1 \text{ is } A_{i1}(x_1) \text{ and } \dots \text{ and }
        x_n \text{ is } A_{in}(x_n) \text{ then } \hat y = c_1 \text{ with } P(c_1\mid r_i),
        \dots,\ \hat y = c_C \text{ with } P(c_C\mid r_i)\ [w_i],

    so a rule may describe several classes and a class may need several rules
    (the number of rules :math:`R` can exceed the number of classes :math:`C`).
    The classifier is a quadratic Bayes classifier whose class-conditional
    densities are mixtures of axis-parallel Gaussians. The clustering uses the
    class labels: the distance of a labeled sample to a cluster is the
    reciprocal of the probability that the cluster generated both the input
    and the label [1]_.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of rules :math:`R` (may be larger than the number of classes).
    m : float, default=2.0
        Fuzziness exponent.
    gamma : float, default=0.0
        Variance regularization :math:`\gamma\in[0,1]`: every antecedent
        variance is shrunk toward the variance of the feature,
        :math:`\sigma^2_{ij}\leftarrow(1-\gamma)\sigma^2_{ij}+\gamma\,\mathrm{var}(x_j)`
        (the diagonal analog of the covariance regularization of
        :class:`~fuzzyclust.cluster.GG`). A small value (0.05–0.2) prevents the
        collapse of a rule onto a few samples, a known weakness of the
        exponential distance.
    var_floor : float, default=1e-6
        Lower bound of the antecedent variances relative to the feature variances.
    normalize_density : bool, default=True
        ``True``: the Gaussians of the distance carry their normalizing
        constants, so :math:`1/D^2_{ik}` is the joint probability
        :math:`p(\mathbf{x}_k, y_k, r_i)` (the Gath–Geva distance of eq. (5)
        and the rule weights of eq. (18) in [1]_ contain these constants, and
        for :math:`m=2` the clustering is then the EM algorithm of the
        mixture). ``False``: the constants are dropped during clustering,
        which is eq. (20) of [1]_ literally; clusters with small variances are
        then not favored by the distance. Classification always uses the
        weights :math:`w_i` of eq. (18).
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="kmeans++"
    n_init : int, default=3
        Number of random restarts; the run with the largest log-likelihood
        :math:`\sum_k\log\sum_i 1/D^2_{ik}` is kept (ignored when ``init``
        is an array).
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centers :math:`v_{ij}` of the Gaussian antecedent sets.
    input_variances_ : ndarray of shape (n_clusters, n_features)
        Variances :math:`\sigma^2_{ij}`.
    class_probs_ : ndarray of shape (n_clusters, n_classes)
        Consequents :math:`P(c_j\mid r_i)`.
    priors_ : ndarray of shape (n_clusters,)
        :math:`P(r_i)`.
    rule_weights_ : ndarray of shape (n_clusters,)
        :math:`w_i = P(r_i)\prod_j (2\pi\sigma^2_{ij})^{-1/2}`.
    log_likelihood_ : float
        :math:`\sum_k\log\sum_i 1/D^2_{ik}` of the kept run.
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer` (``labels_`` are
        *cluster* indices of the training samples, not classes).

    Notes
    -----
    The distance of :math:`\mathbf{z}_k=[\mathbf{x}_k^T\,y_k]^T` to rule :math:`i` is

    .. math::
        \frac{1}{D^2_{ik}} = P(r_i)\prod_{j=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2_{ij}}}
        \exp\!\Big(-\tfrac12\frac{(x_{jk}-v_{ij})^2}{\sigma^2_{ij}}\Big)\;P(c_j=y_k\mid r_i),

    the Gath–Geva distance of the inputs multiplied by the probability that the
    rule describes the class of the sample (eq. 20 of [1]_, written with the
    normalizing constants of the Gaussians so that :math:`1/D^2_{ik}` is the
    joint probability :math:`p(\mathbf{x}_k, y_k, r_i)`; the constants are
    exactly the rule weights :math:`w_i` of the paper). The updates are

    .. math::
        \mathbf{v}_i=\frac{\sum_k\mu_{ik}^m\mathbf{x}_k}{\sum_k\mu_{ik}^m},\quad
        \sigma^2_{ij}=\frac{\sum_k\mu_{ik}^m(x_{jk}-v_{ij})^2}{\sum_k\mu_{ik}^m},\quad
        P(c_j\mid r_i)=\frac{\sum_{k:\,y_k=c_j}\mu_{ik}^m}{\sum_k\mu_{ik}^m},\quad
        P(r_i)=\frac1N\sum_k\mu_{ik}^m ,

    followed by the usual membership update. Classification of a new input
    aggregates the rules with the normalized firing strengths
    :math:`\beta_i(\mathbf{x})=w_i\prod_j A_{ij}(x_j)`:

    .. math::
        P(c_j\mid\mathbf{x}) = \frac{\sum_i \beta_i(\mathbf{x})\,P(c_j\mid r_i)}
        {\sum_i\beta_i(\mathbf{x})},\qquad
        \hat y = \arg\max_j P(c_j\mid\mathbf{x}).

    Hence :meth:`predict_proba` returns **class** posteriors (as any
    scikit-learn classifier), not cluster memberships; the latter are
    available through :meth:`cluster_membership`.

    References
    ----------
    .. [1] J. Abonyi, F. Szeifert, "Supervised fuzzy clustering for the
           identification of fuzzy classifiers", Pattern Recognition Letters,
           24(14), 2195–2207, 2003.

    Examples
    --------
    >>> from fuzzyclust.datasets import load_iris
    >>> from fuzzyclust.models import SupervisedFuzzyClassifier
    >>> iris = load_iris()
    >>> clf = SupervisedFuzzyClassifier(n_clusters=3, random_state=0).fit(iris.data, iris.target)
    >>> clf.predict_proba(iris.data).shape
    (150, 3)
    >>> clf.score(iris.data, iris.target) > 0.9
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        gamma: float = 0.0,
        var_floor: float = 1e-6,
        normalize_density: bool = True,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="kmeans++",
        n_init: int = 3,
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
        self.gamma = gamma
        self.var_floor = var_floor
        self.normalize_density = normalize_density
        self.n_init = n_init

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("SupervisedFuzzyClassifier requires m > 1.")
        if self.var_floor < 0:
            raise ValueError("var_floor must be non-negative.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        self._feature_var = np.maximum(X.var(axis=0), EPS)
        self._var_floor_abs = self.var_floor * self._feature_var + EPS

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None):
        """Identify the rules from labeled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (any hashable values); required.

        Returns
        -------
        self : object
        """
        _require_y("SupervisedFuzzyClassifier", y)
        X, y = validate_data(self, X, y, dtype=float, ensure_min_samples=2)
        check_classification_targets(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        Yc = np.zeros((X.shape[0], self.classes_.shape[0]))
        Yc[np.arange(X.shape[0]), y_enc] = 1.0
        self._Yc = Yc
        try:
            _fit_with_restarts(
                self, X, lambda est: float(np.sum(logsumexp(-est._log_distances(X, est._Yc), axis=1)))
            )
        finally:
            del self._Yc
        return self

    def _update_prototypes(self, X, Um):
        N = X.shape[0]
        s = np.maximum(Um.sum(axis=0), EPS)
        V = (Um.T @ X) / s[:, None]
        S2 = np.empty_like(V)
        for i in range(self.n_clusters):
            S2[i] = (Um[:, i] @ (X - V[i]) ** 2) / s[i]
        if self.gamma > 0:
            S2 = (1.0 - self.gamma) * S2 + self.gamma * self._feature_var
        S2 = np.maximum(S2, self._var_floor_abs)
        self.cluster_centers_ = V
        self.input_variances_ = S2
        self.class_probs_ = (Um.T @ self._Yc) / s[:, None]
        self.priors_ = np.maximum(Um.sum(axis=0) / N, EPS)
        self._log_weights = np.log(self.priors_) - 0.5 * np.sum(_LOG_2PI + np.log(S2), axis=1)
        self.rule_weights_ = np.exp(self._log_weights)

    def _log_firing(self, X, normalized=True):
        """log(w_i prod_j A_ij(x_j)); with ``normalized=False`` the prior replaces the weight w_i."""
        out = np.empty((X.shape[0], self.n_clusters))
        logw = self._log_weights if normalized else np.log(self.priors_)
        for i in range(self.n_clusters):
            d2 = (X - self.cluster_centers_[i]) ** 2 / self.input_variances_[i]
            out[:, i] = logw[i] - 0.5 * d2.sum(axis=1)
        return out

    def _log_distances(self, X, Yc=None):
        logd2 = -self._log_firing(X, normalized=self.normalize_density)
        if Yc is not None:
            logd2 -= np.log(Yc @ self.class_probs_.T + EPS)
        return logd2

    def _distances(self, X):
        return np.exp(np.minimum(self._log_distances(X, getattr(self, "_Yc", None)), _LOG_CAP))

    def _memberships(self, d2):
        logits = -np.log(d2 + EPS) / (self.m - 1.0)
        logits -= logits.max(axis=1, keepdims=True)
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ inference
    def _normalized_firing(self, X):
        logb = self._log_firing(X)
        logb -= logb.max(axis=1, keepdims=True)
        B = np.exp(logb)
        return B / B.sum(axis=1, keepdims=True)

    def cluster_membership(self, X, y=None):
        r"""Memberships in the **rules** (clusters), ``(n_samples, n_clusters)``.

        Without ``y``: normalized firing strengths :math:`p(r_i\mid\mathbf{x})`.
        With ``y``: the class term enters as during fitting,
        :math:`p(r_i\mid\mathbf{x}, y)`.
        """
        check_is_fitted(self, "cluster_centers_")
        if y is None:
            X = validate_data(self, X, dtype=float, reset=False)
            return self._normalized_firing(X)
        X, y = validate_data(self, X, y, dtype=float, reset=False)
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= len(self.classes_)) or np.any(
            self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y
        ):
            raise ValueError("y contains labels unseen during fit.")
        Yc = np.zeros((X.shape[0], len(self.classes_)))
        Yc[np.arange(X.shape[0]), idx] = 1.0
        return self._memberships(np.exp(np.minimum(self._log_distances(X, Yc), _LOG_CAP)))

    def predict_proba(self, X):
        r"""Class posteriors :math:`P(c_j\mid\mathbf{x})`, shape ``(n_samples, n_classes)``."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        P = self._normalized_firing(X) @ self.class_probs_
        return P / np.maximum(P.sum(axis=1, keepdims=True), EPS)

    def predict(self, X):
        """Predicted class labels."""
        P = self.predict_proba(X)
        return self.classes_[P.argmax(axis=1)]

    def fit_predict(self, X, y=None, **kwargs):
        """Fit and return the predicted class labels of the training samples."""
        return self.fit(X, y).predict(X)

    def transform(self, X):
        """Antecedent distances :math:`D_{ik}` of ``X`` to the rules (``(n_samples, n_clusters)``)."""
        check_is_fitted(self, "cluster_centers_")
        X = validate_data(self, X, dtype=float, reset=False)
        return np.sqrt(np.exp(np.minimum(self._log_distances(X), _LOG_CAP)))

    # ------------------------------------------------------------------ feature selection
    def interclass_separability(self, features=None, criterion: str = "det") -> float:
        r"""Fisher interclass separability of the rules, :math:`J=\det\mathbf{F}_B/\det\mathbf{F}_W`.

        With the rule priors :math:`P(r_i)`, centers :math:`\mathbf{v}_i` and
        (diagonal) covariances :math:`\mathbf{F}_i=\operatorname{diag}(\sigma^2_{i1},\dots)`,

        .. math::
            \mathbf{F}_W=\sum_i P(r_i)\mathbf{F}_i,\qquad
            \mathbf{F}_B=\sum_i P(r_i)(\mathbf{v}_i-\mathbf{v}_0)(\mathbf{v}_i-\mathbf{v}_0)^T,\qquad
            \mathbf{v}_0=\sum_i P(r_i)\mathbf{v}_i

        (eq. 27–28 of Abonyi and Szeifert (2003)). A large :math:`J` means well separated, compact
        rules. ``features`` restricts the computation to a subset of the
        features; the importance of a feature is the value of :math:`J`
        obtained *without* it (see :meth:`feature_importance`).

        :math:`\mathbf{F}_B` has rank at most :math:`R-1`, so for fewer rules
        than features its determinant is zero; the product of its
        :math:`\min(R-1, n)` largest eigenvalues (pseudo-determinant) is used
        in that case, which keeps :math:`J` comparable between feature subsets
        of the same size. With ``criterion="trace"`` the classical Fisher
        criterion :math:`\operatorname{tr}(\mathbf{F}_W^{-1}\mathbf{F}_B)` is
        returned instead.
        """
        check_is_fitted(self, "cluster_centers_")
        idx = np.arange(self.n_features_in_) if features is None else np.asarray(features, dtype=int)
        if idx.size == 0:
            return float("nan")
        P = self.priors_ / self.priors_.sum()
        V = self.cluster_centers_[:, idx]
        v0 = P @ V
        d = V - v0
        FB = (d * P[:, None]).T @ d
        fw = P @ self.input_variances_[:, idx]  # diagonal of F_W
        if criterion == "trace":
            return float(np.trace(FB / fw[:, None]))
        if criterion != "det":
            raise ValueError("criterion must be 'det' or 'trace'.")
        rank = min(self.n_clusters - 1, idx.size)
        if rank < 1:
            return 0.0
        w = np.sort(np.linalg.eigvalsh((FB + FB.T) / 2.0))[::-1][:rank]
        if np.any(w <= 0):
            return 0.0
        return float(np.exp(np.sum(np.log(w)) - np.sum(np.log(fw))))

    def feature_importance(self, criterion: str = "det") -> np.ndarray:
        r"""Separability :math:`J_{-j}` of the rule base when feature :math:`j` is left out.

        The feature with the *largest* :math:`J_{-j}` is the one whose removal
        harms the separability least, i.e. the least needed feature; the
        step-wise elimination of Abonyi and Szeifert (2003) is implemented in
        :func:`select_features_by_separability`.

        Returns
        -------
        J_minus : ndarray of shape (``n_features_in_``,)
        """
        check_is_fitted(self, "cluster_centers_")
        n = self.n_features_in_
        return np.array(
            [self.interclass_separability(np.delete(np.arange(n), j), criterion) for j in range(n)]
        )

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def rule_base(self, feature_names=None, class_names=None) -> pd.DataFrame:
        """Rule base as a table: rule weight, prior, antecedent sets and class probabilities."""
        check_is_fitted(self, "cluster_centers_")
        n = self.n_features_in_
        if feature_names is None:
            feature_names = list(getattr(self, "feature_names_in_", [f"x{j + 1}" for j in range(n)]))
        feature_names = list(feature_names)
        if len(feature_names) != n:
            raise ValueError("feature_names must have n_features_in_ entries.")
        if class_names is None:
            class_names = [str(c) for c in self.classes_]
        rows = []
        for i in range(self.n_clusters):
            row = {"rule": i + 1, "weight": self.rule_weights_[i], "prior": self.priors_[i]}
            for j in range(n):
                row[f"{feature_names[j]} center"] = self.cluster_centers_[i, j]
                row[f"{feature_names[j]} sigma"] = np.sqrt(self.input_variances_[i, j])
            for j, cn in enumerate(class_names):
                row[f"P({cn})"] = self.class_probs_[i, j]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_rule_{i}" for i in range(self.n_clusters)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        return tags


def select_features_by_separability(
    X, y, n_features_to_select: int = 1, criterion: str = "det", **classifier_params
):
    r"""Backward feature elimination by the Fisher interclass separability of the rules.

    The step-wise procedure of Abonyi and Szeifert (2003): a
    :class:`SupervisedFuzzyClassifier` is identified on the current feature
    set, the separability :math:`J_{-j}` without each feature is computed
    (:meth:`SupervisedFuzzyClassifier.feature_importance`), the feature whose
    removal reduces :math:`J` least is deleted, and the classifier is
    re-identified on the reduced data. The training accuracy after every step
    is recorded so that the number of features can be chosen where the
    performance starts to drop.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_features_to_select : int, default=1
        Stop when this many features remain.
    criterion : {"det", "trace"}, default="det"
        See :meth:`SupervisedFuzzyClassifier.interclass_separability`.
    **classifier_params
        Passed to :class:`SupervisedFuzzyClassifier` (e.g. ``n_clusters``,
        ``random_state``).

    Returns
    -------
    selected : ndarray of int
        Indices of the remaining features (in the original order).
    eliminated : list of int
        Features in the order of their removal.
    history : pandas.DataFrame
        One row per step: ``n_features``, ``removed`` (``-1`` before the
        first removal), ``separability`` :math:`J` and training ``accuracy``
        of the classifier identified on the features of that step.

    References
    ----------
    .. [1] J. Abonyi, F. Szeifert, "Supervised fuzzy clustering for the
           identification of fuzzy classifiers", Pattern Recognition Letters,
           24(14), 2195–2207, 2003.

    Examples
    --------
    >>> from fuzzyclust.datasets import load_iris
    >>> from fuzzyclust.models import select_features_by_separability
    >>> iris = load_iris()
    >>> sel, order, hist = select_features_by_separability(
    ...     iris.data, iris.target, n_features_to_select=2, n_clusters=3, random_state=0)
    >>> len(sel), len(order), list(hist.columns)
    (2, 2, ['n_features', 'removed', 'separability', 'accuracy'])
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[1]
    if not 1 <= n_features_to_select <= n:
        raise ValueError("n_features_to_select must lie in [1, n_features].")
    remaining = list(range(n))
    eliminated: list[int] = []
    rows = []
    removed = -1
    while True:
        clf = SupervisedFuzzyClassifier(**classifier_params).fit(X[:, remaining], y)
        rows.append(
            {
                "n_features": len(remaining),
                "removed": removed,
                "separability": clf.interclass_separability(criterion=criterion),
                "accuracy": float(np.mean(clf.predict(X[:, remaining]) == y)),
            }
        )
        if len(remaining) <= n_features_to_select:
            break
        drop = int(np.argmax(clf.feature_importance(criterion)))
        removed = remaining.pop(drop)
        eliminated.append(removed)
    return np.asarray(remaining), eliminated, pd.DataFrame(rows)


__all__ = ["SupervisedFuzzyClassifier", "select_features_by_separability"]
