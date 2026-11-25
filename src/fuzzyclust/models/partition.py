# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Supervised fuzzy partitions of continuous features for decision-tree induction."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS
from .modified_gg import _require_y
from .supervised import SupervisedFuzzyClassifier

_SHAPES = ("gaussian", "trapezoid", "triangle")
_ENCODINGS = ("crisp", "membership")
_METHODS = ("supervised", "ruspini")


def membership_1d(x, centers, sigmas, shape: str = "trapezoid") -> np.ndarray:
    r"""Evaluate the fuzzy sets of one feature at the values ``x``.

    The sets are parameterized by a center :math:`v` and a width :math:`\sigma`
    (the parameters of a Gaussian cluster); the shapes are

    * ``"gaussian"``: :math:`A(x)=\exp(-\tfrac12 (x-v)^2/\sigma^2)`;
    * ``"trapezoid"``: the trapezoid with corner points
      :math:`a=v-3\sigma,\ b=v-0.5\sigma,\ c=v+0.5\sigma,\ d=v+3\sigma` (Abonyi 2010);
    * ``"triangle"``: the triangle with corners :math:`v-3\sigma,\ v,\ v+3\sigma`.

    Returns
    -------
    A : ndarray of shape (len(x), n_sets)
    """
    x = np.asarray(x, dtype=float).ravel()[:, None]
    v = np.asarray(centers, dtype=float)[None, :]
    s = np.maximum(np.asarray(sigmas, dtype=float), EPS)[None, :]
    if shape == "gaussian":
        return np.exp(-0.5 * ((x - v) / s) ** 2)
    if shape == "trapezoid":
        a, b, c, d = v - 3 * s, v - 0.5 * s, v + 0.5 * s, v + 3 * s
    elif shape == "triangle":
        a, b, c, d = v - 3 * s, v, v, v + 3 * s
    else:
        raise ValueError(f"shape must be one of {_SHAPES}.")
    up = (x - a) / np.maximum(b - a, EPS)
    down = (d - x) / np.maximum(d - c, EPS)
    return np.clip(np.minimum(np.minimum(up, 1.0), down), 0.0, 1.0)


def set_similarity(A: np.ndarray, B: np.ndarray) -> float:
    r"""Jaccard-type similarity :math:`S=|A\cap B|/|A\cup B|` of two fuzzy sets sampled on a common grid."""
    inter = np.minimum(A, B).sum()
    union = np.maximum(A, B).sum()
    return float(inter / union) if union > 0 else 1.0


def _merge_feature(centers, sigmas, weights, lo, hi, shape, threshold, grid_size=200):
    """Greedy merging of the most similar pair of sets until all similarities are below the threshold."""
    centers = list(np.asarray(centers, dtype=float))
    sigmas = list(np.asarray(sigmas, dtype=float))
    weights = list(np.asarray(weights, dtype=float))
    groups = [[i] for i in range(len(centers))]
    grid = np.linspace(lo, hi, grid_size)
    while len(centers) > 1:
        A = membership_1d(grid, centers, sigmas, shape)
        best, pair = -1.0, None
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                s = set_similarity(A[:, i], A[:, j])
                if s > best:
                    best, pair = s, (i, j)
        if best <= threshold:
            break
        i, j = pair
        wi, wj = weights[i], weights[j]
        w = wi + wj
        v = (wi * centers[i] + wj * centers[j]) / w
        var = (
            wi * (sigmas[i] ** 2 + (centers[i] - v) ** 2) + wj * (sigmas[j] ** 2 + (centers[j] - v) ** 2)
        ) / w
        centers[i], sigmas[i], weights[i] = v, np.sqrt(var), w
        groups[i] = groups[i] + groups[j]
        del centers[j], sigmas[j], weights[j], groups[j]
    order = np.argsort(centers)
    return (
        np.asarray(centers)[order],
        np.asarray(sigmas)[order],
        np.asarray(weights)[order],
        [groups[o] for o in order],
    )


def supervised_fuzzy_partition(
    X,
    y,
    n_clusters: int = 3,
    shape: str = "trapezoid",
    merge_threshold: float = 0.8,
    m: float = 2.0,
    random_state=None,
    **clusterer_params,
):
    r"""Per-feature fuzzy partitions derived from supervised fuzzy clustering.

    :class:`~fuzzyclust.models.SupervisedFuzzyClassifier` is run with
    ``n_clusters`` rules; the Gaussian antecedent sets of the rules,
    projected on every feature (center :math:`v_{ij}`, width
    :math:`\sigma_{ij}`), form an initial partition of :math:`R` sets per
    feature. Sets whose similarity exceeds ``merge_threshold`` are merged
    (weighted by the rule priors) so that every feature ends with as many
    sets as the class structure requires; a feature left with a single set is
    a "don't care" variable [1]_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Class labels.
    n_clusters : int, default=3
        Number of rules = initial number of sets per feature.
    shape : {"gaussian", "trapezoid", "triangle"}, default="trapezoid"
        Shape used to evaluate the sets (see :func:`membership_1d`).
    merge_threshold : float, default=0.8
        Similarity threshold :math:`\theta` of the set merging; ``1.0`` disables merging.
    m : float, default=2.0
    random_state : int, RandomState instance or None, default=None
    **clusterer_params
        Passed to :class:`~fuzzyclust.models.SupervisedFuzzyClassifier`.

    Returns
    -------
    partitions : list of dict, one per feature
        Keys ``"centers"``, ``"sigmas"``, ``"weights"`` (arrays sorted by
        center), ``"groups"`` (which rules were merged into each set),
        ``"range"`` (min, max of the feature) and ``"shape"``.
    classifier : fitted :class:`~fuzzyclust.models.SupervisedFuzzyClassifier`

    References
    ----------
    .. [1] J. Abonyi, "Supervised fuzzy clustering based initialization of
           fuzzy partitions for decision tree induction", in: Soft Computing
           in Industrial Applications, AISC 75, Springer, 31–39, 2010.

    Examples
    --------
    >>> from fuzzyclust.datasets import load_iris
    >>> from fuzzyclust.models import supervised_fuzzy_partition
    >>> iris = load_iris()
    >>> parts, clf = supervised_fuzzy_partition(iris.data, iris.target, n_clusters=3, random_state=0)
    >>> len(parts), [len(p["centers"]) <= 3 for p in parts]
    (4, [True, True, True, True])
    """
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {_SHAPES}.")
    if not 0.0 <= merge_threshold <= 1.0:
        raise ValueError("merge_threshold must lie in [0, 1].")
    clf = SupervisedFuzzyClassifier(n_clusters=n_clusters, m=m, random_state=random_state, **clusterer_params)
    clf.fit(X, y)
    Xa = np.asarray(X, dtype=float)
    parts = []
    for j in range(Xa.shape[1]):
        lo, hi = Xa[:, j].min(), Xa[:, j].max()
        centers, sigmas, weights, groups = _merge_feature(
            clf.cluster_centers_[:, j],
            np.sqrt(clf.input_variances_[:, j]),
            clf.priors_,
            lo,
            hi,
            shape,
            merge_threshold,
        )
        parts.append(
            {
                "centers": centers,
                "sigmas": sigmas,
                "weights": weights,
                "groups": groups,
                "range": (lo, hi),
                "shape": shape,
            }
        )
    return parts, clf


def ruspini_partition(X, n_sets: int = 3):
    r"""Ruspini-type (equidistant triangular) partition of every feature, the unsupervised baseline.

    The :math:`n_\text{sets}` triangular sets have their peaks equally spaced
    between the minimum and the maximum of the feature and their feet at the
    neighboring peaks, so the memberships sum to one everywhere (a Ruspini
    partition). Nothing about the class labels or the distribution of the data
    enters, which is why Abonyi (2010) compares the supervised partition
    against it. The sets are returned in the same ``dict`` structure as
    :func:`supervised_fuzzy_partition` (``sigmas`` equal one third of the
    spacing, so that the triangle of :func:`membership_1d` has its feet at the
    neighboring centers).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_sets : int, default=3

    Returns
    -------
    partitions : list of dict

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.models import ruspini_partition, membership_1d
    >>> parts = ruspini_partition(np.linspace(0, 1, 11)[:, None], n_sets=3)
    >>> parts[0]["centers"]
    array([0. , 0.5, 1. ])
    >>> A = membership_1d([0.25], parts[0]["centers"], parts[0]["sigmas"], "triangle")
    >>> A.round(3)
    array([[0.5, 0.5, 0. ]])
    """
    if n_sets < 2:
        raise ValueError("n_sets must be at least 2.")
    Xa = np.asarray(X, dtype=float)
    parts = []
    for j in range(Xa.shape[1]):
        lo, hi = Xa[:, j].min(), Xa[:, j].max()
        centers = np.linspace(lo, hi, n_sets)
        spacing = max((hi - lo) / (n_sets - 1), EPS)
        parts.append(
            {
                "centers": centers,
                "sigmas": np.full(n_sets, spacing / 3.0),
                "weights": np.full(n_sets, 1.0 / n_sets),
                "groups": [[s] for s in range(n_sets)],
                "range": (lo, hi),
                "shape": "triangle",
            }
        )
    return parts


class FuzzyPartitionDiscretizer(TransformerMixin, BaseEstimator):
    r"""Supervised fuzzy discretization of continuous features.

    A transformer that replaces every continuous feature by the fuzzy sets of
    a partition obtained with :func:`supervised_fuzzy_partition`. The output
    is either the membership of the sample in every set (``encode="membership"``)
    or the index of the set with the highest membership
    (``encode="crisp"``, one integer per feature, sets ordered from small to
    large values), which is the a-priori discretization that ID3-type
    decision-tree learners such as :class:`sklearn.tree.DecisionTreeClassifier`
    can be trained on. Because the partition is derived from a classifier,
    the cut points are placed where the classes change, in contrast to
    equal-width or equal-frequency binning [1]_.

    Parameters
    ----------
    n_sets : int, default=3
        Number of rules of the supervised clustering = initial number of
        sets per feature (merging may reduce it).
    shape : {"gaussian", "trapezoid", "triangle"}, default="trapezoid"
    merge_threshold : float, default=0.8
        Similarity threshold of set merging (``1.0``: no merging).
    encode : {"crisp", "membership"}, default="crisp"
        ``"crisp"`` returns one float column per feature holding the index of
        the set with the highest membership; ``"membership"`` returns the
        memberships in every set.
    method : {"supervised", "ruspini"}, default="supervised"
        ``"supervised"`` derives the sets from supervised fuzzy clustering
        (:func:`supervised_fuzzy_partition`); ``"ruspini"`` is the
        unsupervised, equidistant triangular baseline of
        :func:`ruspini_partition` (``shape``, ``merge_threshold`` and the
        clustering parameters are then ignored).
    drop_single : bool, default=False
        Drop features whose partition collapsed to a single set ("don't care"
        variables).
    m : float, default=2.0
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="kmeans++"
        Initialization of the supervised clustering.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    partitions_ : list of dict
        One partition per input feature (see :func:`supervised_fuzzy_partition`).
    classifier_ : fitted :class:`~fuzzyclust.models.SupervisedFuzzyClassifier` or None
        The supervised clustering (``None`` for ``method="ruspini"``).
    n_sets_ : ndarray of shape (n_features,)
        Number of sets per feature after merging.
    active_features_ : ndarray of int
        Features kept by ``transform``.
    n_iter_ : int
        Iterations of the supervised clustering (0 for ``method="ruspini"``).

    References
    ----------
    .. [1] J. Abonyi, "Supervised fuzzy clustering based initialization of
           fuzzy partitions for decision tree induction", 2010.

    Examples
    --------
    >>> from fuzzyclust.datasets import load_iris
    >>> from fuzzyclust.models import FuzzyPartitionDiscretizer
    >>> iris = load_iris()
    >>> disc = FuzzyPartitionDiscretizer(n_sets=3, random_state=0).fit(iris.data, iris.target)
    >>> disc.transform(iris.data).shape
    (150, 4)
    >>> bool(disc.set_params(encode="membership").transform(iris.data).shape[1] == disc.n_sets_.sum())
    True
    """

    def __init__(
        self,
        n_sets: int = 3,
        shape: str = "trapezoid",
        merge_threshold: float = 0.8,
        encode: str = "crisp",
        method: str = "supervised",
        drop_single: bool = False,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="kmeans++",
        random_state=None,
    ):
        self.n_sets = n_sets
        self.shape = shape
        self.merge_threshold = merge_threshold
        self.encode = encode
        self.method = method
        self.drop_single = drop_single
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state

    def fit(self, X, y=None):
        """Derive the per-feature fuzzy partitions from the labeled data."""
        _require_y("FuzzyPartitionDiscretizer", y)
        X, y = validate_data(self, X, y, dtype=float, ensure_min_samples=2)
        if self.encode not in _ENCODINGS:
            raise ValueError(f"encode must be one of {_ENCODINGS}.")
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}.")
        if self.method == "ruspini":
            self.partitions_ = ruspini_partition(X, n_sets=self.n_sets)
            self.classifier_ = None
            self.n_iter_ = 0
        else:
            self.partitions_, self.classifier_ = supervised_fuzzy_partition(
                X,
                y,
                n_clusters=self.n_sets,
                shape=self.shape,
                merge_threshold=self.merge_threshold,
                m=self.m,
                random_state=self.random_state,
                tol=self.tol,
                max_iter=self.max_iter,
                init=self.init,
            )
            self.n_iter_ = self.classifier_.n_iter_
        self.n_sets_ = np.array([len(p["centers"]) for p in self.partitions_])
        if self.drop_single:
            self.active_features_ = np.flatnonzero(self.n_sets_ > 1)
        else:
            self.active_features_ = np.arange(X.shape[1])
        return self

    def membership(self, X):
        """Memberships in the sets of every active feature: list of arrays ``(n_samples, n_sets_j)``."""
        check_is_fitted(self, "partitions_")
        X = validate_data(self, X, dtype=float, reset=False)
        out = []
        for j in self.active_features_:
            p = self.partitions_[j]
            out.append(membership_1d(X[:, j], p["centers"], p["sigmas"], p["shape"]))
        return out

    def transform(self, X):
        """Encode the features.

        Returns ``(n_samples, n_active)`` set indices (as floats, like
        :class:`sklearn.preprocessing.KBinsDiscretizer` with ordinal encoding)
        or ``(n_samples, sum n_sets_j)`` memberships.
        """
        M = self.membership(X)
        if self.encode == "crisp":
            if not M:
                return np.empty((np.asarray(X).shape[0], 0))
            return np.column_stack([A.argmax(axis=1) for A in M]).astype(float)
        if not M:
            return np.empty((np.asarray(X).shape[0], 0))
        return np.hstack(M)

    def evaluate(self, feature: int, grid):
        """Memberships of the sets of ``feature`` on ``grid`` (for plotting), ``(len(grid), n_sets_j)``."""
        check_is_fitted(self, "partitions_")
        p = self.partitions_[feature]
        return membership_1d(grid, p["centers"], p["sigmas"], p["shape"])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "partitions_")
        n = self.n_features_in_
        if input_features is None:
            input_features = list(getattr(self, "feature_names_in_", [f"x{j}" for j in range(n)]))
        names = []
        for j in self.active_features_:
            if self.encode == "crisp":
                names.append(f"{input_features[j]}_set")
            else:
                names += [f"{input_features[j]}_set{s}" for s in range(self.n_sets_[j])]
        return np.asarray(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags


__all__ = [
    "membership_1d",
    "set_similarity",
    "supervised_fuzzy_partition",
    "ruspini_partition",
    "FuzzyPartitionDiscretizer",
]
