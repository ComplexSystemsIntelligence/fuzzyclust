# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy k-modes and fuzzy k-prototypes for categorical and mixed data.

Huang's k-modes (1998) replaces the mean by the *mode* and the Euclidean
distance by the simple matching dissimilarity so that k-means works on
categorical attributes; k-prototypes combines both for mixed records.
Huang and Ng (1999) gave the fuzzy version of k-modes with the FCM
membership formula and proved that the mode of a fuzzy cluster is the
category with the largest membership mass per attribute. Both are
implemented here on top of the alternating-optimization loop of
:class:`~fuzzyclust.base.BaseFuzzyClusterer`; the categorical columns are
encoded as integer codes internally and decoded for the fitted modes.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import fuzzy_centers, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer
from .dissimilarity import _mask_from_spec

__all__ = ["FuzzyKPrototypes", "FuzzyKModes", "matching_dissimilarity"]


def matching_dissimilarity(A, B) -> np.ndarray:
    r"""Simple matching dissimilarity :math:`\delta(\mathbf{a},\mathbf{b})=\sum_j[a_j\neq b_j]` between rows.

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
    B : array-like of shape (n_other, n_features)

    Returns
    -------
    D : ndarray of shape (n_samples, n_other)
        Number of attributes in which the two records differ.

    Examples
    --------
    >>> from fuzzyclust.relational import matching_dissimilarity
    >>> matching_dissimilarity([["a", "x"], ["b", "x"]], [["a", "x"], ["b", "y"]])
    array([[0., 2.],
           [1., 1.]])
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of attributes.")
    D = np.zeros((A.shape[0], B.shape[0]))
    for j in range(A.shape[1]):
        D += A[:, j][:, None] != B[:, j][None, :]
    return D


def _is_categorical_dtype(dt) -> bool:
    import pandas as pd

    return not pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)


class FuzzyKPrototypes(BaseFuzzyClusterer):
    r"""Fuzzy k-prototypes clustering of mixed numeric and categorical data (Huang, 1998).

    Records with numeric *and* categorical attributes cannot be clustered
    by fuzzy c-means without encoding the categories as numbers, which
    imposes an arbitrary geometry on them. The k-prototypes dissimilarity
    of Huang treats the two kinds of attributes in their own terms -- the
    squared Euclidean distance for the numeric part and the number of
    mismatching categories for the categorical part -- and balances them
    with a weight :math:`\gamma`; the prototype of a cluster is the fuzzy
    mean of the numeric attributes together with the *mode* (most frequent
    category, weighted by the memberships) of every categorical attribute.
    This is the fuzzy version of the algorithm: memberships follow the FCM
    formula with the exponent :math:`m` (Huang and Ng, 1999), and the
    category of a mode is the one with the largest membership mass.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    gamma : float or None, default=None
        Weight :math:`\gamma` of the categorical mismatches against the
        squared numeric distances. ``None`` uses the average standard
        deviation of the (training) numeric columns, the guidance of
        Huang's paper; larger values let the categorical attributes
        dominate the partition, ``0`` ignores them.
    categorical_features : None, "all", boolean mask or list of columns, default=None
        Which columns are categorical. ``None`` detects them from the
        dtypes of a :class:`pandas.DataFrame` (object, string, category
        and boolean columns) and treats every column of a numeric array as
        numeric; column names (DataFrame) or integer positions are accepted.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        ``"random"`` picks :math:`c` distinct records as initial
        prototypes (Huang's start); ``"kmeans++"`` seeds them by
        :math:`D^2`-weighted sampling with the k-prototypes dissimilarity;
        an integer array of length ``n_clusters`` gives the initial
        records, an array of shape ``(n_samples, n_clusters)`` an initial
        partition.
    n_init : int, default=1
        Random restarts; the run with the lowest objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes in the *encoded* representation: numeric columns hold
        the fuzzy means, categorical columns the integer code of the mode.
    prototypes_ : pandas.DataFrame or ndarray of shape (n_clusters, n_features)
        The prototypes with the original category labels (a DataFrame with
        the input's column names when a DataFrame was fitted).
    modes_ : pandas.DataFrame or ndarray of shape (n_clusters, n_categorical)
        The categorical part of the prototypes with original labels.
    numeric_centers_ : ndarray of shape (n_clusters, n_numeric)
    categorical_features_ : ndarray of int
        Positions of the categorical columns; ``numeric_features_`` the rest.
    gamma_ : float
        The weight actually used.
    categories_ : list of ndarray
        Categories of every categorical column (from the encoder).
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_``
        is the square root of the k-prototypes dissimilarity.

    Notes
    -----
    With the numeric attributes :math:`j\in\mathcal{N}` and the categorical
    ones :math:`j\in\mathcal{C}` the dissimilarity of record
    :math:`\mathbf{x}_k` to prototype :math:`\mathbf{v}_i` is

    .. math::
        d(\mathbf{x}_k,\mathbf{v}_i)=\sum_{j\in\mathcal{N}}(x_{kj}-v_{ij})^2
        +\gamma\sum_{j\in\mathcal{C}}\delta(x_{kj},v_{ij}),\qquad
        \delta(a,b)=\begin{cases}0&a=b\\1&a\neq b,\end{cases}

    and the objective :math:`J=\sum_i\sum_k\mu_{ik}^m\,d(\mathbf{x}_k,\mathbf{v}_i)`.
    For fixed prototypes the optimal memberships are
    :math:`\mu_{ik}=1/\sum_j\big(d(\mathbf{x}_k,\mathbf{v}_i)/d(\mathbf{x}_k,\mathbf{v}_j)\big)^{1/(m-1)}`
    (with :math:`\mu_{ik}=1` when :math:`d=0`); for fixed memberships the
    numeric part of the prototype is the fuzzy mean
    :math:`v_{ij}=\sum_k\mu_{ik}^m x_{kj}/\sum_k\mu_{ik}^m` and, by the
    theorem of Huang and Ng, the categorical part is the category with the
    largest membership mass,

    .. math::
        v_{ij}=a^{(r)}_j\quad\text{with}\quad
        \sum_{k:\,x_{kj}=a^{(r)}_j}\mu_{ik}^m\;\ge\;\sum_{k:\,x_{kj}=a^{(t)}_j}\mu_{ik}^m
        \quad\forall t .

    Unknown categories of new records (``predict_proba``) never match a
    mode and contribute one mismatch.

    References
    ----------
    .. [1] Z. Huang, "Extensions to the k-means algorithm for clustering
           large data sets with categorical values", Data Mining and
           Knowledge Discovery 2(3), 283--304, 1998.
    .. [2] Z. Huang, M. K. Ng, "A fuzzy k-modes algorithm for clustering
           categorical data", IEEE Transactions on Fuzzy Systems 7(4),
           446--452, 1999.

    Examples
    --------
    >>> from fuzzyclust.relational import FuzzyKPrototypes, make_mixed_blobs
    >>> data = make_mixed_blobs(random_state=0)
    >>> kp = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(data.data)
    >>> kp.prototypes_.shape, len(kp.categorical_features_)
    ((3, 6), 3)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        gamma: float | None = None,
        categorical_features=None,
        m: float = 2.0,
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
        self.gamma = gamma
        self.categorical_features = categorical_features

    # ------------------------------------------------------------------ encoding
    def _categorical_spec(self):
        return self.categorical_features

    def _validate_and_encode(self, X, reset: bool):
        """Validate input, store feature names, return the encoded float matrix."""
        import pandas as pd

        is_frame = isinstance(X, pd.DataFrame)
        dtypes = list(X.dtypes) if is_frame else None
        columns = list(X.columns) if is_frame else None
        X_arr = validate_data(
            self, X, dtype=None, ensure_all_finite=False, reset=reset, ensure_min_samples=2 if reset else 1
        )
        n = X_arr.shape[1]
        if reset:
            spec = self._categorical_spec()
            if spec is None:
                if is_frame:
                    is_cat = np.array([_is_categorical_dtype(dt) for dt in dtypes])
                elif X_arr.dtype.kind in "fiu":
                    is_cat = np.zeros(n, dtype=bool)
                else:
                    is_cat = np.ones(n, dtype=bool)
            else:
                is_cat = _mask_from_spec(spec, n, columns)
            self.categorical_features_ = np.flatnonzero(is_cat)
            self.numeric_features_ = np.flatnonzero(~is_cat)
            self._columns = columns
            if len(self.categorical_features_):
                self.encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
                )
                self.encoder_.fit(self._object_block(X_arr, self.categorical_features_))
                self.categories_ = list(self.encoder_.categories_)
            else:
                self.encoder_ = None
                self.categories_ = []
        Z = np.empty(X_arr.shape, dtype=float)
        if len(self.numeric_features_):
            try:
                num = X_arr[:, self.numeric_features_].astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Numeric columns must be convertible to float; declare categorical columns with "
                    "categorical_features."
                ) from exc
            if not np.all(np.isfinite(num)):
                raise ValueError("Numeric columns contain NaN or infinite values.")
            Z[:, self.numeric_features_] = num
        if len(self.categorical_features_):
            Z[:, self.categorical_features_] = self.encoder_.transform(
                self._object_block(X_arr, self.categorical_features_)
            )
        return Z

    @staticmethod
    def _object_block(X_arr, cols):
        block = X_arr[:, cols]
        if block.dtype.kind in "fiu":
            block = block.astype(float)
            if not np.all(np.isfinite(block)):
                raise ValueError("Categorical columns contain NaN or infinite values.")
            return block.astype(object)
        block = block.astype(object)
        for v in block.ravel():
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                raise ValueError("Categorical columns contain missing (None/NaN) or infinite values.")
        return block

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("Fuzzy k-prototypes requires m > 1.")
        if self.gamma is not None and self.gamma < 0:
            raise ValueError("gamma must be non-negative.")
        num = self.numeric_features_
        if self.gamma is None:
            self.gamma_ = float(X[:, num].std(axis=0).mean()) if len(num) else 1.0
        else:
            self.gamma_ = float(self.gamma)

    # ------------------------------------------------------------------ dissimilarity
    def _dissimilarity(self, Z, V):
        """k-prototypes dissimilarity of encoded records Z to encoded prototypes V, shape (N, c)."""
        num, cat = self.numeric_features_, self.categorical_features_
        d = np.zeros((Z.shape[0], V.shape[0]))
        if len(num):
            d += sq_euclidean(Z[:, num], V[:, num])
        if len(cat):
            d += self.gamma_ * matching_dissimilarity(Z[:, cat], V[:, cat])
        return d

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        c, N = self.n_clusters, X.shape[0]
        init = self.init
        if isinstance(init, str):
            if init == "random":
                idx = rng.choice(N, size=c, replace=False)
            elif init == "kmeans++":
                idx = [int(rng.randint(N))]
                dmin = self._dissimilarity(X, X[idx])[:, 0]
                while len(idx) < c:
                    dmin[idx] = 0.0
                    probs = dmin / dmin.sum() if dmin.sum() > 0 else np.full(N, 1.0 / N)
                    j = int(rng.choice(N, p=probs))
                    idx.append(j)
                    dmin = np.minimum(dmin, self._dissimilarity(X, X[[j]])[:, 0])
                idx = np.asarray(idx)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
            V = X[idx].copy()
        else:
            init = np.asarray(init)
            if init.shape == (N, c):
                U = init.astype(float)
                return U / U.sum(axis=1, keepdims=True)
            if init.shape == (c,) and np.issubdtype(init.dtype, np.integer):
                if init.min() < 0 or init.max() >= N:
                    raise ValueError("initial record indices out of range.")
                V = X[init].copy()
            elif init.shape == (c, X.shape[1]):
                V = init.astype(float)
            else:
                raise ValueError(
                    "init array must be a partition (n_samples, n_clusters), n_clusters record indices "
                    "or encoded prototypes (n_clusters, n_features)."
                )
        self.cluster_centers_ = V
        return self._memberships(self._dissimilarity(X, V))

    def _update_prototypes(self, X, Um):
        c = self.n_clusters
        V = np.empty((c, X.shape[1]))
        num, cat = self.numeric_features_, self.categorical_features_
        if len(num):
            V[:, num] = fuzzy_centers(X[:, num], Um)
        for j in cat:
            codes = X[:, j].astype(int)
            n_cat = max(int(codes.max()) + 1, 1)
            # membership mass of every category in every cluster: (n_cat, c)
            mass = np.zeros((n_cat, c))
            np.add.at(mass, codes, Um)
            V[:, j] = mass.argmax(axis=0)
        self.cluster_centers_ = V

    def _distances(self, X):
        return self._dissimilarity(X, self.cluster_centers_)

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Cluster mixed-type records.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Records; categorical columns may hold strings, categories,
            booleans or numbers (see ``categorical_features``).
        y : Ignored

        Returns
        -------
        self : object
        """
        Z = self._validate_and_encode(X, reset=True)
        self._check_params(Z)
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        rng = check_random_state(self.random_state)
        n_init = 1 if not isinstance(self.init, str) else self.n_init
        best = None
        for run in range(n_init):
            result = self._fit_single(Z, rng)
            if best is None or result["history"][-1] < best["history"][-1]:
                best = result
            if self.verbose and n_init > 1:
                print(f"restart {run + 1}/{n_init}: J = {result['history'][-1]:.6g}")
        self.__dict__.update(best["state"])
        self.membership_ = best["U"]
        self.labels_ = best["U"].argmax(axis=1)
        self.distances_ = np.sqrt(best["d2"])
        self.objective_history_ = np.asarray(best["history"])
        self.objective_ = best["history"][-1]
        self.n_iter_ = best["n_iter"]
        self._decode_prototypes()
        return self

    def _decode_prototypes(self):
        import pandas as pd

        V = self.cluster_centers_
        num, cat = self.numeric_features_, self.categorical_features_
        self.numeric_centers_ = V[:, num].copy()
        proto = np.empty(V.shape, dtype=object)
        proto[:, num] = V[:, num]
        if len(cat):
            modes = self.encoder_.inverse_transform(V[:, cat].astype(int))
            proto[:, cat] = modes
        else:
            modes = np.empty((V.shape[0], 0), dtype=object)
        if self._columns is not None:
            self.prototypes_ = pd.DataFrame(proto, columns=self._columns)
            self.modes_ = pd.DataFrame(modes, columns=[self._columns[j] for j in cat])
        else:
            self.prototypes_ = proto
            self.modes_ = modes

    def predict_proba(self, X):
        """Memberships of (new) records in the fitted clusters.

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        Z = self._validate_and_encode(X, reset=False)
        return self._memberships(self._distances(Z))

    def transform(self, X):
        """Square root of the k-prototypes dissimilarity of records to the prototypes."""
        check_is_fitted(self, "cluster_centers_")
        Z = self._validate_and_encode(X, reset=False)
        return np.sqrt(self._distances(Z))

    def score(self, X, y=None):
        """Negative objective on ``X`` (higher is better)."""
        check_is_fitted(self, "cluster_centers_")
        Z = self._validate_and_encode(X, reset=False)
        d2 = self._distances(Z)
        U = self._memberships(d2)
        return -self._objective(U**self.m, d2)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags


class FuzzyKModes(FuzzyKPrototypes):
    r"""Fuzzy k-modes clustering of categorical data (Huang and Ng, 1999).

    Every attribute is categorical, the dissimilarity between two records
    is the number of attributes in which they differ (simple matching),
    and the prototype of a cluster is its *mode*: per attribute the
    category that carries the largest membership mass. The algorithm is
    :class:`FuzzyKPrototypes` with all columns categorical and
    :math:`\gamma=1`, i.e. the fuzzy c-means functional with the matching
    dissimilarity

    .. math::
        J=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\sum_{j=1}^{n}\delta(x_{kj},v_{ij}),

    minimized by alternating the FCM membership formula (with the
    dissimilarity in the role of :math:`D^2_{ik}`; :math:`\mu_{ik}=1` when
    the record *is* the mode) and the mode update of Huang and Ng's
    theorem. Because the dissimilarity is integer-valued, many records are
    equally far from several modes and the partition is fuzzier than that
    of FCM on numeric data; the exponent :math:`m` should be kept small
    (Huang and Ng use :math:`m\approx1.1`--$2$), and ties in the mode
    update are resolved by the first category in the encoder's order.

    Parameters
    ----------
    n_clusters : int, default=3
    m : float, default=1.5
        Fuzziness exponent :math:`m>1`.
    tol, max_iter, init, n_init, random_state, verbose
        See :class:`FuzzyKPrototypes`.

    Attributes
    ----------
    modes_ : pandas.DataFrame or ndarray of shape (n_clusters, n_features)
        The modes with the original category labels (same as ``prototypes_``).
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        The modes as integer codes (stored as floats).
    categories_, membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`FuzzyKPrototypes`.

    References
    ----------
    .. [1] Z. Huang, M. K. Ng, "A fuzzy k-modes algorithm for clustering
           categorical data", IEEE Transactions on Fuzzy Systems 7(4),
           446--452, 1999.
    .. [2] Z. Huang, "Extensions to the k-means algorithm for clustering
           large data sets with categorical values", Data Mining and
           Knowledge Discovery 2(3), 283--304, 1998.

    Examples
    --------
    >>> from fuzzyclust.relational import FuzzyKModes, make_categorical_blobs
    >>> data = make_categorical_blobs(random_state=0)
    >>> km = FuzzyKModes(n_clusters=3, random_state=0).fit(data.data)
    >>> km.modes_.shape, km.membership_.shape
    ((3, 6), (300, 3))
    """

    # the two k-prototypes parameters are fixed by the algorithm (class attributes, so that
    # the sklearn constructor contract -- no extra attributes set in __init__ -- holds)
    gamma = 1.0
    categorical_features = "all"

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 1.5,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        BaseFuzzyClusterer.__init__(
            self,
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )

    def _categorical_spec(self):
        return "all"
