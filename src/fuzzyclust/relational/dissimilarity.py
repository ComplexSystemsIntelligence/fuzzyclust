# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Dissimilarity matrices for objects that are not feature vectors.

Relational clustering starts from an :math:`N\times N` matrix
:math:`\mathbf{R}=[r_{jk}]` of pairwise dissimilarities instead of an
:math:`N\times n` data matrix. The helpers of this module build such
matrices for the common cases: any metric of :mod:`scipy.spatial.distance`
(:func:`pairwise_dissimilarity`), time series compared by dynamic time
warping (:func:`dtw_distance_matrix`) or by their autocorrelation
structure (:func:`autocorrelation_distance_matrix`), and records with mixed
numeric and categorical attributes (:func:`gower_distance_matrix`). Every
function also accepts a second set of objects ``Y`` and then returns the
rectangular ``(n_samples_X, n_samples_Y)`` matrix that the relational
estimators need to assign *new* objects to the fitted clusters.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "pairwise_dissimilarity",
    "check_dissimilarity_matrix",
    "dtw_distance_matrix",
    "dtw_alignment",
    "autocorrelation_features",
    "autocorrelation_distance_matrix",
    "gower_distance_matrix",
]


# --------------------------------------------------------------------------- generic
def check_dissimilarity_matrix(R, n_objects: int | None = None, square: bool = True) -> np.ndarray:
    """Validate a (square or rectangular) dissimilarity matrix.

    Parameters
    ----------
    R : array-like of shape (n_samples, n_samples) or (n_samples, n_objects)
    n_objects : int, optional
        Required number of columns (the training objects).
    square : bool, default=True
        Require a symmetric square matrix with a zero diagonal and
        non-negative entries.

    Returns
    -------
    R : ndarray of float
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("A dissimilarity matrix must be two-dimensional.")
    if not np.all(np.isfinite(R)):
        raise ValueError("The dissimilarity matrix contains NaN or infinite entries.")
    if np.any(R < 0):
        raise ValueError("Negative values in data: dissimilarities must be non-negative.")
    if n_objects is not None and R.shape[1] != n_objects:
        raise ValueError(
            f"Expected dissimilarities to the {n_objects} training objects as columns, got {R.shape[1]}."
        )
    if square:
        if R.shape[0] != R.shape[1]:
            raise ValueError(f"A square dissimilarity matrix is required, got shape {R.shape}.")
        if not np.allclose(R, R.T, atol=1e-8 * max(1.0, np.abs(R).max())):
            raise ValueError("The dissimilarity matrix must be symmetric.")
        if np.any(np.abs(np.diag(R)) > 1e-8 * max(1.0, np.abs(R).max())):
            raise ValueError("The dissimilarity matrix must have a zero diagonal.")
    return R


def pairwise_dissimilarity(X, Y=None, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Pairwise dissimilarities of feature vectors by any SciPy metric.

    Thin wrapper around :func:`scipy.spatial.distance.pdist` /
    :func:`~scipy.spatial.distance.cdist` that always returns a dense matrix.
    ``metric="precomputed"`` validates and returns ``X`` itself, mirroring the
    convention of scikit-learn: the caller already holds the dissimilarities.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Objects, or the dissimilarity matrix when ``metric="precomputed"``.
    Y : array-like of shape (n_other, n_features), optional
        Second set of objects; the result is then ``(n_samples, n_other)``.
    metric : str, default="euclidean"
        Any metric name accepted by SciPy (``"euclidean"``, ``"sqeuclidean"``,
        ``"cityblock"``, ``"cosine"``, ``"hamming"``, ``"jaccard"``, ...), or
        ``"precomputed"``.
    **kwargs
        Passed on to SciPy (e.g. ``p=`` for the Minkowski metric).

    Returns
    -------
    R : ndarray of shape (n_samples, n_samples) or (n_samples, n_other)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.relational import pairwise_dissimilarity
    >>> X = np.array([[0.0, 0.0], [3.0, 4.0]])
    >>> pairwise_dissimilarity(X)
    array([[0., 5.],
           [5., 0.]])
    >>> pairwise_dissimilarity(X, metric="cityblock")[0, 1]
    np.float64(7.0)
    """
    if metric == "precomputed":
        if Y is not None:
            raise ValueError("Y cannot be given with metric='precomputed'.")
        return check_dissimilarity_matrix(X)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Y is None:
        if X.shape[0] == 1:
            return np.zeros((1, 1))
        return squareform(pdist(X, metric=metric, **kwargs))
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    return cdist(X, Y, metric=metric, **kwargs)


# --------------------------------------------------------------------------- dynamic time warping
def _as_series_array(X) -> np.ndarray:
    """Return series as a 3-D array ``(n_series, length, n_channels)``."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :, None]
    elif X.ndim == 2:
        X = X[:, :, None]
    elif X.ndim != 3:
        raise ValueError("Series must be given as (n_series, length) or (n_series, length, n_channels).")
    if not np.all(np.isfinite(X)):
        raise ValueError("Series contain NaN or infinite values.")
    return X


def _check_window(window, T1: int, T2: int) -> int:
    if window is None:
        return max(T1, T2)
    if isinstance(window, float) and 0.0 <= window <= 1.0 and not float(window).is_integer():
        window = int(np.ceil(window * max(T1, T2)))
    window = int(window)
    if window < 0:
        raise ValueError("window must be non-negative (or None for no constraint).")
    # the band must be wide enough to reach the opposite corner when lengths differ
    return max(window, abs(T1 - T2))


def dtw_distance_matrix(X, Y=None, window=None) -> np.ndarray:
    r"""Dynamic time warping distances between all pairs of (equal-length) series.

    Dynamic time warping aligns two series by a monotone, continuous path
    through the matrix of local costs :math:`c(t,s)=\norm{x_t-y_s}^2` so
    that the summed cost along the path is minimal (Sakoe and Chiba, 1978),

    .. math::
        \mathrm{DTW}(\mathbf{x},\mathbf{y})^2=\min_{\text{paths}}\sum_{(t,s)\in\text{path}} \norm{x_t-y_s}^2 ,

    with the symmetric step pattern :math:`(t-1,s),(t,s-1),(t-1,s-1)` and the
    Sakoe--Chiba band :math:`|t-s|\le w` as the global constraint. The square
    root of the minimal cost is returned, so that for ``window=0`` the result
    is the Euclidean distance of the two series. The recursion runs over
    all pairs simultaneously (vectorized over the pairs), so the cost is
    :math:`O(N^2\,T\,w)` with ``n_series`` :math:`N`, length :math:`T` and
    band width :math:`w`.

    Parameters
    ----------
    X : array-like of shape (n_series, length) or (n_series, length, n_channels)
        The series (multichannel series use the squared Euclidean norm of
        the channel difference as local cost).
    Y : array-like of the same kind, optional
        Second set of series (same length); the result is then rectangular.
    window : int, float in (0, 1) or None, default=None
        Half width :math:`w` of the Sakoe--Chiba band in samples, or as a
        fraction of the length; ``None`` imposes no constraint, ``0`` gives
        the Euclidean distance.

    Returns
    -------
    D : ndarray of shape (n_series, n_series) or (n_series, n_other)
        DTW distances (not squared). Note that DTW is not a metric: the
        triangle inequality may fail, so the squared matrix is in general
        *not* Euclidean and relational c-means needs the
        :math:`\beta`-spread (:class:`~fuzzyclust.relational.NERFCM`).

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition", IEEE Transactions on Acoustics,
           Speech, and Signal Processing 26(1), 43--49, 1978.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.relational import dtw_distance_matrix
    >>> t = np.linspace(0, 1, 30)
    >>> X = np.vstack([np.sin(2 * np.pi * t), np.sin(2 * np.pi * (t - 0.1)), np.cos(2 * np.pi * t)])
    >>> D = dtw_distance_matrix(X, window=5)
    >>> bool(D[0, 1] < D[0, 2]), D.shape
    (True, (3, 3))
    """
    A = _as_series_array(X)
    B = A if Y is None else _as_series_array(Y)
    if A.shape[2] != B.shape[2]:
        raise ValueError("X and Y must have the same number of channels.")
    N1, T1, _ = A.shape
    N2, T2, _ = B.shape
    w = _check_window(window, T1, T2)
    inf = np.inf
    # accumulated cost of the previous and the current row, for all pairs at once
    prev = np.full((N1, N2, T2 + 1), inf)
    prev[:, :, 0] = 0.0
    cur = np.empty_like(prev)
    for t in range(T1):
        cur.fill(inf)
        lo, hi = max(0, t - w), min(T2 - 1, t + w)
        # local costs of row t for every pair and every s in the band: (N1, N2, hi-lo+1)
        diff = A[:, None, t, None, :] - B[None, :, lo : hi + 1, :]
        cost = np.einsum("abse,abse->abs", diff, diff)
        for s in range(lo, hi + 1):
            best = np.minimum(np.minimum(prev[:, :, s + 1], prev[:, :, s]), cur[:, :, s])
            cur[:, :, s + 1] = cost[:, :, s - lo] + best
        prev, cur = cur, prev
    D = np.sqrt(prev[:, :, T2])
    if Y is None:
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
    return D


def dtw_alignment(x, y, window=None):
    r"""Optimal warping path between two series (for inspection and plotting).

    Parameters
    ----------
    x, y : array-like of shape (length,) or (length, n_channels)
    window : int, float or None, default=None
        Sakoe--Chiba band, see :func:`dtw_distance_matrix`.

    Returns
    -------
    distance : float
        The DTW distance (square root of the minimal accumulated cost).
    path : ndarray of shape (n_steps, 2)
        Index pairs ``(t, s)`` of the optimal path from ``(0, 0)`` to
        ``(len(x)-1, len(y)-1)``.
    """
    a = _as_series_array(x)[0]
    b = _as_series_array(y)[0]
    T1, T2 = a.shape[0], b.shape[0]
    w = _check_window(window, T1, T2)
    acc = np.full((T1 + 1, T2 + 1), np.inf)
    acc[0, 0] = 0.0
    for t in range(1, T1 + 1):
        lo, hi = max(1, t - w), min(T2, t + w)
        diff = a[t - 1][None, :] - b[lo - 1 : hi]
        cost = np.einsum("se,se->s", diff, diff)
        for s in range(lo, hi + 1):
            acc[t, s] = cost[s - lo] + min(acc[t - 1, s], acc[t, s - 1], acc[t - 1, s - 1])
    t, s = T1, T2
    path = [(t - 1, s - 1)]
    while t > 1 or s > 1:
        candidates = ((acc[t - 1, s - 1], t - 1, s - 1), (acc[t - 1, s], t - 1, s), (acc[t, s - 1], t, s - 1))
        _, t, s = min(candidates, key=lambda c: c[0])
        path.append((t - 1, s - 1))
    return float(np.sqrt(acc[T1, T2])), np.asarray(path[::-1], dtype=int)


# --------------------------------------------------------------------------- autocorrelation
def autocorrelation_features(X, n_lags: int = 10) -> np.ndarray:
    r"""Sample autocorrelations :math:`\hat\rho(1),\dots,\hat\rho(L)` of every series.

    .. math::
        \hat\rho_k(l)=\frac{\sum_{t=l+1}^{T}(x_{k,t}-\bar x_k)(x_{k,t-l}-\bar x_k)}
                           {\sum_{t=1}^{T}(x_{k,t}-\bar x_k)^2}.

    Parameters
    ----------
    X : array-like of shape (n_series, length)
    n_lags : int, default=10
        Number of lags :math:`L` (must be smaller than the length).

    Returns
    -------
    A : ndarray of shape (n_series, n_lags)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("Univariate series of shape (n_series, length) are required.")
    T = X.shape[1]
    if not 1 <= int(n_lags) < T:
        raise ValueError("n_lags must satisfy 1 <= n_lags < length.")
    Z = X - X.mean(axis=1, keepdims=True)
    denom = np.einsum("kt,kt->k", Z, Z)
    denom = np.where(denom <= 0, 1.0, denom)
    A = np.empty((X.shape[0], int(n_lags)))
    for l in range(1, int(n_lags) + 1):
        A[:, l - 1] = np.einsum("kt,kt->k", Z[:, l:], Z[:, :-l]) / denom
    return A


def autocorrelation_distance_matrix(X, Y=None, n_lags: int = 10, weights=None) -> np.ndarray:
    r"""Autocorrelation-based distance between time series (D'Urso and Maharaj, 2009).

    Two series are considered similar when their *dependence structure* is
    similar, whatever their level or phase: the series are represented by
    their estimated autocorrelation functions up to lag :math:`L` and
    compared by the (optionally weighted) Euclidean distance

    .. math::
        d_{\mathrm{ACF}}(\mathbf{x}_j,\mathbf{x}_k)=
        \Big(\sum_{l=1}^{L} w_l\,\big(\hat\rho_j(l)-\hat\rho_k(l)\big)^2\Big)^{1/2}.

    Because the representation is a feature vector, the squared matrix is
    Euclidean and relational c-means on it coincides with fuzzy c-means on
    the autocorrelation features.

    Parameters
    ----------
    X : array-like of shape (n_series, length)
    Y : array-like of shape (n_other, length), optional
    n_lags : int, default=10
        Number of lags :math:`L`.
    weights : array-like of shape (n_lags,), optional
        Lag weights :math:`w_l` (uniform by default; geometrically decaying
        weights emphasize the first lags).

    Returns
    -------
    D : ndarray of shape (n_series, n_series) or (n_series, n_other)

    References
    ----------
    .. [1] P. D'Urso, E. A. Maharaj, "Autocorrelation-based fuzzy clustering
           of time series", Fuzzy Sets and Systems 160(24), 3565--3589, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.relational import autocorrelation_distance_matrix
    >>> rng = np.random.RandomState(0)
    >>> e = rng.randn(3, 200)
    >>> smooth = np.cumsum(e[:2], axis=1)  # strongly autocorrelated
    >>> X = np.vstack([smooth, e[2:]])      # plus a white-noise series
    >>> D = autocorrelation_distance_matrix(X, n_lags=5)
    >>> bool(D[0, 1] < D[0, 2])
    True
    """
    A = autocorrelation_features(X, n_lags=n_lags)
    B = A if Y is None else autocorrelation_features(Y, n_lags=n_lags)
    if weights is None:
        w = np.ones(A.shape[1])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != (A.shape[1],) or np.any(w < 0):
            raise ValueError("weights must be n_lags non-negative numbers.")
    D = cdist(A * np.sqrt(w), B * np.sqrt(w))
    if Y is None:
        np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------- Gower
def _split_mixed(X, categorical_features):
    """Return (object array, boolean mask of categorical columns, column names or None)."""
    columns = None
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None
    if pd is not None and isinstance(X, pd.DataFrame):
        columns = list(X.columns)
        if categorical_features is None:
            is_cat = np.array(
                [not pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt) for dt in X.dtypes]
            )
        else:
            is_cat = _mask_from_spec(categorical_features, X.shape[1], columns)
        arr = X.to_numpy(dtype=object)
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("Data must be two-dimensional.")
        if categorical_features is None:
            is_cat = np.zeros(arr.shape[1], dtype=bool) if arr.dtype.kind in "fiub" else np.ones(
                arr.shape[1], dtype=bool
            )
        else:
            is_cat = _mask_from_spec(categorical_features, arr.shape[1], None)
        arr = arr.astype(object)
    return arr, is_cat, columns


def _mask_from_spec(spec, n_features, columns):
    if isinstance(spec, str):
        if spec == "all":
            return np.ones(n_features, dtype=bool)
        raise ValueError("categorical_features must be None, 'all', a boolean mask or a list of columns.")
    spec = np.asarray(spec)
    if spec.dtype == bool:
        if spec.shape != (n_features,):
            raise ValueError("A boolean categorical_features mask must have one entry per column.")
        return spec
    mask = np.zeros(n_features, dtype=bool)
    for item in spec.ravel():
        if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            mask[int(item)] = True
        elif columns is not None and item in columns:
            mask[columns.index(item)] = True
        else:
            raise ValueError(f"Unknown column {item!r} in categorical_features.")
    return mask


def _is_missing(col):
    out = np.zeros(col.shape[0], dtype=bool)
    for k, v in enumerate(col):
        if v is None:
            out[k] = True
        elif isinstance(v, float) and np.isnan(v):
            out[k] = True
    return out


def gower_distance_matrix(X, Y=None, categorical_features=None, ranges=None, feature_weights=None):
    r"""Gower's general coefficient of dissimilarity for mixed data (Gower, 1971).

    Every attribute contributes a score in :math:`[0,1]`: for a numeric
    attribute the absolute difference scaled by the attribute's range
    :math:`R_j`, for a categorical attribute :math:`0` when the two records
    agree and :math:`1` otherwise. The dissimilarity is the weighted mean
    of the scores over the attributes for which both values are present,

    .. math::
        d_G(\mathbf{x}_k,\mathbf{x}_l)=
        \frac{\sum_j w_j\,\delta_{klj}\,s_{klj}}{\sum_j w_j\,\delta_{klj}},\qquad
        s_{klj}=\begin{cases}|x_{kj}-x_{lj}|/R_j & \text{numeric}\\
        [x_{kj}\neq x_{lj}] & \text{categorical,}\end{cases}

    with :math:`\delta_{klj}=0` when either value is missing (``NaN`` or
    ``None``). Gower showed that :math:`\sqrt{d_G}` is Euclidean-embeddable
    when there are no missing values, so :math:`\mathbf{R}=[d_G]` (the
    *squared* embedded distance) is a valid input of relational c-means.

    Parameters
    ----------
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
    Y : array-like of shape (n_other, n_features), optional
    categorical_features : None, "all", boolean mask or list of columns, default=None
        Which columns are categorical. ``None`` detects them from the
        dtypes of a DataFrame (non-numeric and boolean columns); for a
        numeric array all columns are numeric, for an object array all are
        categorical.
    ranges : array-like of shape (n_numeric,), optional
        Ranges :math:`R_j` of the numeric columns (default: observed range
        of ``X``, which must be reused for new data to keep the scale).
    feature_weights : array-like of shape (n_features,), optional
        Attribute weights :math:`w_j` (uniform by default).

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples) or (n_samples, n_other)
        Gower dissimilarities in :math:`[0, 1]`.

    References
    ----------
    .. [1] J. C. Gower, "A general coefficient of similarity and some of its
           properties", Biometrics 27(4), 857--871, 1971.

    Examples
    --------
    >>> import pandas as pd
    >>> from fuzzyclust.relational import gower_distance_matrix
    >>> df = pd.DataFrame({"age": [20.0, 40.0, 60.0], "job": ["a", "a", "b"]})
    >>> gower_distance_matrix(df).round(3)
    array([[0.  , 0.25, 1.  ],
           [0.25, 0.  , 0.75],
           [1.  , 0.75, 0.  ]])
    """
    A, is_cat, _ = _split_mixed(X, categorical_features)
    if Y is None:
        B = A
    else:
        B, _, _ = _split_mixed(Y, categorical_features if categorical_features is not None else is_cat)
        if B.shape[1] != A.shape[1]:
            raise ValueError("X and Y must have the same number of columns.")
    n = A.shape[1]
    if feature_weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(feature_weights, dtype=float).ravel()
        if w.shape != (n,) or np.any(w < 0):
            raise ValueError("feature_weights must hold one non-negative weight per column.")
    num_cols = np.flatnonzero(~is_cat)
    if ranges is None:
        rng_ = np.empty(len(num_cols))
        for p, j in enumerate(num_cols):
            col = A[:, j].astype(float)
            col = col[np.isfinite(col)]
            rng_[p] = (col.max() - col.min()) if col.size else 0.0
    else:
        rng_ = np.asarray(ranges, dtype=float).ravel()
        if rng_.shape != (len(num_cols),):
            raise ValueError("ranges must hold one value per numeric column.")
    rng_ = np.where(rng_ <= 0, 1.0, rng_)
    num = np.zeros((A.shape[0], B.shape[0]))
    den = np.zeros_like(num)
    p = 0
    for j in range(n):
        if is_cat[j]:
            a, b = A[:, j], B[:, j]
            miss = _is_missing(a)[:, None] | _is_missing(b)[None, :]
            score = (a[:, None] != b[None, :]).astype(float)
        else:
            a, b = A[:, j].astype(float), B[:, j].astype(float)
            miss = ~np.isfinite(a)[:, None] | ~np.isfinite(b)[None, :]
            score = np.abs(np.nan_to_num(a)[:, None] - np.nan_to_num(b)[None, :]) / rng_[p]
            np.minimum(score, 1.0, out=score)
            p += 1
        valid = (~miss).astype(float) * w[j]
        num += valid * score
        den += valid
    D = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    if Y is None:
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
    return D
