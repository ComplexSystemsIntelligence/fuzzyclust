# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Single-pass fuzzy c-means (spFCM)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import sq_euclidean
from ..base import BaseFuzzyClusterer
from .weighted import WeightedFCM, extend_partition

__all__ = ["SinglePassFCM"]


def _check_chunk_size(chunk_size, c: int) -> int:
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_size < c:
        raise ValueError(f"chunk_size={chunk_size} must be at least n_clusters={c}.")
    return int(chunk_size)


def _split_chunks(N: int, chunk_size: int, shuffle: bool, rng) -> list[np.ndarray]:
    """Row indices of consecutive chunks of about ``chunk_size`` samples (randomly ordered if ``shuffle``)."""
    order = rng.permutation(N) if shuffle else np.arange(N)
    n_chunks = max(1, int(np.ceil(N / chunk_size)))
    return np.array_split(order, n_chunks)


class SinglePassFCM(BaseFuzzyClusterer):
    r"""Single-pass fuzzy c-means (spFCM) for very large and streaming data.

    The data are processed in chunks that fit into memory. The first chunk
    is clustered with FCM; the resulting :math:`c` prototypes, each
    weighted by the total membership it collected, are *carried over* as
    :math:`c` extra weighted samples into the next chunk, which is
    clustered with weighted FCM (:class:`WeightedFCM`) starting from the
    previous prototypes, and so on. The prototypes of the last chunk
    summarize the whole data set; the partition of all samples is obtained
    by one extension step. Every chunk is seen once, the working memory is
    :math:`O(\text{chunk size}\cdot n)` and :meth:`partial_fit` processes
    a stream chunk by chunk (Hore, Hall and Goldgof 2007; Havens et al.
    2012).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    chunk_size : int, default=10000
        Samples per chunk :math:`n_l` (``fit`` splits the data into
        :math:`s=\lceil N/n_l\rceil` chunks).
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
        Termination of the (weighted) FCM run on every chunk.
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization of the first chunk; later chunks start from the
        carried-over prototypes.
    n_init : int, default=1
        Random restarts of the first chunk.
    shuffle : bool, default=True
        Randomly permute the samples before chunking in ``fit`` (the
        chunks of the paper are random subsets). Set ``False`` to keep the
        order of the data, e.g. to simulate a stream; ``partial_fit``
        never reorders.
    forgetting_factor : float, default=1.0
        Factor :math:`\lambda\in[0,1]` by which the weights of the
        carried prototypes are multiplied before they enter the next chunk.
        ``1.0`` is the algorithm of the paper (all past observations count
        fully); a smaller value lets the model follow a drifting stream,
        the old data being down-weighted geometrically (an effective memory
        of about :math:`1/(1-\lambda)` chunks). ``0.0`` clusters every
        chunk on its own, started from the previous prototypes.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes after the last chunk.
    center_weights_ : ndarray of shape (n_clusters,)
        Weight :math:`w_i=\sum_k w_k\mu_{ik}` of every prototype after
        the last chunk (the number of observations it stands for).
    chunk_centers_ : ndarray of shape (n_chunks, n_clusters, n_features)
        Prototypes after every chunk (the chunk timeline).
    chunk_weights_ : ndarray of shape (n_chunks, n_clusters)
    chunk_objective_ : ndarray of shape (n_chunks,)
        Final :math:`J_w` of every chunk run.
    chunk_n_iter_ : ndarray of shape (n_chunks,)
    n_chunks_ : int
    membership_ : ndarray of shape (n_samples, n_clusters)
        Extended partition of the data passed to ``fit`` (after
        ``partial_fit``: of the last chunk).
    labels_ : ndarray of shape (n_samples,)
    distances_ : ndarray of shape (n_samples, n_clusters)
    objective_ : float
        :math:`J` of the extended partition.
    objective_history_ : ndarray of shape (n_chunks,)
        Same as ``chunk_objective_``.
    n_iter_ : int
        Total number of (weighted) FCM iterations over all chunks.

    Notes
    -----
    With chunks :math:`\mathbf{X}_1,\dots,\mathbf{X}_s` the algorithm is

    .. math::
        (\mathbf{U}_1,\mathbf{V}_1)&=\text{wFCM}(\mathbf{X}_1,\ \mathbf{w}=\mathbf{1}),\\
        w'_{i}&=\sum_{k} w_k\,\mu_{ik}\quad\text{(weight of prototype } \mathbf{v}_i\text{)},\\
        (\mathbf{U}_l,\mathbf{V}_l)&=\text{wFCM}\big([\mathbf{X}_l;\mathbf{V}_{l-1}],\
        [\mathbf{1};\mathbf{w}']\big)\ \text{initialized with } \mathbf{V}_{l-1},\quad l=2,\dots,s,

    and the result is :math:`\mathbf{V}=\mathbf{V}_s` with the extension
    :math:`\mathbf{U}=\text{extend}(\mathbf{X},\mathbf{V})`. The weights
    make a carried prototype count as the observations it represents, so
    the final prototypes approximate those of FCM on the complete data;
    the approximation is good when every chunk contains all clusters and
    deteriorates for tiny chunks.

    The ``forgetting_factor`` :math:`\lambda` is an addition of this
    implementation for streams whose clusters drift: the carried weights
    become :math:`\lambda\,w'_i`. It is not part of the published
    algorithm, which corresponds to :math:`\lambda=1`.

    References
    ----------
    .. [1] T. C. Havens, J. C. Bezdek, C. Leckie, L. O. Hall,
           M. Palaniswami, "Fuzzy c-means algorithms for very large data",
           IEEE Transactions on Fuzzy Systems 20(6), 1130--1146, 2012.
    .. [2] P. Hore, L. O. Hall, D. B. Goldgof, "Single pass fuzzy c means",
           Proc. IEEE International Conference on Fuzzy Systems, 1--7, 2007.

    Examples
    --------
    >>> from fuzzyclust.large import SinglePassFCM, make_large_blobs
    >>> X, y = make_large_blobs(n_samples=5000, n_clusters=3, random_state=0)
    >>> sp = SinglePassFCM(n_clusters=3, chunk_size=1000, random_state=0).fit(X)
    >>> sp.n_chunks_, sp.chunk_centers_.shape, sp.membership_.shape
    (5, (5, 3, 2), (5000, 3))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        chunk_size: int = 10_000,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="random",
        n_init: int = 1,
        shuffle: bool = True,
        forgetting_factor: float = 1.0,
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
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.forgetting_factor = forgetting_factor

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("SinglePassFCM requires m > 1.")
        _check_chunk_size(self.chunk_size, self.n_clusters)
        lam = self.forgetting_factor
        numeric = isinstance(lam, (int, float, np.integer, np.floating)) and not isinstance(lam, bool)
        if not numeric or not 0.0 <= lam <= 1.0:
            raise ValueError("forgetting_factor must be a number in [0, 1].")

    # ------------------------------------------------------------------ chunk processing
    def _reset(self):
        self._rng = check_random_state(self.random_state)
        self._history = {"centers": [], "weights": [], "objective": [], "n_iter": []}
        for attr in ("cluster_centers_", "center_weights_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _process_chunk(self, Xc: np.ndarray) -> WeightedFCM:
        if not hasattr(self, "cluster_centers_"):
            wfcm = WeightedFCM(
                n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
                init=self.init, n_init=self.n_init, random_state=self._rng, verbose=self.verbose,
            ).fit(Xc)
            w = np.ones(Xc.shape[0])
        else:
            Xa = np.vstack([Xc, self.cluster_centers_])
            w = np.concatenate([np.ones(Xc.shape[0]), self.forgetting_factor * self.center_weights_])
            wfcm = WeightedFCM(
                n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
                init=self.cluster_centers_, random_state=self._rng, verbose=self.verbose,
            ).fit(Xa, sample_weight=w)
        self.cluster_centers_ = wfcm.cluster_centers_
        self.center_weights_ = (w[:, None] * wfcm.membership_).sum(axis=0)
        h = self._history
        h["centers"].append(self.cluster_centers_.copy())
        h["weights"].append(self.center_weights_.copy())
        h["objective"].append(wfcm.objective_)
        h["n_iter"].append(wfcm.n_iter_)
        return wfcm

    def _store_history(self):
        h = self._history
        self.chunk_centers_ = np.asarray(h["centers"])
        self.chunk_weights_ = np.asarray(h["weights"])
        self.chunk_objective_ = np.asarray(h["objective"], dtype=float)
        self.chunk_n_iter_ = np.asarray(h["n_iter"], dtype=int)
        self.n_chunks_ = len(h["centers"])
        self.objective_history_ = self.chunk_objective_
        self.n_iter_ = int(self.chunk_n_iter_.sum())

    def _extend(self, X):
        U, d2 = extend_partition(X, self.cluster_centers_, self.m, int(self.chunk_size))
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.distances_ = np.sqrt(d2)
        self.objective_ = float(np.sum(U**self.m * d2))

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Process the data chunk by chunk and extend the final prototypes to all samples.

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
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        self._reset()
        for idx in _split_chunks(X.shape[0], int(self.chunk_size), self.shuffle, self._rng):
            self._process_chunk(X[idx])
        self._store_history()
        self._extend(X)
        return self

    def partial_fit(self, X, y=None):
        """Process one more chunk of a stream (the first call initializes the model).

        The prototypes and their weights are carried over from the
        previous call; ``membership_`` and ``labels_`` refer to the chunk
        just processed. Use :meth:`predict_proba` for other data.

        Parameters
        ----------
        X : array-like of shape (n_chunk, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        first = not hasattr(self, "_history")
        X = validate_data(self, X, dtype=float, ensure_min_samples=1, reset=first)
        self._check_params(X)
        if first:
            if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
                raise ValueError("n_init must be a positive integer.")
            self._reset()
        self._process_chunk(X)
        self._store_history()
        self._extend(X)
        return self

    # ------------------------------------------------------------------ base API hooks
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _distances(self, X):
        check_is_fitted(self, "cluster_centers_")
        return sq_euclidean(X, self.cluster_centers_)
