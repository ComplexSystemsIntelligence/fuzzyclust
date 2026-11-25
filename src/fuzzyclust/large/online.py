# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Online fuzzy c-means (oFCM)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import sq_euclidean
from ..base import BaseFuzzyClusterer
from .single_pass import _check_chunk_size, _split_chunks
from .weighted import WeightedFCM, extend_partition

__all__ = ["OnlineFCM"]


class OnlineFCM(BaseFuzzyClusterer):
    r"""Online fuzzy c-means (oFCM) for very large and streaming data.

    Every chunk is clustered on its own with FCM; the :math:`c` prototypes
    of each chunk, weighted by the total membership they collected, are
    kept. When all chunks have been seen, the :math:`s\cdot c` weighted
    prototypes are clustered into :math:`c` clusters with weighted FCM
    (:class:`WeightedFCM`), and the partition of all samples is obtained
    by one extension step. In contrast to :class:`SinglePassFCM` the
    chunks do not influence each other, so the chunk clusterings can run
    in parallel or arrive from a stream in any order; the price is the
    storage of :math:`s\cdot c` prototypes and the final clustering of
    them (Hore et al. 2009; Havens et al. 2012).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    chunk_size : int, default=10000
        Samples per chunk :math:`n_l`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization of the FCM run on the first chunk.
    n_init : int, default=1
        Random restarts of the first chunk.
    chunk_init : {"previous", "independent"}, default="previous"
        Start the FCM run on a chunk from the prototypes of the previous
        chunk (fewer iterations, aligned cluster order) or from a fresh
        initialization with ``init``/``n_init`` (fully independent chunk
        clusterings, as in the original formulation).
    shuffle : bool, default=True
        Randomly permute the samples before chunking in ``fit``;
        ``partial_fit`` never reorders.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        The prototypes of the final weighted clustering of the chunk prototypes.
    center_weights_ : ndarray of shape (n_clusters,)
        Weight (number of observations) collected by every final prototype.
    chunk_centers_ : ndarray of shape (n_chunks, n_clusters, n_features)
        Prototypes of every chunk.
    chunk_weights_ : ndarray of shape (n_chunks, n_clusters)
        Weights :math:`w_{li}=\sum_k\mu_{ik}` of the chunk prototypes.
    chunk_objective_, chunk_n_iter_ : ndarray of shape (n_chunks,)
    n_chunks_ : int
    centers_model_ : :class:`WeightedFCM` or None
        The final weighted clusterer of the chunk prototypes (``None``
        after a single chunk, when there is nothing to merge).
    membership_ : ndarray of shape (n_samples, n_clusters)
        Extended partition of the data passed to ``fit`` (after
        ``partial_fit``: of the last chunk).
    labels_, distances_, objective_
        Of the extended partition.
    objective_history_ : ndarray
        :math:`J_w` per iteration of the final weighted clustering.
    n_iter_ : int
        Iterations of the final weighted clustering.

    Notes
    -----
    .. math::
        (\mathbf{U}_l,\mathbf{V}_l)&=\text{FCM}(\mathbf{X}_l),\qquad
        w_{li}=\sum_{k}\mu_{l,ik},\qquad l=1,\dots,s,\\
        \mathbf{V}&=\text{wFCM}\big([\mathbf{V}_1;\dots;\mathbf{V}_s],\
        [\mathbf{w}_1;\dots;\mathbf{w}_s]\big),\\
        \mathbf{U}&=\text{extend}(\mathbf{X},\mathbf{V}).

    The final weighted clustering is initialized with the prototypes of
    the last chunk. :meth:`partial_fit` appends the prototypes of a new
    chunk and repeats the (cheap) final clustering, so the model is usable
    after every chunk of a stream.

    References
    ----------
    .. [1] T. C. Havens, J. C. Bezdek, C. Leckie, L. O. Hall,
           M. Palaniswami, "Fuzzy c-means algorithms for very large data",
           IEEE Transactions on Fuzzy Systems 20(6), 1130--1146, 2012.
    .. [2] P. Hore, L. O. Hall, D. B. Goldgof, Y. Gu, A. A. Maudsley,
           A. Darkazanli, "A scalable framework for segmenting magnetic
           resonance images", Journal of Signal Processing Systems 54(1--3),
           183--203, 2009.

    Examples
    --------
    >>> from fuzzyclust.large import OnlineFCM, make_large_blobs
    >>> X, y = make_large_blobs(n_samples=5000, n_clusters=3, random_state=0)
    >>> of = OnlineFCM(n_clusters=3, chunk_size=1000, random_state=0).fit(X)
    >>> of.n_chunks_, of.chunk_weights_.shape, of.membership_.shape
    (5, (5, 3), (5000, 3))
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
        chunk_init: str = "previous",
        shuffle: bool = True,
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
        self.chunk_init = chunk_init
        self.shuffle = shuffle

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("OnlineFCM requires m > 1.")
        _check_chunk_size(self.chunk_size, self.n_clusters)
        if self.chunk_init not in ("previous", "independent"):
            raise ValueError("chunk_init must be 'previous' or 'independent'.")

    # ------------------------------------------------------------------ chunk processing
    def _reset(self):
        self._rng = check_random_state(self.random_state)
        self._history = {"centers": [], "weights": [], "objective": [], "n_iter": []}
        for attr in ("cluster_centers_", "center_weights_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _process_chunk(self, Xc: np.ndarray) -> WeightedFCM:
        h = self._history
        if h["centers"] and self.chunk_init == "previous":
            init, n_init = h["centers"][-1], 1
        else:
            init, n_init = self.init, self.n_init
        # FCM on the chunk; WeightedFCM with unit weights is FCM with plain
        # Euclidean distances (identical result, cheaper distance evaluation)
        fcm = WeightedFCM(
            n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
            init=init, n_init=n_init, random_state=self._rng, verbose=self.verbose,
        ).fit(Xc)
        h["centers"].append(fcm.cluster_centers_.copy())
        h["weights"].append(fcm.membership_.sum(axis=0))
        h["objective"].append(fcm.objective_)
        h["n_iter"].append(fcm.n_iter_)
        return fcm

    def _cluster_centers(self):
        h = self._history
        self.chunk_centers_ = np.asarray(h["centers"])
        self.chunk_weights_ = np.asarray(h["weights"])
        self.chunk_objective_ = np.asarray(h["objective"], dtype=float)
        self.chunk_n_iter_ = np.asarray(h["n_iter"], dtype=int)
        self.n_chunks_ = len(h["centers"])
        V_all = self.chunk_centers_.reshape(-1, self.chunk_centers_.shape[-1])
        w_all = self.chunk_weights_.ravel()
        if self.n_chunks_ == 1:  # a single chunk: nothing to merge
            self.centers_model_ = None
            self.cluster_centers_ = V_all.copy()
            self.center_weights_ = w_all.copy()
            self.objective_history_ = self.chunk_objective_.copy()
            self.n_iter_ = int(self.chunk_n_iter_[0])
            return
        wfcm = WeightedFCM(
            n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
            init=h["centers"][-1], random_state=self._rng, verbose=self.verbose,
        ).fit(V_all, sample_weight=w_all)
        self.centers_model_ = wfcm
        self.cluster_centers_ = wfcm.cluster_centers_
        self.center_weights_ = (w_all[:, None] * wfcm.membership_).sum(axis=0)
        self.objective_history_ = wfcm.objective_history_
        self.n_iter_ = wfcm.n_iter_

    def _extend(self, X):
        U, d2 = extend_partition(X, self.cluster_centers_, self.m, int(self.chunk_size))
        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.distances_ = np.sqrt(d2)
        self.objective_ = float(np.sum(U**self.m * d2))

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Cluster every chunk, cluster the weighted chunk prototypes, extend to all samples.

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
        self._cluster_centers()
        self._extend(X)
        return self

    def partial_fit(self, X, y=None):
        """Add one chunk of a stream: cluster it, store its weighted prototypes, update the model.

        ``membership_`` and ``labels_`` refer to the chunk just processed;
        use :meth:`predict_proba` for other data.

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
        self._cluster_centers()
        self._extend(X)
        return self

    # ------------------------------------------------------------------ base API hooks
    def _update_prototypes(self, X, Um):  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _distances(self, X):
        check_is_fitted(self, "cluster_centers_")
        return sq_euclidean(X, self.cluster_centers_)
