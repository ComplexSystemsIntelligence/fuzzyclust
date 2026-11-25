# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy clustering with volume prototypes and adaptive cluster merging (Kaymak & Setnes)."""

from __future__ import annotations

import numpy as np

from .._utils import (
    EPS,
    fuzzy_centers,
    fuzzy_covariances,
    memberships_from_distances,
    regularize_covariance,
    sq_euclidean,
    sq_mahalanobis,
)
from ..base import BaseFuzzyClusterer
from ..cluster.merging import cluster_similarity, reference_fuzziness

_NORMS = ("euclidean", "adaptive")


class VolumePrototypeFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means with volume prototypes and optional adaptive cluster merging (Kaymak & Setnes, 2002).

    A point prototype measures the distance to the *center* of a cluster, so
    a sample in the middle of a large cluster is farther from its own
    prototype than a sample at the edge of a small one: this is why FCM and
    GK tend to equalize cluster sizes and split large clusters. A *volume
    prototype* replaces the point by a hyperellipsoidal core of radius
    :math:`r_i` (in the norm of the cluster) inside which the distance is
    zero; only the distance to the *surface* of the core counts,

    .. math::
        \tilde D_{ik}=\max\bigl(0,\;D_{ik}-r_i\bigr),

    so every sample inside the core belongs fully to the cluster and a
    large cluster can absorb its members without having to compete for
    them point by point. The core radius adapts to the cluster. Kaymak and
    Setnes combine the volume prototypes with *adaptive cluster merging*:
    the algorithm is started with more clusters than expected and, whenever
    two clusters have become similar as fuzzy sets, they are merged and the
    iteration continues with one cluster less, so that the number of
    clusters is found during the clustering itself.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c` (the initial, maximal number when
        ``merge=True``; the final number is ``n_clusters_``).
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    norm : {"euclidean", "adaptive"}, default="euclidean"
        Distance to the center: Euclidean (spherical cores) or the
        Gustafson--Kessel norm :math:`\mathbf{A}_i=[\det\mathbf{F}_i]^{1/n}\mathbf{F}_i^{-1}`
        (ellipsoidal cores aligned with the clusters).
    radius : {"fuzzy_mean"} or float or array-like of shape (n_clusters,), default="fuzzy_mean"
        Core radius. ``"fuzzy_mean"`` re-estimates
        :math:`r_i=\rho\sum_k\mu_{ik}^mD_{ik}/\sum_k\mu_{ik}^m` (the
        membership-weighted mean distance of the data to the center, in the
        cluster norm) at every iteration; a number or an array fixes the
        radii. ``0`` recovers FCM/GK.
    radius_scale : float, default=1.0
        Factor :math:`\rho` of the estimated radius; larger cores make the
        memberships crisper and the clusters greedier.
    merge : bool, default=False
        Adaptive cluster merging: whenever the partition has converged for
        the current number of clusters, the most similar pair of clusters
        with :math:`S_{ij}` above ``similarity_threshold`` is merged (their
        membership columns are added) and the iteration continues with one
        cluster less until no similar pair remains.
    similarity_threshold : float, default=0.2
        Threshold on the fuzzy-set similarity :math:`S_{ij}` of two clusters.
        The similarity is evaluated on the memberships that the center
        distances alone would give (the cores make the memberships nearly
        crisp and would hide the overlap), rescaled to the reference
        exponent :math:`m=2` as in :class:`~fuzzyclust.cluster.CompatibleClusterMerging`.
    gamma : float, default=0.0
    beta : float, default=1e15
        Covariance regularization of the adaptive norm, as in :class:`~fuzzyclust.cluster.GK`.
    tol : float, default=1e-4
    max_iter : int, default=300
        Maximum number of iterations; with ``merge=True`` per number of
        clusters (the iterations after every merge are counted afresh), so
        that a long merge sequence is not cut short.
    init : {"random", "kmeans++"} or ndarray, default="random"
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (``n_clusters_``, n_features)
    radii_ : ndarray of shape (``n_clusters_``,)
        Core radii :math:`r_i` at convergence.
    n_clusters_ : int
        Final number of clusters (``n_clusters`` unless merging reduced it).
    membership_ : ndarray of shape (n_samples, ``n_clusters_``)
    covariances_ : ndarray of shape (``n_clusters_``, n_features, n_features)
        Fuzzy covariance matrices of the final partition.
    norm_matrices_ : ndarray of shape (``n_clusters_``, n_features, n_features)
        Norm-inducing matrices (identity for the Euclidean norm).
    merge_history_ : list of dict
        One entry per merge: ``iteration``, ``n_clusters`` (before), ``pair``, ``similarity``.
    labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_`` and
        ``transform`` return :math:`\tilde D_{ik}`, zero inside the cores.

    Notes
    -----
    The functional is the c-means functional with the core distance,
    :math:`J=\sum_i\sum_k\mu_{ik}^m\tilde D^2_{ik}`, with
    :math:`\tilde D_{ik}=\max(0,D_{ik\mathbf{A}_i}-r_i)`,
    :math:`D^2_{ik\mathbf{A}_i}=(\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{A}_i(\mathbf{x}_k-\mathbf{v}_i)`.
    Memberships follow the FCM formula (a sample inside exactly one core
    has membership one there), centers are the :math:`\mu^m`-weighted means,
    the norms are those of GK when ``norm="adaptive"``, and the radii are
    re-estimated from the current partition. The similarity of two
    clusters is :math:`S_{ij}=\sum_k\min(\mu_{ik},\mu_{jk})/\sum_k\max(\mu_{ik},\mu_{jk})`
    (Setnes et al.). This differs from Kaymak and Setnes, who use the
    inclusion measure :math:`\max(I_{ij},I_{ji})`,
    :math:`I_{ij}=\sum_k\min(\mu_{ik},\mu_{jk})/\sum_k\mu_{ik}`, merge during
    the iterations with the threshold :math:`1/(c-1)` and derive the radius
    from the fuzzy covariance; here clusters are merged after convergence
    with a fixed threshold, and the default radius is a fuzzy mean distance.

    References
    ----------
    .. [1] U. Kaymak, M. Setnes, "Fuzzy clustering with volume prototypes and
           adaptive cluster merging", IEEE Transactions on Fuzzy Systems
           10(6), 705--712, 2002.
    .. [2] M. Setnes, R. Babuška, U. Kaymak, H. R. van Nauta Lemke,
           "Similarity measures in fuzzy rule base simplification", IEEE
           Transactions on Systems, Man, and Cybernetics B 28(3), 376--386, 1998.

    Examples
    --------
    >>> from fuzzyclust.extensions import VolumePrototypeFCM
    >>> from fuzzyclust.datasets import make_different_densities
    >>> X, _ = make_different_densities(random_state=0)
    >>> vp = VolumePrototypeFCM(n_clusters=3, random_state=0).fit(X)
    >>> vp.radii_.shape
    (3,)
    >>> vpm = VolumePrototypeFCM(n_clusters=8, merge=True, random_state=0).fit(X)
    >>> int(vpm.n_clusters_) < 8
    True
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        norm: str = "euclidean",
        radius="fuzzy_mean",
        radius_scale: float = 1.0,
        merge: bool = False,
        similarity_threshold: float = 0.2,
        gamma: float = 0.0,
        beta: float = 1e15,
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
        self.norm = norm
        self.radius = radius
        self.radius_scale = radius_scale
        self.merge = merge
        self.similarity_threshold = similarity_threshold
        self.gamma = gamma
        self.beta = beta

    # ------------------------------------------------------------------ checks
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("VolumePrototypeFCM requires m > 1.")
        if self.norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}.")
        if isinstance(self.radius, str):
            if self.radius != "fuzzy_mean":
                raise ValueError("radius must be 'fuzzy_mean', a number or an array.")
        else:
            r = np.asarray(self.radius, dtype=float)
            if r.ndim > 1 or (r.ndim == 1 and r.shape[0] != self.n_clusters) or np.any(r < 0):
                raise ValueError("radius must be a non-negative scalar or an array of n_clusters values.")
        if self.radius_scale < 0:
            raise ValueError("radius_scale must be non-negative.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must lie in [0, 1].")
        self._F0 = np.cov(X, rowvar=False) if X.shape[1] > 1 else np.atleast_2d(X.var())

    # ------------------------------------------------------------------ hooks
    def _center_distances(self, X):
        """Distances (not squared) to the centers in the cluster norm."""
        if self.norm == "euclidean":
            return np.sqrt(sq_euclidean(X, self.cluster_centers_))
        return np.sqrt(sq_mahalanobis(X, self.cluster_centers_, self.norm_matrices_))

    def _update_prototypes(self, X, Um):
        c = Um.shape[1]
        n = X.shape[1]
        V = fuzzy_centers(X, Um)
        F = fuzzy_covariances(X, Um, V)
        A = np.empty_like(F)
        if self.norm == "adaptive":
            for i in range(c):
                Fi = regularize_covariance(F[i], self._F0, self.gamma, self.beta)
                det = max(np.linalg.det(Fi), EPS)
                A[i] = det ** (1.0 / n) * np.linalg.pinv(Fi)
                F[i] = Fi
        else:
            A[:] = np.eye(n)
        self.cluster_centers_ = V
        self.covariances_ = F
        self.norm_matrices_ = A
        if isinstance(self.radius, str):
            D = self._center_distances(X)
            self.radii_ = self.radius_scale * (Um * D).sum(axis=0) / np.maximum(Um.sum(axis=0), EPS)
        else:
            self.radii_ = np.broadcast_to(np.asarray(self.radius, dtype=float), (c,)).copy()

    def _distances(self, X):
        D = self._center_distances(X)
        return np.maximum(D - self.radii_[None, :], 0.0) ** 2

    def _similarity_partition(self, X):
        """Partition from the center distances alone (cores removed), rescaled to m = 2.

        Inside the cores the memberships are (nearly) crisp, which would make
        two clusters sharing one group look dissimilar; the center distances
        give the FCM memberships, for which the similarity threshold is
        calibrated.
        """
        U = memberships_from_distances(self._center_distances(X) ** 2, self.m)
        return reference_fuzziness(U, self.m)

    def _init_state(self, X, rng):
        U = super()._init_state(X, rng)
        self.n_clusters_ = U.shape[1]
        return U

    def _fit_single(self, X, rng):
        if not self.merge:
            self.merge_history_ = []
            result = super()._fit_single(X, rng)
            result["state"]["n_clusters_"] = result["U"].shape[1]
            return result
        U = self._init_state(X, rng)
        history, merges = [], []
        it = 0
        stage_it = 0  # iterations since the last merge
        while True:
            it += 1
            stage_it += 1
            Um = U**self.m
            self._update_prototypes(X, Um)
            d2 = self._distances(X)
            U_new = self._memberships(d2)
            history.append(self._objective(Um, d2))
            if self.verbose:
                print(f"iter {it:4d}  c = {U.shape[1]}  J = {history[-1]:.6g}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol or stage_it >= self.max_iter:
                if U.shape[1] <= 1:
                    break
                S = cluster_similarity(self._similarity_partition(X))
                np.fill_diagonal(S, 0.0)
                i, j = np.unravel_index(int(S.argmax()), S.shape)
                if S[i, j] < self.similarity_threshold:
                    break
                i, j = sorted((int(i), int(j)))
                merges.append(
                    {"iteration": it, "n_clusters": U.shape[1], "pair": (i, j), "similarity": float(S[i, j])}
                )
                U_merged = np.delete(U, j, axis=1)
                U_merged[:, i] = U[:, i] + U[:, j]
                U = U_merged
                # one more prototype step so that the returned state matches the merged partition
                Um = U**self.m
                self._update_prototypes(X, Um)
                d2 = self._distances(X)
                U = self._memberships(d2)
                stage_it = 0
        self.merge_history_ = merges
        self.n_clusters_ = U.shape[1]
        state = {k: v for k, v in self.__dict__.items() if k.endswith("_") and not k.startswith("_")}
        return {"U": U, "d2": d2, "history": history, "n_iter": it, "state": state}

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "cluster_centers_")
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters_)], dtype=object)


__all__ = ["VolumePrototypeFCM"]
