# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Compatible cluster merging: reducing a deliberately large number of clusters by merging compatible pairs.

Instead of scanning the number of clusters with validity indices, the
algorithm is started with more clusters than expected and pairs of clusters
that are *compatible* are merged, re-clustering after every merge, until no
compatible pair is left. Two families of criteria are implemented: the
geometric criteria of Krishnapuram and Freg for thin (line- or plane-like)
clusters---parallel normals, centers on the same hyperplane, centers close
---and the similarity of the membership columns (Setnes et al.), used for
merging by Kaymak and Setnes, for clusters of any shape.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_covariances
from .gk import GK

_CRITERIA = ("geometric", "similarity")


def cluster_similarity(U: np.ndarray) -> np.ndarray:
    r"""Pairwise similarity of fuzzy clusters from their membership columns (fuzzy Jaccard index).

    .. math::
        S_{ij}=\frac{\sum_{k=1}^{N}\min(\mu_{ik},\mu_{jk})}{\sum_{k=1}^{N}\max(\mu_{ik},\mu_{jk})},

    the fuzzy Jaccard similarity of the clusters seen as fuzzy sets over the
    samples (Setnes et al., 1998): one for identical columns, zero for
    disjoint supports. Kaymak and Setnes (2002), whose adaptive cluster
    merging motivates this criterion, use a fuzzy inclusion measure instead,
    :math:`S_{ij}=\max(I_{ij},I_{ji})` with
    :math:`I_{ij}=\sum_k\min(\mu_{ik},\mu_{jk})/\sum_k\mu_{ik}`; the toolbox
    uses the symmetric Jaccard form with a fixed threshold.

    Parameters
    ----------
    U : ndarray of shape (n_samples, n_clusters)

    Returns
    -------
    S : ndarray of shape (n_clusters, n_clusters)
        Symmetric, with ones on the diagonal.
    """
    U = np.asarray(U, dtype=float)
    c = U.shape[1]
    S = np.eye(c)
    for i in range(c):
        for j in range(i + 1, c):
            num = np.minimum(U[:, i], U[:, j]).sum()
            den = np.maximum(U[:, i], U[:, j]).sum()
            S[i, j] = S[j, i] = num / max(den, EPS)
    return S


def reference_fuzziness(U: np.ndarray, m: float, m_ref: float = 2.0) -> np.ndarray:
    r"""Rescale an FCM-type partition obtained with exponent ``m`` to the one exponent ``m_ref`` would give.

    For memberships of the form :math:`\mu_{ik}\propto(D^2_{ik})^{-1/(m-1)}`
    the partition for another exponent follows from the same distances by
    :math:`\mu'_{ik}\propto\mu_{ik}^{(m-1)/(m_{\mathrm{ref}}-1)}`; a hard
    partition (``m <= 1``) is returned unchanged.
    """
    U = np.asarray(U, dtype=float)
    if m <= 1.0 or m == m_ref:
        return U
    W = (U + EPS) ** ((m - 1.0) / (m_ref - 1.0))
    return W / W.sum(axis=1, keepdims=True)


def geometric_compatibility(V, F, parallel_threshold=0.95, hyperplane_threshold=0.5, distance_threshold=2.0,
                            thin_threshold=0.2):
    r"""Compatibility of cluster pairs from centers and covariances (Krishnapuram & Freg).

    Let :math:`\boldsymbol\phi_i` be the eigenvector of the smallest
    eigenvalue of :math:`\mathbf{F}_i` (the normal of a thin cluster),
    :math:`\lambda_{i,\max}` its largest eigenvalue and
    :math:`\sigma_i=\sqrt{\lambda_{i,\max}}` the extent of the cluster along
    its main axis. Clusters :math:`i` and :math:`j` are compatible if

    1. their normals are parallel, :math:`|\boldsymbol\phi_i\T\boldsymbol\phi_j|\ge\gamma_1`;
    2. their centers lie on the same hyperplane,
       :math:`|\tfrac12(\boldsymbol\phi_i+\boldsymbol\phi_j)\T(\mathbf{v}_i-\mathbf{v}_j)|
       \le\gamma_2\,\tfrac12(\sigma_i+\sigma_j)`;
    3. their centers are close, :math:`\|\mathbf{v}_i-\mathbf{v}_j\|\le\gamma_3(\sigma_i+\sigma_j)`;

    and both are thin, :math:`\lambda_{\min}/\lambda_{\max}\le\gamma_4`. The
    compatibility degree returned for a compatible pair is the normal
    alignment :math:`|\boldsymbol\phi_i\T\boldsymbol\phi_j|` reduced by the
    relative hyperplane offset and center distance, so that the most
    compatible pair can be chosen; incompatible pairs get zero.

    Parameters
    ----------
    V : ndarray of shape (n_clusters, n_features)
    F : ndarray of shape (n_clusters, n_features, n_features)
    parallel_threshold, hyperplane_threshold, distance_threshold, thin_threshold : float
        :math:`\gamma_1,\dots,\gamma_4`.

    Returns
    -------
    C : ndarray of shape (n_clusters, n_clusters)
        Compatibility degrees (zero for incompatible pairs and on the diagonal).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    c = V.shape[0]
    w, Q = np.linalg.eigh(F)
    normals = Q[:, :, 0]  # (c, n): eigenvector of the smallest eigenvalue
    extent = np.sqrt(np.maximum(w[:, -1], EPS))
    thin = w[:, 0] / np.maximum(w[:, -1], EPS) <= thin_threshold
    C = np.zeros((c, c))
    for i in range(c):
        for j in range(i + 1, c):
            if not (thin[i] and thin[j]):
                continue
            align = abs(float(normals[i] @ normals[j]))
            if align < parallel_threshold:
                continue
            # make the normals point the same way before averaging them
            nj = normals[j] if normals[i] @ normals[j] >= 0 else -normals[j]
            dv = V[i] - V[j]
            offset = abs(float(0.5 * (normals[i] + nj) @ dv)) / (0.5 * (extent[i] + extent[j]))
            if offset > hyperplane_threshold:
                continue
            dist = float(np.linalg.norm(dv)) / (extent[i] + extent[j])
            if dist > distance_threshold:
                continue
            C[i, j] = C[j, i] = align * (1.0 - offset / hyperplane_threshold / 2.0) * (
                1.0 - dist / distance_threshold / 2.0
            )
    return C


class CompatibleClusterMerging(ClusterMixin, TransformerMixin, BaseEstimator):
    r"""Compatible cluster merging: find the number of clusters by merging from a large start.

    A base estimator (Gustafson--Kessel by default) is fitted with
    ``start_n_clusters`` clusters. The compatibility of every pair of
    clusters is evaluated, the most compatible pair above the thresholds is
    merged---their membership columns are added---and the base estimator is
    refitted from the merged partition with one cluster less. This repeats
    until no compatible pair remains or ``min_n_clusters`` is reached. The
    *geometric* criterion of Krishnapuram and Freg is meant for clusters
    that stand for lines, planes or local linear models: two thin clusters
    are compatible if their normals are parallel, their centers lie on the
    same hyperplane and are close. The *similarity* criterion (after Kaymak and
    Setnes) works for any shape: two clusters are compatible if their
    membership columns are similar as fuzzy sets.

    Parameters
    ----------
    base_estimator : fuzzyclust clusterer or None, default=None
        Cloned and refitted at every step; ``None`` means
        ``GK(random_state=0)``. Its ``n_clusters`` and ``init`` are
        overridden. The geometric criterion needs ``covariances_`` (FCM, GK,
        GG and most extensions provide it; otherwise the fuzzy covariances
        are computed from the partition).
    start_n_clusters : int, default=8
        Deliberately large initial number of clusters :math:`c_{\max}`.
    min_n_clusters : int, default=1
        Merging stops at this number of clusters at the latest.
    criterion : {"geometric", "similarity"}, default="geometric"
        Compatibility criterion, see :func:`geometric_compatibility` and
        :func:`cluster_similarity`.
    parallel_threshold : float, default=0.95
        :math:`\gamma_1`: minimal :math:`|\boldsymbol\phi_i\T\boldsymbol\phi_j|` (geometric).
    hyperplane_threshold : float, default=0.5
        :math:`\gamma_2`: maximal offset of the centers from the common
        hyperplane, relative to the mean extent (geometric).
    distance_threshold : float, default=2.0
        :math:`\gamma_3`: maximal center distance relative to the sum of the
        extents (geometric).
    thin_threshold : float, default=0.2
        :math:`\gamma_4`: maximal :math:`\lambda_{\min}/\lambda_{\max}` for a
        cluster to count as thin (geometric).
    similarity_threshold : float, default=0.2
        Minimal :math:`S_{ij}` for a merge (similarity criterion). The
        similarity is evaluated on the partition rescaled to the reference
        exponent :math:`m=2` (:math:`\mu_{ik}^{m-1}`, renormalized, see
        :func:`reference_fuzziness`), so that the threshold does not depend
        on the exponent of the base estimator. Two FCM clusters that share
        one Gaussian group have :math:`S_{ij}\approx0.35` at :math:`m=2`
        whatever :math:`c`, two separate groups well below :math:`0.2`.

    Attributes
    ----------
    estimator_ : fitted base estimator
        The final model with ``n_clusters_`` clusters.
    n_clusters_ : int
    cluster_centers_, membership_, labels_, distances_
        Taken from ``estimator_``.
    merge_history_ : list of dict
        One entry per merge with ``n_clusters`` (before the merge), ``pair``
        (the merged clusters) and ``compatibility`` (degree or similarity).
    compatibility_ : ndarray of shape (``n_clusters_``, ``n_clusters_``)
        Compatibility (or similarity) matrix of the final clusters; all
        entries are below the thresholds.
    n_features_in_, feature_names_in_

    Notes
    -----
    Merging by adding membership columns and refitting with ``init`` set to
    the merged partition is the generic form of Kelly's merging formulas
    for the center and covariance of the union: the first iteration of the
    refit computes exactly the membership-weighted center and covariance of
    the merged cluster. Because the number of clusters only decreases, the
    procedure costs at most :math:`c_{\max}-c^*` refits, each started close
    to its solution, against :math:`c_{\max}-1` refits from scratch for a
    validity scan.

    References
    ----------
    .. [1] R. Krishnapuram, C.-P. Freg, "Fitting an unknown number of lines
           and planes to image data through compatible cluster merging",
           Pattern Recognition 25(4), 385--400, 1992.
    .. [2] U. Kaymak, M. Setnes, "Fuzzy clustering with volume prototypes and
           adaptive cluster merging", IEEE Transactions on Fuzzy Systems
           10(6), 705--712, 2002.
    .. [3] U. Kaymak, R. Babuška, "Compatible cluster merging for fuzzy
           modeling", Proc. FUZZ-IEEE/IFES, Yokohama, 897--904, 1995.

    Examples
    --------
    >>> from fuzzyclust.cluster import CompatibleClusterMerging, GK
    >>> from fuzzyclust.datasets import make_lines
    >>> X, _ = make_lines(n_lines=3, random_state=0)
    >>> ccm = CompatibleClusterMerging(GK(random_state=0), start_n_clusters=8).fit(X)
    >>> int(ccm.n_clusters_)
    3
    """

    def __init__(
        self,
        base_estimator=None,
        start_n_clusters: int = 8,
        min_n_clusters: int = 1,
        criterion: str = "geometric",
        parallel_threshold: float = 0.95,
        hyperplane_threshold: float = 0.5,
        distance_threshold: float = 2.0,
        thin_threshold: float = 0.2,
        similarity_threshold: float = 0.2,
    ):
        self.base_estimator = base_estimator
        self.start_n_clusters = start_n_clusters
        self.min_n_clusters = min_n_clusters
        self.criterion = criterion
        self.parallel_threshold = parallel_threshold
        self.hyperplane_threshold = hyperplane_threshold
        self.distance_threshold = distance_threshold
        self.thin_threshold = thin_threshold
        self.similarity_threshold = similarity_threshold

    def _compatibility(self, est, X):
        U = est.membership_
        if self.criterion == "similarity":
            S = cluster_similarity(reference_fuzziness(U, getattr(est, "m", 2.0)))
            np.fill_diagonal(S, 0.0)
            S[S < self.similarity_threshold] = 0.0
            return S
        V = est.cluster_centers_
        F = getattr(est, "covariances_", None)
        if F is None or np.asarray(F).shape != (V.shape[0], X.shape[1], X.shape[1]):
            F = fuzzy_covariances(X, U ** getattr(est, "m", 2.0), V)
        return geometric_compatibility(
            V,
            F,
            self.parallel_threshold,
            self.hyperplane_threshold,
            self.distance_threshold,
            self.thin_threshold,
        )

    def fit(self, X, y=None):
        """Fit with ``start_n_clusters`` clusters and merge compatible pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        X = validate_data(self, X, dtype=float, ensure_min_samples=2)
        if self.criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}.")
        if not isinstance(self.start_n_clusters, (int, np.integer)) or self.start_n_clusters < 1:
            raise ValueError("start_n_clusters must be a positive integer.")
        if not 1 <= self.min_n_clusters <= self.start_n_clusters:
            raise ValueError("Need 1 <= min_n_clusters <= start_n_clusters.")
        base = GK(random_state=0) if self.base_estimator is None else self.base_estimator
        c = min(self.start_n_clusters, X.shape[0])
        est = clone(base).set_params(n_clusters=c).fit(X)
        history = []
        while True:
            C = self._compatibility(est, X)
            if c <= self.min_n_clusters or C.max() <= 0.0:
                break
            i, j = np.unravel_index(int(C.argmax()), C.shape)
            i, j = sorted((int(i), int(j)))
            history.append({"n_clusters": c, "pair": (i, j), "compatibility": float(C[i, j])})
            U = est.membership_
            U_merged = np.delete(U, j, axis=1)
            U_merged[:, i] = U[:, i] + U[:, j]
            c -= 1
            est = clone(base).set_params(n_clusters=c, init=U_merged).fit(X)
        self.estimator_ = est
        self.compatibility_ = C
        self.merge_history_ = history
        self.n_clusters_ = c
        self.cluster_centers_ = est.cluster_centers_
        self.membership_ = est.membership_
        self.labels_ = est.labels_
        self.distances_ = est.distances_
        return self

    def predict_proba(self, X):
        """Memberships of (new) samples in the final clusters."""
        check_is_fitted(self, "estimator_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """Hard labels of (new) samples."""
        check_is_fitted(self, "estimator_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self.estimator_.predict(X)

    def transform(self, X):
        """Distances of samples to the final prototypes, shape ``(n_samples, n_clusters_)``."""
        check_is_fitted(self, "estimator_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self.estimator_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def score(self, X, y=None):
        """Score of the final estimator (negative objective)."""
        check_is_fitted(self, "estimator_")
        X = validate_data(self, X, dtype=float, reset=False)
        return self.estimator_.score(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "estimator_")
        return np.asarray([f"dist_cluster_{i}" for i in range(self.n_clusters_)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


__all__ = ["CompatibleClusterMerging", "cluster_similarity", "geometric_compatibility", "reference_fuzziness"]
