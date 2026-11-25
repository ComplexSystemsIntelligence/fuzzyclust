# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Linear projection of data and cluster prototypes by principal component analysis."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.utils import check_array

__all__ = ["pca_projection"]


def pca_projection(X, n_components: int = 2, centers=None):
    r"""Project the data and the cluster prototypes onto the first principal components.

    PCA finds the :math:`q`-dimensional linear subspace that captures the
    largest part of the variance of the data. The eigenvectors
    :math:`\mathbf{w}_1,\dots,\mathbf{w}_q` of the data covariance matrix

    .. math::
        \mathbf{F} = \frac{1}{N}\sum_{k=1}^{N}(\mathbf{x}_k-\bar{\mathbf{x}})(\mathbf{x}_k-\bar{\mathbf{x}})^T

    with the :math:`q` largest eigenvalues span the projection plane and the
    embedding of a sample is :math:`\mathbf{y}_k = \mathbf{W}^T(\mathbf{x}_k-\bar{\mathbf{x}})`.
    The cluster prototypes are mapped by the *same* affine transform, so the
    projected centers :math:`\mathbf{z}_i` sit where the fuzzy means of the
    projected data would be (for Euclidean prototypes). The transform
    preserves variance, not distances, so compact high-dimensional clusters
    may overlap in the plane; this is the baseline the Sammon-type methods of
    this module improve on.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix :math:`\mathbf{X}`.
    n_components : int, default=2
        Dimension :math:`q` of the projection.
    centers : array-like of shape (n_clusters, n_features), optional
        Cluster prototypes :math:`\mathbf{v}_i` to project with the same transform.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Projected data :math:`\mathbf{y}_k`.
    Z : ndarray of shape (n_clusters, n_components) or None
        Projected prototypes :math:`\mathbf{z}_i`; ``None`` if ``centers`` is not given.
    pca : sklearn.decomposition.PCA
        The fitted PCA object (``components_``, ``explained_variance_ratio_``,
        ``transform`` for further data).

    References
    ----------
    .. [1] I. T. Jolliffe, "Principal Component Analysis", Springer, 2002.
    .. [2] J. Vesanto, "Neural network tool for data mining: SOM Toolbox",
           Proc. TOOLMET 2000, pp. 184-196, 2000.

    Examples
    --------
    >>> from fuzzyclust import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.viz import pca_projection
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, n_features=5, random_state=0)
    >>> fcm = FCM(n_clusters=3, random_state=0).fit(X)
    >>> Y, Z, pca = pca_projection(X, 2, centers=fcm.cluster_centers_)
    >>> Y.shape, Z.shape
    ((300, 2), (3, 2))
    """
    X = check_array(X, dtype=float)
    q = int(n_components)
    if q < 1 or q > min(X.shape):
        raise ValueError(f"n_components must be in [1, min(n_samples, n_features)] = [1, {min(X.shape)}].")
    pca = PCA(n_components=q).fit(X)
    Y = pca.transform(X)
    Z = None
    if centers is not None:
        centers = check_array(centers, dtype=float, ensure_min_samples=1)
        if centers.shape[1] != X.shape[1]:
            raise ValueError("centers must have the same number of features as X.")
        Z = pca.transform(centers)
    return Y, Z, pca
