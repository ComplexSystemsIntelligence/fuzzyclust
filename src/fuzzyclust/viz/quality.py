# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Quality measures of a projection of fuzzy clustering results (MATLAB ``projeval``)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_array

from .._utils import check_membership, memberships_from_distances, sq_euclidean
from .sammon import sammon_stress

__all__ = ["recomputed_membership", "projection_quality"]


def recomputed_membership(Y, Z, m: float = 2.0) -> np.ndarray:
    r"""Fuzzy memberships recomputed in the projected space.

    Every projected cluster is a point :math:`\mathbf{z}_i`; the membership
    of the projected sample :math:`\mathbf{y}_k` follows the FCM formula with
    the Euclidean distance :math:`d^*_{ik}=\|\mathbf{y}_k-\mathbf{z}_i\|`:

    .. math::
        \mu^*_{ik} = \frac{1}{\sum_{j=1}^{c}\left(d^*_{ik}/d^*_{jk}\right)^{2/(m-1)}} .

    The matrix :math:`\mathbf{U}^*=[\mu^*_{ik}]` is what a viewer "sees" in
    the plot; comparing it with the original partition :math:`\mathbf{U}`
    quantifies how faithfully the projection represents the clustering.

    Parameters
    ----------
    Y : array-like of shape (n_samples, n_components)
        Projected data.
    Z : array-like of shape (n_clusters, n_components)
        Projected cluster centers.
    m : float, default=2.0
        Fuzziness exponent of the clustering.

    Returns
    -------
    U_star : ndarray of shape (n_samples, n_clusters)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.viz import recomputed_membership
    >>> Y = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> Z = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> recomputed_membership(Y, Z).round(6)
    array([[1., 0.],
           [0., 1.]])
    """
    Y = check_array(Y, dtype=float)
    Z = check_array(Z, dtype=float, ensure_min_samples=1)
    if Y.shape[1] != Z.shape[1]:
        raise ValueError("Y and Z must have the same number of columns.")
    return memberships_from_distances(sq_euclidean(Y, Z), m)


def projection_quality(U, Y, Z, m: float = 2.0, X=None) -> dict:
    r"""Measures of how well a 2-D (or q-D) projection represents a fuzzy partition.

    The projected prototypes :math:`\mathbf{z}_i` and data :math:`\mathbf{y}_k`
    define a recomputed partition :math:`\mathbf{U}^*` (see
    :func:`recomputed_membership`). The following numbers are returned
    (Kovács & Abonyi 2002; Feil, Balasko & Abonyi 2007; MATLAB ``projeval`` and
    ``samstr``):

    * ``P`` -- mean absolute difference of the original and recomputed memberships,
      :math:`P = \frac{1}{Nc}\sum_{i,k}|\mu_{ik}-\mu^*_{ik}|` (eq. (1.48) of the
      toolbox manual; 0 means that the plot shows exactly the fitted partition);
    * ``F`` -- partition coefficient of the original partition,
      :math:`F=\frac{1}{N}\sum_{i,k}\mu_{ik}^2`;
    * ``F_star`` -- partition coefficient of the recomputed partition; a large
      difference :math:`F^*-F` means that the plot looks crisper (or fuzzier)
      than the clustering really is;
    * ``E`` -- Sammon stress of the embedding with respect to the Euclidean
      distances of ``X`` (only if ``X`` is given), see
      :func:`~fuzzyclust.viz.sammon_stress`.

    Parameters
    ----------
    U : array-like of shape (n_samples, n_clusters)
        Fuzzy partition :math:`\mathbf{U}^T` of the clustering.
    Y : array-like of shape (n_samples, n_components)
        Projected data.
    Z : array-like of shape (n_clusters, n_components)
        Projected prototypes.
    m : float, default=2.0
        Fuzziness exponent used to recompute the memberships.
    X : array-like of shape (n_samples, n_features), optional
        Original data; if given, the Sammon stress ``E`` is added.

    Returns
    -------
    quality : dict
        Keys ``"P"``, ``"F"``, ``"F_star"`` and, if ``X`` is given, ``"E"``.

    References
    ----------
    .. [1] A. Kovács, J. Abonyi, "Visualization of fuzzy clustering results by
           modified Sammon mapping", Proc. 3rd Int. Symp. of Hungarian
           Researchers on Computational Intelligence, 177-188, 2002.
    .. [2] B. Feil, B. Balasko, J. Abonyi, "Visualization of fuzzy clusters by
           fuzzy Sammon mapping projection: application to the analysis of phase
           space trajectories", Soft Computing 11, 479-488, 2007.

    Examples
    --------
    >>> from fuzzyclust import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.viz import pca_projection, projection_quality
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, n_features=4, random_state=0)
    >>> fcm = FCM(n_clusters=3, random_state=0).fit(X)
    >>> Y, Z, _ = pca_projection(X, 2, fcm.cluster_centers_)
    >>> q = projection_quality(fcm.membership_, Y, Z, m=2.0, X=X)
    >>> sorted(q)
    ['E', 'F', 'F_star', 'P']
    """
    Y = check_array(Y, dtype=float)
    U = check_membership(U, Y.shape[0])
    U_star = recomputed_membership(Y, Z, m)
    if U_star.shape != U.shape:
        raise ValueError(f"U has shape {U.shape} but Z defines {U_star.shape[1]} clusters.")
    out = {
        "P": float(np.mean(np.abs(U - U_star))),
        "F": float(np.sum(U**2) / U.shape[0]),
        "F_star": float(np.sum(U_star**2) / U.shape[0]),
    }
    if X is not None:
        out["E"] = sammon_stress(X, Y)
    return out
