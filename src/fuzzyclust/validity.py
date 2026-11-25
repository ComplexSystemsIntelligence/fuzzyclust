# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Cluster validity indices and model-order (number of clusters) selection.

All index functions share the signature ``index(X, U, V, m=2.0)`` where

* ``X`` is the data matrix :math:`\mathbf{X}` of shape ``(n_samples, n_features)``,
* ``U`` the fuzzy partition :math:`\mathbf{U}^T` of shape ``(n_samples, n_clusters)``,
* ``V`` the prototypes :math:`\mathbf{V}` of shape ``(n_clusters, n_features)``,
* ``m`` the fuzziness exponent used by the clustering algorithm.

Indices that need only the partition (PC, CE, MPC) ignore ``X`` and ``V``;
indices that evaluate crisp clusters (DI, ADI, silhouette) harden the
partition first. Distances between data and prototypes are Euclidean, which
is the common convention in the validity literature; algorithm specific norms
are therefore *not* used here.

Two further groups of tools live here: *external* comparisons of two fuzzy
partitions (the fuzzy Rand, adjusted Rand and Jaccard indices of
Hüllermeier et al. and of Campello) and the *stability-based* selection of
the number of clusters (Ben-Hur et al.; Lange et al.), which resamples the
data and asks for which ``n_clusters`` the partitions agree.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone

from ._utils import EPS, check_membership, fuzzy_centers, fuzzy_covariances, harden, sq_euclidean

__all__ = [
    "partition_coefficient",
    "classification_entropy",
    "modified_partition_coefficient",
    "partition_index",
    "separation_index",
    "xie_beni",
    "fukuyama_sugeno",
    "dunn_index",
    "alternative_dunn_index",
    "silhouette",
    "fuzzy_hypervolume",
    "partition_density",
    "average_partition_density",
    "pbmf_index",
    "kwon_index",
    "pcaes_index",
    "fuzzy_rand_index",
    "fuzzy_adjusted_rand_index",
    "fuzzy_jaccard_index",
    "stability_selection",
    "validity_indices",
    "scan_n_clusters",
    "suggest_n_clusters",
    "INDEX_DIRECTION",
]

#: For every index: ``"max"`` if the best partition maximizes it, ``"min"`` otherwise.
INDEX_DIRECTION = {
    "PC": "max",
    "CE": "min",
    "MPC": "max",
    "SC": "min",
    "S": "min",
    "XB": "min",
    "FS": "min",
    "DI": "max",
    "ADI": "max",
    "SIL": "max",
    "FHV": "min",
    "PD": "max",
    "APD": "max",
    "PBMF": "max",
    "KWON": "min",
    "PCAES": "max",
}


def _prep(X, U, V):
    X = np.asarray(X, dtype=float)
    U = check_membership(U, X.shape[0])
    V = np.asarray(V, dtype=float)
    return X, U, V


# ----------------------------------------------------------- partition-only indices
def partition_coefficient(X, U, V=None, m=2.0) -> float:
    r"""Partition coefficient (Bezdek).

    .. math:: PC = \frac{1}{N}\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2

    Measures the amount of overlap between clusters; :math:`PC=1` for a hard
    partition and :math:`1/c` for the fuzziest one. Higher is better. Its
    drawback is the monotone dependence on :math:`c` and the lack of a direct
    connection to the data.
    """
    U = check_membership(U)
    return float(np.sum(U**2) / U.shape[0])


def classification_entropy(X, U, V=None, m=2.0) -> float:
    r"""Classification (partition) entropy (Bezdek).

    .. math:: CE = -\frac{1}{N}\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}\log\mu_{ik}

    Zero for a hard partition, :math:`\log c` for the fuzziest one. Lower is better.
    """
    U = check_membership(U)
    return float(-np.sum(U * np.log(U + EPS)) / U.shape[0])


def modified_partition_coefficient(X, U, V=None, m=2.0) -> float:
    r"""Modified partition coefficient (Davé, 1996).

    .. math:: MPC = 1 - \frac{c}{c-1}\left(1 - PC\right)

    Rescales PC to :math:`[0,1]` to remove its monotone trend in :math:`c`.
    Higher is better.
    """
    U = check_membership(U)
    c = U.shape[1]
    if c < 2:
        return 1.0
    return float(1.0 - c / (c - 1.0) * (1.0 - partition_coefficient(X, U)))


# ------------------------------------------------------ compactness / separation
def partition_index(X, U, V, m=2.0) -> float:
    r"""Partition index SC (Bensaid et al., 1996).

    .. math::
        SC = \sum_{i=1}^{c}\frac{\sum_{k=1}^{N}\mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|^2}
                                {N_i\sum_{j=1}^{c}\|\mathbf{v}_j-\mathbf{v}_i\|^2},
        \qquad N_i=\sum_{k=1}^{N}\mu_{ik}

    Ratio of compactness to separation summed over clusters, normalized by
    the fuzzy cardinality :math:`N_i`. Lower is better; useful to compare
    partitions with the same number of clusters.
    """
    X, U, V = _prep(X, U, V)
    if V.shape[0] < 2:
        return float("nan")  # separation of prototypes is undefined for one cluster
    d2 = sq_euclidean(X, V)
    comp = np.sum(U**m * d2, axis=0)
    card = U.sum(axis=0)
    sep = sq_euclidean(V, V).sum(axis=1)
    return float(np.sum(comp / (card * sep + EPS)))


def separation_index(X, U, V, m=2.0) -> float:
    r"""Separation index S: Xie--Beni in its original form, with squared memberships (Xie and Beni, 1991).

    .. math::
        S = \frac{\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2\|\mathbf{x}_k-\mathbf{v}_i\|^2}
                 {N\min_{i\neq j}\|\mathbf{v}_j-\mathbf{v}_i\|^2}

    Like SC but with a minimum-distance separation term. Lower is better.
    """
    X, U, V = _prep(X, U, V)
    if V.shape[0] < 2:
        return float("nan")  # separation of prototypes is undefined for one cluster
    d2 = sq_euclidean(X, V)
    comp = np.sum(U**2 * d2)
    vv = sq_euclidean(V, V)
    np.fill_diagonal(vv, np.inf)
    return float(comp / (X.shape[0] * vv.min() + EPS))


def xie_beni(X, U, V, m=2.0) -> float:
    r"""Xie–Beni index (Xie & Beni, 1991).

    .. math::
        XB = \frac{\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|^2}
                  {N\min_{i\neq j}\|\mathbf{v}_j-\mathbf{v}_i\|^2}

    Total within-cluster variation over the minimal separation of
    prototypes. Lower is better; the index tends to decrease monotonically for
    large :math:`c`, so it is usually inspected for the first pronounced minimum.
    """
    X, U, V = _prep(X, U, V)
    if V.shape[0] < 2:
        return float("nan")  # separation of prototypes is undefined for one cluster
    d2 = sq_euclidean(X, V)
    comp = np.sum(U**m * d2)
    vv = sq_euclidean(V, V)
    np.fill_diagonal(vv, np.inf)
    return float(comp / (X.shape[0] * vv.min() + EPS))


def fukuyama_sugeno(X, U, V, m=2.0) -> float:
    r"""Fukuyama–Sugeno index (1989).

    .. math::
        FS = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\left(\|\mathbf{x}_k-\mathbf{v}_i\|^2
             - \|\mathbf{v}_i-\bar{\mathbf{x}}\|^2\right)

    Compactness minus separation from the grand mean :math:`\bar{\mathbf{x}}`.
    Lower is better.
    """
    X, U, V = _prep(X, U, V)
    d2 = sq_euclidean(X, V)
    vbar = np.sum((V - X.mean(axis=0)) ** 2, axis=1)
    return float(np.sum(U**m * (d2 - vbar[None, :])))


# ------------------------------------------------------------- crisp indices
def _crisp_groups(X, U):
    labels = harden(U)
    return labels, [X[labels == i] for i in range(U.shape[1])]


def dunn_index(X, U, V=None, m=2.0) -> float:
    r"""Dunn's index (1974) on the hardened partition.

    .. math::
        DI = \min_{i}\min_{j\neq i}\left\{
             \frac{\min_{\mathbf{x}\in C_i,\mathbf{y}\in C_j} d(\mathbf{x},\mathbf{y})}
                  {\max_{l}\max_{\mathbf{x},\mathbf{y}\in C_l} d(\mathbf{x},\mathbf{y})}\right\}

    Smallest inter-cluster distance over the largest cluster diameter. Higher
    is better. Designed for compact, well separated clusters; unreliable for
    overlapping ones and :math:`O(N^2)` to compute.
    """
    X, U, _ = _prep(X, U, np.zeros((U.shape[1], X.shape[1])))
    _, groups = _crisp_groups(X, U)
    c = len(groups)
    diam = 0.0
    for g in groups:
        if g.shape[0] > 1:
            diam = max(diam, cdist(g, g).max())
    sep = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            if groups[i].size and groups[j].size:
                sep = min(sep, cdist(groups[i], groups[j]).min())
    if not np.isfinite(sep) or diam == 0.0:
        return float("nan")
    return float(sep / diam)


def alternative_dunn_index(X, U, V, m=2.0) -> float:
    r"""Alternative Dunn index (ADI) of the MATLAB toolbox.

    The inter-cluster distance :math:`\min d(\mathbf{x},\mathbf{y})` of Dunn's
    index is bounded from below via the triangle inequality
    :math:`d(\mathbf{x},\mathbf{y})\ge|d(\mathbf{y},\mathbf{v}_j)-d(\mathbf{x},\mathbf{v}_j)|`,
    which is much cheaper to evaluate:

    .. math::
        ADI = \min_{i}\min_{j\neq i}\left\{
              \frac{\min_{\mathbf{x}\in C_i,\mathbf{y}\in C_j}
                    |d(\mathbf{y},\mathbf{v}_j)-d(\mathbf{x},\mathbf{v}_j)|}
                   {\max_{l}\max_{\mathbf{x},\mathbf{y}\in C_l} d(\mathbf{x},\mathbf{y})}\right\}.

    Higher is better.
    """
    X, U, V = _prep(X, U, V)
    _, groups = _crisp_groups(X, U)
    c = len(groups)
    diam = 0.0
    for g in groups:
        if g.shape[0] > 1:
            diam = max(diam, cdist(g, g).max())
    sep = np.inf
    for i in range(c):
        for j in range(c):
            if i == j or not groups[i].size or not groups[j].size:
                continue
            dj = np.linalg.norm(groups[j] - V[j], axis=1)
            di = np.linalg.norm(groups[i] - V[j], axis=1)
            sep = min(sep, np.abs(dj[:, None] - di[None, :]).min())
    if not np.isfinite(sep) or diam == 0.0:
        return float("nan")
    return float(sep / diam)


def silhouette(X, U, V=None, m=2.0) -> float:
    """Mean silhouette coefficient (Rousseeuw, 1987) of the hardened partition."""
    from sklearn.metrics import silhouette_score

    X = np.asarray(X, dtype=float)
    labels = harden(check_membership(U, X.shape[0]))
    if np.unique(labels).size < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


# --------------------------------------------------------------------------- #
# Density-based indices of Gath and Geva (1989)
# --------------------------------------------------------------------------- #
def _gg_prep(X, U, V, m):
    X, U, V = _prep(X, U, V)
    if V is None or V.ndim == 0:
        V = fuzzy_centers(X, U**m)
    F = fuzzy_covariances(X, U**m, V)
    return X, U, V, F


def _central_members(X, U, V, F):
    """Sum of memberships of the samples inside the unit Mahalanobis ellipsoid of each cluster."""
    S = np.empty(V.shape[0])
    for i in range(V.shape[0]):
        xv = X - V[i]
        maha = np.einsum("kj,jl,kl->k", xv, np.linalg.pinv(F[i]), xv)
        S[i] = U[maha < 1.0, i].sum()
    return S


def fuzzy_hypervolume(X, U, V, m=2.0) -> float:
    r"""Fuzzy hypervolume :math:`FHV = \sum_i \sqrt{\det\mathbf{F}_i}` (Gath & Geva, 1989). Min is best.

    Measures the total volume of the clusters; compact clusters give small
    values. It is the natural companion of the GG algorithm because it uses
    the same fuzzy covariance matrices.
    """
    X, U, V, F = _gg_prep(X, U, V, m)
    return float(sum(np.sqrt(max(np.linalg.det(Fi), 0.0)) for Fi in F))


def partition_density(X, U, V, m=2.0) -> float:
    r"""Partition density :math:`PD = \sum_i S_i / FHV` (Gath & Geva, 1989). Max is best.

    :math:`S_i=\sum_{k:\,(\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{F}_i^{-1}(\mathbf{x}_k-\mathbf{v}_i)<1}\mu_{ik}`
    is the membership mass inside the unit Mahalanobis ellipsoid of cluster :math:`i`.
    """
    X, U, V, F = _gg_prep(X, U, V, m)
    fhv = sum(np.sqrt(max(np.linalg.det(Fi), 0.0)) for Fi in F)
    if fhv <= 0:
        return float("inf")
    return float(_central_members(X, U, V, F).sum() / fhv)


def average_partition_density(X, U, V, m=2.0) -> float:
    r"""Average partition density (Gath & Geva). Max is best.

    .. math:: APD = \frac1c\sum_i S_i/\sqrt{\det\mathbf{F}_i}
    """
    X, U, V, F = _gg_prep(X, U, V, m)
    S = _central_members(X, U, V, F)
    vols = np.array([np.sqrt(max(np.linalg.det(Fi), EPS)) for Fi in F])
    return float(np.mean(S / vols))


def pbmf_index(X, U, V, m=2.0) -> float:
    r"""PBM-F index (Pakhira, Bandyopadhyay & Maulik, 2004). Max is best.

    .. math::
        PBMF = \left(\frac{1}{c}\cdot\frac{E_1}{J_m}\cdot D_c\right)^2,\quad
        E_1=\sum_k\|\mathbf{x}_k-\bar{\mathbf{x}}\|,\;
        J_m=\sum_{i,k}\mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|,\;
        D_c=\max_{i,j}\|\mathbf{v}_i-\mathbf{v}_j\|
    """
    X, U, V = _prep(X, U, V)
    c = V.shape[0]
    E1 = np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=1)).sum()
    Jm = np.sum(U**m * np.sqrt(sq_euclidean(X, V)))
    Dc = np.sqrt(sq_euclidean(V, V)).max() if c > 1 else 0.0
    return float((E1 / (c * max(Jm, EPS)) * Dc) ** 2)


def kwon_index(X, U, V, m=2.0) -> float:
    r"""Kwon's index (1998). Min is best.

    .. math::
        K=\frac{\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2\|\mathbf{x}_k-\mathbf{v}_i\|^2
                 +\frac1c\sum_{i=1}^{c}\|\mathbf{v}_i-\bar{\mathbf{x}}\|^2}
                {\min_{i\neq j}\|\mathbf{v}_i-\mathbf{v}_j\|^2}

    The Xie--Beni ratio with a punishing term in the numerator, the mean
    squared distance of the prototypes from the grand mean
    :math:`\bar{\mathbf{x}}`, which grows when prototypes are added and so
    removes the monotone decrease of XB for large :math:`c`.
    """
    X, U, V = _prep(X, U, V)
    c = V.shape[0]
    if c < 2:
        return float("nan")  # separation of prototypes is undefined for one cluster
    comp = np.sum(U**2 * sq_euclidean(X, V))
    punish = np.sum((V - X.mean(axis=0)) ** 2) / c
    vv = sq_euclidean(V, V)
    np.fill_diagonal(vv, np.inf)
    return float((comp + punish) / (vv.min() + EPS))


def pcaes_index(X, U, V, m=2.0) -> float:
    r"""Partition coefficient and exponential separation, PCAES (Wu & Yang, 2005). Max is best.

    .. math::
        PCAES=\sum_{i=1}^{c}\frac{\sum_{k=1}^{N}\mu_{ik}^2}{\mu_M}
              -\sum_{i=1}^{c}\exp\Bigl(-\frac{\min_{j\neq i}\|\mathbf{v}_i-\mathbf{v}_j\|^2}{\beta_T}\Bigr),
        \quad \mu_M=\min_i\sum_k\mu_{ik}^2,\ \
        \beta_T=\frac1c\sum_{i=1}^{c}\|\mathbf{v}_i-\bar{\mathbf{x}}\|^2 .

    The first sum is a partition coefficient normalized per cluster by the
    least compact cluster (a cluster-wise measure of how well the members
    are covered), the second penalizes every prototype by how close its
    nearest neighbor is on the scale :math:`\beta_T` of the spread of the
    prototypes; a cluster with a neighbor much closer than
    :math:`\sqrt{\beta_T}` costs nearly one unit. Since the first sum is
    at least :math:`c` and the penalty at most :math:`c`, the index is
    nonnegative. Because :math:`\mu_M` is the mass of the *smallest*
    cluster (as published), a spurious tiny cluster inflates the index;
    inspect the scan curve rather than only its maximum.
    """
    X, U, V = _prep(X, U, V)
    c = V.shape[0]
    if c < 2:
        return float("nan")  # separation of prototypes is undefined for one cluster
    mass = np.sum(U**2, axis=0)
    mu_M = max(mass.min(), EPS)
    beta_T = max(np.sum((V - X.mean(axis=0)) ** 2) / c, EPS)
    vv = sq_euclidean(V, V)
    np.fill_diagonal(vv, np.inf)
    return float(np.sum(mass / mu_M) - np.sum(np.exp(-vv.min(axis=1) / beta_T)))


_INDEX_FUNCS = {
    "PC": partition_coefficient,
    "CE": classification_entropy,
    "MPC": modified_partition_coefficient,
    "SC": partition_index,
    "S": separation_index,
    "XB": xie_beni,
    "FS": fukuyama_sugeno,
    "DI": dunn_index,
    "ADI": alternative_dunn_index,
    "SIL": silhouette,
    "FHV": fuzzy_hypervolume,
    "PD": partition_density,
    "APD": average_partition_density,
    "PBMF": pbmf_index,
    "KWON": kwon_index,
    "PCAES": pcaes_index,
}


# --------------------------------------------------------------------------- #
# Comparing two fuzzy partitions
# --------------------------------------------------------------------------- #
_TNORMS = {
    "product": lambda a, b: a * b,
    "minimum": np.minimum,
    "lukasiewicz": lambda a, b: np.maximum(a + b - 1.0, 0.0),
}


def _pair_mask(N):
    return np.triu(np.ones((N, N), dtype=bool), k=1)


def _equivalence_huellermeier(U):
    """Degree to which two samples share a cluster: 1 - L1/2 distance of their membership rows."""
    U = np.asarray(U, dtype=float)
    d = 0.5 * np.abs(U[:, None, :] - U[None, :, :]).sum(axis=2)
    return 1.0 - np.clip(d, 0.0, 1.0)


def _agreement_campello(U, tnorm):
    r"""Campello's degrees that a pair is in the same (V) or in different (X) clusters."""
    U = np.asarray(U, dtype=float)
    N, c = U.shape
    T = _TNORMS[tnorm]
    if tnorm == "product":
        V = U @ U.T
        Xd = np.outer(U.sum(axis=1), U.sum(axis=1)) - V
    else:
        V = np.zeros((N, N))
        Xd = np.zeros((N, N))
        for i in range(c):
            for j in range(c):
                t = T(U[:, i][:, None], U[:, j][None, :])
                if i == j:
                    V += t
                else:
                    Xd += t
    return np.clip(V, 0.0, 1.0), np.clip(Xd, 0.0, 1.0)


def _pair_counts(U1, U2, method="huellermeier", tnorm=None):
    """Fuzzy pair counts a, b, c, d: together in both, only in the first, only in the second, in neither."""
    U1 = check_membership(U1)
    U2 = check_membership(U2, U1.shape[0])
    if U1.shape[0] < 2:
        raise ValueError("At least two samples are needed to compare partitions.")
    mask = _pair_mask(U1.shape[0])
    if method == "huellermeier":
        T = _TNORMS["lukasiewicz" if tnorm is None else tnorm]
        E1, E2 = _equivalence_huellermeier(U1), _equivalence_huellermeier(U2)
        a = T(E1, E2)[mask].sum()
        b = T(E1, 1.0 - E2)[mask].sum()
        c = T(1.0 - E1, E2)[mask].sum()
        d = T(1.0 - E1, 1.0 - E2)[mask].sum()
    elif method == "campello":
        tn = "product" if tnorm is None else tnorm
        T = _TNORMS[tn]
        V1, X1 = _agreement_campello(U1, tn)
        V2, X2 = _agreement_campello(U2, tn)
        a = T(V1, V2)[mask].sum()
        b = T(V1, X2)[mask].sum()
        c = T(X1, V2)[mask].sum()
        d = T(X1, X2)[mask].sum()
    else:
        raise ValueError("method must be 'huellermeier' or 'campello'.")
    return float(a), float(b), float(c), float(d)


def fuzzy_rand_index(U1, U2, method: str = "huellermeier", tnorm: str | None = None) -> float:
    r"""Fuzzy Rand index between two fuzzy partitions of the same samples.

    The Rand index counts the pairs of samples on which two partitions
    agree---together in both or apart in both---over all pairs. Two fuzzy
    generalizations are implemented.

    * ``"huellermeier"`` (Hüllermeier, Rifqi, Henzgen & Senge, 2012): every
      partition defines a fuzzy equivalence relation
      :math:`E_U(k,l)=1-\tfrac12\sum_i|\mu_{ik}-\mu_{il}|` (the degree to
      which samples :math:`k` and :math:`l` are in the same cluster), and
      the index is one minus the normalized distance between the two
      relations,

      .. math::
          R_E(\mathbf{U}_1,\mathbf{U}_2)=1-\frac{\sum_{k<l}|E_1(k,l)-E_2(k,l)|}{N(N-1)/2} .

      This distance-based form is the definition of the paper and is what
      ``tnorm=None`` computes. Passing a t-norm instead evaluates the
      count-based form :math:`(a+d)/(a+b+c+d)` with
      :math:`a=\sum\top(E_1,E_2)`, :math:`b=\sum\top(E_1,1-E_2)`, etc.;
      for soft partitions the two forms differ (the four degrees of a pair
      need not sum to one), for hard partitions both equal the classical
      Rand index. The adjusted Rand and Jaccard variants use the counts
      (Łukasiewicz t-norm by default).
    * ``"campello"`` (Campello, 2007): for every pair the degrees of being
      in the same cluster, :math:`V_{kl}=\sum_i\top(\mu_{ik},\mu_{il})`,
      and in different clusters, :math:`X_{kl}=\sum_{i\neq j}\top(\mu_{ik},\mu_{jl})`,
      are formed with a t-norm :math:`\top` for both partitions, and the
      four counts are :math:`a=\sum_{k<l}\top(V^{(1)}_{kl},V^{(2)}_{kl})`,
      :math:`b=\sum\top(V^{(1)},X^{(2)})`, :math:`c=\sum\top(X^{(1)},V^{(2)})`,
      :math:`d=\sum\top(X^{(1)},X^{(2)})`; the Rand index is
      :math:`(a+d)/(a+b+c+d)`.

    Both reduce to the classical Rand index for hard partitions.

    Parameters
    ----------
    U1, U2 : array-like of shape (n_samples, n_clusters_1) and (n_samples, n_clusters_2)
        Membership matrices (rows sum to one); hard partitions as one-hot
        matrices (:func:`fuzzyclust._utils.labels_to_membership`).
    method : {"huellermeier", "campello"}, default="huellermeier"
    tnorm : {"product", "minimum", "lukasiewicz"} or None, default=None
        t-norm; ``None`` gives the distance-based index for
        ``"huellermeier"`` and the product t-norm for ``"campello"``.

    Returns
    -------
    float in [0, 1]

    References
    ----------
    .. [1] E. Hüllermeier, M. Rifqi, S. Henzgen, R. Senge, "Comparing fuzzy
           partitions: a generalization of the Rand index and related
           measures", IEEE Trans. Fuzzy Systems 20(3), 546--556, 2012.
    .. [2] R. J. G. B. Campello, "A fuzzy extension of the Rand index and
           other related indexes for clustering and classification
           assessment", Pattern Recognition Letters 28(7), 833--841, 2007.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM, GK
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.validity import fuzzy_rand_index
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> U1 = FCM(n_clusters=3, random_state=0).fit(X).membership_
    >>> U2 = GK(n_clusters=3, random_state=0).fit(X).membership_
    >>> round(fuzzy_rand_index(U1, U2), 2) > 0.9
    True
    """
    if method == "huellermeier" and tnorm is None:
        U1 = check_membership(U1)
        U2 = check_membership(U2, U1.shape[0])
        if U1.shape[0] < 2:
            raise ValueError("At least two samples are needed to compare partitions.")
        mask = _pair_mask(U1.shape[0])
        diff = np.abs(_equivalence_huellermeier(U1) - _equivalence_huellermeier(U2))[mask]
        return float(1.0 - diff.mean())
    a, b, c, d = _pair_counts(U1, U2, method, tnorm)
    return float((a + d) / max(a + b + c + d, EPS))


def fuzzy_adjusted_rand_index(U1, U2, method: str = "huellermeier", tnorm: str | None = None) -> float:
    r"""Fuzzy adjusted Rand index (chance-corrected), from the fuzzy pair counts of :func:`fuzzy_rand_index`.

    .. math::
        ARI=\frac{a-\frac{(a+b)(a+c)}{a+b+c+d}}{\frac12\bigl((a+b)+(a+c)\bigr)-\frac{(a+b)(a+c)}{a+b+c+d}}

    (Hubert and Arabie's correction applied to the fuzzy pair counts). Zero
    in expectation for independent partitions, one for identical ones;
    coincides with ``sklearn.metrics.adjusted_rand_score`` for hard partitions.
    """
    a, b, c, d = _pair_counts(U1, U2, method, tnorm)
    n = a + b + c + d
    expected = (a + b) * (a + c) / max(n, EPS)
    maximum = 0.5 * ((a + b) + (a + c))
    if abs(maximum - expected) < EPS:
        return 1.0 if abs(a - expected) < EPS else 0.0
    return float((a - expected) / (maximum - expected))


def fuzzy_jaccard_index(U1, U2, method: str = "huellermeier", tnorm: str | None = None) -> float:
    r"""Fuzzy Jaccard index :math:`a/(a+b+c)` from the fuzzy pair counts of :func:`fuzzy_rand_index`.

    Ignores the pairs that are apart in both partitions, which dominate the
    Rand index when the number of clusters is large.
    """
    a, b, c, _ = _pair_counts(U1, U2, method, tnorm)
    return float(a / max(a + b + c, EPS))


_STABILITY_INDICES = {
    "fuzzy_ari": fuzzy_adjusted_rand_index,
    "fuzzy_rand": fuzzy_rand_index,
    "fuzzy_jaccard": fuzzy_jaccard_index,
}


def _seeded(estimator, c, rng):
    """Clone ``estimator`` with ``n_clusters=c`` and, if it has one, a seed drawn from ``rng``."""
    est = clone(estimator).set_params(n_clusters=c)
    if "random_state" in est.get_params() and est.get_params()["random_state"] is None:
        est.set_params(random_state=int(rng.randint(np.iinfo(np.int32).max)))
    return est


def stability_selection(
    estimator,
    X,
    n_clusters_range: Iterable[int] = range(2, 11),
    n_splits: int = 20,
    method: str = "subsample",
    subsample: float = 0.8,
    index: str = "fuzzy_ari",
    random_state=None,
) -> pd.DataFrame:
    r"""Stability of the partition against resampling, for every number of clusters.

    A number of clusters that reflects structure in the data gives the same
    partition on different subsets of the data; a wrong one splits or
    merges groups arbitrarily and the partitions of two subsets disagree.
    Two protocols are implemented.

    * ``"subsample"`` (Ben-Hur, Elisseeff & Guyon, 2002): two random
      sub-samples of a fraction ``subsample`` of the data are clustered
      independently and their partitions compared on the samples they
      share.
    * ``"split"`` (Lange, Roth, Braun & Buhmann, 2004): the data are split
      into two halves, the estimator is fitted on each, the model of the
      first half predicts the memberships of the second half
      (``predict_proba`` acts as the classifier of the original method),
      and the prediction is compared with the second half's own partition.

    The comparison uses a fuzzy partition index, by default the fuzzy
    adjusted Rand index, which is zero for chance agreement and so needs no
    further normalization (the raw fuzzy Rand index grows with :math:`c`
    because pairs apart in both partitions count as agreement). The
    statistic to inspect is the distribution over splits: a stable
    :math:`c` has a mean near one and a small spread.

    Parameters
    ----------
    estimator : fuzzyclust clusterer
        Cloned and refitted for every value of ``n_clusters`` and every split.
    X : array-like of shape (n_samples, n_features)
    n_clusters_range : iterable of int, default=range(2, 11)
    n_splits : int, default=20
        Number of resampling repetitions per ``n_clusters``.
    method : {"subsample", "split"}, default="subsample"
    subsample : float, default=0.8
        Fraction of the samples in every sub-sample (``"subsample"`` only).
    index : {"fuzzy_ari", "fuzzy_rand", "fuzzy_jaccard"}, default="fuzzy_ari"
    random_state : int, RandomState instance or None

    Returns
    -------
    pandas.DataFrame
        Indexed by ``n_clusters`` with one column per split (``split_0``,
        ...); ``table.mean(axis=1).idxmax()`` is the most stable :math:`c`.

    References
    ----------
    .. [1] A. Ben-Hur, A. Elisseeff, I. Guyon, "A stability based method for
           discovering structure in clustered data", Pacific Symposium on
           Biocomputing 7, 6--17, 2002.
    .. [2] T. Lange, V. Roth, M. L. Braun, J. M. Buhmann, "Stability-based
           validation of clustering solutions", Neural Computation 16(6),
           1299--1323, 2004.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.validity import stability_selection
    >>> X, _ = make_fuzzy_blobs(n_clusters=4, random_state=0)
    >>> table = stability_selection(FCM(random_state=0), X, range(2, 7), n_splits=5, random_state=0)
    >>> int(table.mean(axis=1).idxmax())
    4
    """
    from sklearn.utils import check_random_state

    if method not in ("subsample", "split"):
        raise ValueError("method must be 'subsample' or 'split'.")
    if index not in _STABILITY_INDICES:
        raise ValueError(f"index must be one of {tuple(_STABILITY_INDICES)}.")
    if not 0.0 < subsample < 1.0:
        raise ValueError("subsample must lie in (0, 1).")
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    rng = check_random_state(random_state)
    score = _STABILITY_INDICES[index]
    n_sub = max(int(round(subsample * N)), 2)
    rows = {}
    for c in n_clusters_range:
        vals = []
        for _ in range(n_splits):
            if method == "subsample":
                idx1 = np.sort(rng.choice(N, n_sub, replace=False))
                idx2 = np.sort(rng.choice(N, n_sub, replace=False))
                common = np.intersect1d(idx1, idx2)
                est1 = _seeded(estimator, int(c), rng).fit(X[idx1])
                est2 = _seeded(estimator, int(c), rng).fit(X[idx2])
                U1 = est1.predict_proba(X[common])
                U2 = est2.predict_proba(X[common])
            else:
                perm = rng.permutation(N)
                half1, half2 = perm[: N // 2], perm[N // 2 :]
                est1 = _seeded(estimator, int(c), rng).fit(X[half1])
                est2 = _seeded(estimator, int(c), rng).fit(X[half2])
                U1 = est1.predict_proba(X[half2])
                U2 = est2.membership_
            vals.append(score(U1, U2))
        rows[int(c)] = vals
    df = pd.DataFrame.from_dict(rows, orient="index", columns=[f"split_{s}" for s in range(n_splits)])
    df.index.name = "n_clusters"
    return df


def validity_indices(X, U, V, m: float = 2.0, indices: Iterable[str] | None = None) -> dict:
    """Evaluate several validity indices at once.

    Parameters
    ----------
    X, U, V, m
        See module docstring.
    indices : iterable of str, optional
        Subset of ``{"PC","CE","MPC","SC","S","XB","FS","DI","ADI","SIL","FHV","PD","APD","PBMF",
        "KWON","PCAES"}``. Default: all.

    Returns
    -------
    dict
        Index name -> value.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.validity import validity_indices
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> fcm = FCM(n_clusters=3, random_state=0).fit(X)
    >>> res = validity_indices(X, fcm.membership_, fcm.cluster_centers_, m=fcm.m)
    >>> sorted(res)[:3]
    ['ADI', 'APD', 'CE']
    """
    names = list(indices) if indices is not None else list(_INDEX_FUNCS)
    return {name: _INDEX_FUNCS[name](X, U, V, m) for name in names}


def scan_n_clusters(
    estimator,
    X,
    n_clusters_range: Iterable[int] = range(2, 11),
    indices: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fit the estimator for several numbers of clusters and tabulate validity indices.

    Parameters
    ----------
    estimator : fitted or unfitted fuzzyclust estimator
        Cloned and refitted for every value of ``n_clusters``.
    X : array-like of shape (n_samples, n_features)
    n_clusters_range : iterable of int, default=range(2, 11)
    indices : iterable of str, optional

    Returns
    -------
    pandas.DataFrame
        Indexed by ``n_clusters`` with one column per index.

    Examples
    --------
    >>> from fuzzyclust.cluster import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.validity import scan_n_clusters
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> table = scan_n_clusters(FCM(random_state=0), X, range(2, 6), indices=["PC", "XB"])
    >>> list(table.columns)
    ['PC', 'XB']
    """
    rows = {}
    for c in n_clusters_range:
        est = clone(estimator).set_params(n_clusters=int(c)).fit(X)
        m = getattr(est, "m", 2.0)
        rows[int(c)] = validity_indices(X, est.membership_, est.cluster_centers_, m, indices)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "n_clusters"
    return df


def suggest_n_clusters(table: pd.DataFrame) -> pd.Series:
    """Optimal number of clusters suggested by every index of a scan table.

    Uses :data:`INDEX_DIRECTION` to pick the arg-max or arg-min of each column.
    PC and CE decrease/increase monotonically with :math:`c`, so for them the
    knee rather than the extremum is often the better choice; the raw
    suggestion is still reported for completeness.
    """
    out = {}
    for col in table.columns:
        s = table[col].dropna()
        if s.empty:
            out[col] = np.nan
            continue
        out[col] = s.idxmax() if INDEX_DIRECTION.get(col, "min") == "max" else s.idxmin()
    return pd.Series(out, name="suggested_n_clusters")
