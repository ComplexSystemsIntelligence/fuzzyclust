# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Relational fuzzy c-means (RFCM) and its non-Euclidean extension (NERF c-means).

Both algorithms minimize the fuzzy c-means functional although only the
pairwise dissimilarities :math:`\mathbf{R}=[r_{jk}]` of the objects are
known: the prototypes are convex combinations of the objects
(:math:`\mathbf{v}_i=\mathbf{U}_i^m/\sum_k\mu_{ik}^m`) and the distance of an
object to such a prototype follows from :math:`\mathbf{R}` alone when
:math:`\mathbf{R}` holds squared Euclidean distances (Hathaway, Davenport
and Bezdek, 1989). When it does not, the relational distances can become
negative; the :math:`\beta`-spread transformation of Hathaway and Bezdek
(1994) adds a constant to all off-diagonal dissimilarities until they are
not.
"""

from __future__ import annotations

import numpy as np

from .._utils import EPS, memberships_from_distances
from ..base import BaseFuzzyClusterer
from .dissimilarity import check_dissimilarity_matrix, pairwise_dissimilarity

__all__ = ["RelationalFCM", "NERFCM", "relational_distances"]


def relational_distances(R, V, beta: float = 0.0) -> np.ndarray:
    r"""Squared relational distances of the training objects to relational prototypes.

    .. math::
        d^2_{ik}=(\mathbf{R}\mathbf{v}_i)_k-\tfrac12\,\mathbf{v}_i\T\mathbf{R}\mathbf{v}_i
                 +\tfrac{\beta}{2}\norm{\mathbf{e}_k-\mathbf{v}_i}^2 ,

    where :math:`\mathbf{v}_i\in\mathbb{R}^N` sums to one and
    :math:`\mathbf{e}_k` is the :math:`k`-th unit vector. The last term is
    the effect of the :math:`\beta`-spread
    :math:`\mathbf{R}_\beta=\mathbf{R}+\beta(\mathbf{1}\mathbf{1}\T-\mathbf{I})`
    (Hathaway and Bezdek, 1994), computed without forming :math:`\mathbf{R}_\beta`.

    Parameters
    ----------
    R : ndarray of shape (n_samples, n_samples)
        Square dissimilarity matrix of the objects.
    V : ndarray of shape (n_clusters, n_samples)
        Relational prototypes (rows sum to one).
    beta : float, default=0.0

    Returns
    -------
    d2 : ndarray of shape (n_samples, n_clusters)

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.relational import relational_distances
    >>> R = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])  # points 0, 1, 2 on a line
    >>> V = np.array([[0.5, 0.5, 0.0]])  # prototype at 0.5
    >>> relational_distances(R, V).ravel()
    array([0.25, 0.25, 2.25])
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    if R.shape[0] != R.shape[1] or V.shape[1] != R.shape[0]:
        raise ValueError("R must be square (n_samples, n_samples) and V of shape (n_clusters, n_samples).")
    vRv = np.einsum("ij,jk,ik->i", V, R, V)
    d2 = R @ V.T - 0.5 * vRv[None, :]
    if beta > 0:
        d2 = d2 + 0.5 * beta * _unit_minus_v_sq(V, R.shape[0], same_objects=True)
    return d2


def _unit_minus_v_sq(V: np.ndarray, n_samples: int, same_objects: bool) -> np.ndarray:
    r"""``||e_k - v_i||^2`` for every sample k and prototype i, shape (n_samples, c).

    For the training objects :math:`\norm{\mathbf{e}_k-\mathbf{v}_i}^2=1-2v_{ik}+\norm{\mathbf{v}_i}^2`;
    a new object is orthogonal to the training simplex, so the term is
    :math:`1+\norm{\mathbf{v}_i}^2`.
    """
    v2 = np.einsum("ij,ij->i", V, V)[None, :]
    if same_objects:
        return 1.0 - 2.0 * V.T + v2
    return np.broadcast_to(1.0 + v2, (n_samples, V.shape[0])).copy()


class RelationalFCM(BaseFuzzyClusterer):
    r"""Relational fuzzy c-means (Hathaway, Davenport and Bezdek, 1989).

    Fuzzy c-means needs feature vectors to compute prototypes and distances.
    When only the pairwise dissimilarities :math:`\mathbf{R}=[r_{jk}]` of the
    :math:`N` objects are available (a distance matrix, a dynamic-time-warping
    matrix, a dissimilarity between graphs or documents), the *relational
    dual* of FCM still minimizes the same functional. Its prototypes are
    weight vectors over the objects, :math:`\mathbf{v}_i\in\mathbb{R}^N`, and
    the squared distance of object :math:`k` to prototype :math:`i` is
    expressed through :math:`\mathbf{R}` alone. If :math:`\mathbf{R}`
    consists of squared Euclidean distances of some feature vectors, RFCM
    produces exactly the FCM partition of those vectors (the duality
    theorem of the paper); if not, the relational distances may become
    negative and the algorithm is undefined -- then use :class:`NERFCM`.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    metric : str, default="euclidean"
        How ``fit(X)`` obtains :math:`\mathbf{R}`. ``"precomputed"``: ``X``
        *is* the :math:`N\times N` dissimilarity matrix (symmetric, zero
        diagonal, non-negative), whose entries play the role of **squared**
        distances :math:`r_{jk}=d^2(\mathbf{x}_j,\mathbf{x}_k)`. Any other
        value is a metric name of :func:`scipy.spatial.distance.pdist`; the
        pairwise distances of the rows of ``X`` are computed and squared
        (``"euclidean"`` therefore reproduces FCM).
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        ``"random"`` draws a random partition :math:`\mathbf{U}` (the paper
        starts from an arbitrary partition); ``"kmeans++"`` seeds :math:`c`
        objects by :math:`D^2`-weighted sampling on :math:`\mathbf{R}` and
        derives :math:`\mathbf{U}` from the dissimilarities to them; an
        array of shape ``(n_samples, n_clusters)`` is an initial partition,
        an integer array of length ``n_clusters`` gives initial objects.
    n_init : int, default=1
        Number of random restarts; the run with the lowest :math:`J` is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    relational_centers_ : ndarray of shape (n_clusters, n_samples)
        Relational prototypes :math:`\mathbf{v}_i=\mathbf{U}_i^m/\sum_k\mu_{ik}^m`
        (rows sum to one): the weight of every training object in cluster
        :math:`i`.
    cluster_centers_ : ndarray of shape (n_clusters, n_features) or (n_clusters, n_samples)
        With a feature-based ``metric`` the dual centers in feature space,
        :math:`\sum_k v_{ik}\mathbf{x}_k` (equal to the FCM centers for the
        Euclidean metric); with ``metric="precomputed"`` the same as
        ``relational_centers_``, since no feature space exists.
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        The matrix :math:`\mathbf{R}` the algorithm ran on (squared
        distances when computed from features).
    membership_, labels_, distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_``
        is :math:`\sqrt{d^2_{ik}}` of the relational distances.
    beta_ : float
        Always ``0.0`` for RFCM (kept for a uniform interface with :class:`NERFCM`).

    Notes
    -----
    RFCM minimizes the c-means functional :math:`J=\sum_i\sum_k\mu_{ik}^m d^2_{ik}`
    by alternating

    .. math::
        \mathbf{v}_i=\frac{(\mu_{i1}^m,\dots,\mu_{iN}^m)\T}{\sum_{k=1}^{N}\mu_{ik}^m},\qquad
        d^2_{ik}=(\mathbf{R}\mathbf{v}_i)_k-\tfrac12\,\mathbf{v}_i\T\mathbf{R}\mathbf{v}_i ,

    and the FCM membership update
    :math:`\mu_{ik}=1/\sum_j(d^2_{ik}/d^2_{jk})^{1/(m-1)}`. When
    :math:`r_{jk}=\norm{\mathbf{x}_j-\mathbf{x}_k}^2`, the identity
    :math:`\norm{\mathbf{x}_k-\sum_l v_{il}\mathbf{x}_l}^2=\sum_l v_{il}r_{kl}
    -\tfrac12\sum_{l,q}v_{il}v_{iq}r_{lq}` (valid for any weights summing to one) makes :math:`d^2_{ik}` the
    squared Euclidean distance to the FCM center, so RFCM and FCM generate
    identical iterates from identical starts. The identity holds for *any*
    object whose squared distances to the :math:`N` training objects are
    known, which is how :meth:`predict_proba` assigns new objects: pass
    their ``(n_new, n_samples)`` dissimilarities (with
    ``metric="precomputed"``) or their feature vectors.

    A matrix that is not Euclidean -- squared city-block or Chebyshev
    distances, squared dynamic-time-warping distances, dissimilarities that
    violate the triangle inequality -- can produce :math:`d^2_{ik}<0`; RFCM
    then raises an error and recommends :class:`NERFCM`. (Unsquared
    Euclidean distances, or any power :math:`\norm{\cdot}^{p}` with
    :math:`p\le2`, still form a Euclidean matrix by Schoenberg's theorem:
    RFCM runs on them, but it is then the dual of FCM on a *different*
    point configuration, not on :math:`\mathbf{X}`.) Memory and time are
    :math:`O(N^2)`.

    References
    ----------
    .. [1] R. J. Hathaway, J. W. Davenport, J. C. Bezdek, "Relational duals
           of the c-means clustering algorithms", Pattern Recognition 22(2),
           205--212, 1989.
    .. [2] J. C. Bezdek, "Pattern Recognition with Fuzzy Objective Function
           Algorithms", Plenum Press, 1981.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.cluster import FCM
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.relational import RelationalFCM, pairwise_dissimilarity
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> R = pairwise_dissimilarity(X, metric="sqeuclidean")
    >>> U0 = np.random.RandomState(0).rand(X.shape[0], 3)
    >>> rfcm = RelationalFCM(n_clusters=3, metric="precomputed", init=U0).fit(R)
    >>> fcm = FCM(n_clusters=3, init=U0).fit(X)
    >>> bool(np.allclose(rfcm.membership_, fcm.membership_, atol=1e-6))
    True
    """

    _spread = False

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
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
        self.metric = metric

    # ------------------------------------------------------------------ data handling
    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("Relational c-means requires m > 1.")
        if not isinstance(self.metric, str):
            raise ValueError("metric must be a string (a SciPy metric name or 'precomputed').")
        self._prepare_relation(X)

    def _relation(self, X, Y=None):
        """Dissimilarities (as the algorithm uses them: 'squared') between rows of X and Y."""
        if self.metric == "precomputed":
            return X
        D = pairwise_dissimilarity(X, Y, metric=self.metric)
        return D**2

    def _prepare_relation(self, X):
        if self.metric == "precomputed":
            R = check_dissimilarity_matrix(X)
        else:
            R = self._relation(X)
        self._X_fit = X
        self.dissimilarity_matrix_ = R

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        c, N = self.n_clusters, X.shape[0]
        R = self.dissimilarity_matrix_
        self.beta_ = 0.0
        self.n_spread_updates_ = 0
        init = self.init
        if isinstance(init, str):
            if init == "random":
                U = rng.rand(N, c)
                return U / U.sum(axis=1, keepdims=True)
            if init == "kmeans++":
                idx = [int(rng.randint(N))]
                dmin = R[:, idx[0]].copy()
                while len(idx) < c:
                    dmin[idx] = 0.0
                    probs = dmin / dmin.sum() if dmin.sum() > 0 else np.full(N, 1.0 / N)
                    j = int(rng.choice(N, p=probs))
                    idx.append(j)
                    dmin = np.minimum(dmin, R[:, j])
                return memberships_from_distances(R[:, idx], self.m)
            raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
        init = np.asarray(init)
        if init.shape == (N, c):
            U = init.astype(float)
            return U / U.sum(axis=1, keepdims=True)
        if init.shape == (c,) and np.issubdtype(init.dtype, np.integer):
            if init.min() < 0 or init.max() >= N:
                raise ValueError("initial object indices out of range.")
            return memberships_from_distances(R[:, init], self.m)
        raise ValueError(
            "init array must be a partition of shape (n_samples, n_clusters) or n_clusters object indices."
        )

    def _update_prototypes(self, X, Um):
        s = Um.sum(axis=0)
        s = np.where(s == 0, EPS, s)
        V = (Um / s[None, :]).T  # (c, N), rows sum to one
        self.relational_centers_ = V
        R = self.dissimilarity_matrix_
        self._vRv = np.einsum("ij,jk,ik->i", V, R, V)
        if self.metric == "precomputed":
            self.cluster_centers_ = V
        else:
            self.cluster_centers_ = V @ X

    def _distances(self, X):
        V = self.relational_centers_
        training = X is self._X_fit
        if training:
            R = self.dissimilarity_matrix_
        else:
            R = self._relation(X, self._X_fit)
            if self.metric == "precomputed":
                R = check_dissimilarity_matrix(R, n_objects=V.shape[1], square=False)
        d2 = R @ V.T - 0.5 * self._vRv[None, :]
        e_v = _unit_minus_v_sq(V, R.shape[0], same_objects=training)
        if self.beta_ > 0:
            d2 = d2 + 0.5 * self.beta_ * e_v
        if training and self._spread:
            d2 = self._spread_if_needed(d2, e_v)
        if np.any(d2 < -1e-9 * max(1.0, float(np.abs(R).max()))):
            raise ValueError(
                "Negative relational distance: the dissimilarity matrix is not (squared) Euclidean, "
                "so RFCM is undefined on it. Use NERFCM, which applies the beta-spread transformation."
            )
        return np.maximum(d2, 0.0)

    def _spread_if_needed(self, d2, e_v):  # pragma: no cover - overridden in NERFCM
        return d2

    def _memberships(self, d2):
        return memberships_from_distances(d2, self.m)

    # ------------------------------------------------------------------ sklearn plumbing
    def fit(self, X, y=None):
        r"""Cluster the objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples) or (n_samples, n_features)
            The dissimilarity matrix :math:`\mathbf{R}` when
            ``metric="precomputed"``, otherwise the feature vectors from
            which :math:`\mathbf{R}` is computed with ``metric``.
        y : Ignored

        Returns
        -------
        self : object
        """
        return super().fit(X, y)

    def predict_proba(self, X):
        """Memberships of (new) objects in the fitted clusters.

        Parameters
        ----------
        X : array-like of shape (n_new, n_samples) or (n_new, n_features)
            With ``metric="precomputed"`` the dissimilarities of the new
            objects to the ``n_samples`` *training* objects (same kind of
            entries as the training matrix); otherwise feature vectors.

        Returns
        -------
        U : ndarray of shape (n_new, n_clusters)
        """
        return super().predict_proba(X)

    def transform(self, X):
        r"""Relational distances :math:`\sqrt{d^2_{ik}}` of (new) objects to the prototypes."""
        return super().transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.positive_only = self.metric == "precomputed"  # dissimilarities are >= 0
        return tags


class NERFCM(RelationalFCM):
    r"""Non-Euclidean relational fuzzy c-means (Hathaway and Bezdek, 1994).

    Relational fuzzy c-means (:class:`RelationalFCM`) is the exact dual of
    FCM only when the dissimilarity matrix :math:`\mathbf{R}` is Euclidean
    (squared distances of some point configuration). Many relational data
    are not -- dynamic-time-warping distances, Gower dissimilarities with
    missing values, subjective ratings -- and then the relational distance
    :math:`d^2_{ik}=(\mathbf{R}\mathbf{v}_i)_k-\tfrac12\mathbf{v}_i\T\mathbf{R}\mathbf{v}_i`
    can be negative. NERF c-means uses the :math:`\beta`-*spread*
    transformation :math:`\mathbf{R}_\beta=\mathbf{R}+\beta(\mathbf{1}\mathbf{1}\T-\mathbf{I})`:
    adding :math:`\beta` to every off-diagonal dissimilarity makes the
    matrix Euclidean for a large enough :math:`\beta`, and the paper shows
    that this shifts the relational distances by
    :math:`\tfrac{\beta}{2}\norm{\mathbf{e}_k-\mathbf{v}_i}^2` -- a term that
    is computed without ever forming :math:`\mathbf{R}_\beta`. The spread
    is applied *adaptively*: :math:`\beta` starts at zero and is increased
    only when, and only as much as, a negative distance appears, so the
    partition stays as close to the RFCM result as the data allow.

    Parameters
    ----------
    n_clusters, metric, m, tol, max_iter, init, n_init, random_state, verbose
        As in :class:`RelationalFCM`. ``metric="precomputed"`` is the
        typical use; a feature metric that is not Euclidean (e.g.
        ``"cityblock"``, ``"cosine"``) also leads to a non-Euclidean
        :math:`\mathbf{R}` after squaring.

    Attributes
    ----------
    beta_ : float
        Final spread :math:`\beta`. ``0.0`` means the matrix behaved as a
        Euclidean one and the result is the RFCM partition.
    n_spread_updates_ : int
        How many times :math:`\beta` had to be increased during the run.
    relational_centers_, cluster_centers_, dissimilarity_matrix_, membership_, labels_,
    distances_, objective_, objective_history_, n_iter_
        See :class:`RelationalFCM`. The objective is that of the spread
        matrix :math:`\mathbf{R}_\beta`.

    Notes
    -----
    At every iteration the relational distances are computed with the
    current :math:`\beta`; if some :math:`d^2_{ik}<0`, the increment

    .. math::
        \Delta\beta=\max_{i,k}\frac{-2\,d^2_{ik}}{\norm{\mathbf{e}_k-\mathbf{v}_i}^2},\qquad
        \beta\leftarrow\beta+\Delta\beta,\qquad
        d^2_{ik}\leftarrow d^2_{ik}+\tfrac{\Delta\beta}{2}\norm{\mathbf{e}_k-\mathbf{v}_i}^2

    is the smallest shift that makes all distances non-negative (the paper's
    Step 3), and the iteration continues with the shifted distances. Since
    :math:`\norm{\mathbf{e}_k-\mathbf{v}_i}^2=1-2v_{ik}+\norm{\mathbf{v}_i}^2`
    is larger for objects far from the prototype, the shift affects weakly
    related objects most and leaves the memberships of well-clustered
    objects nearly unchanged; a large final :math:`\beta` nevertheless
    flattens the partition toward :math:`1/c`, which is the price of a
    strongly non-Euclidean matrix. New objects (``predict_proba``) receive
    the same shift with :math:`\norm{\mathbf{e}-\mathbf{v}_i}^2=1+\norm{\mathbf{v}_i}^2`,
    the value for an object outside the training set.

    References
    ----------
    .. [1] R. J. Hathaway, J. C. Bezdek, "NERF c-means: non-Euclidean
           relational fuzzy clustering", Pattern Recognition 27(3), 429--437,
           1994.
    .. [2] R. J. Hathaway, J. W. Davenport, J. C. Bezdek, "Relational duals
           of the c-means clustering algorithms", Pattern Recognition 22(2),
           205--212, 1989.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.relational import NERFCM, pairwise_dissimilarity
    >>> X, _ = make_fuzzy_blobs(random_state=0)
    >>> R = pairwise_dissimilarity(X, metric="cityblock") ** 2   # squared L1 distances: not Euclidean
    >>> nerf = NERFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    >>> bool(nerf.beta_ > 0), nerf.n_spread_updates_, nerf.membership_.shape
    (True, 1, (300, 3))
    """

    _spread = True

    def _spread_if_needed(self, d2, e_v):
        """Step 3 of NERF: raise beta by the smallest amount that removes negative distances."""
        neg = d2 < 0
        if np.any(neg):
            delta = float(np.max(-2.0 * d2[neg] / np.maximum(e_v[neg], EPS)))
            self.beta_ += delta
            self.n_spread_updates_ += 1
            if self.verbose:
                print(f"beta-spread: beta += {delta:.4g} -> {self.beta_:.4g}")
            d2 = d2 + 0.5 * delta * e_v
        return d2
