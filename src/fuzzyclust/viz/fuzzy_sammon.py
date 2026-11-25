# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""FUZZSAM: fuzzy Sammon mapping of fuzzy clustering results."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from .._utils import EPS, check_membership, fuzzy_centers, sq_euclidean
from .quality import recomputed_membership
from .sammon import _clip_displacements

__all__ = ["FuzzySammon"]


class FuzzySammon(BaseEstimator):
    r"""Fuzzy Sammon mapping (FUZZSAM) of a fitted fuzzy clustering.

    Classical Sammon mapping preserves the :math:`N(N-1)/2` Euclidean
    inter-point distances of the data and therefore (i) is expensive for
    large :math:`N` and (ii) ignores the distance measure of the clustering
    algorithm (the adaptive norms of GK/GG, or any non-point prototype).
    FUZZSAM instead preserves only the :math:`N\times c` distances between the
    samples and the cluster prototypes, *measured by the clustering algorithm
    itself*, weighted by the memberships. In the plane every cluster is a
    single point :math:`\mathbf{z}_i` and the geometry around it is Euclidean,
    so a cluster whose prototype fits the data well appears as a compact,
    roughly circular group, whatever its shape in the original space.

    Parameters
    ----------
    n_components : int, default=2
        Dimension :math:`q` of the embedding.
    max_iter : int, default=100
        Maximum number of iterations.
    alpha : float, default=0.4
        Step size of the pseudo-Newton update (0.3-0.4 recommended).
    init : {"pca", "random"} or ndarray of shape (n_samples, n_components), default="pca"
        Initial configuration of the projected data: principal component
        scores of ``X`` (requires ``X``), uniform random points in
        :math:`[-0.5,0.5]^q`, or an explicit matrix (e.g. the result of
        :class:`~fuzzyclust.viz.Sammon`, as in the MATLAB demo).
    tol : float, default=1e-6
        Stop when the relative decrease of :math:`E_\text{fuzz}` is below ``tol``.
    backtracking : bool, default=True
        Safeguard of the Newton step as in :class:`~fuzzyclust.viz.Sammon`:
        clip outlying displacements and halve the step (at most
        ``max_backtracks`` times) when it would increase :math:`E_\text{fuzz}`.
    max_backtracks : int, default=10
    m : float or None, default=None
        Fuzziness exponent used for the weights and the recomputed memberships.
        ``None`` takes ``estimator.m`` (2.0 if the estimator has no ``m``).
    weight_exponent : float or None, default=None
        Exponent of the membership weights :math:`\mu_{ik}^{w}` in the cost and
        in the projected-center formula. ``None`` uses :math:`w=m` (FUZZSAM of
        Abonyi & Babuška 2004 and of the MATLAB toolbox); ``1.0`` gives the
        first variant of Kovács & Abonyi (2002), which weighted with
        :math:`\mu_{ik}` itself.
    random_state : int, RandomState instance or None, default=None
        Seed for ``init="random"``.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Projected data :math:`\mathbf{y}_k`.
    embedded_centers_ : ndarray of shape (n_clusters, n_components)
        Projected cluster centers :math:`\mathbf{z}_i`.
    stress_ : float
        Final value of :math:`E_\text{fuzz}`.
    stress_history_ : ndarray of shape (``n_iter_``,)
    n_iter_ : int
    membership_ : ndarray of shape (n_samples, n_clusters)
        The partition that was mapped (copied from the estimator).
    distances_ : ndarray of shape (n_samples, n_clusters)
        The prototype distances :math:`d_{ik}` that were mapped.
    m_ : float
        Fuzziness exponent actually used.

    Notes
    -----
    FUZZSAM minimizes

    .. math::
        E_\text{fuzz} = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m
        \left(d(\mathbf{x}_k,\mathbf{v}_i) - d^*_{ik}\right)^2,\qquad
        d^*_{ik}=\|\mathbf{y}_k-\mathbf{z}_i\|,

    where :math:`d(\mathbf{x}_k,\mathbf{v}_i)` is the distance *of the
    clustering algorithm* (``estimator.transform(X)``: Euclidean for FCM,
    Mahalanobis-type for GK, the exponential distance of GG, ...). The
    projected points are moved by the Sammon-type diagonal Newton step

    .. math::
        y_{kl} \leftarrow y_{kl} - \alpha\,
        \frac{\partial E_\text{fuzz}/\partial y_{kl}}{|\partial^2 E_\text{fuzz}/\partial y_{kl}^2|},\quad
        \frac{\partial E_\text{fuzz}}{\partial y_{kl}} = -2\sum_{i=1}^{c}\mu_{ik}^m
        \frac{d_{ik}-d^*_{ik}}{d^*_{ik}}(y_{kl}-z_{il}),

    .. math::
        \frac{\partial^2 E_\text{fuzz}}{\partial y_{kl}^2} = -2\sum_{i=1}^{c}
        \frac{\mu_{ik}^m}{d^*_{ik}}\left[(d_{ik}-d^*_{ik})
        -\frac{(y_{kl}-z_{il})^2}{d^*_{ik}}\left(1+\frac{d_{ik}-d^*_{ik}}{d^*_{ik}}\right)\right],

    and after every sweep the projected centers are recomputed as the
    membership-weighted means
    :math:`\mathbf{z}_i=\sum_k\mu_{ik}^m\mathbf{y}_k/\sum_k\mu_{ik}^m`.
    Only :math:`N c` distances are needed per iteration instead of
    :math:`N(N-1)/2`. The quality of the map is judged with
    :func:`~fuzzyclust.viz.projection_quality`.

    References
    ----------
    .. [1] A. Kovács, J. Abonyi, "Visualization of fuzzy clustering results by
           modified Sammon mapping", Proc. 3rd Int. Symp. of Hungarian
           Researchers on Computational Intelligence, 177-188, 2002.
    .. [2] J. Abonyi, R. Babuška, "FUZZSAM - visualization of fuzzy clustering
           results by modified Sammon mapping", Proc. IEEE Int. Conf. on Fuzzy
           Systems (FUZZ-IEEE), 2004.
    .. [3] B. Feil, B. Balasko, J. Abonyi, "Visualization of fuzzy clusters by
           fuzzy Sammon mapping projection: application to the analysis of phase
           space trajectories", Soft Computing 11, 479-488, 2007.

    Examples
    --------
    >>> from fuzzyclust import GK
    >>> from fuzzyclust.datasets import make_fuzzy_blobs
    >>> from fuzzyclust.viz import FuzzySammon
    >>> X, _ = make_fuzzy_blobs(n_clusters=3, n_features=5, random_state=0)
    >>> gk = GK(n_clusters=3, random_state=0).fit(X)
    >>> fs = FuzzySammon(max_iter=50).fit(gk, X)
    >>> fs.embedding_.shape, fs.embedded_centers_.shape
    ((300, 2), (3, 2))
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        alpha: float = 0.4,
        init="pca",
        tol: float = 1e-6,
        backtracking: bool = True,
        max_backtracks: int = 10,
        m=None,
        weight_exponent=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.init = init
        self.tol = tol
        self.backtracking = backtracking
        self.max_backtracks = max_backtracks
        self.m = m
        self.weight_exponent = weight_exponent
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers
    def _resolve_inputs(self, estimator, X, membership, distances):
        """Return (X or None, U, D, m) from the accepted input combinations."""
        if estimator is not None:
            check_is_fitted(estimator, "cluster_centers_")
            if X is not None:
                X = check_array(X, dtype=float)
                D = np.asarray(estimator.transform(X), dtype=float)
                U = np.asarray(estimator.predict_proba(X), dtype=float)
            else:
                D = np.asarray(estimator.distances_, dtype=float)
                U = np.asarray(estimator.membership_, dtype=float)
            m = getattr(estimator, "m", 2.0) if self.m is None else self.m
        else:
            if membership is None or distances is None:
                raise ValueError("Pass a fitted estimator or both membership= and distances=.")
            U = np.asarray(membership, dtype=float)
            D = np.asarray(distances, dtype=float)
            if X is not None:
                X = check_array(X, dtype=float)
            m = 2.0 if self.m is None else self.m
        U = check_membership(U)
        if D.shape != U.shape:
            raise ValueError(f"distances {D.shape} and membership {U.shape} must have the same shape.")
        if X is not None and X.shape[0] != U.shape[0]:
            raise ValueError("X must have as many rows as the membership matrix.")
        if not m > 1.0:
            raise ValueError("m must be > 1.")
        return X, U, D, float(m)

    def _initial_embedding(self, X, N, rng):
        q = self.n_components
        if isinstance(self.init, str):
            if self.init == "pca":
                if X is None:
                    raise ValueError("init='pca' needs the data matrix X; pass X or use init='random'.")
                from ._pca import pca_projection

                Y, _, _ = pca_projection(X, min(q, min(X.shape)))
                if Y.shape[1] < q:
                    Y = np.hstack([Y, np.zeros((N, q - Y.shape[1]))])
                return Y
            if self.init == "random":
                return rng.rand(N, q) - 0.5
            raise ValueError("init must be 'pca', 'random' or an array of shape (n_samples, n_components).")
        Y0 = np.array(self.init, dtype=float, copy=True)
        if Y0.shape != (N, q):
            raise ValueError(f"init array must have shape ({N}, {q}), got {Y0.shape}.")
        return Y0

    # ------------------------------------------------------------------ fitting
    def fit(self, estimator=None, X=None, *, membership=None, distances=None):
        """Map the result of a fuzzy clustering to ``n_components`` dimensions.

        Parameters
        ----------
        estimator : fitted fuzzyclust clusterer, optional
            Provides the memberships (``predict_proba``/``membership_``), the
            algorithm-specific distances (``transform``/``distances_``) and ``m``.
        X : array-like of shape (n_samples, n_features), optional
            The clustered data. Needed for ``init="pca"``; when given together
            with ``estimator`` the memberships and distances are re-evaluated
            on it (so any data set the estimator can score may be mapped).
        membership : array-like of shape (n_samples, n_clusters), optional
            Partition :math:`\\mathbf{U}^T` (alternative to ``estimator``).
        distances : array-like of shape (n_samples, n_clusters), optional
            Prototype distances :math:`d_{ik}` (alternative to ``estimator``).

        Returns
        -------
        self : object
        """
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not self.alpha > 0:
            raise ValueError("alpha must be positive.")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be >= 0.")
        X, U, D, m = self._resolve_inputs(estimator, X, membership, distances)
        w_exp = m if self.weight_exponent is None else float(self.weight_exponent)
        if w_exp <= 0:
            raise ValueError("weight_exponent must be positive.")
        rng = check_random_state(self.random_state)
        N = U.shape[0]
        W = U**w_exp  # weights mu_ik^w

        Y = self._initial_embedding(X, N, rng)
        Y = Y - Y.mean(axis=0)
        Z = fuzzy_centers(Y, W)

        def _stress(Y, Z):
            Ds = np.sqrt(sq_euclidean(Y, Z))
            return float(np.sum(W * (D - Ds) ** 2))

        prev = _stress(Y, Z)
        history = []
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007
            diff = Y[:, None, :] - Z[None, :, :]  # (N, c, q): y_k - z_i
            Ds = np.sqrt(np.einsum("kij,kij->ki", diff, diff))  # d*_ik
            ok = Ds > 0
            inv_ds = np.where(ok, 1.0 / np.where(ok, Ds, 1.0), 0.0)
            dq = np.where(ok, D - Ds, 0.0)
            term = W * dq * inv_ds  # mu^w (d - d*) / d*
            grad = -2.0 * np.einsum("ki,kij->kj", term, diff)
            term2 = W * inv_ds**2 * (1.0 + dq * inv_ds)  # mu^w (1 + (d-d*)/d*) / d*^2
            hess = -2.0 * (term.sum(axis=1)[:, None] - np.einsum("ki,kij->kj", term2, diff**2))
            hess_abs = np.abs(hess)
            step = np.divide(grad, hess_abs, out=np.zeros_like(grad), where=hess_abs > EPS)
            if self.backtracking:
                step = _clip_displacements(step)
            scale = self.alpha
            for _ in range(self.max_backtracks + 1):
                Y_new = Y - scale * step
                Y_new = Y_new - Y_new.mean(axis=0)
                Z_new = fuzzy_centers(Y_new, W)
                stress = _stress(Y_new, Z_new)
                if stress <= prev or not self.backtracking:
                    break
                scale *= 0.5
            Y, Z = Y_new, Z_new
            history.append(stress)
            if abs(prev - stress) < self.tol * max(prev, EPS):
                break
            prev = stress

        self.embedding_ = Y
        self.embedded_centers_ = Z
        self.stress_history_ = np.asarray(history)
        self.stress_ = history[-1]
        self.n_iter_ = it
        self.membership_ = U
        self.distances_ = D
        self.m_ = m
        return self

    def fit_transform(self, estimator=None, X=None, *, membership=None, distances=None):
        """Fit and return the embedding ``(n_samples, n_components)``."""
        return self.fit(estimator, X, membership=membership, distances=distances).embedding_

    def recomputed_membership(self) -> np.ndarray:
        r"""Memberships :math:`\mathbf{U}^*` implied by the plot (FCM formula on the projected distances)."""
        check_is_fitted(self, "embedding_")
        return recomputed_membership(self.embedding_, self.embedded_centers_, self.m_)

    def quality(self, X=None) -> dict:
        """Projection quality measures ``P``, ``F``, ``F_star`` (and ``E`` if ``X`` is given).

        See :func:`~fuzzyclust.viz.projection_quality`.
        """
        from .quality import projection_quality

        check_is_fitted(self, "embedding_")
        return projection_quality(self.membership_, self.embedding_, self.embedded_centers_, self.m_, X=X)
