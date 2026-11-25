# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-shells (FCS) and fuzzy c-spherical shells (FCSS) clustering."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_centers, labels_to_membership, sq_euclidean
from ..base import BaseFuzzyClusterer
from ..cluster import FCM


class FuzzyCShells(BaseFuzzyClusterer):
    r"""Fuzzy c-(spherical) shells: circles / hyperspheres as prototypes.

    Hollow clusters -- rings in 2-D, spherical shells in higher dimension --
    have no interior points, so the point prototype of FCM placed at their
    center is far from every member, and concentric rings cannot be told
    apart at all. Shell clustering describes cluster :math:`i` by a center
    :math:`\mathbf{v}_i` **and** a radius :math:`r_i` and measures the
    distance of a sample as its deviation from the sphere. Compared with FCM
    only the prototype and the distance change; the membership formula is
    the same. Two published algorithms are implemented and selected by
    ``algorithm``:

    * ``"fcss"`` -- the *fuzzy c-spherical shells* algorithm of
      Krishnapuram, Nasraoui and Frigui [2]_. The squared *algebraic*
      distance :math:`(\|\mathbf{x}_k-\mathbf{v}_i\|^2-r_i^2)^2` is quadratic
      in a reparametrised prototype, so the prototype update is a
      closed-form weighted least-squares fit of a circle/sphere.
    * ``"fcs"`` -- the original *fuzzy c-shells* algorithm of Davé [1]_ with
      the squared *radial* (geometric) distance
      :math:`(\|\mathbf{x}_k-\mathbf{v}_i\|-r_i)^2`. Its necessary
      conditions are coupled nonlinear equations in :math:`\mathbf{v}_i`
      and :math:`r_i`, which Davé solved by Newton's method in every
      iteration; here ``newton_steps`` Gauss--Newton steps on the radial
      residuals are started from the closed-form FCSS fit.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of shells :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    algorithm : {"fcss", "fcs"}, default="fcss"
        Shell distance and prototype update, see above.
    newton_steps : int, default=2
        Gauss--Newton steps per iteration for ``algorithm="fcs"``.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"fcm", "random", "kmeans++"} or ndarray, default="fcm"
        ``"fcm"`` runs :class:`~fuzzyclust.cluster.FCM` and starts from its
        *hardened* partition, which for intersecting or separate shells
        puts the groups on different shells and avoids the poor local
        minimum of one big circle through everything. For *concentric*
        shells FCM halves every ring, so use ``n_init > 1`` (the further
        restarts are random) or ``init="random"``.
        ``"random"``/``"kmeans++"``/array as in the base class; the first
        prototype update turns the point prototypes into shells.
    n_init : int, default=1
        Number of restarts; the solution with the smallest objective is
        kept. With ``init="fcm"`` the first restart uses the hardened FCM
        partition and the further ones random point prototypes (FCM itself
        converges to nearly the same partition from most starts, so
        repeating it would not diversify the restarts).
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Shell centers :math:`\mathbf{v}_i`.
    radii_ : ndarray of shape (n_clusters,)
        Shell radii :math:`r_i`.
    membership_, labels_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.
    distances_ : ndarray of shape (n_samples, n_clusters)
        :math:`D_{ik}` at convergence: :math:`|\,\|\mathbf{x}_k-\mathbf{v}_i\|^2-r_i^2|`
        for FCSS, the absolute radial deviation
        :math:`|\,\|\mathbf{x}_k-\mathbf{v}_i\|-r_i|` for FCS (the same holds
        for :meth:`transform`; :meth:`radial_deviation` gives the signed
        radial deviation for both).

    Notes
    -----
    Both algorithms minimize the c-means functional with a shell distance,

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m D^2_{ik},\qquad
        D^2_{ik} = \left(\|\mathbf{x}_k-\mathbf{v}_i\|^2 - r_i^2\right)^2
        \ \text{(FCSS)}\quad\text{or}\quad
        D^2_{ik} = \left(\|\mathbf{x}_k-\mathbf{v}_i\| - r_i\right)^2
        \ \text{(FCS)},

    under :math:`\sum_i\mu_{ik}=1`, so the memberships follow the FCM
    formula with the shell distance. FCSS writes the algebraic distance as
    :math:`\|\mathbf{x}_k\|^2+\mathbf{p}_i^T\mathbf{y}_k` with
    :math:`\mathbf{p}_i=[-2\mathbf{v}_i^T,\ \mathbf{v}_i^T\mathbf{v}_i-r_i^2]^T`
    and regressors :math:`\mathbf{y}_k=[\mathbf{x}_k^T,\,1]^T`, so that the
    weighted least-squares solution

    .. math::
        \mathbf{p}_i = -\Big(\sum_k\mu_{ik}^m\mathbf{y}_k\mathbf{y}_k^T\Big)^{-1}
                        \sum_k\mu_{ik}^m\|\mathbf{x}_k\|^2\,\mathbf{y}_k

    gives :math:`\mathbf{v}_i` and :math:`r_i^2=\mathbf{v}_i^T\mathbf{v}_i-p_{i,n+1}`
    in closed form. For FCS the necessary conditions are

    .. math::
        r_i = \frac{\sum_k\mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|}{\sum_k\mu_{ik}^m},
        \qquad
        \sum_k\mu_{ik}^m\Big(1-\frac{r_i}{\|\mathbf{x}_k-\mathbf{v}_i\|}\Big)
        (\mathbf{x}_k-\mathbf{v}_i) = \mathbf{0},

    which have no closed form. Near the shell the two distances are related
    by :math:`\|\mathbf{x}\|^2-r^2=(\|\mathbf{x}\|-r)(\|\mathbf{x}\|+r)\approx 2r(\|\mathbf{x}\|-r)`,
    so the algebraic distance weights the same radial deviation more on a
    large shell than on a small one: FCSS memberships favor small shells,
    a known bias that matters when shells of similar radius are nested or
    when only arcs are observed; FCS does not have it but costs a Newton
    iteration per cluster and iteration.

    References
    ----------
    .. [1] R. N. Davé, "Fuzzy shell-clustering and applications to circle
           detection in digital images", Int. J. General Systems 16(4),
           343--355, 1990.
    .. [2] R. Krishnapuram, O. Nasraoui, H. Frigui, "The fuzzy c spherical
           shells algorithm: a new approach", IEEE Trans. Neural Networks
           3(5), 663--671, 1992.

    Examples
    --------
    >>> from fuzzyclust.extensions import FuzzyCShells
    >>> from fuzzyclust.datasets import make_rings
    >>> X, y = make_rings(radii=(1.0, 3.0), random_state=0)
    >>> fcs = FuzzyCShells(n_clusters=2, n_init=5, random_state=0).fit(X)
    >>> sorted(fcs.radii_.round(1).tolist())
    [1.0, 3.0]
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        algorithm: str = "fcss",
        newton_steps: int = 2,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="fcm",
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
        self.algorithm = algorithm
        self.newton_steps = newton_steps

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("FuzzyCShells requires m > 1.")
        if self.algorithm not in ("fcss", "fcs"):
            raise ValueError("algorithm must be 'fcss' or 'fcs'.")
        if not isinstance(self.newton_steps, (int, np.integer)) or self.newton_steps < 1:
            raise ValueError("newton_steps must be a positive integer.")
        self._n_starts = 0

    def _init_state(self, X, rng):
        self._n_starts += 1
        if isinstance(self.init, str) and self.init == "fcm":
            if self._n_starts == 1:
                fcm = FCM(
                    n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
                    random_state=rng,
                ).fit(X)
                return labels_to_membership(fcm.labels_, self.n_clusters)
            # further restarts: random point prototypes in the bounding box (base-class rule)
            mean = X.mean(axis=0)
            half_range = np.abs(X - mean).max(axis=0)
            V = 2.0 * half_range * (rng.rand(self.n_clusters, X.shape[1]) - 0.5) + mean
            return self._init_memberships(sq_euclidean(X, V))
        return super()._init_state(X, rng)

    # ------------------------------------------------------------ prototype fits
    @staticmethod
    def _lstsq_shell(X, w):
        """Weighted algebraic least-squares circle/sphere fit (FCSS update)."""
        n = X.shape[1]
        Y = np.c_[X, np.ones(X.shape[0])]
        b = np.einsum("ij,ij->i", X, X)
        G = (Y * w[:, None]).T @ Y
        h = -(Y * w[:, None]).T @ b
        p = np.linalg.lstsq(G, h, rcond=None)[0]
        v = -0.5 * p[:n]
        r2 = v @ v - p[n]
        return v, np.sqrt(max(r2, EPS))

    @staticmethod
    def _newton_shell(X, w, v, r, steps):
        """Gauss-Newton refinement of the geometric (radial) shell fit (FCS)."""
        for _ in range(steps):
            diff = X - v
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            safe = np.maximum(dist, EPS)
            e = dist - r
            # Jacobian of e_k with respect to [v, r]: [-(x_k - v)/|x_k - v|, -1]
            Jm = np.c_[-diff / safe[:, None], -np.ones(X.shape[0])]
            JW = Jm * w[:, None]
            H = JW.T @ Jm
            g = JW.T @ e
            step = np.linalg.lstsq(H, g, rcond=None)[0]
            v = v - step[:-1]
            r = max(r - step[-1], EPS)
        return v, r

    def _update_prototypes(self, X, Um):
        c, n = self.n_clusters, X.shape[1]
        V = np.empty((c, n))
        R = np.empty(c)
        for i in range(c):
            w = Um[:, i]
            if w.sum() <= EPS:  # empty cluster: keep a point prototype
                V[i] = fuzzy_centers(X, Um)[i]
                R[i] = EPS
                continue
            v, r = self._lstsq_shell(X, w)
            if self.algorithm == "fcs":
                v, r = self._newton_shell(X, w, v, r, self.newton_steps)
            V[i], R[i] = v, r
        self.cluster_centers_ = V
        self.radii_ = R

    def _distances(self, X):
        d2 = sq_euclidean(X, self.cluster_centers_)
        if self.algorithm == "fcss":
            return (d2 - self.radii_[None, :] ** 2) ** 2
        return (np.sqrt(d2) - self.radii_[None, :]) ** 2

    def radial_deviation(self, X):
        r"""Signed radial deviation :math:`\|\mathbf{x}_k-\mathbf{v}_i\|-r_i` of samples from every shell.

        Negative values lie inside the shell, positive values outside; the
        absolute value is the geometric shell distance of FCS.

        Returns
        -------
        E : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self, "radii_")
        X = validate_data(self, X, dtype=float, reset=False)
        return np.sqrt(sq_euclidean(X, self.cluster_centers_)) - self.radii_[None, :]
