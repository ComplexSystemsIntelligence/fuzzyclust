# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Kernelised fuzzy c-means."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise

from .._utils import EPS, fuzzy_centers
from ..base import BaseFuzzyClusterer

#: Kernels accepted by :class:`KernelFCM` (names of :func:`sklearn.metrics.pairwise.pairwise_kernels`).
KERNELS = ("rbf", "poly", "linear", "sigmoid", "cosine", "laplacian")


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    r"""Gaussian (RBF) kernel matrix :math:`K_{kl}=\exp(-\gamma\|\mathbf{x}_k-\mathbf{y}_l\|^2)`.

    Thin wrapper around :func:`sklearn.metrics.pairwise.rbf_kernel` with a
    mandatory width, kept for backward compatibility.
    """
    return pairwise.rbf_kernel(X, Y, gamma=gamma)


class KernelFCM(BaseFuzzyClusterer):
    r"""Kernel fuzzy c-means: FCM in the feature space of a Mercer kernel.

    FCM measures distances with the Euclidean norm of the input space and
    therefore finds clusters that are convex and roughly spherical. Kernel
    methods map the data with a (usually unknown) nonlinear map
    :math:`\Phi` into a high-dimensional *feature space* :math:`\mathcal{H}`
    where inner products are given by a kernel,
    :math:`K(\mathbf{x},\mathbf{y})=\langle\Phi(\mathbf{x}),\Phi(\mathbf{y})\rangle`,
    and run a linear algorithm there. Kernel FCM runs FCM in
    :math:`\mathcal{H}`: the objective and the membership formula are
    those of FCM, only the distance is the feature-space distance
    :math:`\|\Phi(\mathbf{x}_k)-\mathbf{w}_i\|^2`, which is evaluated with
    the *kernel trick* without ever computing :math:`\Phi`. Spheres in
    :math:`\mathcal{H}` are curved regions in the input space, so rings,
    moons and other nonconvex groups become separable. Two variants are
    implemented, following the comparative study of Graves and Pedrycz
    [2]_ (who call them KFCM-K and KFCM-F, with "feature space" meaning the
    *input* space in their terminology):

    * ``prototypes="feature"`` (default) -- the prototypes
      :math:`\mathbf{w}_i` live in :math:`\mathcal{H}` as weighted means
      of the mapped samples, :math:`\mathbf{w}_i=\sum_k w_{ik}\Phi(\mathbf{x}_k)`
      with :math:`w_{ik}=\mu_{ik}^m/\sum_l\mu_{il}^m`, and all distances are
      kernel evaluations. Any Mercer kernel can be used; the clusters can be
      nonconvex in the input space, at the price of :math:`O(N^2)` memory
      for the kernel matrix and of prototypes that have no exact pre-image.
    * ``prototypes="input"`` -- the KFCM of Zhang and Chen [1]_: the
      prototypes :math:`\mathbf{v}_i` stay in the input space and the
      distance is :math:`\|\Phi(\mathbf{x}_k)-\Phi(\mathbf{v}_i)\|^2
      =2(1-K(\mathbf{x}_k,\mathbf{v}_i))` for the Gaussian kernel. The
      distance saturates for far-away samples, so outliers lose their
      leverage on the prototypes: this variant is a *robust* FCM, but,
      because the distance is a monotone function of
      :math:`\|\mathbf{x}_k-\mathbf{v}_i\|`, its hard partition boundaries
      are the same bisecting hyperplanes as those of FCM.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`.
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    kernel : {"rbf", "poly", "linear", "sigmoid", "cosine", "laplacian"}, default="rbf"
        Kernel function, evaluated with
        :func:`sklearn.metrics.pairwise.pairwise_kernels`. With
        ``prototypes="input"`` only ``"rbf"`` is allowed, because the
        prototype update of Zhang and Chen is specific to it.
    gamma : float or None, default=None
        Kernel coefficient of the ``"rbf"`` (:math:`\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2)`),
        ``"laplacian"``, ``"poly"`` and ``"sigmoid"`` kernels. ``None`` uses
        :math:`1/(n\,\mathrm{Var}(\mathbf{X}))` (the ``"scale"`` rule of
        scikit-learn's SVMs). Large :math:`\gamma` gives very local kernels.
    degree : int, default=3
        Degree of the ``"poly"`` kernel :math:`(\gamma\,\mathbf{x}^T\mathbf{y}+c_0)^d`.
    coef0 : float, default=1.0
        Constant :math:`c_0` of the ``"poly"`` and ``"sigmoid"`` kernels.
    prototypes : {"feature", "input"}, default="feature"
        Where the prototypes live, see above.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"random", "kmeans++"} or ndarray, default="random"
        Initialization in the input space (as in the base class); the
        initial partition is derived from Euclidean distances.
    n_init : int, default=1
        Number of random restarts; the run with the lowest objective is
        kept. Kernel FCM with a local kernel has many local optima, so
        restarts matter more than for FCM.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes in the input space. For ``prototypes="feature"`` these
        are the membership-weighted means :math:`\sum_k w_{ik}\mathbf{x}_k`
        of the samples, i.e. only an approximate pre-image of the
        feature-space prototypes, provided for plotting.
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    gamma_ : float
        Kernel coefficient used.
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`; ``distances_`` are
        feature-space distances.

    Notes
    -----
    Both variants minimize :math:`J=\sum_i\sum_k\mu_{ik}^m D^2_{ik}` with the
    FCM membership formula. For the feature-space prototypes, expanding
    :math:`\|\Phi(\mathbf{x}_k)-\mathbf{w}_i\|^2` gives

    .. math::
        D^2_{ik} = K(\mathbf{x}_k,\mathbf{x}_k) - 2\sum_l w_{il}K(\mathbf{x}_k,\mathbf{x}_l)
                   + \sum_{l,l'} w_{il}w_{il'}K(\mathbf{x}_l,\mathbf{x}_{l'}),\qquad
        w_{il} = \frac{\mu_{il}^m}{\sum_{l'}\mu_{il'}^m},

    the fuzzy counterpart of kernel k-means [3]_; the same formula with
    :math:`K(\mathbf{x},\mathbf{x}_l)` for a new sample :math:`\mathbf{x}`
    gives its memberships in :meth:`predict_proba`. For the input-space
    prototypes and the Gaussian kernel

    .. math::
        D^2_{ik} = 2\left(1-K(\mathbf{x}_k,\mathbf{v}_i)\right),\qquad
        \mathbf{v}_i = \frac{\sum_k\mu_{ik}^m K(\mathbf{x}_k,\mathbf{v}_i)\,\mathbf{x}_k}
                            {\sum_k\mu_{ik}^m K(\mathbf{x}_k,\mathbf{v}_i)},

    a fixed-point update in which every sample is additionally weighted by
    its kernel similarity to the current prototype (far samples get weight
    :math:`\approx 0`). The kernel width :math:`\gamma` is the decisive
    parameter: :math:`\gamma\to0` recovers FCM (the RBF kernel becomes
    linear in the squared distance), a very large :math:`\gamma` makes all
    samples mutually orthogonal in :math:`\mathcal{H}` and the memberships
    uninformative.

    References
    ----------
    .. [1] D.-Q. Zhang, S.-C. Chen, "Clustering incomplete data using
           kernel-based fuzzy c-means algorithm", Neural Processing Letters
           18, 155--162, 2003.
    .. [2] D. Graves, W. Pedrycz, "Kernel-based fuzzy clustering and fuzzy
           clustering: a comparative experimental study", Fuzzy Sets and
           Systems 161(4), 522--543, 2010.
    .. [3] M. Girolami, "Mercer kernel-based clustering in feature space",
           IEEE Trans. Neural Networks 13(3), 780--784, 2002.

    Examples
    --------
    >>> from fuzzyclust.extensions import KernelFCM
    >>> from fuzzyclust.datasets import make_rings
    >>> X, y = make_rings(random_state=0)
    >>> kfcm = KernelFCM(n_clusters=2, gamma=2.0, n_init=3, random_state=0).fit(X)
    >>> kfcm.membership_.shape
    (300, 2)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        kernel: str = "rbf",
        gamma=None,
        degree: int = 3,
        coef0: float = 1.0,
        prototypes: str = "feature",
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
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.prototypes = prototypes

    # ------------------------------------------------------------ kernel helpers
    def _kernel_params(self):
        if self.kernel in ("rbf", "laplacian"):
            return {"gamma": self.gamma_}
        if self.kernel == "poly":
            return {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        if self.kernel == "sigmoid":
            return {"gamma": self.gamma_, "coef0": self.coef0}
        return {}

    def _kernel(self, X, Y):
        """Kernel matrix ``(len(X), len(Y))``."""
        return pairwise.pairwise_kernels(X, Y, metric=self.kernel, **self._kernel_params())

    def _self_kernel(self, X):
        r"""Diagonal :math:`K(\mathbf{x}_k,\mathbf{x}_k)` without forming the full matrix."""
        if self.kernel in ("rbf", "laplacian", "cosine"):
            return np.ones(X.shape[0])
        x2 = np.einsum("ij,ij->i", X, X)
        if self.kernel == "linear":
            return x2
        if self.kernel == "poly":
            return (self.gamma_ * x2 + self.coef0) ** self.degree
        return np.tanh(self.gamma_ * x2 + self.coef0)  # sigmoid

    def _check_params(self, X):
        super()._check_params(X)
        if self.m <= 1.0:
            raise ValueError("KernelFCM requires m > 1.")
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {self.kernel!r}.")
        if self.prototypes not in ("input", "feature"):
            raise ValueError("prototypes must be 'input' or 'feature'.")
        if self.prototypes == "input" and self.kernel != "rbf":
            raise ValueError("prototypes='input' (Zhang-Chen KFCM) is only defined for kernel='rbf'.")
        if self.gamma is None:
            var = X.var()
            self.gamma_ = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        else:
            if self.gamma <= 0:
                raise ValueError("gamma must be positive.")
            self.gamma_ = float(self.gamma)
        if self.prototypes == "feature":
            self._X_fit = X
            self._K_fit = self._kernel(X, X)
            self._k_self_fit = np.diag(self._K_fit).copy()

    def _init_state(self, X, rng):
        U = super()._init_state(X, rng)
        self.cluster_centers_ = fuzzy_centers(X, U**self.m)
        return U

    # ---------------------------------------------------------------- hooks
    def _update_prototypes(self, X, Um):
        if self.prototypes == "input":
            K = self._kernel(X, self.cluster_centers_)
            self.cluster_centers_ = fuzzy_centers(X, Um * K)
        else:
            W = Um / np.maximum(Um.sum(axis=0, keepdims=True), EPS)  # (N, c), columns sum to one
            self._W = W
            self._ww = np.einsum("ki,kl,li->i", W, self._K_fit, W)
            self.cluster_centers_ = W.T @ X  # approximate pre-image, for plotting only

    def _distances(self, X):
        if self.prototypes == "input":
            return 2.0 * (1.0 - self._kernel(X, self.cluster_centers_))
        if X is self._X_fit:
            K, k_self = self._K_fit, self._k_self_fit
        else:
            K, k_self = self._kernel(X, self._X_fit), self._self_kernel(X)
        d2 = k_self[:, None] - 2.0 * K @ self._W + self._ww[None, :]
        np.maximum(d2, 0.0, out=d2)
        return d2
