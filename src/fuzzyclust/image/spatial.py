# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Spatially constrained fuzzy c-means for image segmentation.

Pixel-wise FCM ignores that neighboring pixels of an image almost always
belong to the same region, so every noisy pixel is misclassified on its
own. The algorithms collected in :class:`SpatialFCM` add the neighborhood
to the c-means functional in different ways (``method``): a neighborhood
term in the distance (FCM_S), a filtered image as a second observation
(FCM_S1, FCM_S2), a linearly weighted sum image whose gray-level histogram
is clustered (EnFCM), a locally adaptive weighted image and its histogram
(FGFCM), and a fuzzy local information factor (FLICM).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .._utils import EPS, fuzzy_centers, memberships_from_distances, sq_euclidean
from ..large.weighted import WeightedFCM

__all__ = ["SpatialFCM", "METHODS", "neighborhood_sum", "local_similarity_image", "weighted_sum_image"]

#: Segmentation variants of :class:`SpatialFCM`.
METHODS = ("fcm", "fcm_s", "fcm_s1", "fcm_s2", "enfcm", "fgfcm", "flicm")
_HISTOGRAM_METHODS = ("enfcm", "fgfcm")


# --------------------------------------------------------------------------- neighborhood helpers
def _check_window(window_size) -> int:
    if (
        not isinstance(window_size, (int, np.integer))
        or isinstance(window_size, bool)
        or window_size < 3
        or window_size % 2 == 0
    ):
        raise ValueError("window_size must be an odd integer >= 3.")
    return int(window_size)


def _offsets(window_size: int):
    """Row/column offsets of the neighbors in a square window, center excluded."""
    r = window_size // 2
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]


def _shifted(img: np.ndarray, dy: int, dx: int, r: int) -> np.ndarray:
    """``img`` shifted by ``(dy, dx)`` with reflected borders (``img`` is 2-D or 3-D)."""
    pad = ((r, r), (r, r)) + ((0, 0),) * (img.ndim - 2)
    P = np.pad(img, pad, mode="symmetric")  # numpy's "symmetric" is scipy.ndimage's "reflect"
    H, W = img.shape[:2]
    return P[r + dy : r + dy + H, r + dx : r + dx + W]


def neighborhood_sum(img, window_size: int = 3, kernel=None):
    r"""Sum of an image over the neighbors of every pixel, :math:`\sum_{r\in N_k} y_r` (center excluded).

    Parameters
    ----------
    img : ndarray of shape (H, W) or (H, W, C)
        Image (or stack of images along the last axis).
    window_size : int, default=3
        Side of the square neighborhood :math:`N_k`; the center pixel is
        not counted. Borders are reflected.
    kernel : ndarray of shape (window_size, window_size), optional
        Weights of the neighbors instead of ones (the center entry is
        ignored and set to zero).

    Returns
    -------
    out : ndarray, same shape as ``img``

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.image import neighborhood_sum
    >>> neighborhood_sum(np.ones((3, 3)))[1, 1]
    np.float64(8.0)
    """
    img = np.asarray(img, dtype=float)
    w = _check_window(window_size)
    if kernel is None:
        K = np.ones((w, w))
    else:
        K = np.array(kernel, dtype=float, copy=True)
        if K.shape != (w, w):
            raise ValueError(f"kernel must have shape ({w}, {w}).")
    K[w // 2, w // 2] = 0.0
    if img.ndim == 2:
        return ndimage.convolve(img, K, mode="reflect")
    out = np.empty_like(img)
    for ch in range(img.shape[2]):
        out[..., ch] = ndimage.convolve(img[..., ch], K, mode="reflect")
    return out


def weighted_sum_image(img, alpha: float = 1.0, window_size: int = 3):
    r"""Linearly weighted sum image of EnFCM.

    .. math::
        \xi_k=\frac{1}{1+\alpha}\Big(x_k+\frac{\alpha}{N_R}\sum_{r\in N_k}x_r\Big)

    Parameters
    ----------
    img : ndarray of shape (H, W) or (H, W, C)
    alpha : float, default=1.0
        Weight :math:`\alpha` of the neighborhood mean.
    window_size : int, default=3

    Returns
    -------
    xi : ndarray, same shape as ``img``
    """
    img = np.asarray(img, dtype=float)
    N_R = _check_window(window_size) ** 2 - 1
    return (img + (alpha / N_R) * neighborhood_sum(img, window_size)) / (1.0 + alpha)


def local_similarity_image(img, lambda_s: float = 3.0, lambda_g: float = 6.0, window_size: int = 3):
    r"""Locally adaptive weighted image of FGFCM, :math:`\xi_k=\sum_{r\in N_k}S_{kr}x_r/\sum_{r}S_{kr}`.

    The similarity of pixel :math:`k` and its neighbor :math:`r` combines
    a spatial and a gray-level factor,

    .. math::
        S_{kr}=S^{s}_{kr}\,S^{g}_{kr},\qquad
        S^{s}_{kr}=\exp\Big(-\frac{\max(|p_k-p_r|,|q_k-q_r|)}{\lambda_s}\Big),\qquad
        S^{g}_{kr}=\exp\Big(-\frac{\norm{\mathbf{x}_k-\mathbf{x}_r}^2}{\lambda_g\,\sigma_k^2}\Big),

    with :math:`(p,q)` the pixel coordinates and the local scale
    :math:`\sigma_k^2=\frac{1}{N_R}\sum_{r\in N_k}\norm{\mathbf{x}_r-\mathbf{x}_k}^2`,
    which makes the gray-level factor adapt to the local contrast (a
    neighbor that differs by more than the typical local difference gets a
    small weight -- a noise pixel does not spoil its neighbors, and an
    edge is not blurred).

    Parameters
    ----------
    img : ndarray of shape (H, W) or (H, W, C)
    lambda_s : float, default=3.0
        Scale of the spatial factor.
    lambda_g : float, default=6.0
        Scale of the gray-level factor.
    window_size : int, default=3

    Returns
    -------
    xi : ndarray, same shape as ``img``
    """
    img = np.asarray(img, dtype=float)
    w = _check_window(window_size)
    if lambda_s <= 0 or lambda_g <= 0:
        raise ValueError("lambda_s and lambda_g must be positive.")
    r = w // 2
    multi = img.ndim == 3
    offs = _offsets(w)
    shifted = [_shifted(img, dy, dx, r) for dy, dx in offs]
    d2 = [((s - img) ** 2).sum(axis=2) if multi else (s - img) ** 2 for s in shifted]
    sigma2 = np.mean(d2, axis=0)
    sigma2 = np.where(sigma2 > 0, sigma2, EPS)  # flat regions: all neighbors equally similar
    num = np.zeros_like(img)
    den = np.zeros(img.shape[:2])
    for (dy, dx), s, dd in zip(offs, shifted, d2, strict=True):
        S = np.exp(-max(abs(dy), abs(dx)) / lambda_s) * np.exp(-dd / (lambda_g * sigma2))
        num += S[..., None] * s if multi else S * s
        den += S
    return num / (den[..., None] if multi else den)


# --------------------------------------------------------------------------- the estimator
class SpatialFCM(ClusterMixin, BaseEstimator):
    r"""Spatially constrained fuzzy c-means for image segmentation.

    Segments a gray-level or multichannel image into ``n_clusters`` regions
    by fuzzy c-means on the pixel values, with the spatial context of every
    pixel brought into the objective in one of the published ways:

    ``"fcm"``
        Plain pixel-wise FCM (the reference; every pixel on its own).
    ``"fcm_s"``
        Ahmed et al. (2002): the distance of a pixel to a prototype is
        augmented by the mean distance of its neighbors to the same
        prototype, weighted by :math:`\alpha`. (The paper's second
        component, the estimation of a smooth intensity bias field of MRI
        data, is not included here.) Cost: a neighborhood sum per cluster
        and iteration.
    ``"fcm_s1"``, ``"fcm_s2"``
        Chen and Zhang (2004): the neighborhood sum is replaced by the
        *mean*- (S1) or *median*-filtered (S2) image computed once in
        advance; the filtered pixel is a second observation weighted by
        :math:`\alpha`. The median variant is robust to salt-and-pepper
        noise.
    ``"enfcm"``
        Szilágyi et al. (2003): the linearly weighted sum image
        :math:`\xi` of pixel and neighborhood mean is formed once, and its
        gray-level *histogram* is clustered -- the cost per iteration is
        :math:`O(q\,c)` for :math:`q` gray levels instead of :math:`O(Nc)`.
    ``"fgfcm"``
        Cai, Chen and Zhang (2007): like EnFCM, but the summed image uses
        locally adaptive weights :math:`S_{kr}` that combine spatial and
        gray-level similarity (:func:`local_similarity_image`), so that
        edges are kept and noise pixels do not spread.
    ``"flicm"``
        Krinidis and Chatzis (2010): a *fuzzy factor* :math:`G_{ki}` that
        weighs each neighbor's distance to the prototype by its spatial
        distance and by :math:`(1-\mu_{ir})^m`, its lack of membership in
        the cluster, is added to the distance; the factor needs no
        parameter and is recomputed every iteration.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of regions :math:`c`.
    method : {"fcm", "fcm_s", "fcm_s1", "fcm_s2", "enfcm", "fgfcm", "flicm"}, default="flicm"
    m : float, default=2.0
        Fuzziness exponent :math:`m>1`.
    alpha : float, default=1.0
        Weight :math:`\alpha` of the neighborhood term (``fcm_s``,
        ``fcm_s1``, ``fcm_s2``, ``enfcm``). ``0`` gives plain FCM; larger
        values smooth more and blur fine structure.
    window_size : int, default=3
        Side of the square neighborhood :math:`N_k` (odd, :math:`\ge3`);
        :math:`N_R=` ``window_size**2 - 1`` neighbors.
    lambda_s : float, default=3.0
    lambda_g : float, default=6.0
        Scales of the spatial and gray-level similarity of ``fgfcm``.
    n_levels : int, default=256
        Number of gray-level bins of the histogram methods (``enfcm``,
        ``fgfcm``); the summed image is quantized into this many equal
        bins between its minimum and maximum.
    tol : float, default=1e-4
        Termination tolerance on :math:`\max|\Delta\mu_{ik}|`.
    max_iter : int, default=100
    init : {"random", "kmeans++"} or ndarray, default="random"
        ``"random"`` draws prototypes inside the range of the pixel values,
        ``"kmeans++"`` seeds them on the pixels; an array of shape
        ``(n_clusters, n_channels)`` gives initial prototypes, an array of
        shape ``(H, W, n_clusters)`` or ``(H*W, n_clusters)`` an initial
        partition.
    n_init : int, default=1
        Random restarts; the run with the lowest final objective is kept.
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_channels)
        Prototypes :math:`\mathbf{v}_i` in pixel-value units.
    membership_ : ndarray of shape (H, W, n_clusters)
        Membership image of every cluster.
    labels_ : ndarray of shape (H, W)
        Label image (argmax of the memberships).
    filtered_image_ : ndarray of shape (H, W) or (H, W, C)
        The filtered/summed image the method works with (mean or median
        image for ``fcm_s1``/``fcm_s2``, :math:`\xi` for ``enfcm`` and
        ``fgfcm``); absent for the other methods.
    histogram_ : ndarray of shape (n_levels,)
        Pixel counts :math:`\gamma_l` per gray-level bin (histogram methods).
    level_values_ : ndarray of shape (n_levels,)
        Gray level of every bin (bin centers).
    level_membership_ : ndarray of shape (n_levels, n_clusters)
        Memberships of the gray levels (histogram methods).
    objective_ : float
    objective_history_ : ndarray of shape (``n_iter_``,)
    n_iter_ : int
    image_shape_ : tuple
        ``(H, W)`` of the training image.
    n_channels_ : int

    Notes
    -----
    Let :math:`\mathbf{x}_k` be the value of pixel :math:`k` (a scalar gray
    level or a vector of :math:`C` channels), :math:`N_k` the set of its
    :math:`N_R` neighbors in the window, and :math:`\mathbf{v}_i` the
    prototype of region :math:`i`. The methods minimize, by alternating
    optimization,

    .. math::
        J_{\text{FCM\_S}}&=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\norm{\mathbf{x}_k-\mathbf{v}_i}^2
          +\frac{\alpha}{N_R}\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m
          \sum_{r\in N_k}\norm{\mathbf{x}_r-\mathbf{v}_i}^2,\\
        J_{\text{FCM\_S1,2}}&=\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\norm{\mathbf{x}_k-\mathbf{v}_i}^2
          +\alpha\sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\norm{\bar{\mathbf{x}}_k-\mathbf{v}_i}^2,\\
        J_{\text{EnFCM,FGFCM}}&=\sum_{i=1}^{c}\sum_{l=1}^{q}\gamma_l\,\mu_{il}^m\,(\xi_l-v_i)^2,\\
        J_{\text{FLICM}}&=\sum_{i=1}^{c}\sum_{k=1}^{N}\Big[\mu_{ik}^m\norm{\mathbf{x}_k-\mathbf{v}_i}^2+G_{ki}\Big],
        \qquad G_{ki}=\sum_{r\in N_k}\frac{1}{d_{kr}+1}(1-\mu_{ir})^m\norm{\mathbf{x}_r-\mathbf{v}_i}^2,

    where :math:`\bar{\mathbf{x}}_k` is the mean (S1) or median (S2) of
    the window, :math:`\xi_l` the gray level of histogram bin :math:`l`
    with :math:`\gamma_l` pixels, and :math:`d_{kr}` the Euclidean distance
    of the pixel coordinates. All membership updates have the FCM form
    with an *effective* squared distance,

    .. math::
        \mu_{ik}=\frac{1}{\sum_{j=1}^{c}\big(\tilde D^2_{ik}/\tilde D^2_{jk}\big)^{1/(m-1)}},

    with

    .. math::
        \tilde D^2_{ik}=\begin{cases}
        \norm{\mathbf{x}_k-\mathbf{v}_i}^2
          +\frac{\alpha}{N_R}\sum_{r\in N_k}\norm{\mathbf{x}_r-\mathbf{v}_i}^2 & \text{FCM\_S}\\
        \norm{\mathbf{x}_k-\mathbf{v}_i}^2
          +\alpha\norm{\bar{\mathbf{x}}_k-\mathbf{v}_i}^2 & \text{FCM\_S1, S2}\\
        (\xi_l-v_i)^2 & \text{EnFCM, FGFCM}\\
        \norm{\mathbf{x}_k-\mathbf{v}_i}^2+G_{ki} & \text{FLICM.}
        \end{cases}

    The prototype updates are

    .. math::
        \mathbf{v}_i&=\frac{\sum_k\mu_{ik}^m\big(\mathbf{x}_k
          +\frac{\alpha}{N_R}\sum_{r\in N_k}\mathbf{x}_r\big)}{(1+\alpha)\sum_k\mu_{ik}^m}\ \text{(FCM\_S)},\\
        \mathbf{v}_i&=\frac{\sum_k\mu_{ik}^m(\mathbf{x}_k+\alpha\bar{\mathbf{x}}_k)}{(1+\alpha)\sum_k\mu_{ik}^m}
        \ \text{(FCM\_S1, S2)},

    .. math::
        v_i=\frac{\sum_l\gamma_l\mu_{il}^m\xi_l}{\sum_l\gamma_l\mu_{il}^m}\ \text{(EnFCM, FGFCM)},\qquad
        \mathbf{v}_i=\frac{\sum_k\mu_{ik}^m\mathbf{x}_k}{\sum_k\mu_{ik}^m}\ \text{(FLICM)}.

    The histogram methods are weighted FCM (:class:`~fuzzyclust.large.WeightedFCM`)
    on the gray levels with the histogram as weights; a pixel inherits
    the membership of its gray level. All neighborhood operations use
    :mod:`scipy.ndimage` with reflected borders.

    References
    ----------
    .. [1] M. N. Ahmed, S. M. Yamany, N. Mohamed, A. A. Farag, T. Moriarty,
           "A modified fuzzy c-means algorithm for bias field estimation and
           segmentation of MRI data", IEEE Transactions on Medical Imaging
           21(3), 193--199, 2002.
    .. [2] S. Chen, D. Zhang, "Robust image segmentation using FCM with
           spatial constraints based on new kernel-induced distance
           measure", IEEE Transactions on Systems, Man, and Cybernetics,
           Part B 34(4), 1907--1916, 2004.
    .. [3] L. Szilágyi, Z. Benyó, S. M. Szilágyi, H. S. Adam, "MR brain
           image segmentation using an enhanced fuzzy c-means algorithm",
           Proc. 25th Annual International Conference of the IEEE EMBS,
           724--726, 2003.
    .. [4] W. Cai, S. Chen, D. Zhang, "Fast and robust fuzzy c-means
           clustering algorithms incorporating local information for image
           segmentation", Pattern Recognition 40(3), 825--838, 2007.
    .. [5] S. Krinidis, V. Chatzis, "A robust fuzzy local information
           c-means clustering algorithm", IEEE Transactions on Image
           Processing 19(5), 1328--1337, 2010.

    Examples
    --------
    >>> from fuzzyclust.image import SpatialFCM, make_synthetic_image
    >>> data = make_synthetic_image(shape=(48, 48), n_classes=3, noise_level=0.1, random_state=0)
    >>> seg = SpatialFCM(n_clusters=3, method="flicm", random_state=0).fit(data.image)
    >>> seg.labels_.shape, seg.membership_.shape, seg.cluster_centers_.shape
    ((48, 48), (48, 48, 3), (3, 1))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        method: str = "flicm",
        m: float = 2.0,
        alpha: float = 1.0,
        window_size: int = 3,
        lambda_s: float = 3.0,
        lambda_g: float = 6.0,
        n_levels: int = 256,
        tol: float = 1e-4,
        max_iter: int = 100,
        init="random",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.m = m
        self.alpha = alpha
        self.window_size = window_size
        self.lambda_s = lambda_s
        self.lambda_g = lambda_g
        self.n_levels = n_levels
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ validation
    def _check_image(self, X, reset: bool) -> np.ndarray:
        img = np.asarray(X, dtype=float)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3 or img.shape[0] < 1 or img.shape[1] < 1:
            raise ValueError("The image must be a 2-D array (H, W) or a 3-D array (H, W, C).")
        if not np.all(np.isfinite(img)):
            raise ValueError("The image contains NaN or inf.")
        if reset:
            self.image_shape_ = img.shape[:2]
            self.n_channels_ = img.shape[2]
        elif img.shape[2] != self.n_channels_:
            raise ValueError(
                f"The image has {img.shape[2]} channels, the model was fitted with {self.n_channels_}."
            )
        return img

    def _check_params(self, img: np.ndarray) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}.")
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        N = img.shape[0] * img.shape[1]
        if self.n_clusters > N:
            raise ValueError(f"n_clusters={self.n_clusters} must be <= number of pixels={N}.")
        if self.m <= 1.0:
            raise ValueError("m must be > 1.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        _check_window(self.window_size)
        if self.lambda_s <= 0 or self.lambda_g <= 0:
            raise ValueError("lambda_s and lambda_g must be positive.")
        if not isinstance(self.n_levels, (int, np.integer)) or self.n_levels < 2:
            raise ValueError("n_levels must be an integer >= 2.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if self.method in _HISTOGRAM_METHODS and img.shape[2] != 1:
            raise ValueError(
                f"method={self.method!r} clusters the gray-level histogram and needs a single-channel image."
            )

    # ------------------------------------------------------------------ preprocessing
    def _filtered(self, img: np.ndarray):
        """Filtered / summed image of the method (``None`` for fcm, fcm_s, flicm)."""
        w = int(self.window_size)
        channels = range(img.shape[2])
        if self.method in ("fcm_s1", "fcm_s2"):
            filt = ndimage.uniform_filter if self.method == "fcm_s1" else ndimage.median_filter
            return np.stack([filt(img[..., ch], size=w, mode="reflect") for ch in channels], axis=2)
        if self.method == "enfcm":
            return weighted_sum_image(img, self.alpha, w)
        if self.method == "fgfcm":
            return local_similarity_image(img, self.lambda_s, self.lambda_g, w)
        return None

    def _flicm_kernel(self) -> np.ndarray:
        w = int(self.window_size)
        r = w // 2
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        K = 1.0 / (np.hypot(yy, xx) + 1.0)
        K[r, r] = 0.0
        return K

    def _quantize(self, xi: np.ndarray):
        """Histogram of a single-channel image: bin index per pixel, counts, bin centers."""
        v = xi[..., 0]
        lo, hi = float(v.min()), float(v.max())
        q = int(self.n_levels)
        if hi <= lo:
            idx = np.zeros(v.shape, dtype=int)
            return idx, np.array([v.size], dtype=float), np.array([lo]), lo, hi
        idx = np.clip(((v - lo) / (hi - lo) * q).astype(int), 0, q - 1)
        counts = np.bincount(idx.ravel(), minlength=q).astype(float)
        values = lo + (np.arange(q) + 0.5) * (hi - lo) / q
        return idx, counts, values, lo, hi

    # ------------------------------------------------------------------ core iteration
    def _effective_distances(self, img, P, d2, U):
        """Effective squared distances of the pixel-based methods, shape (N, c)."""
        H, W = img.shape[:2]
        c = d2.shape[1]
        method = self.method
        if method == "fcm":
            return d2
        if method in ("fcm_s1", "fcm_s2"):
            return d2 + self.alpha * sq_euclidean(self._Pbar, self.cluster_centers_)
        D = d2.reshape(H, W, c)
        if method == "fcm_s":
            N_R = int(self.window_size) ** 2 - 1
            S = neighborhood_sum(D, int(self.window_size))
            return d2 + (self.alpha / N_R) * S.reshape(-1, c)
        # flicm
        G = neighborhood_sum((1.0 - U.reshape(H, W, c)) ** self.m * D, int(self.window_size), kernel=self._K)
        return d2 + G.reshape(-1, c)

    def _update_centers(self, P, Um):
        method = self.method
        if method == "fcm_s":
            N_R = int(self.window_size) ** 2 - 1
            return fuzzy_centers(P + (self.alpha / N_R) * self._Pnb, Um) / (1.0 + self.alpha)
        if method in ("fcm_s1", "fcm_s2"):
            return fuzzy_centers(P + self.alpha * self._Pbar, Um) / (1.0 + self.alpha)
        return fuzzy_centers(P, Um)

    def _objective(self, Um, d2_eff, U, d2):
        if self.method == "flicm":
            return float(np.sum(Um * d2 + (d2_eff - d2)))  # the G term is not weighted by u^m
        return float(np.sum(Um * d2_eff))

    def _init_partition(self, P, rng):
        c, C = self.n_clusters, P.shape[1]
        init = self.init
        if isinstance(init, str):
            if init == "random":
                mean = P.mean(axis=0)
                half_range = np.abs(P - mean).max(axis=0)
                V = 2.0 * half_range * (rng.rand(c, C) - 0.5) + mean
            elif init == "kmeans++":
                from sklearn.cluster import kmeans_plusplus

                V, _ = kmeans_plusplus(P, n_clusters=c, random_state=rng)
            else:
                raise ValueError(f"Unknown init='{init}'. Use 'random', 'kmeans++' or an array.")
            return memberships_from_distances(sq_euclidean(P, V), self.m)
        init = np.asarray(init, dtype=float)
        if init.shape == (c, C):
            return memberships_from_distances(sq_euclidean(P, init), self.m)
        if init.shape == (P.shape[0], c):
            return init / init.sum(axis=1, keepdims=True)
        if init.ndim == 3 and init.shape[2] == c and init.shape[0] * init.shape[1] == P.shape[0]:
            flat = init.reshape(-1, c)
            return flat / flat.sum(axis=1, keepdims=True)
        raise ValueError(
            "init array must have shape (n_clusters, n_channels), (n_pixels, n_clusters) "
            "or (H, W, n_clusters)."
        )

    def _run_pixels(self, img, P, rng):
        U = self._init_partition(P, rng)
        history = []
        for it in range(1, self.max_iter + 1):
            Um = U**self.m
            self.cluster_centers_ = self._update_centers(P, Um)
            d2 = sq_euclidean(P, self.cluster_centers_)
            d2_eff = self._effective_distances(img, P, d2, U)
            U_new = memberships_from_distances(d2_eff, self.m)
            history.append(self._objective(Um, d2_eff, U, d2))
            if self.verbose:
                print(f"iter {it:4d}  J = {history[-1]:.6g}")
            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < self.tol:
                break
        return {"U": U, "V": self.cluster_centers_, "history": history, "n_iter": it}

    def _run_histogram(self, xi, rng):
        idx, counts, values, lo, hi = self._quantize(xi)
        keep = counts > 0
        levels = values[keep][:, None]
        init = self.init
        if isinstance(init, np.ndarray) and init.shape != (self.n_clusters, 1):
            raise ValueError("The histogram methods accept an init array of prototypes (n_clusters, 1) only.")
        n_clusters = min(self.n_clusters, levels.shape[0])
        if n_clusters < self.n_clusters:
            raise ValueError(
                f"Only {levels.shape[0]} distinct gray levels in the summed image, fewer than n_clusters."
            )
        wfcm = WeightedFCM(
            n_clusters=self.n_clusters, m=self.m, tol=self.tol, max_iter=self.max_iter,
            init=init, n_init=self.n_init, random_state=rng, verbose=self.verbose,
        ).fit(levels, sample_weight=counts[keep])
        level_U = np.zeros((values.shape[0], self.n_clusters))
        level_U[keep] = wfcm.membership_
        # memberships of empty bins (never used by a pixel) follow the prototypes
        if not keep.all():
            d2_empty = sq_euclidean(values[~keep][:, None], wfcm.cluster_centers_)
            level_U[~keep] = memberships_from_distances(d2_empty, self.m)
        self.histogram_ = counts
        self.level_values_ = values
        self.level_membership_ = level_U
        self.level_model_ = wfcm
        return {
            "U": level_U[idx.ravel()], "V": wfcm.cluster_centers_,
            "history": list(wfcm.objective_history_), "n_iter": wfcm.n_iter_,
        }

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Segment an image.

        Parameters
        ----------
        X : ndarray of shape (H, W) or (H, W, C)
            Gray-level or multichannel image (any float range; integer
            images are converted).
        y : Ignored

        Returns
        -------
        self : object
        """
        img = self._check_image(X, reset=True)
        self._check_params(img)
        H, W, C = img.shape
        P = img.reshape(-1, C)
        rng = check_random_state(self.random_state)
        filt = self._filtered(img)
        if filt is not None:
            self.filtered_image_ = filt[..., 0] if C == 1 else filt
        if self.method in _HISTOGRAM_METHODS:
            best = self._run_histogram(filt, rng)
        else:
            if self.method in ("fcm_s1", "fcm_s2"):
                self._Pbar = filt.reshape(-1, C)
            if self.method == "fcm_s":
                self._Pnb = neighborhood_sum(img, int(self.window_size)).reshape(-1, C)
            if self.method == "flicm":
                self._K = self._flicm_kernel()
            n_init = 1 if not isinstance(self.init, str) else self.n_init
            best = None
            for run in range(n_init):
                result = self._run_pixels(img, P, rng)
                if best is None or result["history"][-1] < best["history"][-1]:
                    best = result
                if self.verbose and n_init > 1:
                    print(f"restart {run + 1}/{n_init}: J = {result['history'][-1]:.6g}")
        self.cluster_centers_ = best["V"]
        self.membership_ = best["U"].reshape(H, W, self.n_clusters)
        self.labels_ = self.membership_.argmax(axis=2)
        self.objective_history_ = np.asarray(best["history"], dtype=float)
        self.objective_ = float(self.objective_history_[-1])
        self.n_iter_ = int(best["n_iter"])
        return self

    def predict_proba(self, X):
        """Membership images of a (new) image for the fitted prototypes.

        The method's own membership formula is applied with the prototypes
        held fixed: one step for the FCM_S family and the histogram methods
        (whose filtered image is recomputed for the new image), and an
        iteration to convergence of the membership update for FLICM, whose
        fuzzy factor depends on the memberships themselves.

        Parameters
        ----------
        X : ndarray of shape (H, W) or (H, W, C)

        Returns
        -------
        U : ndarray of shape (H, W, n_clusters)
        """
        check_is_fitted(self, "cluster_centers_")
        img = self._check_image(X, reset=False)
        H, W, C = img.shape
        P = img.reshape(-1, C)
        V = self.cluster_centers_
        if self.method in _HISTOGRAM_METHODS:
            xi = self._filtered(img).reshape(-1, 1)
            U = memberships_from_distances(sq_euclidean(xi, V), self.m)
            return U.reshape(H, W, -1)
        d2 = sq_euclidean(P, V)
        if self.method == "fcm":
            d2_eff = d2
        elif self.method in ("fcm_s1", "fcm_s2"):
            d2_eff = d2 + self.alpha * sq_euclidean(self._filtered(img).reshape(-1, C), V)
        elif self.method == "fcm_s":
            N_R = int(self.window_size) ** 2 - 1
            S = neighborhood_sum(d2.reshape(H, W, -1), int(self.window_size))
            d2_eff = d2 + (self.alpha / N_R) * S.reshape(-1, V.shape[0])
        else:  # flicm: fixed-point iteration of the membership update
            U = memberships_from_distances(d2, self.m)
            K = self._flicm_kernel()
            D = d2.reshape(H, W, -1)
            for _ in range(self.max_iter):
                lack = (1.0 - U.reshape(H, W, -1)) ** self.m
                G = neighborhood_sum(lack * D, int(self.window_size), kernel=K)
                U_new = memberships_from_distances(d2 + G.reshape(-1, V.shape[0]), self.m)
                delta = np.max(np.abs(U_new - U))
                U = U_new
                if delta < self.tol:
                    break
            return U.reshape(H, W, -1)
        return memberships_from_distances(d2_eff, self.m).reshape(H, W, -1)

    def predict(self, X):
        """Label image of a (new) image (argmax of :meth:`predict_proba`)."""
        return self.predict_proba(X).argmax(axis=2)

    def fit_predict(self, X, y=None, **kwargs):
        """Segment the image and return its label image."""
        return self.fit(X).labels_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        return tags
