# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Synthetic test images for segmentation: piecewise-constant shapes under noise."""

from __future__ import annotations

import numpy as np
from sklearn.utils import Bunch, check_random_state

__all__ = ["make_synthetic_image", "add_image_noise"]


def add_image_noise(
    image, noise: str = "gaussian", noise_level: float = 0.1, random_state=None, value_range=None
):
    r"""Corrupt an image with Gaussian, salt-and-pepper or both kinds of noise.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, C)
    noise : {"gaussian", "salt_pepper", "both", "none"}, default="gaussian"
    noise_level : float, default=0.1
        Standard deviation of the Gaussian noise (in the units of the
        image) and/or the fraction of pixels hit by salt-and-pepper noise.
    random_state : int, RandomState instance or None
    value_range : tuple (low, high), optional
        Values of "pepper" and "salt"; default the minimum and maximum of
        the image.

    Returns
    -------
    noisy : ndarray, same shape as ``image``
    """
    img = np.asarray(image, dtype=float)
    if noise not in ("gaussian", "salt_pepper", "both", "none"):
        raise ValueError("noise must be 'gaussian', 'salt_pepper', 'both' or 'none'.")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")
    rng = check_random_state(random_state)
    out = img.copy()
    if noise in ("gaussian", "both"):
        out = out + rng.randn(*img.shape) * noise_level
    if noise in ("salt_pepper", "both"):
        lo, hi = (float(img.min()), float(img.max())) if value_range is None else value_range
        hit = rng.rand(*img.shape[:2]) < min(noise_level, 1.0)
        salt = rng.rand(*img.shape[:2]) < 0.5
        out[hit & salt] = hi
        out[hit & ~salt] = lo
    return out


def make_synthetic_image(
    shape=(128, 128),
    n_classes: int = 4,
    n_channels: int = 1,
    noise: str = "gaussian",
    noise_level: float = 0.1,
    random_state=None,
):
    r"""A piecewise-constant image of simple shapes with known labels, plus a noisy copy.

    The image is tiled into ``n_classes`` polygonal regions (the Voronoi
    cells of seed points spread over the image by farthest-point
    sampling, so the regions have comparable areas and straight edges),
    and every class additionally owns one disc placed inside another
    class's cell (curved edges and small isolated regions). Gray levels
    are equally spaced in :math:`[0,1]`; with ``n_channels > 1`` every
    class gets a random color vector in :math:`[0,1]^C`. The noisy copy
    is produced by :func:`add_image_noise`. Spatially unaware pixel
    clustering fails on such an image as soon as the noise is comparable
    to the gap between the gray levels, which is exactly what the
    spatially constrained methods are made for.

    Parameters
    ----------
    shape : tuple (H, W), default=(128, 128)
    n_classes : int, default=4
    n_channels : int, default=1
    noise : {"gaussian", "salt_pepper", "both", "none"}, default="gaussian"
    noise_level : float, default=0.1
    random_state : int, RandomState instance or None

    Returns
    -------
    data : :class:`sklearn.utils.Bunch`
        ``image`` -- the noisy image ``(H, W)`` or ``(H, W, C)``;
        ``clean`` -- the noise-free image; ``truth`` -- label image
        ``(H, W)`` with values ``0..n_classes-1``; ``levels`` -- the class
        values ``(n_classes,)`` or ``(n_classes, C)``.

    Examples
    --------
    >>> from fuzzyclust.image import make_synthetic_image
    >>> data = make_synthetic_image(shape=(64, 64), n_classes=3, random_state=0)
    >>> data.image.shape, data.truth.shape, sorted(set(data.truth.ravel().tolist()))
    ((64, 64), (64, 64), [0, 1, 2])
    """
    H, W = int(shape[0]), int(shape[1])
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1.")
    rng = check_random_state(random_state)
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.c_[yy.ravel() / H, xx.ravel() / W]
    # seeds by farthest-point sampling from a random start: cells of comparable area
    seeds = [rng.uniform(0.15, 0.85, size=2)]
    while len(seeds) < n_classes:
        d = np.min([((coords - s) ** 2).sum(axis=1) for s in seeds], axis=0)
        inner = (coords[:, 0] > 0.1) & (coords[:, 0] < 0.9) & (coords[:, 1] > 0.1) & (coords[:, 1] < 0.9)
        d = np.where(inner, d, -1.0)
        seeds.append(coords[int(d.argmax())])
    seeds = np.asarray(seeds)
    d2 = ((coords[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    truth = d2.argmin(axis=1).reshape(H, W)
    # one disc per class inside the cell of another class
    r = 0.09 * min(H, W)
    for cls in range(n_classes):
        host = (cls + 1) % n_classes
        cy, cx = seeds[host, 0] * H, seeds[host, 1] * W
        shift = rng.uniform(-0.6, 0.6, size=2) * r
        mask = (yy - (cy + shift[0])) ** 2 + (xx - (cx + shift[1])) ** 2 <= r**2
        truth[mask] = cls
    if n_channels == 1:
        levels = np.linspace(0.0, 1.0, n_classes)
        clean = levels[truth]
    else:
        levels = rng.rand(n_classes, n_channels)
        clean = levels[truth]
    noisy = add_image_noise(
        clean, noise=noise, noise_level=noise_level, random_state=rng, value_range=(0.0, 1.0)
    )
    return Bunch(image=noisy, clean=clean, truth=truth, levels=levels)
