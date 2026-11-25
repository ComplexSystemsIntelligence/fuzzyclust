# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Synthetic regression, classification and dynamic-system data for model identification.

The generators follow the conventions of :mod:`fuzzyclust.datasets`: static
problems return ``(X, y)`` (plus a component label where one exists), dynamic
processes return the input and output *sequences* ``(u, y)`` from which
:func:`fuzzyclust.preprocessing.make_regressors` builds the NARX regression
matrix. Every generator accepts ``random_state``.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = [
    "make_sinc",
    "make_piecewise_linear",
    "make_switching_lines",
    "make_second_order_plant",
    "make_mimo_plant",
    "make_two_class_overlap",
    "make_banded_classes",
]


def make_sinc(n_samples: int = 200, noise: float = 0.02, n_features: int = 1, random_state=None):
    r"""Samples of the sinc function, the classic static benchmark of fuzzy modeling.

    .. math::
        y = \frac{\sin x}{x}\quad (n=1),\qquad
        y = \frac{\sin x_1}{x_1}\,\frac{\sin x_2}{x_2}\quad (n=2),

    with :math:`x_j` drawn uniformly from :math:`[-10, 10]` and additive
    Gaussian noise of standard deviation ``noise``.

    Parameters
    ----------
    n_samples : int, default=200
    noise : float, default=0.02
    n_features : {1, 2}, default=1
    random_state : int, RandomState instance or None, default=None

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.models import make_sinc
    >>> X, y = make_sinc(n_samples=50, random_state=0)
    >>> X.shape, y.shape
    ((50, 1), (50,))
    """
    if n_features not in (1, 2):
        raise ValueError("n_features must be 1 or 2.")
    rng = check_random_state(random_state)
    X = rng.uniform(-10.0, 10.0, size=(n_samples, n_features))
    y = np.prod(np.sinc(X / np.pi), axis=1)  # numpy sinc is sin(pi x)/(pi x)
    y = y + noise * rng.randn(n_samples)
    return X, y


def make_piecewise_linear(n_samples: int = 400, noise: float = 0.05, random_state=None):
    r"""A continuous piecewise-linear function of two variables (sum of two hinges).

    .. math::
        y = \max(x_1 + x_2,\; 0.5 - x_1 + 0.5 x_2) + \max(0,\; x_2 - 0.5 x_1 - 0.5)

    on :math:`[-2, 2]^2`, i.e. two hinging-hyperplane functions whose hinges
    cross. The surface has four affine pieces and is the natural test case for
    :class:`~fuzzyclust.models.HingingHyperplaneTree`.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    """
    rng = check_random_state(random_state)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    y = piecewise_linear_function(X) + noise * rng.randn(n_samples)
    return X, y


def piecewise_linear_function(X) -> np.ndarray:
    """Noise-free value of the function behind :func:`make_piecewise_linear`."""
    X = np.asarray(X, dtype=float)
    x1, x2 = X[:, 0], X[:, 1]
    return np.maximum(x1 + x2, 0.5 - x1 + 0.5 * x2) + np.maximum(0.0, x2 - 0.5 * x1 - 0.5)


def make_switching_lines(
    n_samples: int = 200,
    slopes=(1.0, -1.0),
    intercepts=(0.0, 1.0),
    noise: float = 0.1,
    x_range=(-2.0, 2.0),
    random_state=None,
):
    r"""Switching regression data: samples of several lines over a common input range.

    Every sample :math:`(x_k, y_k)` belongs to one of :math:`c` lines
    :math:`y = a_i x + b_i`, chosen at random; because the lines share the
    input interval they cross each other and cannot be separated by clustering
    the inputs alone. This is the setting of fuzzy c-regression models.

    Parameters
    ----------
    n_samples : int, default=200
    slopes, intercepts : sequences of length c
    noise : float, default=0.1
    x_range : tuple, default=(-2, 2)
    random_state : int, RandomState instance or None, default=None

    Returns
    -------
    X : ndarray of shape (n_samples, 1)
    y : ndarray of shape (n_samples,)
    labels : ndarray of shape (n_samples,)
        Index of the generating line.
    """
    slopes = np.asarray(slopes, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    if slopes.shape != intercepts.shape:
        raise ValueError("slopes and intercepts must have the same length.")
    rng = check_random_state(random_state)
    labels = rng.randint(0, slopes.shape[0], size=n_samples)
    x = rng.uniform(x_range[0], x_range[1], size=n_samples)
    y = slopes[labels] * x + intercepts[labels] + noise * rng.randn(n_samples)
    return x[:, None], y, labels


def second_order_plant_function(y1, y2):
    r"""Nonlinear component :math:`f(y(k-1), y(k-2))` of :func:`make_second_order_plant`."""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    return y1 * y2 * (y1 - 0.5) / (1.0 + y1**2 + y2**2)


def make_second_order_plant(
    n_samples: int = 200, excitation: str = "random", noise: float = 0.0, random_state=None
):
    r"""The second-order nonlinear benchmark plant of Abonyi, Babuška and Szeifert (2002).

    .. math::
        y(k) = f\big(y(k-1), y(k-2)\big) + u(k),\qquad
        f(y_1, y_2) = \frac{y_1 y_2 (y_1 - 0.5)}{1 + y_1^2 + y_2^2}.

    Following the paper, the identification data use a random input uniformly
    distributed in :math:`[-1.5, 1.5]` and the validation data the sinusoid
    :math:`u(k)=\sin(2\pi k/25)`.

    Parameters
    ----------
    n_samples : int, default=200
    excitation : {"random", "sine"}, default="random"
    noise : float, default=0.0
        Standard deviation of white noise added to the output.
    random_state : int, RandomState instance or None, default=None

    Returns
    -------
    u : ndarray of shape (n_samples,)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.models import make_second_order_plant
    >>> u, y = make_second_order_plant(n_samples=100, random_state=0)
    >>> u.shape, y.shape
    ((100,), (100,))
    """
    rng = check_random_state(random_state)
    if excitation == "random":
        u = rng.uniform(-1.5, 1.5, size=n_samples)
    elif excitation == "sine":
        u = np.sin(2.0 * np.pi * np.arange(1, n_samples + 1) / 25.0)
    else:
        raise ValueError("excitation must be 'random' or 'sine'.")
    y = np.zeros(n_samples)
    for k in range(2, n_samples):
        y[k] = second_order_plant_function(y[k - 1], y[k - 2]) + u[k]
    if noise > 0:
        y = y + noise * rng.randn(n_samples)
    return u, y


def _random_steps(rng, n_samples, n_inputs, hold, low=-1.0, high=1.0):
    n_steps = int(np.ceil(n_samples / hold))
    levels = rng.uniform(low, high, size=(n_steps, n_inputs))
    return np.repeat(levels, hold, axis=0)[:n_samples]


def make_mimo_plant(n_samples: int = 600, hold: int = 10, noise: float = 0.0, random_state=None):
    r"""A two-input two-output nonlinear process with operating-point-dependent gains.

    .. math::
        \begin{aligned}
        y_1(k+1) &= 0.8\,y_1(k) - 0.1\,y_2(k) + g_1(y_1(k))\,u_1(k) + 0.1\,u_2(k),\\
        y_2(k+1) &= 0.1\,y_1(k) + 0.7\,y_2(k) + 0.2\,u_1(k) + g_2(y_2(k))\,u_2(k),
        \end{aligned}

    with the gains :math:`g_1(y) = 0.2 + 0.4/(1+y^2)` and
    :math:`g_2(y) = 0.1 + 0.5\,\exp(-y^2)`: the process is linear in the lagged
    variables but its input gains shrink as the outputs move away from zero,
    exactly the kind of behavior that a Takagi–Sugeno model with scheduling on
    the outputs interpolates between local linear models. The inputs are random
    step sequences in :math:`[-1, 1]`, held for ``hold`` samples.

    Returns
    -------
    u : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples, 2)

    Examples
    --------
    >>> from fuzzyclust.models import make_mimo_plant
    >>> u, y = make_mimo_plant(n_samples=100, random_state=0)
    >>> u.shape, y.shape
    ((100, 2), (100, 2))
    """
    rng = check_random_state(random_state)
    u = _random_steps(rng, n_samples, 2, hold)
    y = np.zeros((n_samples, 2))
    for k in range(n_samples - 1):
        y1, y2 = y[k]
        g1 = 0.2 + 0.4 / (1.0 + y1**2)
        g2 = 0.1 + 0.5 * np.exp(-(y2**2))
        y[k + 1, 0] = 0.8 * y1 - 0.1 * y2 + g1 * u[k, 0] + 0.1 * u[k, 1]
        y[k + 1, 1] = 0.1 * y1 + 0.7 * y2 + 0.2 * u[k, 0] + g2 * u[k, 1]
    if noise > 0:
        y = y + noise * rng.randn(*y.shape)
    return u, y


def make_two_class_overlap(n_samples: int = 400, spread: float = 0.6, random_state=None):
    r"""Two overlapping classes, each a mixture of two Gaussian blobs (XOR layout).

    Class 0 occupies the blobs centered at :math:`(-1.5, 0)` and :math:`(1.5, 0)`,
    class 1 those at :math:`(0, 1.5)` and :math:`(0, -1.5)`. Neither class is a
    single ellipsoid, so a classifier with one rule per class fails while a
    fuzzy classifier with more rules than classes succeeds.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
        Class labels in ``{0, 1}``.

    Examples
    --------
    >>> from fuzzyclust.models import make_two_class_overlap
    >>> X, y = make_two_class_overlap(n_samples=100, random_state=0)
    >>> sorted(set(y.tolist()))
    [0, 1]
    """
    rng = check_random_state(random_state)
    centers = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 1.5], [0.0, -1.5]])
    comp = rng.randint(0, 4, size=n_samples)
    X = centers[comp] + spread * rng.randn(n_samples, 2)
    y = (comp >= 2).astype(int)
    return X, y


def make_banded_classes(n_samples: int = 450, random_state=None):
    r"""Three classes separated by thresholds on two features, plus one irrelevant feature.

    Class 0 lies at :math:`x_1\approx-2` with a wide spread in :math:`x_2`;
    classes 1 and 2 share :math:`x_1\approx0.5` and are separated along
    :math:`x_2` (centers :math:`1.8` and :math:`-1.0`). The third feature is
    standard normal noise unrelated to the class. The class boundaries are
    thus axis-parallel but sit at positions that neither equal-width nor
    equal-frequency binning hits, and one feature is a "don't care"
    variable -- the situation in which a supervised discretization pays off
    (Abonyi 2010).

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
    y : ndarray of shape (n_samples,)
        Class labels in ``{0, 1, 2}``.

    Examples
    --------
    >>> from fuzzyclust.models import make_banded_classes
    >>> X, y = make_banded_classes(n_samples=90, random_state=0)
    >>> X.shape, sorted(set(y.tolist()))
    ((90, 3), [0, 1, 2])
    """
    rng = check_random_state(random_state)
    y = rng.randint(0, 3, size=n_samples)
    m1, s1 = np.array([-2.0, 0.5, 0.5]), np.array([0.4, 0.4, 0.4])
    m2, s2 = np.array([0.0, 1.8, -1.0]), np.array([1.2, 0.4, 0.5])
    X = np.empty((n_samples, 3))
    X[:, 0] = m1[y] + s1[y] * rng.randn(n_samples)
    X[:, 1] = m2[y] + s2[y] * rng.randn(n_samples)
    X[:, 2] = rng.randn(n_samples)
    return X, y
