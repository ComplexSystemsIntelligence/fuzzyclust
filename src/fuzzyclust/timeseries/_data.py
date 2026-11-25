# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Synthetic time series and dynamic systems for the time-series examples.

Same conventions as :mod:`fuzzyclust.datasets`: every generator accepts
``random_state`` and returns plain :class:`numpy.ndarray` objects. The
benchmarks reproduce the systems used in the papers of the time-series
sub-package: the piecewise-stationary multivariate series of the
segmentation paper, the continuous polymerization reactor of the FNN
model-order paper, the chaotic systems of the state-space reconstruction
paper and the mixed discrete/continuous regression problem of the
structure-selection paper.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = [
    "make_piecewise_timeseries",
    "make_mackey_glass",
    "make_lorenz",
    "make_rossler",
    "make_nonlinear_io_system",
    "make_mixed_input_system",
]


# ----------------------------------------------------------------------------- segmentation
def make_piecewise_timeseries(
    n_samples: int = 2000,
    n_features: int = 6,
    n_latent: int = 2,
    change_points=(500, 1000, 1500),
    change_kinds=("correlation", "mean", "both"),
    mean_shift: float = 4.0,
    noise: float = 0.2,
    latent_noise: float = 0.5,
    random_state=None,
):
    r"""Multivariate series with planted regime changes (segmentation benchmark).

    Mimics the synthetic data set of Abonyi et al. (2005): ``n_latent`` hidden
    variables :math:`\mathbf{y}_k` are drawn around a piecewise-constant mean,
    and the observed variables are linear mixtures
    :math:`\mathbf{x}_k=\mathbf{W}\mathbf{y}_k+\boldsymbol{\varepsilon}_k`.
    At every change point one of three things happens:

    * ``"mean"`` -- the mean of the latent variables jumps (the hyperplane of
      the observations is shifted but keeps its orientation);
    * ``"correlation"`` -- the mixing matrix :math:`\mathbf{W}` is replaced
      (the correlation structure of the observations changes while the mean
      is kept);
    * ``"both"`` -- both change.

    The true segment of every sample is returned, so that segmentation
    algorithms can be evaluated. A segmentation that uses only one variable
    cannot see a pure correlation change.

    Parameters
    ----------
    n_samples : int, default=2000
    n_features : int, default=6
        Number of observed variables :math:`n`.
    n_latent : int, default=2
        Number of latent variables (the rank of the local PCA models).
    change_points : sequence of int, default=(500, 1000, 1500)
        Sample indices where a new segment starts.
    change_kinds : sequence of {"mean", "correlation", "both"}
        What changes at every change point (same length as ``change_points``).
    mean_shift : float, default=4.0
        Scale of the latent mean jumps.
    noise : float, default=0.2
        Standard deviation of the observation noise.
    latent_noise : float, default=0.5
        Standard deviation of the latent variables around their mean.
    random_state : int, RandomState instance or None

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        True segment index of every sample (``0, 1, ...`` in time order).

    Examples
    --------
    >>> from fuzzyclust.timeseries import make_piecewise_timeseries
    >>> X, seg = make_piecewise_timeseries(n_samples=400, change_points=(200,),
    ...                                    change_kinds=("mean",), random_state=0)
    >>> X.shape, seg.max()
    ((400, 6), np.int64(1))
    """
    rng = check_random_state(random_state)
    change_points = list(change_points)
    change_kinds = list(change_kinds)
    if len(change_points) != len(change_kinds):
        raise ValueError("change_points and change_kinds must have the same length.")
    if any(k not in ("mean", "correlation", "both") for k in change_kinds):
        raise ValueError("change_kinds must contain 'mean', 'correlation' or 'both'.")
    bounds = [0, *change_points, n_samples]
    if any(b <= a for a, b in zip(bounds[:-1], bounds[1:], strict=True)):
        raise ValueError("change_points must be increasing and inside (0, n_samples).")

    def random_mixing():
        W = rng.randn(n_features, n_latent)
        W /= np.linalg.norm(W, axis=0)
        return W

    mean = np.zeros(n_latent)
    W = random_mixing()
    X = np.empty((n_samples, n_features))
    y = np.empty(n_samples, dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        if s > 0:
            kind = change_kinds[s - 1]
            if kind in ("mean", "both"):
                mean = mean + mean_shift * rng.randn(n_latent)
            if kind in ("correlation", "both"):
                W = random_mixing()
        latent = mean + latent_noise * rng.randn(b - a, n_latent)
        X[a:b] = latent @ W.T + noise * rng.randn(b - a, n_features)
        y[a:b] = s
    return X, y


# ----------------------------------------------------------------------------- chaotic systems
def _rk4(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def make_mackey_glass(
    n_samples: int = 2000,
    tau: int = 17,
    a: float = 0.2,
    b: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    x0: float = 1.2,
    discard: int = 1000,
    substeps: int = 10,
):
    r"""Mackey–Glass delay-differential series (chaotic for :math:`\tau=17`).

    .. math::
        \frac{dx}{dt} = \frac{a\,x(t-\tau)}{1+x(t-\tau)^{n}} - b\,x(t)

    integrated with a fourth-order Runge–Kutta scheme on a grid of
    ``dt / substeps`` with linear interpolation of the delayed value, and
    sampled every ``dt``. The first ``discard`` samples (transient) are
    dropped. ``tau=17`` gives the classical benchmark with a fractal
    dimension of about 2.1, ``tau=30`` a higher-dimensional attractor.

    Returns
    -------
    x : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.timeseries import make_mackey_glass
    >>> x = make_mackey_glass(n_samples=500)
    >>> x.shape, bool(0.2 < x.min() < x.max() < 1.6)
    ((500,), True)
    """
    h = dt / substeps
    delay_steps = int(round(tau / h))
    total = (n_samples + discard) * substeps
    buf = np.full(total + delay_steps + 1, x0, dtype=float)

    def f(x, xd):
        return a * xd / (1.0 + xd**n) - b * x

    for i in range(delay_steps, total + delay_steps):
        xd = buf[i - delay_steps]
        xd_half = 0.5 * (buf[i - delay_steps] + buf[i - delay_steps + 1])
        xd_full = buf[i - delay_steps + 1]
        x = buf[i]
        k1 = f(x, xd)
        k2 = f(x + 0.5 * h * k1, xd_half)
        k3 = f(x + 0.5 * h * k2, xd_half)
        k4 = f(x + h * k3, xd_full)
        buf[i + 1] = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    series = buf[delay_steps::substeps][: n_samples + discard]
    return series[discard:].copy()


def make_lorenz(
    n_samples: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0=(1.0, 1.0, 1.0),
    discard: int = 1000,
):
    r"""Trajectory of the Lorenz system (RK4, fixed step).

    .. math::
        \dot x = \sigma(y-x),\quad \dot y = x(\rho-z)-y,\quad \dot z = xy-\beta z .

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        State trajectory after the transient of ``discard`` steps.

    Examples
    --------
    >>> from fuzzyclust.timeseries import make_lorenz
    >>> make_lorenz(n_samples=100).shape
    (100, 3)
    """

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    s = np.asarray(x0, dtype=float)
    out = np.empty((n_samples + discard, 3))
    for k in range(n_samples + discard):
        out[k] = s
        s = _rk4(f, s, dt)
    return out[discard:]


def make_rossler(
    n_samples: int = 5000,
    dt: float = 0.05,
    a: float = 0.38,
    b: float = 0.3,
    c: float = 4.5,
    x0=(0.0, 0.0, 0.0),
    discard: int = 1000,
):
    r"""Trajectory of the Rössler system with the parameters of Abonyi et al. (2004).

    .. math::
        \dot x_1 = -(x_2+x_3),\quad \dot x_2 = x_1 + a x_2,\quad \dot x_3 = b + x_3(x_1-c).

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
    """

    def f(s):
        x1, x2, x3 = s
        return np.array([-(x2 + x3), x1 + a * x2, b + x3 * (x1 - c)])

    s = np.asarray(x0, dtype=float)
    out = np.empty((n_samples + discard, 3))
    for k in range(n_samples + discard):
        out[k] = s
        s = _rk4(f, s, dt)
    return out[discard:]


# ----------------------------------------------------------------------------- input-output benchmark
def _polymerization_ode(x, u):
    x1, x2, x3, x4 = x
    sx2 = np.sqrt(max(x2, 0.0))
    return np.array(
        [
            10.0 * (6.0 - x1) - 2.4568 * x1 * sx2,
            80.0 * u - 10.1022 * x2,
            0.0024121 * x1 * sx2 + 0.112191 * x2 - 10.0 * x3,
            245.978 * x1 * sx2 - 10.0 * x4,
        ]
    )


def make_nonlinear_io_system(
    n_samples: int = 2000,
    dt: float = 0.2,
    u_range=(0.007, 0.015),
    substeps: int = 10,
    discard: int = 50,
    random_state=None,
):
    r"""Continuous polymerization reactor benchmark of the FNN model-order paper.

    Free-radical polymerization of methyl methacrylate (Doyle, Ogunnaike &
    Pearson, 1995) as used by Feil, Abonyi & Szeifert:

    .. math::
        \dot x_1 &= 10(6-x_1) - 2.4568\,x_1\sqrt{x_2},\\
        \dot x_2 &= 80u - 10.1022\,x_2,\\
        \dot x_3 &= 0.0024121\,x_1\sqrt{x_2} + 0.112191\,x_2 - 10x_3,\\
        \dot x_4 &= 245.978\,x_1\sqrt{x_2} - 10x_4,\qquad y = x_4/x_3 .

    The input :math:`u` (dimensionless initiator flow) is a uniformly
    distributed random sequence in ``u_range`` held constant over each
    sampling period ``dt`` (zero-order hold); the output :math:`y` is the
    number-average molecular weight, roughly in the range 26 000–34 000.
    (The FNN paper prints the first coefficient of :math:`\dot x_3` as
    0.024121; the value 0.0024121 of the original model is used here, which
    reproduces the stated output range.)
    A NARX model with one lagged output and two lagged inputs
    (:math:`n_a=1`, :math:`n_b=2`) describes the system well, which is the
    reference answer of the model-order selection example.

    Returns
    -------
    u : ndarray of shape (n_samples,)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from fuzzyclust.timeseries import make_nonlinear_io_system
    >>> u, y = make_nonlinear_io_system(n_samples=200, random_state=0)
    >>> u.shape, bool(20000 < y.mean() < 40000)
    ((200,), True)
    """
    rng = check_random_state(random_state)
    u = rng.uniform(u_range[0], u_range[1], size=n_samples + discard)
    # steady state for the mean input as the initial condition
    u0 = 0.5 * (u_range[0] + u_range[1])
    x2 = 80.0 * u0 / 10.1022
    x1 = 60.0 / (10.0 + 2.4568 * np.sqrt(x2))
    x3 = (0.0024121 * x1 * np.sqrt(x2) + 0.112191 * x2) / 10.0
    x4 = 245.978 * x1 * np.sqrt(x2) / 10.0
    x = np.array([x1, x2, x3, x4])
    h = dt / substeps
    y = np.empty(n_samples + discard)
    for k in range(n_samples + discard):
        y[k] = x[3] / x[2]
        for _ in range(substeps):
            x = _rk4(_polymerization_ode, x, h, u[k])
    return u[discard:], y[discard:]


# ----------------------------------------------------------------------------- mixed inputs
def make_mixed_input_system(n_samples: int = 1000, noise: float = 0.1, n_dummy: int = 2, random_state=None):
    r"""Regression problem with one discrete and several continuous inputs.

    Example B of Girimonte, Babuška & Abonyi:

    .. math::
        y = \begin{cases} x_1^2 + \epsilon & \text{if } x_2 = 1,\\
                          x_4^2 + x_3 + \epsilon & \text{if } x_2 = 0,\end{cases}
        \qquad \epsilon\sim\mathcal{N}(0, \text{noise}^2),

    with :math:`x_1, x_3, x_4\sim U[0,1]`, the switching input
    :math:`x_2\in\{0,1\}` drawn at random, and ``n_dummy`` irrelevant
    uniform inputs appended (:math:`x_5, x_6`). The correct structure is
    :math:`\{x_1, x_2, x_3, x_4\}`.

    Returns
    -------
    X : ndarray of shape (n_samples, 4 + n_dummy)
    y : ndarray of shape (n_samples,)
    discrete : ndarray of bool, shape (4 + n_dummy,)
        Mask of the discrete (categorical) inputs (only :math:`x_2`).

    Examples
    --------
    >>> from fuzzyclust.timeseries import make_mixed_input_system
    >>> X, y, disc = make_mixed_input_system(n_samples=50, random_state=0)
    >>> X.shape, disc.tolist()
    ((50, 6), [False, True, False, False, False, False])
    """
    rng = check_random_state(random_state)
    n_in = 4 + n_dummy
    X = rng.rand(n_samples, n_in)
    X[:, 1] = (rng.rand(n_samples) < 0.5).astype(float)
    eps = noise * rng.randn(n_samples)
    y = np.where(X[:, 1] == 1.0, X[:, 0] ** 2, X[:, 3] ** 2 + X[:, 2]) + eps
    discrete = np.zeros(n_in, dtype=bool)
    discrete[1] = True
    return X, y, discrete
