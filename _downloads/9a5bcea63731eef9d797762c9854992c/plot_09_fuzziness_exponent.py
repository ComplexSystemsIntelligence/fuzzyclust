r"""
The fuzziness exponent m: from crisp cells to uniform memberships
==================================================================

The fuzzy c-means functional :math:`J=\sum_i\sum_k\mu_{ik}^m D^2_{ik}` has a
single parameter besides the number of clusters: the weighting exponent
:math:`m>1`. For fixed prototypes the memberships are

.. math:: \mu_{ik} = \frac{1}{\sum_{j=1}^{c}\left(D_{ik}/D_{jk}\right)^{2/(m-1)}},

so :math:`m` controls how fast the membership of a sample decays with its
relative distance to the prototypes. As :math:`m\to1` the exponent
:math:`2/(m-1)` grows without bound and the partition becomes crisp (k-means);
as :math:`m\to\infty` every membership tends to :math:`1/c` and the prototypes
collapse onto the grand mean of the data. This example shows the membership
functions along a line for several values of :math:`m`, the corresponding
partitions of overlapping clusters, and how the partition coefficient, the
spread of the prototypes, and the number of iterations depend on :math:`m`.
"""

# %%
# Membership functions between two fixed prototypes
# -------------------------------------------------
# Two prototypes sit at :math:`v_1=0` and :math:`v_2=1` on a line. The
# membership of a point :math:`x` in the first cluster depends only on the
# ratio of its distances to the two prototypes; the plot is the textbook
# illustration of the role of :math:`m`.
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust import FCM
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.plotting import PALETTE, plot_membership_contours

# [snippet:membership1d]
ms = [1.1, 1.5, 2.0, 3.0, 5.0]
x = np.linspace(-1.0, 2.0, 601)
v = np.array([0.0, 1.0])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for j, m in enumerate(ms):
    d2 = (x[:, None] - v[None, :]) ** 2 + 1e-10  # squared distances to v_1, v_2
    mu = d2 ** (-1.0 / (m - 1.0))
    mu /= mu.sum(axis=1, keepdims=True)
    ax.plot(x, mu[:, 0], lw=2.0, color=PALETTE[j], label=f"m = {m}")
# [/snippet:membership1d]
ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
ax.axhline(0.5, color="0.8", lw=0.8)
ax.set_xlabel("$x$")
ax.set_ylabel(r"membership $\mu_{1}(x)$ in the cluster of $v_1=0$")
ax.set_title(r"membership functions for two prototypes $v_1=0$, $v_2=1$")
ax.legend(frameon=False)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# For :math:`m` close to one the function is a step at the midpoint between
# the prototypes (the Voronoi boundary of k-means); for :math:`m=2` it is the
# classical bell-like curve :math:`\mu_1(x)=(x-1)^2/((x-1)^2+x^2)`; for large
# :math:`m` it is flat at :math:`1/2` almost everywhere. The membership is
# one *only* at the prototype itself and tends to :math:`1/c` far away from
# both prototypes in either direction, a consequence of the probabilistic
# constraint :math:`\sum_i\mu_{ik}=1`.

# %%
# Partitions of overlapping clusters for several m
# ------------------------------------------------
# Three Gaussian clusters whose spread is large relative to the distance
# between them.
X, y_true = make_fuzzy_blobs(n_clusters=3, n_samples_per_cluster=150, spread=1.3, box=8.0, random_state=3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
for ax, m in zip(axes, (1.2, 2.0, 4.0), strict=True):
    model = FCM(n_clusters=3, m=m, random_state=0).fit(X)
    plot_membership_contours(model, X, ax=ax, labels=False)
    ax.set_title(f"FCM with m = {m}")
plt.tight_layout()

# %%
# Quantities that depend on m
# ---------------------------
# We fit FCM on a fine grid of exponents and record the partition
# coefficient :math:`PC=\frac1N\sum_{ik}\mu_{ik}^2`, the mean distance of the
# prototypes from the grand mean of the data (a measure of how far the
# prototypes have collapsed), and the number of iterations.
grand_mean = X.mean(axis=0)
grid = np.array([1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
pc, spread, iters = [], [], []
for m in grid:
    model = FCM(n_clusters=3, m=m, random_state=0, tol=1e-6).fit(X)
    pc.append(np.sum(model.membership_**2) / X.shape[0])
    spread.append(np.linalg.norm(model.cluster_centers_ - grand_mean, axis=1).mean())
    iters.append(model.n_iter_)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
for ax, values, label in zip(axes, (pc, spread, iters),
                             ("partition coefficient PC", "mean distance of prototypes\nfrom the grand mean", "iterations to converge"), strict=True):
    ax.plot(grid, values, marker="o", color=PALETTE[0], lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("fuzziness exponent $m$")
    ax.set_title(label)
    ax.grid(True, color="0.9")
axes[0].axhline(1 / 3, color="0.6", ls="--", lw=0.8)
axes[0].text(1.12, 1 / 3 + 0.02, "$1/c$", color="0.4")
plt.tight_layout()

for m, p, s, it in zip(grid, pc, spread, iters, strict=True):
    print(f"m = {m:5.2f}   PC = {p:.3f}   prototype spread = {s:.2f}   iterations = {it}")

# %%
# **Take-away.** The exponent trades the sharpness of the partition against
# its robustness: small :math:`m` gives crisp, k-means-like partitions that
# react to every sample; large :math:`m` gives fuzzy partitions whose
# prototypes drift toward the grand mean and eventually coincide (the
# partition coefficient then approaches :math:`1/c`). Above :math:`m\approx2`
# the iteration count grows with :math:`m`, because the memberships change
# more slowly; near :math:`m=1` it is also higher, because the nearly crisp
# partition takes several iterations to settle.
# :math:`m=2` is the common default, values in :math:`[1.5, 2.5]` are worth
# trying, and the validity scan can be run over :math:`m` as well as over
# :math:`c`.
