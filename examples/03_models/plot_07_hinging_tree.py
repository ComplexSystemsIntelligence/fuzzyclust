r"""
Hinging-hyperplane regression tree identified by fuzzy c-regression
===================================================================

A *hinging hyperplane* (Breiman) is the continuous piecewise-linear function
:math:`h(\mathbf{x})=\max(\mathbf{x}_e^T\boldsymbol\theta_1,\ \mathbf{x}_e^T\boldsymbol\theta_2)`
or the corresponding :math:`\min`: two planes glued together along the
*hinge* :math:`\{\mathbf{x}:\ \mathbf{x}_e^T(\boldsymbol\theta_1-\boldsymbol\theta_2)=0\}`.
Kenesei and Abonyi observed that fuzzy c-regression with :math:`c=2` finds
two local linear models and their fuzzy operating regions at once, that is,
one hinge, provided the two planes cross inside the data (a constraint added
to the weighted least-squares step). The sign of the hinge function then
splits the data into the two operating regions, each of which is again
modeled by a hinge, and so on. The result is a binary regression tree with
*oblique* splits whose leaves are hinge functions. No splitting variable has
to be chosen, the tree is piecewise linear, and every node reports how well
the crisp split agrees with the fuzzy partition. This example fits a surface
made of two crossing hinges and studies the depth, the leaf size, and the
growth strategy.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuzzyclust.models import (
    FuzzyCRegression,
    HingingHyperplaneTree,
    make_piecewise_linear,
    piecewise_linear_function,
    plot_hinge_tree_surface,
)
from fuzzyclust.plotting import PALETTE

# %%
# One hinge from fuzzy c-regression (one input)
# ---------------------------------------------
# The data are noisy samples of :math:`y=\max(0.8x,\ -1.2x+1)`. FCRM with
# two models finds the two lines; the hinge is where they cross, and the
# membership functions of the two models form a soft version of the split.
rng = np.random.RandomState(0)
x = rng.uniform(-2, 3, 200)
y1d = np.maximum(0.8 * x, -1.2 * x + 1.0) + 0.1 * rng.randn(200)
fcrm = FuzzyCRegression(n_clusters=2, n_init=5, random_state=0).fit(x[:, None], y1d)
hinge1d = HingingHyperplaneTree(max_depth=0, random_state=0).fit(x[:, None], y1d)
grid = np.linspace(-2, 3, 300)[:, None]
fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
axes[0].scatter(x, y1d, s=10, color="0.5", alpha=0.7, label="data")
for i in range(2):
    axes[0].plot(grid[:, 0], fcrm.predict(grid)[:, i], color=PALETTE[i], lw=1.3, ls="--", label=f"FCRM model {i + 1}")
axes[0].plot(grid[:, 0], hinge1d.predict(grid), "k-", lw=2.2, label=f"hinge function ({hinge1d.tree_.kind})")
x_hinge = -(hinge1d.tree_.theta[0, 1] - hinge1d.tree_.theta[1, 1]) / (hinge1d.tree_.theta[0, 0] - hinge1d.tree_.theta[1, 0])
axes[0].axvline(x_hinge, color="0.3", ls=":", lw=1)
axes[0].set_ylabel("y")
axes[0].set_title(f"Two FCRM lines form a hinge at x = {x_hinge:.2f} (true 0.5)")
axes[0].legend(frameon=False, fontsize=8)
U = fcrm.predict_proba(x[:, None], y1d)
for i in range(2):
    axes[1].scatter(x, U[:, i], s=8, color=PALETTE[i])
axes[1].axvline(x_hinge, color="0.3", ls=":", lw=1)
axes[1].set_xlabel("x")
axes[1].set_ylabel(r"$\mu_{ik}$")
axes[1].set_title("FCRM memberships: the fuzzy operating regions of the two lines")
plt.tight_layout()

# %%
# The node condition :math:`\varrho=1-|m_1-\sum_k\mu_{1k}|/\sum_{ik}\mu_{ik}`
# compares the number of samples the crisp hinge sends to the left with the
# membership mass of the first model; a value near one means the hinge
# agrees with the fuzzy partition.
print(f"condition of the root node: {hinge1d.tree_.condition:.3f}, RMSE {hinge1d.tree_.rmse:.3f}")

# %%
# A surface with two crossing hinges
# ----------------------------------
# :func:`~fuzzyclust.models.make_piecewise_linear` samples
# :math:`y=\max(x_1+x_2,\ 0.5-x_1+0.5x_2)+\max(0,\ x_2-0.5x_1-0.5)`, which
# consists of four affine pieces separated by two oblique creases. A tree of
# depth one (a root hinge and two leaf hinges, four planes) could represent
# the surface exactly if the root hinge coincided with the first crease.
# However, the root hinge is fitted to all the data, which are not planar on
# either side of the crease, so it is only approximately right, and deeper
# trees refine the pieces.
X, y = make_piecewise_linear(n_samples=400, noise=0.05, random_state=0)
X_test, _ = make_piecewise_linear(n_samples=2000, noise=0.0, random_state=1)
y_test = piecewise_linear_function(X_test)

# [snippet:fit]
tree = HingingHyperplaneTree(max_depth=2, min_samples_leaf=10, random_state=0).fit(X, y)
print(tree.describe(feature_names=["x1", "x2"], digits=3))
print(tree.node_table().round(3).to_string(index=False))
print(f"test RMSE: {np.sqrt(np.mean((tree.predict(X_test) - y_test) ** 2)):.4f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
gx, gy = np.meshgrid(np.linspace(-2, 2, 120), np.linspace(-2, 2, 120))
G = np.c_[gx.ravel(), gy.ravel()]
cs = axes[0].contourf(gx, gy, piecewise_linear_function(G).reshape(gx.shape), levels=14, cmap="viridis", alpha=0.85)
plt.colorbar(cs, ax=axes[0], label="$y$", fraction=0.046, pad=0.03)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=8, edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
axes[0].set_title("True surface and the 400 noisy samples")
plot_hinge_tree_surface(tree, X, ax=axes[1])
axes[1].set_title(f"Hinge tree of depth {tree.depth_} ({tree.n_leaves_} leaves): internal hinges black, leaf hinges white")
plt.tight_layout()

# %%
# The root hinge (thick black line) splits the plane obliquely close to the
# main crease of the surface. The hinges of the children (thin dashed white
# lines) model the remaining kinks inside their regions, and one child is
# split once more. The root reports a condition of one (its crisp split and
# the fuzzy partition agree), whereas the deepest node has a lower
# :math:`\varrho`: its two planes fit a region whose boundary the hinge
# follows only roughly, and this is where the remaining error lies.
#
# Parameter study: depth, leaf size, and growth strategy
# ------------------------------------------------------
# A deeper tree has more pieces. ``min_samples_leaf`` prevents splits that
# leave too few samples for a two-plane fit. ``growth="greedy"`` splits the
# worst leaf first until ``max_leaves`` leaves exist, which gives unbalanced
# trees that spend their pieces where the error is. ``leaf_model="linear"``
# replaces every leaf hinge by a single plane.

# [snippet:scan]
rows = []
for depth in (0, 1, 2, 3, 4):
    for leaf in (10, 40):
        t = HingingHyperplaneTree(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)
        rows.append(
            {
                "growth": "balanced",
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "leaves": t.n_leaves_,
                "train RMSE": np.sqrt(np.mean((t.predict(X) - y) ** 2)),
                "test RMSE": np.sqrt(np.mean((t.predict(X_test) - y_test) ** 2)),
            }
        )
for n_leaves in (2, 3, 4, 6, 8):
    t = HingingHyperplaneTree(max_depth=6, growth="greedy", max_leaves=n_leaves, random_state=0).fit(X, y)
    rows.append(
        {
            "growth": "greedy",
            "max_depth": t.depth_,
            "min_samples_leaf": 10,
            "leaves": t.n_leaves_,
            "train RMSE": np.sqrt(np.mean((t.predict(X) - y) ** 2)),
            "test RMSE": np.sqrt(np.mean((t.predict(X_test) - y_test) ** 2)),
        }
    )
scan = pd.DataFrame(rows)
# [/snippet:scan]
print(scan.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4))
for i, ((growth, leaf), sub) in enumerate(scan.groupby(["growth", "min_samples_leaf"])):
    label = f"{growth}, min_samples_leaf={leaf}" if growth == "balanced" else "greedy (max_leaves 2..8)"
    ax.plot(sub["leaves"], sub["test RMSE"], "o-", color=PALETTE[i], label=label)
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("number of leaves")
ax.set_ylabel("test RMSE (noise-free surface)")
ax.set_title("Approximation error versus tree size")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

linear_leaves = HingingHyperplaneTree(max_depth=2, leaf_model="linear", random_state=0).fit(X, y)
print(
    f"depth 2 with linear leaves: {linear_leaves.n_leaves_} planes, test RMSE "
    f"{np.sqrt(np.mean((linear_leaves.predict(X_test) - y_test) ** 2)):.4f}"
)

# %%
# **Take-away.** Fuzzy c-regression with two models is a hinge finder. Used
# recursively, it grows a regression tree whose splits are oblique hinges and
# whose leaves are piecewise-linear models. The result is a transparent
# nonlinear model that is directly usable in model-based control, because
# every leaf is a pair of local linear models. The depth and the greedy leaf
# budget control the number of pieces. The test error drops steeply with the
# first splits and then slowly, because every hinge is fitted to data that
# are only approximately piecewise planar; the greedy growth reaches a given
# error with fewer leaves than the balanced one. A large ``min_samples_leaf``
# stops the growth early, and linear leaves need more splits than hinge
# leaves. The node condition :math:`\varrho` flags splits where the hinge
# overrides the fuzzy partition.
