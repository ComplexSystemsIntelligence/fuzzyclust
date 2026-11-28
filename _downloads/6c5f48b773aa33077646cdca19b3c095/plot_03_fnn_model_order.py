r"""
Model-order selection by false nearest neighbors with a cluster-based threshold
===============================================================================

Before a nonlinear dynamic model :math:`y_k=f(\mathbf{x}_k)` can be
identified, the regression vector
:math:`\mathbf{x}_k=[y_{k-1},\dots,y_{k-n_a},u_{k-1},\dots,u_{k-n_b}]^T`
has to be chosen. The *false nearest neighbors* (FNN) test is geometric:
if the regressors carry enough information, two regression vectors that
are close have close outputs, so the ratio :math:`|y_k-y_j|/\|\mathbf{x}_k-\mathbf{x}_j\|`
of a sample and its nearest neighbor is bounded by the norm of the
gradient of :math:`f`. A neighbor that violates the bound is *false*;
the percentage of false neighbors drops to (nearly) zero once the
regression vector is complete. The weak point is the bound :math:`R`.
For a nonlinear system the gradient varies over the operating range, so
one constant is either too loose or too strict. Feil, Abonyi, and Szeifert
estimate the gradient *locally* from a Gath–Geva clustering of the
regression data (every cluster is a local linear model whose slope is
read off the eigenvector of the smallest eigenvalue of the cluster
covariance) and use the membership-weighted slope as a point-wise
threshold :math:`R_k`.

The benchmark is the continuous polymerization reactor of the paper
(free-radical polymerization of methyl methacrylate), whose output, the
number-average molecular weight, is well described by one lagged output
and two lagged inputs.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust.plotting import PALETTE
from fuzzyclust.preprocessing import make_regressors
from fuzzyclust.timeseries import (
    cluster_based_threshold,
    false_nearest_neighbors,
    fnn_model_order,
    make_nonlinear_io_system,
    plot_fnn_table,
    select_model_order,
)

# %%
# Identification data of the polymerization reactor
# -------------------------------------------------
# [snippet:data]
u, y = make_nonlinear_io_system(n_samples=600, random_state=0)
# [/snippet:data]
fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
axes[0].plot(u, color=PALETTE[0], lw=0.8)
axes[0].set_ylabel("initiator flow $u$")
axes[0].set_title("Random input and molecular-weight output of the polymerization reactor")
axes[1].plot(y, color=PALETTE[1], lw=0.8)
axes[1].set_ylabel("molecular weight $y$")
axes[1].set_xlabel("sample $k$")
plt.tight_layout()

# %%
# FNN table with the cluster-based threshold
# ------------------------------------------
# For every order :math:`(n_a, n_b)`, the regression data are clustered
# with six Gath–Geva clusters, the point-wise thresholds :math:`R_k` are
# computed, and the percentage of false neighbors is tabulated. The
# selected order is the smallest :math:`n_a+n_b` whose percentage is
# below ``fnn_tol`` (5 % by default).

# [snippet:fit]
res = fnn_model_order(y, u, max_na=3, max_nb=3, n_clusters=6, random_state=0)
print(res.table.round(2))
print("selected order: n_a =", res.n_a, " n_b =", res.n_b)
# [/snippet:fit]

fig, ax = plt.subplots(figsize=(5.2, 4.2))
plot_fnn_table(res.table, ax=ax, selected=(res.n_a, res.n_b))
ax.set_title("False nearest neighbors [%], cluster-based threshold")
plt.tight_layout()

# %%
# The percentage collapses in the corner :math:`n_a\ge1`, :math:`n_b\ge2`,
# and the smallest order in that corner, :math:`(1, 2)`, is the structure
# the paper reports (and the one a MARS model of the same system needs).
#
# Why a point-wise threshold: the local slopes
# --------------------------------------------
# The thresholds :math:`R_k` are the norms of the local gradients of the
# regression function (in standardized coordinates). For the selected
# order they differ by more than 60 % between the operating regions found
# by the six clusters, so a constant :math:`R` cannot be right everywhere:
# it is too strict in the flat regions or too loose in the steep ones.
Phi, target, names = make_regressors(y, u, n_a=1, n_b=2, n_k=1)
Phi_s = (Phi - Phi.mean(0)) / Phi.std(0)
target_s = (target - target.mean()) / target.std()
R, gg = cluster_based_threshold(Phi_s, target_s, n_clusters=6, random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].scatter(Phi[:, 1], R, c=gg.labels_, cmap="tab10", s=6)
axes[0].set_xlabel(names[1])
axes[0].set_ylabel("threshold $R_k$")
axes[0].set_title("Point-wise thresholds $R_k$ along the lagged input, colored by cluster", fontsize=9)
axes[1].hist(R, bins=30, color=PALETTE[0], alpha=0.8)
axes[1].set_xlabel("threshold $R_k$")
axes[1].set_ylabel("count")
axes[1].set_title("Distribution of the local gradient norms")
plt.tight_layout()

# %%
# Parameter study: constant thresholds and the classical FNN test
# ---------------------------------------------------------------
# With a constant threshold, the result depends on its value. A loose one
# (:math:`R=3`, above every local slope) declares almost every neighbor
# true, so wrong orders of the same size, such as :math:`(0, 3)`, pass the
# test as well; a strict one (:math:`R=0.5`, below every local slope) keeps
# flagging more than a third of the neighbors at every order, so no corner
# appears. The classical Kennel test with its two heuristic criteria, shown
# for comparison, is looser still and accepts orders that are too small.

# [snippet:scan]
tables = {"cluster-based": res.table}
for R_const in (0.5, 1.0, 3.0):
    tables[f"R = {R_const:g}"] = fnn_model_order(y, u, max_na=3, max_nb=3, threshold=R_const).table
tables["Kennel ($R_{tol}=15$, $A_{tol}=2$)"] = false_nearest_neighbors(y, u, max_na=3, max_nb=3)
# [/snippet:scan]
fig, axes = plt.subplots(1, len(tables), figsize=(3.3 * len(tables), 3.4))
for ax, (name, table) in zip(axes, tables.items(), strict=True):
    im = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100, origin="lower")
    ax.set_title(name, fontsize=9)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlabel("$n_b$")
    ax.set_ylabel("$n_a$")
    for r in range(4):
        for q in range(4):
            v = table.iloc[r, q]
            ax.text(q, r, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else "black")
plt.tight_layout()

# %%
# The number of clusters is the only parameter of the cluster-based
# threshold, and the selected order is stable over a wide range:
print(f"n_clusters =  6: selected order {(res.n_a, res.n_b)}")
for c in (3, 10):
    order = select_model_order(y, u, max_na=3, max_nb=3, n_clusters=c, random_state=0)
    print(f"n_clusters = {c:2d}: selected order {order}")

# %%
# Checking the selected order with linear models
# ----------------------------------------------
# A model-free structure test should agree with model-based validation.
# The one-step-ahead errors of linear ARX models over the same grid of
# orders confirm the selection: the error drops sharply up to :math:`(1, 2)`
# and then flattens.
errors = np.full((4, 4), np.nan)
for na in range(4):
    for nb in range(4):
        if na == 0 and nb == 0:
            continue
        P, tgt, _ = make_regressors(y, u, n_a=na, n_b=nb, n_k=1)
        Pe = np.column_stack([P, np.ones(P.shape[0])])
        theta, *_ = np.linalg.lstsq(Pe, tgt, rcond=None)
        errors[na, nb] = np.sqrt(np.mean((tgt - Pe @ theta) ** 2)) / tgt.std()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for nb in range(4):
    ax.plot(range(4), errors[:, nb], "o-", color=PALETTE[nb], label=f"$n_b = {nb}$")
ax.set_xlabel("$n_a$")
ax.set_ylabel("normalized RMSE of a linear ARX model")
ax.set_title("Model-based check of the order selection")
ax.set_xticks(range(4))
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# **Take-away.** The FNN test turns model-order selection into a geometric
# question that needs no model, except for the threshold. Estimating the
# threshold from Gath–Geva clusters, which are local linear models of the
# regression surface, makes it adapt to the nonlinearity and removes the
# tuning problem. The order is read off the corner of the table where the
# percentage of false neighbors collapses, and it is stable with respect to
# the number of clusters. A constant threshold has to be guessed, and a
# wrong guess shifts the corner.
