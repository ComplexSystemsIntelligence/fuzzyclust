r"""
Subtractive clustering: the number and location of clusters from density
========================================================================

Every alternating-optimization algorithm needs the number of clusters and a
starting point. The *mountain method* of Yager and Filev and its data-point
version, Chiu's *subtractive clustering*, provide both without iteration.
Every candidate point (a grid node or a data point) receives a *potential*,
the sum of the Gaussian influences of all samples, with influence radius
:math:`r_a`:

.. math:: P_k=\sum_{l=1}^{N}\exp\bigl(-4\|\mathbf{x}_k-\mathbf{x}_l\|^2/r_a^2\bigr).

The point of highest potential becomes the first center; its influence, with
a larger radius :math:`r_b=\eta r_a`, is subtracted from all potentials, and
the next peak is examined, until the remaining potential is a small fraction
of the first. The radius is the one parameter that matters: it is the scale
at which the data are viewed. This example shows the potential, the effect
of the radius, and the two classical uses of the result: the initialization
of Gustafson--Kessel and the number of rules of a Takagi--Sugeno model.
"""

# %%
# Eight elongated groups of different size
# ----------------------------------------
# The groups have between 40 and 150 samples and random orientations. The
# data are scaled to the unit square internally, so the radius is a
# fraction of the data range.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from fuzzyclust.cluster import GK, MountainClustering, SubtractiveClustering
from fuzzyclust.models import TakagiSugeno, make_sinc
from fuzzyclust.plotting import PALETTE, plot_clusters

rng = np.random.RandomState(0)
group_centers = [(-6, -4), (-6, 0), (-6, 4), (0, -5), (0, 0), (0, 5), (6, -3), (6, 3)]
sizes = [150, 40, 80, 40, 150, 60, 100, 40]
parts, y = [], []
for i, (center, n) in enumerate(zip(group_centers, sizes, strict=True)):
    theta = rng.rand() * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    parts.append((rng.randn(n, 2) * np.array([1.0, 0.35])) @ R.T + center)
    y.append(np.full(n, i))
X, y = np.vstack(parts), np.concatenate(y)

# [snippet:fit]
sub = SubtractiveClustering(radii=0.25).fit(X)
print(f"{sub.n_clusters_} centers, relative potentials at selection: {np.round(sub.potentials_, 2)}")
mountain = MountainClustering(radii=0.25).fit(X)  # 100 x 100 grid nodes
print(f"mountain method: {mountain.n_clusters_} centers on a {mountain.grid_size_} x {mountain.grid_size_} grid")
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
sc = axes[0].scatter(X[:, 0], X[:, 1], c=sub.initial_potentials_, cmap="viridis", s=12)
fig.colorbar(sc, ax=axes[0], label="potential $P_k$")
for order, center in enumerate(sub.cluster_centers_, start=1):
    axes[0].plot(*center, "x", color="k", ms=9, mew=2)
    axes[0].annotate(str(order), center, xytext=(5, 5), textcoords="offset points", fontsize=9, weight="bold")
axes[0].set_title("subtractive clustering: potential of the samples\nand centers in the order of selection")
side = mountain.grid_size_
G = mountain.grid_.reshape(side, side, 2)
M = mountain.mountain_.reshape(side, side)
axes[1].contourf(G[..., 0], G[..., 1], M, levels=15, cmap="viridis")
axes[1].plot(X[:, 0], X[:, 1], ".", color="w", ms=2, alpha=0.6)
axes[1].plot(mountain.cluster_centers_[:, 0], mountain.cluster_centers_[:, 1], "x", color="k", ms=9, mew=2)
axes[1].set_title("mountain method: mountain function on the grid\nand the selected grid nodes")
plt.tight_layout()

# %%
# The potential is a kernel density estimate with bandwidth
# :math:`r_a/\sqrt8`: the cores of the large groups have the highest
# potential and are selected first, and the small groups follow once the
# large ones have been subtracted. Both methods find the eight groups. The
# mountain method needs :math:`100^2` grid nodes for two features, which is
# why subtractive clustering, whose candidates are the samples, is the method
# of choice in more than two or three dimensions.

# %%
# Initializing Gustafson--Kessel
# ------------------------------
# The centers are passed as ``init`` together with their number. GK then
# starts close to its solution, needs no random restarts, and converges in
# fewer iterations than from random partitions.

# [snippet:init]
gk = GK(n_clusters=sub.n_clusters_, init=sub.cluster_centers_).fit(X)
# [/snippet:init]
random_runs = [GK(n_clusters=8, random_state=s).fit(X) for s in range(20)]
iters = np.array([r.n_iter_ for r in random_runs])
print(f"GK from subtractive centers: {gk.n_iter_} iterations, J = {gk.objective_:.1f}, "
      f"ARI = {adjusted_rand_score(y, gk.labels_):.2f}")
print(f"GK from 20 random partitions: median {np.median(iters):.0f} iterations "
      f"(range {iters.min()}-{iters.max()}), best J = {min(r.objective_ for r in random_runs):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
plot_clusters(X, gk.labels_, centers=gk.cluster_centers_, ax=axes[0], legend=False)
(start,) = axes[0].plot(sub.cluster_centers_[:, 0], sub.cluster_centers_[:, 1], "o", mfc="none", mec="k", ms=10)
axes[0].legend([start], ["subtractive centers (start)"], frameon=False, loc="lower right")
axes[0].set_title(f"GK started from the subtractive centers ({gk.n_iter_} iterations)")
axes[1].hist(iters, bins=np.arange(0, iters.max() + 10, 8), color=PALETTE[0], alpha=0.8)
axes[1].axvline(gk.n_iter_, color=PALETTE[1], lw=2, label="subtractive start")
axes[1].set_xlabel("iterations to converge")
axes[1].set_ylabel("random starts")
axes[1].set_title("GK from 20 random partitions")
axes[1].legend(frameon=False)
plt.tight_layout()

# %%
# Parameter study: the radius
# ---------------------------
# The number of centers falls as the radius grows. A *plateau*, a range of
# radii over which the number does not change, indicates a stable
# structure; here the plateau at eight spans :math:`r_a\in[0.18,0.30]` for
# the default squash factor. Below it the elongated groups are cut into
# pieces; above it neighboring groups fall within one radius and are counted
# once. The squash factor :math:`\eta` shifts the plateau: what is
# subtracted after a selection is a bump of radius :math:`\eta r_a`, so a
# larger :math:`\eta` removes more potential around every center, and the
# same number of centers is reached with a smaller :math:`r_a`.
radii = np.round(np.arange(0.08, 0.62, 0.02), 2)
counts = {eta: [SubtractiveClustering(radii=r, squash=eta).fit(X).n_clusters_ for r in radii] for eta in (1.0, 1.25, 1.5)}
counts_mountain = [MountainClustering(radii=r, grid_size=60).fit(X).n_clusters_ for r in radii]
for eta, cs in counts.items():
    print(f"squash {eta}: radii with 8 centers: {radii[np.array(cs) == 8]}")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for j, (eta, cs) in enumerate(counts.items()):
    ax.step(radii, cs, where="mid", color=PALETTE[j], lw=1.8, label=rf"subtractive, $\eta={eta}$")
ax.step(radii, counts_mountain, where="mid", color=PALETTE[3], lw=1.4, ls="--", label=r"mountain, $\eta=1.25$")
ax.axhline(8, color="0.6", lw=0.8, ls=":")
ax.set_xlabel(r"radius $r_a$ (fraction of the data range)")
ax.set_ylabel("number of centers")
ax.set_title("number of clusters against the neighborhood radius")
ax.legend(frameon=False)
ax.grid(True, color="0.92")
plt.tight_layout()

# %%
# The number of rules of a Takagi--Sugeno model
# ---------------------------------------------
# Chiu proposed subtractive clustering for fuzzy model identification:
# clustering the product space :math:`\mathbf{z}=[\mathbf{x}^T\,y]^T` of
# inputs and output gives one cluster, and thus one rule, per region of the
# input space in which the function needs its own local model. We estimate
# the number of rules of a model of :math:`\sin(x)/x` for several radii and
# compare it with the test error of models with 2 to 12 rules.
Xs, ys = make_sinc(n_samples=400, noise=0.02, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.5, random_state=0)

# [snippet:rules]
Z = np.column_stack([X_train, y_train])  # product space of input and output
for r in (0.2, 0.3, 0.4):
    n_rules = SubtractiveClustering(radii=r).fit(Z).n_clusters_
    ts = TakagiSugeno(n_clusters=n_rules, random_state=0, n_init=3).fit(X_train, y_train)
    rmse = np.sqrt(np.mean((ts.predict(X_test) - y_test) ** 2))
    print(f"radius {r:.1f}: {n_rules:2d} rules, test RMSE = {rmse:.4f}")
# [/snippet:rules]

estimates = {r: SubtractiveClustering(radii=r).fit(Z).n_clusters_ for r in (0.15, 0.2, 0.3, 0.4, 0.5)}
rule_range = np.arange(2, 13)
rmse_curve = []
for c in rule_range:
    ts = TakagiSugeno(n_clusters=int(c), random_state=0, n_init=3).fit(X_train, y_train)
    rmse_curve.append(np.sqrt(np.mean((ts.predict(X_test) - y_test) ** 2)))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(rule_range, rmse_curve, marker="o", color=PALETTE[0], lw=1.8, label="test RMSE")
for j, (r, c) in enumerate(sorted(estimates.items())):
    ax.axvline(c, color=PALETTE[1 + j % 4], ls="--", lw=1.2, label=f"subtractive estimate, $r_a={r}$: {c} rules")
ax.set_yscale("log")
ax.set_xlabel("number of rules $c$")
ax.set_ylabel("test RMSE")
ax.set_title(r"Takagi–Sugeno models of $\sin(x)/x$")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(True, color="0.92")
plt.tight_layout()
for c, e in zip(rule_range, rmse_curve, strict=True):
    print(f"{c:2d} rules: test RMSE = {e:.4f}")

# %%
# **Take-away.** Subtractive clustering turns a single, interpretable
# parameter, the neighborhood radius as a fraction of the data range, into a
# number of clusters and a set of centers, in one pass and without random
# initialization. Reading the number of centers over a range of radii and
# looking for a plateau is a quick, model-free check of how many clusters
# the data support. The centers are good starting points for FCM and GK,
# and on product-space data the number of centers estimates the number of
# rules of a Takagi--Sugeno model; since a larger radius yields fewer,
# coarser rules, the radius sets the trade-off between model size and
# accuracy. The mountain method gives the same answer in two dimensions, but
# its grid grows exponentially with the number of features.
