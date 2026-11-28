r"""
Inconsistent edges: Zahn's criteria and the neighborhood depth
==============================================================

Removing the longest edges of the minimal spanning tree works as long as
all clusters have the same density. When a diffuse group lies next to
tight ones, the edges *inside* the diffuse group are as long as the edges
that *separate* the tight groups, and a global length threshold
(Criterion 1, :math:`w(e)>\lambda\,\bar w`) either leaves the tight groups
merged or shatters the diffuse one. Zahn's local criterion (Criterion 2)
compares every edge with the edges in its own tree neighborhood. An edge
is inconsistent when it is at least :math:`f` times longer than the mean
of the edges within :math:`k` tree steps of its end points (ratio form),
or when it exceeds that mean by more than a given number of standard
deviations (deviation form). The factor :math:`f` and the depth :math:`k`
(``inconsistency``, ``inconsistency_std``, ``n_hops``) are the parameters
of the classical algorithm, and this example shows what they do. The
splitting phase is switched off (``min_gain=1.0``) and no refinement is
run, so only the length criteria act.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.graph import MSTClustering, make_density_blobs, plot_mst
from fuzzyclust.plotting import PALETTE

X, y = make_density_blobs(sizes=(200, 40, 40), spreads=(1.2, 0.25, 0.25), random_state=0)

# %%
# Global threshold versus local inconsistency
# -------------------------------------------
# The MST of the data has about 280 edges, whose mean length is dominated
# by the diffuse group. A threshold of twice the mean cuts many edges inside
# the diffuse group (its tails become small clusters), a threshold of four
# times the mean misses the gap between the two tight groups, and no value
# in between yields exactly the three groups. The local criterion with
# :math:`f=2.5` and a depth of three tree steps finds them.

# [snippet:criteria]
common = dict(min_gain=1.0, refine=None)  # length criteria only
global_2 = MSTClustering(threshold=2.0, inconsistency=None, **common).fit(X)
global_4 = MSTClustering(threshold=4.0, inconsistency=None, **common).fit(X)
local = MSTClustering(inconsistency=2.5, n_hops=3, **common).fit(X)
for name, model in (("threshold 2.0", global_2), ("threshold 4.0", global_4), ("inconsistency 2.5, 3 hops", local)):
    print(f"{name:26s}: {model.n_clusters_} clusters, ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"sizes {np.bincount(model.tree_labels_).tolist()}")
# [/snippet:criteria]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharex=True, sharey=True)
for ax, model, name in zip(
    axes, (global_2, global_4, local),
    (r"Criterion 1, $\lambda=2$", r"Criterion 1, $\lambda=4$", r"Criterion 2, $f=2.5$, $k=3$"),
    strict=True,
):
    plot_mst(X, model, ax=ax, legend=False)
    ax.set_title(f"{name}: {model.n_clusters_} clusters, ARI = {adjusted_rand_score(y, model.labels_):.2f}")
plt.tight_layout()

# %%
# The edge-length distribution explains why. The edge between the two
# tight groups is no longer than the longest edges inside the diffuse
# group, so no global threshold can single it out, but each separating
# edge is far longer than its own neighbors. A scan of the threshold
# confirms that no value returns the three groups.
w = local.mst_weights_
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.hist(w, bins=40, color="0.7", label="all tree edges")
cut_w = np.linalg.norm(X[local.cut_edges_[:, 0]] - X[local.cut_edges_[:, 1]], axis=1)
ax.vlines(cut_w, 0, ax.get_ylim()[1] * 0.6, color=PALETTE[1], lw=2.5, label="edges cut by Criterion 2")
ax.axvline(2.0 * w.mean(), color=PALETTE[0], ls="--", label=r"$2\,\bar w$")
ax.axvline(4.0 * w.mean(), color=PALETTE[2], ls="--", label=r"$4\,\bar w$")
ax.set_xlabel("edge length")
ax.set_ylabel("count")
ax.set_title("Edge lengths of the tree: the edge between the tight groups is not among the longest")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

for lam in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
    model = MSTClustering(threshold=lam, inconsistency=None, **common).fit(X)
    print(f"threshold {lam:.1f} x mean: {model.n_clusters_} clusters, ARI = {adjusted_rand_score(y, model.labels_):.2f}")

# %%
# Parameter study: factor :math:`f` and depth :math:`k`
# -----------------------------------------------------
# A small factor or a shallow neighborhood flags every local gap (the
# random spacing inside the diffuse group creates many of them), whereas a
# very large factor flags nothing. A deeper neighborhood averages over more
# edges and makes the reference length more stable, so the result becomes
# insensitive to :math:`f` over a wide range; the two parameters interact.

# [snippet:scan]
factors = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
depths = (1, 2, 3, 5, 8)
n_found = np.array([[MSTClustering(inconsistency=f, n_hops=k, **common).fit(X).n_clusters_
                     for k in depths] for f in factors])
# [/snippet:scan]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for j, k in enumerate(depths):
    axes[0].plot(factors, n_found[:, j], "o-", color=PALETTE[j], label=f"depth $k$ = {k}")
axes[0].axhline(3, color="0.5", ls=":")
axes[0].set_xlabel("inconsistency factor $f$")
axes[0].set_ylabel("number of clusters found")
axes[0].set_title("Ratio form: clusters versus $f$ and the neighborhood depth")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(True, color="0.9")

# the deviation form needs a deep neighborhood: the standard deviation of a
# handful of edges is a poor reference
n_std = (2.0, 2.5, 3.0, 4.0)
found_std = np.array([[MSTClustering(inconsistency=None, inconsistency_std=s, n_hops=k, **common).fit(X).n_clusters_
                       for k in depths] for s in n_std])
for j, k in enumerate(depths):
    axes[1].plot(n_std, found_std[:, j], "s-", color=PALETTE[j], label=f"depth $k$ = {k}")
axes[1].axhline(3, color="0.5", ls=":")
axes[1].set_xlabel("standard deviations above the local mean")
axes[1].set_ylabel("number of clusters found")
axes[1].set_title("Deviation form: clusters versus the threshold and the depth")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(True, color="0.9")
plt.tight_layout()

print("ratio form, number of clusters (rows: f, columns: depth k)")
print("      " + "".join(f"k={k:<5d}" for k in depths))
for f, row in zip(factors, n_found, strict=True):
    print(f"f={f:<4.1f}" + "".join(f"{n:<7d}" for n in row))

# %%
# Small fragments are the usual failure mode when the criteria are too
# eager. ``min_cluster_size`` refuses cuts that would create a component
# below that size, which makes a loose criterion usable again.
for size in (2, 5, 10, 20):
    model = MSTClustering(inconsistency=2.0, n_hops=2, min_cluster_size=size, **common).fit(X)
    print(f"f = 2.0, k = 2, min_cluster_size = {size:2d}: {model.n_clusters_} clusters, "
          f"ARI = {adjusted_rand_score(y, model.labels_):.2f}")

# %%
# **Take-away.** A global length threshold presupposes a single density.
# Zahn's local criterion adapts the reference length to the surroundings of
# every edge and separates groups of very different density. Its two
# parameters interact: the deeper the neighborhood (``n_hops``), the more
# stable the reference and the wider the range of factors ``inconsistency``
# that works (2.5–4 with :math:`k=3` or 5 here). The deviation form needs an even
# deeper neighborhood because it estimates a standard deviation. Fragments
# cut off by an eager criterion are suppressed by ``min_cluster_size`` and,
# in the hybrid algorithm, by the subsequent volume criterion, which only
# accepts cuts that reduce the total volume substantially.
