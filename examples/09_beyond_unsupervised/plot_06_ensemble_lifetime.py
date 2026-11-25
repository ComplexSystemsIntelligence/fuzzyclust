r"""
The lifetime criterion of evidence accumulation
===============================================

Fred and Jain's *evidence accumulation* turns the choice of the number of
clusters into a property of the co-association matrix. The members of
the library deliberately *over-segment* the data (many more clusters
than expected, a different number for each member); objects of the same
natural group stay together in most members, objects of different groups
rarely. Hierarchical clustering of the distance :math:`1-s_{kl}` then
gives a dendrogram, and the chosen number of clusters is the one that
persists over the widest range of thresholds, that is, the one with the
longest *lifetime*. Neither the members nor the consensus needs to know
the number of groups.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.ensemble import (
    FuzzyClusterEnsemble,
    make_overlapping_blobs,
    plot_coassociation,
    plot_lifetimes,
)
from fuzzyclust.plotting import PALETTE, plot_clusters

X, y = make_overlapping_blobs(n_clusters=5, separation=5.0, box=12.0, n_samples_per_cluster=60, random_state=0)

# %%
# Thirty over-segmenting members
# ------------------------------
# Every member is an FCM run with a number of clusters drawn from 6 to 12.
# With ``n_clusters=None`` the ensemble builds the average-link dendrogram
# of :math:`1-s_{kl}`, picks the number of clusters with the longest
# lifetime and partitions the co-association matrix into that many
# groups. Here the lifetime of five clusters is by far the longest, and
# the consensus recovers the five groups.

# [snippet:fit]
ens = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=range(6, 13), n_estimators=30, linkage="average",
                           random_state=0).fit(X)
print(f"lifetime criterion: {ens.n_clusters_} clusters, ARI = {adjusted_rand_score(y, ens.labels_):.2f}")
print("lifetimes of k = 1..8:", np.round(ens.lifetimes_[:8], 3))
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
plot_clusters(X, ens.labels_, ax=axes[0], legend=False)
axes[0].set_title(f"Consensus with {ens.n_clusters_} clusters")
axes[0].set_aspect("equal")
plot_coassociation(ens, ax=axes[1], cbar=False)
axes[1].set_title("Co-association, ordered by the consensus")
plot_lifetimes(ens, ax=axes[2], max_show=10)
axes[2].set_title("Lifetime of $k$ clusters in the dendrogram")
plt.tight_layout()

# %%
# The lifetimes are read off the dendrogram: the height of the merge that
# creates the :math:`k`-cluster partition and the height of the merge that
# destroys it bound the interval of thresholds that cut the tree into
# :math:`k` groups. The shaded band marks the interval of the five-cluster
# partition; below it the groups fall apart into the fragments the
# members produced, above it neighboring groups merge.
fig, ax = plt.subplots(figsize=(9, 3.8))
dendrogram(ens.linkage_, truncate_mode="lastp", p=30, no_labels=True, ax=ax, color_threshold=0,
           above_threshold_color="0.35")
heights = ens.linkage_[:, 2]
N = X.shape[0]
k = ens.n_clusters_
lo, hi = heights[N - k - 1], heights[N - k]
ax.axhspan(lo, hi, color=PALETTE[1], alpha=0.2, lw=0)
ax.text(ax.get_xlim()[1] * 0.99, (lo + hi) / 2, f"{k} clusters: lifetime {hi - lo:.2f}", ha="right",
        va="center", fontsize=9)
ax.set_ylabel(r"distance $1-s_{kl}$ (average link)")
ax.set_title("Top of the co-association dendrogram (30 leaves)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# %%
# Parameter study: the member cluster range and the linkage
# ---------------------------------------------------------
# The number of clusters of the members is drawn from five ranges, and the
# dendrogram is built with average and with single link. Members with
# *fewer* clusters than groups (2--4) merge groups in every run, so the
# evidence never separates them and the criterion returns three clusters.
# From 4--8 clusters on, average link finds the five groups for every
# range. Single link needs no balance between the groups but is fragile
# with soft members: the co-association between neighboring groups never
# drops to zero, a chain of intermediate objects merges them at a low
# height, and the one-cluster partition gets the longest lifetime. The
# printed table repeats the average-link study on four and six groups:
# four groups are found from every range, six groups from members with
# 6--12 or 10--20 clusters (members with 4--8 clusters give five).

# [snippet:scan]
ranges = ((2, 4), (4, 8), (6, 12), (10, 20), (20, 30))
chosen = {"average": [], "single": []}
for link in chosen:
    for a, b in ranges:
        model = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=range(a, b + 1), n_estimators=20,
                                     linkage=link, random_state=0).fit(X)
        chosen[link].append(model.n_clusters_)
# [/snippet:scan]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
xpos = np.arange(len(ranges))
for j, (link, ks) in enumerate(chosen.items()):
    ax.bar(xpos + (j - 0.5) * 0.36, ks, width=0.36, color=PALETTE[j], label=f"{link} link")
ax.axhline(5, color="0.4", ls=":", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{a}-{b}" for a, b in ranges])
ax.set_xlabel("number of clusters of the members")
ax.set_ylabel("clusters chosen by the lifetime")
ax.set_title("Lifetime criterion on five groups")
ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

for n_groups in (4, 6):
    Xg, yg = make_overlapping_blobs(n_clusters=n_groups, separation=5.0, box=12.0, n_samples_per_cluster=60,
                                    random_state=0)
    row = []
    for a, b in ((4, 8), (6, 12), (10, 20)):
        model = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=range(a, b + 1), n_estimators=20,
                                     random_state=0).fit(Xg)
        row.append(f"members {a}-{b}: k = {model.n_clusters_} (ARI {adjusted_rand_score(yg, model.labels_):.2f})")
    print(f"{n_groups} groups, average link -> " + "; ".join(row))

# %%
# **Take-away.** Evidence accumulation reads the number of clusters off
# the co-association dendrogram: the partition that survives the widest
# range of thresholds wins. The recipe has two ingredients. The members
# must over-segment, because with fewer clusters than groups no member
# ever separates some of the groups, and neither can the evidence; beyond
# that, the exact range hardly matters. With fuzzy members, whose
# co-associations between neighboring groups never vanish, average link
# is the robust choice; single link reports a single cluster as soon as a
# chain of intermediate objects connects the groups.
