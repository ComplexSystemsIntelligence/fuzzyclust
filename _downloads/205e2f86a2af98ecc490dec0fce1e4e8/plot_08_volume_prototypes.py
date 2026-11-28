r"""
Volume prototypes and adaptive cluster merging
==============================================

A point prototype measures the distance to the *center* of a cluster, so a
sample in the outer part of a large cluster is farther from its own
prototype than a sample at the edge of a small neighboring cluster. This is
why FCM and Gustafson--Kessel tend to equalize the clusters: a large,
diffuse group is split and its pieces are handed to smaller groups. Kaymak
and Setnes replace the point by a *volume prototype*, a hypersphere (or
hyperellipsoid) of radius :math:`r_i` inside which the distance is zero,

.. math:: \tilde D_{ik}=\max\bigl(0,\;D_{ik}-r_i\bigr),

with the radius estimated from the cluster itself. Samples inside the core
belong fully to their cluster, and only the distance to the *surface* of a
core counts in the competition between clusters. Combined with *adaptive
cluster merging*, which starts with too many clusters and merges the most
similar pair whenever the partition has converged, the method finds the
number of clusters during the clustering.
"""

# %%
# One large diffuse group and two small dense ones
# ------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.datasets import make_different_densities
from fuzzyclust.extensions import VolumePrototypeFCM
from fuzzyclust.plotting import PALETTE, plot_membership_contours

X, y = make_different_densities(sizes=(300, 60, 60), spreads=(1.5, 0.3, 0.6), random_state=0)

# [snippet:fit]
fcm = FCM(n_clusters=3, n_init=5, random_state=0).fit(X)
vp = VolumePrototypeFCM(n_clusters=3, n_init=5, random_state=0).fit(X)
print(f"ARI: FCM {adjusted_rand_score(y, fcm.labels_):.2f}, volume prototypes {adjusted_rand_score(y, vp.labels_):.2f}")
print("core radii r_i:", np.round(vp.radii_, 2))
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
plot_membership_contours(fcm, X, ax=axes[0])
axes[0].set_title(f"FCM: the large group is split (ARI = {adjusted_rand_score(y, fcm.labels_):.2f})")
plot_membership_contours(vp, X, ax=axes[1])
t = np.linspace(0, 2 * np.pi, 200)
for i, (v, r) in enumerate(zip(vp.cluster_centers_, vp.radii_, strict=True)):
    axes[1].plot(v[0] + r * np.cos(t), v[1] + r * np.sin(t), color=PALETTE[i], lw=2.2, ls="--")
axes[1].set_title(f"volume prototypes (dashed: cores) (ARI = {adjusted_rand_score(y, vp.labels_):.2f})")
for ax in axes:
    ax.title.set_fontsize(10)
    ax.set_aspect("equal")
plt.tight_layout()

# %%
# The core radii adapt to the clusters: the core of the large group is
# about two and three times as large as those of the two small ones. The
# membership contours of the large cluster no longer shrink toward its
# center, and the small clusters no longer claim its outer members.

# %%
# Memberships along a line through two clusters
# ---------------------------------------------
# The profile along the line from the center of the large group to the
# center of a small one shows the mechanism. Inside a core the distance is
# zero and the membership one; between the cores the memberships are
# decided by the distances to the core *surfaces*, so the border moves
# toward the small cluster: the two memberships cross at 0.61 of the way
# for the volume prototypes and at 0.43 for FCM, whose small-cluster
# prototype has also been pulled toward the large group.
big = int(np.argmax(vp.radii_))
small = int(np.argmin(vp.radii_))
a, b = vp.cluster_centers_[big], vp.cluster_centers_[small]
s = np.linspace(-0.3, 1.3, 400)
line = a[None, :] + s[:, None] * (b - a)[None, :]
U_vp = vp.predict_proba(line)
fcm_big = int(np.argmin(np.linalg.norm(fcm.cluster_centers_ - a, axis=1)))
fcm_small = int(np.argmin(np.linalg.norm(fcm.cluster_centers_ - b, axis=1)))
U_fcm = fcm.predict_proba(line)

fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(s, U_fcm[:, fcm_big], color=PALETTE[0], lw=1.5, ls=":", label="FCM, large cluster")
ax.plot(s, U_fcm[:, fcm_small], color=PALETTE[1], lw=1.5, ls=":", label="FCM, small cluster")
ax.plot(s, U_vp[:, big], color=PALETTE[0], lw=2.2, label="volume prototypes, large cluster")
ax.plot(s, U_vp[:, small], color=PALETTE[1], lw=2.2, label="volume prototypes, small cluster")
dist = np.linalg.norm(b - a)
ax.axvspan(-vp.radii_[big] / dist, vp.radii_[big] / dist, color=PALETTE[0], alpha=0.08)
ax.axvspan(1 - vp.radii_[small] / dist, 1 + vp.radii_[small] / dist, color=PALETTE[1], alpha=0.12)
ax.set_xlabel("position on the line (0: center of the large group, 1: center of the small group)")
ax.set_ylabel("membership")
ax.set_title("membership profiles; shaded: the cores")
ax.legend(frameon=False, fontsize=8, loc="center left")
ax.grid(True, color="0.92")
plt.tight_layout()

# %%
# Adaptive cluster merging
# ------------------------
# Started with eight clusters, the algorithm iterates to convergence,
# computes the similarity of every pair of clusters as fuzzy sets,
# :math:`S_{ij}=\sum_k\min(\mu_{ik},\mu_{jk})/\sum_k\max(\mu_{ik},\mu_{jk})`
# (from the center distances, rescaled to :math:`m=2`), merges the most
# similar pair if :math:`S_{ij}` exceeds the threshold, and continues with
# one cluster fewer.

# [snippet:merge]
vpm = VolumePrototypeFCM(n_clusters=8, merge=True, similarity_threshold=0.2, random_state=0).fit(X)
for step in vpm.merge_history_:
    print(f"iteration {step['iteration']:3d}: c = {step['n_clusters']} -> merge {step['pair']} "
          f"(S = {step['similarity']:.2f})")
print(f"final number of clusters: {vpm.n_clusters_}, ARI = {adjusted_rand_score(y, vpm.labels_):.2f}")
# [/snippet:merge]

start = VolumePrototypeFCM(n_clusters=8, random_state=0).fit(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
plot_membership_contours(start, X, ax=axes[0], labels=False)
axes[0].set_title("eight volume prototypes before merging")
plot_membership_contours(vpm, X, ax=axes[1])
for i, (v, r) in enumerate(zip(vpm.cluster_centers_, vpm.radii_, strict=True)):
    axes[1].plot(v[0] + r * np.cos(t), v[1] + r * np.sin(t), color=PALETTE[i], lw=2.2, ls="--")
axes[1].set_title(f"after adaptive merging: {vpm.n_clusters_} clusters")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()

# %%
# Parameter study: core size, threshold and initial number of clusters
# --------------------------------------------------------------------
# The radius is the membership-weighted mean distance of the cluster
# members from the center, multiplied by ``radius_scale``; ``0`` is FCM.
# Larger cores make the clusters greedier, and a core that grows beyond
# its group swallows neighbors. The similarity threshold decides when to
# stop merging, and the initial number of clusters should not matter.
scales = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
scale_ari = [adjusted_rand_score(y, VolumePrototypeFCM(n_clusters=3, radius_scale=r, n_init=5, random_state=0)
                                 .fit(X).labels_) for r in scales]
thresholds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
final_c = [VolumePrototypeFCM(n_clusters=8, merge=True, similarity_threshold=th, random_state=0).fit(X).n_clusters_
           for th in thresholds]
starts = [4, 5, 6, 8, 10, 12]
start_c = [VolumePrototypeFCM(n_clusters=c0, merge=True, random_state=0).fit(X).n_clusters_ for c0 in starts]

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].plot(scales, scale_ari, marker="o", color=PALETTE[0], lw=1.8)
axes[0].set_xlabel("radius_scale")
axes[0].set_ylabel("adjusted Rand index")
axes[0].set_title("core size ($0$: FCM), $c=3$")
axes[1].step(thresholds, final_c, where="mid", marker="o", color=PALETTE[1], lw=1.8)
axes[1].set_xlabel("similarity threshold")
axes[1].set_ylabel("final number of clusters")
axes[1].set_title("merging from $c=8$")
axes[2].plot(starts, start_c, marker="o", color=PALETTE[2], lw=1.8)
axes[2].set_xlabel("initial number of clusters")
axes[2].set_ylabel("final number of clusters")
axes[2].set_ylim(0, 6)
axes[2].set_title("merging with threshold 0.2")
for ax in axes:
    ax.grid(True, color="0.92")
plt.tight_layout()
for r, a_ in zip(scales, scale_ari, strict=True):
    print(f"radius_scale {r:4.2f}: ARI = {a_:.2f}")
for th, c in zip(thresholds, final_c, strict=True):
    print(f"threshold {th:.2f}: {c} clusters")
print("initial c -> final c:", dict(zip(starts, start_c, strict=True)))

# %%
# The agreement rises from the FCM value at ``radius_scale=0`` to its
# maximum at the default ``1`` and collapses beyond it, when the core of
# a small cluster grows over the large group. The merging threshold has a
# plateau at three clusters around the default 0.2: two clusters sharing a
# group have :math:`S_{ij}\approx0.25`--:math:`0.35`, so a much higher
# threshold stops too early and a much lower one also merges distinct
# groups. Whatever the initial number of clusters, the merging ends at
# three.

# %%
# **Take-away.** Volume prototypes remove the bias of point prototypes
# toward clusters of equal extent: a large, diffuse group keeps its
# members because only the distance beyond its core counts. Together with
# adaptive merging the method needs only an upper bound on the number of
# clusters and a similarity threshold, and it is insensitive to the bound.
# Keep the core scale at its default; larger cores make the partition
# unstable. Gath--Geva (example ``plot_04_gg_densities.py``) solves the
# same problem through densities and priors but needs a good
# initialization; volume prototypes stay within the robust FCM iteration.
