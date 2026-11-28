r"""
Geodesic-distance-based fuzzy clustering versus fuzzy c-means
=============================================================

Fuzzy c-means measures the distance between a sample and a prototype with
the Euclidean norm, so it can only find groups that are convex in the
feature space. When the data lie on a curved manifold (a spiral, two
interleaved moons, concentric rings), two points that are close in the
Euclidean sense may be far apart *along the manifold*. Feil and Abonyi
replace the Euclidean distance by the *geodesic* distance, the length of
the shortest path in a :math:`k`-nearest-neighbor graph whose edges are
weighted by Euclidean lengths (the Isomap approximation of the manifold
distance). Since the geodesic distance is only defined between samples,
the prototypes are restricted to data points. The algorithm is thus a
fuzzy c-medoid clustering with the FCM membership formula, each medoid
being the sample that minimizes
:math:`\sum_k\mu_{ik}^m d_g(\mathbf{x}_k,\cdot)^2`. The number of
neighbors :math:`k` determines how the graph sees the manifold: if it is
too small, the manifold falls apart into components; if it is too large,
edges "short-circuit" neighboring arms.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.datasets import make_curved_manifold, make_rings, make_two_moons
from fuzzyclust.graph import GeodesicFCM, plot_neighbors_graph
from fuzzyclust.plotting import PALETTE, plot_clusters

datasets = {
    "two spiral arms": make_curved_manifold(n_samples=400, noise=0.05, random_state=0),
    "two moons": make_two_moons(n_samples=300, noise=0.05, random_state=0),
    "two rings": make_rings(n_samples_per_ring=150, radii=(1.0, 3.0), noise=0.1, random_state=0),
}

# %%
# FCM versus geodesic c-medoids on three manifolds
# ------------------------------------------------
# With :math:`k=8` the spiral arms form one connected graph in which the
# path between the arms runs all the way through the center, so the two
# medoids sit in the middle of each arm and every sample follows its own
# arm. The moons and the rings fall into two graph components. The
# components are bridged through their closest pair of points with a
# penalty (``bridge_penalty``), so all distances are finite, but a path
# across the bridge is always longer than any path inside a component.

# [snippet:fit]
k = 8
results = {}
for name, (X, y) in datasets.items():
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    geo = GeodesicFCM(n_clusters=2, n_neighbors=k, random_state=0).fit(X)
    results[name] = (fcm, geo)
    print(f"{name:16s}: FCM ARI = {adjusted_rand_score(y, fcm.labels_):.2f}, "
          f"geodesic c-medoids ARI = {adjusted_rand_score(y, geo.labels_):.2f} "
          f"({geo.n_components_} graph component(s), {geo.n_iter_} iterations)")
# [/snippet:fit]

fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
for j, (name, (X, y)) in enumerate(datasets.items()):
    fcm, geo = results[name]
    plot_clusters(X, fcm.labels_, fcm.cluster_centers_, ax=axes[0, j], legend=False)
    axes[0, j].set_title(f"FCM on {name}: ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
    plot_neighbors_graph(X, geo, ax=axes[1, j])
    axes[1, j].set_title(f"Geodesic c-medoids ($k={k}$): ARI = {adjusted_rand_score(y, geo.labels_):.2f}")
plt.tight_layout()

# %%
# The memberships are fuzzy along the manifold, not across it. On the
# spiral, the membership in the "own" arm decreases smoothly from the
# medoid toward the ends of the arm, whereas a point of the other arm that
# is close in the Euclidean sense keeps a membership near zero.
X, y = datasets["two spiral arms"]
_, geo = results["two spiral arms"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for i, ax in enumerate(axes):
    sc = ax.scatter(X[:, 0], X[:, 1], c=geo.membership_[:, i], cmap="viridis", s=14, vmin=0, vmax=1)
    ax.scatter(*geo.cluster_centers_[i], marker="X", s=160, color="black", edgecolor="white", zorder=5)
    ax.set_title(f"Membership in cluster {i + 1} (medoid marked)")
    ax.set_aspect("equal")
plt.colorbar(sc, ax=axes, fraction=0.025, label=r"$\mu_{ik}$")

# %%
# Parameter study I: the number of neighbors :math:`k`
# -----------------------------------------------------
# For every :math:`k`, the number of graph components and the agreement
# with the true groups are recorded. On the spiral, a small :math:`k`
# fragments the sparsely sampled outer parts of the arms into components
# that have to be bridged (the result degrades gracefully), whereas a large
# :math:`k` connects the arms where they come close, and the geodesic
# distance loses its meaning; in between lies a wide plateau. The moons
# with little noise behave similarly: the plateau is :math:`5\le k\le8`,
# and from :math:`k=10` on an edge jumps across the gap and the tips of the
# moons switch clusters. Noisier moons, whose gap is narrower than the
# sample spacing, form one component already at :math:`k=3`. No :math:`k`
# works for them, because the graph cannot resolve a gap that the sampling
# does not resolve.

# [snippet:scan_k]
ks = (3, 4, 5, 6, 8, 10, 15, 20, 30)
noisy_moons = make_two_moons(n_samples=300, noise=0.08, random_state=0)
scan = {}
for name, (X, y) in {**datasets, "noisier moons": noisy_moons}.items():
    scan[name] = [
        (GeodesicFCM(n_clusters=2, n_neighbors=kk, random_state=0).fit(X), y) for kk in ks
    ]
# [/snippet:scan_k]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for j, (name, fits) in enumerate(scan.items()):
    axes[0].plot(ks, [adjusted_rand_score(y, m.labels_) for m, y in fits], "o-", color=PALETTE[j], label=name)
    axes[1].plot(ks, [m.n_components_ for m, _ in fits], "o-", color=PALETTE[j], label=name)
axes[0].set_xlabel("number of neighbors $k$")
axes[0].set_ylabel("ARI")
axes[0].set_title("Agreement with the true groups versus $k$")
axes[0].set_xscale("log")
axes[0].set_xticks(ks)
axes[0].set_xticklabels(ks)
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(True, color="0.9")
axes[1].set_xlabel("number of neighbors $k$")
axes[1].set_ylabel("graph components")
axes[1].set_title("Connected components of the $k$-neighbor graph")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xticks(ks)
axes[1].set_xticklabels(ks)
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(True, color="0.9")
plt.tight_layout()

Xn, yn = noisy_moons
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, kk in zip(axes, (4, 15), strict=True):
    model = GeodesicFCM(n_clusters=2, n_neighbors=kk, random_state=0).fit(Xn)
    plot_neighbors_graph(Xn, model, ax=ax)
    ax.set_title(f"Noisier moons, $k={kk}$: ARI = {adjusted_rand_score(yn, model.labels_):.2f}")
plt.tight_layout()

# %%
# Parameter study II: the fuzziness exponent :math:`m` and the bridges
# ----------------------------------------------------------------------
# As in FCM, :math:`m` controls how soft the partition is; the medoids,
# being data points, hardly move. Without bridging
# (``connect_components=False``), the geodesic distance between components
# is infinite, so a sample belongs with membership one to the component of
# its medoid, and a component without a medoid has no prototype at all.
# With a plain Euclidean bridge (``bridge_penalty=0``), the far side of the
# outer ring is closer to the inner medoid *through the bridge* than to its
# own medoid along the ring, and the clustering fails.
X, y = datasets["two spiral arms"]
print("spiral, k = 8:")
for m in (1.3, 2.0, 3.0):
    model = GeodesicFCM(n_clusters=2, n_neighbors=8, m=m, random_state=0).fit(X)
    print(f"  m = {m:.1f}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean maximal membership = {model.membership_.max(axis=1).mean():.2f}")
Xr, yr = datasets["two rings"]
print("rings, k = 8:")
for kwargs in (dict(), dict(connect_components=False), dict(bridge_penalty=0.0)):
    model = GeodesicFCM(n_clusters=2, n_neighbors=8, random_state=0, **kwargs).fit(Xr)
    finite = np.isfinite(model.geodesic_distances_).all()
    print(f"  {kwargs or 'default (penalized bridge)'!s:36s}: ARI = {adjusted_rand_score(yr, model.labels_):.2f}, "
          f"all distances finite: {finite}")

# %%
# **Take-away.** Replacing the Euclidean distance by the geodesic distance
# lets the c-means functional follow the manifold. The medoids lie on the
# arms, and the memberships fade along them rather than across them, where
# FCM cuts straight through. The one structural parameter is :math:`k`. It
# must be large enough to keep each manifold connected and small enough
# not to connect different manifolds; the component count and the plateau
# of the ARI curve show the usable range. Disconnected components are a
# feature, not a failure: bridged with a penalty, they stay distinct while
# every distance remains finite. The fuzziness exponent plays its usual
# role.
