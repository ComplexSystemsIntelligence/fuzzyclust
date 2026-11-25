r"""
Fuzzy c-varieties: lines as prototypes, and the elliptotype compromise
======================================================================

Data that lie along lines (or planes) are grouped not by their position but
by the line they belong to. The fuzzy c-varieties (FCV) algorithm of Bezdek,
Coray, Gunderson and Watson replaces the point prototype of FCM by an
:math:`r`-dimensional linear variety through a center :math:`\mathbf{v}_i`
spanned by orthonormal directions :math:`\mathbf{s}_{ij}`, and measures the
squared *orthogonal* distance

.. math::
    D^2_{ik} = \|\mathbf{x}_k-\mathbf{v}_i\|^2
               - \sum_{j=1}^{r}\left((\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{s}_{ij}\right)^2 .

The directions are the leading eigenvectors of the fuzzy scatter matrix of
the cluster, that is, a membership-weighted PCA; with :math:`r=1` the method is
*fuzzy c-lines*. Because a line has no extent, collinear groups are merged;
the *fuzzy c-elliptotypes* mix in a share :math:`\alpha` of the point
distance to keep clusters local. The example compares FCV with
Gustafson--Kessel on crossing lines and shows the effect of :math:`\alpha`
on collinear segments.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM, GK
from fuzzyclust.extensions import (
    FuzzyCVarieties,
    make_collinear_segments,
    make_crossing_lines,
    plot_varieties,
)
from fuzzyclust.plotting import PALETTE, plot_covariance_ellipses, plot_membership_contours

X, y = make_crossing_lines(n_lines=3, n_samples_per_line=80, length=8.0, noise=0.1, random_state=0)

# %%
# Three lines through one point: FCM, GK and fuzzy c-lines
# --------------------------------------------------------
# Near the crossing, the points of all three lines are close together.
# FCM tiles the plane into three cells, and GK, whose ellipsoids still
# measure the position along the line, cannot stretch one ellipsoid along
# each line without swallowing the center of the others: both cut the
# lines into pieces. Fuzzy c-lines only asks on which line a point lies,
# and recovers the three lines and their orientations.

# [snippet:fit]
fcm = FCM(n_clusters=3, n_init=5, random_state=0).fit(X)
gk = GK(n_clusters=3, n_init=5, random_state=0).fit(X)
fcl = FuzzyCVarieties(n_clusters=3, n_dims=1, n_init=5, random_state=0).fit(X)
angles = np.degrees(np.arctan2(fcl.directions_[:, 0, 1], fcl.directions_[:, 0, 0])) % 180
print(f"FCM ARI = {adjusted_rand_score(y, fcm.labels_):.2f}, GK ARI = {adjusted_rand_score(y, gk.labels_):.2f}, "
      f"fuzzy c-lines ARI = {adjusted_rand_score(y, fcl.labels_):.2f}")
print(f"line orientations found: {np.sort(angles).round(1)} degrees (true: 0, 60, 120)")
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
plot_membership_contours(fcm, X, ax=axes[0], labels=False)
axes[0].set_title(f"FCM: ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
plot_membership_contours(gk, X, ax=axes[1], labels=False)
plot_covariance_ellipses(gk.cluster_centers_, gk.covariances_, ax=axes[1])
axes[1].set_title(f"GK: ARI = {adjusted_rand_score(y, gk.labels_):.2f}")
plot_varieties(X, fcl, ax=axes[2])
axes[2].set_title(f"fuzzy c-lines: ARI = {adjusted_rand_score(y, fcl.labels_):.2f}")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()

# %%
# The membership of a point in a line cluster depends only on its distance
# to the line, so the membership contours are bands around the lines
# (left). The eigenvalues of the scatter matrices show how line-like each
# cluster is: the leading one is the spread *along* the line, the
# remaining one the scatter *around* it (right), that is, the squared noise
# level times the membership mass.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_membership_contours(fcl, X, ax=axes[0], labels=False)
axes[0].set_aspect("equal")
axes[0].set_title("fuzzy c-lines: maximal membership is a band around each line")
width = np.arange(3)
axes[1].bar(width - 0.2, fcl.eigenvalues_[:, 0], 0.4, color=PALETTE[0], label="along the line")
axes[1].bar(width + 0.2, fcl.eigenvalues_[:, 1], 0.4, color=PALETTE[1], label="around the line")
axes[1].set_yscale("log")
axes[1].set_xticks(width, [f"cluster {i + 1}" for i in width])
axes[1].set_ylabel("eigenvalues of the fuzzy scatter matrix")
axes[1].set_title("Spread along versus around the lines")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(True, axis="y", color="0.9")
plt.tight_layout()

# %%
# Parameter study I: the elliptotype weight :math:`\alpha`
# ----------------------------------------------------------
# Two collinear segments separated by a gap, plus an oblique third segment.
# Pure fuzzy c-lines (:math:`\alpha=0`) sees the two collinear segments as
# one line and wastes a prototype on noise. A small share of the point
# distance (:math:`\alpha=0.05`) is enough to separate them by position,
# while the oblique segment is still found as a line. Every larger
# :math:`\alpha` up to :math:`\alpha=1` (FCM) works here because the
# segments are compact, but the crossing lines lose accuracy from
# :math:`\alpha=0.1` on and are no longer found from :math:`\alpha=0.2`.
Xc, yc = make_collinear_segments(n_samples_per_segment=60, length=3.0, gap=3.0, noise=0.1, random_state=0)
alphas = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ari_c = [adjusted_rand_score(yc, FuzzyCVarieties(n_clusters=3, alpha=a, n_init=5, random_state=0).fit(Xc).labels_)
         for a in alphas]
ari_x = [adjusted_rand_score(y, FuzzyCVarieties(n_clusters=3, alpha=a, n_init=5, random_state=0).fit(X).labels_)
         for a in alphas]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, a in zip(axes[:2], (0.0, 0.3), strict=True):
    model = FuzzyCVarieties(n_clusters=3, alpha=a, n_init=5, random_state=0).fit(Xc)
    plot_varieties(Xc, model, ax=ax, length=4.0)
    ax.set_aspect("equal")
    ax.set_title(rf"$\alpha$ = {a}: ARI = {adjusted_rand_score(yc, model.labels_):.2f}")
axes[2].plot(alphas, ari_c, "o-", color=PALETTE[0], label="collinear segments")
axes[2].plot(alphas, ari_x, "s-", color=PALETTE[1], label="crossing lines")
axes[2].set_xlabel(r"elliptotype weight $\alpha$")
axes[2].set_ylabel("adjusted Rand index")
axes[2].set_title(r"Agreement with the true groups against $\alpha$")
axes[2].grid(True, color="0.9")
axes[2].legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
# Parameter study II: restarts, the exponent and the variety dimension
# --------------------------------------------------------------------
# Lines through a common point are a hard problem for a random start: a
# single run often puts two prototypes on one line and lets the third
# cover the other two; ``n_init`` keeps the best of several runs by the
# objective. :math:`m` plays its usual role. In three dimensions,
# ``n_dims=2`` fits planes, and the discarded eigenvector is the normal of
# each plane.
print("crossing lines: ARI over 10 seeds")
for n_init in (1, 3, 10):
    aris = [adjusted_rand_score(y, FuzzyCVarieties(n_clusters=3, n_init=n_init, random_state=s).fit(X).labels_)
            for s in range(10)]
    print(f"  n_init = {n_init:2d}: mean {np.mean(aris):.2f}, min {np.min(aris):.2f}")
for m in (1.3, 2.0, 3.0):
    model = FuzzyCVarieties(n_clusters=3, m=m, n_init=10, random_state=0).fit(X)
    print(f"m = {m}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean max membership {model.membership_.max(axis=1).mean():.2f}")
rng = np.random.RandomState(0)
plane1 = np.c_[rng.uniform(-3, 3, (150, 2)), rng.randn(150) * 0.05]
plane2 = np.c_[rng.uniform(-3, 3, 150), rng.randn(150) * 0.05 + 1.5, rng.uniform(-3, 3, 150)]
X3 = np.vstack([plane1, plane2])
y3 = np.repeat([0, 1], 150)
planes = FuzzyCVarieties(n_clusters=2, n_dims=2, n_init=5, random_state=0).fit(X3)
normals = [np.cross(S[0], S[1]).round(2) for S in planes.directions_]
print(f"two planes in 3-D: ARI = {adjusted_rand_score(y3, planes.labels_):.2f}, normals {normals}")

# %%
# **Take-away.** Fuzzy c-varieties keeps the FCM functional and membership
# formula and changes the prototype: a line (or an :math:`r`-dimensional
# variety) found by a membership-weighted PCA of each cluster, with the
# orthogonal distance in place of the Euclidean one. It separates crossing
# lines that point and ellipsoid prototypes cut into pieces, but it also
# merges collinear groups. The remedy is the elliptotype weight
# :math:`\alpha` of the point distance; keep it small (0.05 here), since
# lines that cross already suffer at :math:`\alpha=0.1`.
# Use several restarts, because lines through a common point produce
# local optima.
