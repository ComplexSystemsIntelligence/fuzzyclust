r"""
Possibilistic c-means: typicalities, scales and coincident clusters
===================================================================

Fuzzy c-means forces the memberships of every sample to sum to one, so an
outlier far from all prototypes still receives a membership of about
:math:`1/c` in every cluster and drags every prototype toward itself. The
possibilistic c-means (PCM) of Krishnapuram and Keller drops this constraint:
the *typicality*

.. math::
    t_{ik} = \frac{1}{1 + \left(D^2_{ik}/\eta_i\right)^{1/(m-1)}}

of a sample in cluster :math:`i` depends only on its distance to that
prototype relative to a cluster scale :math:`\eta_i` (the squared distance at
which the typicality is 0.5). Outliers become atypical of *every* cluster and
lose their influence on the prototypes. The scales are estimated from an FCM
run, :math:`\eta_i = K\sum_k\mu_{ik}^m D^2_{ik}/\sum_k\mu_{ik}^m`. The
example shows the role of the factor :math:`K` and of the exponent
:math:`m`, and the price of independent clusters: when the scales are too
wide, two prototypes converge to the same dense region.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from fuzzyclust import FCM
from fuzzyclust.datasets import make_noisy_blobs
from fuzzyclust.extensions import PCM, make_close_blobs, plot_typicality
from fuzzyclust.plotting import PALETTE, plot_clusters, plot_membership_contours

X, y = make_noisy_blobs(n_clusters=3, n_samples_per_cluster=100, n_outliers=60, random_state=0)
inliers = y >= 0
true_centers = np.array([X[inliers][y[inliers] == i].mean(axis=0) for i in range(3)])


def center_error(V):
    """Largest distance between the matched estimated and true centers."""
    D = np.linalg.norm(V[:, None] - true_centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max()


# %%
# FCM versus PCM on three groups with 60 uniformly scattered outliers
# -------------------------------------------------------------------
# PCM is initialized with FCM and takes its scales from the FCM partition.
# With the exponent :math:`m=1.5` recommended for PCM, the typicality decays
# with the *fourth* power of the distance, so a sample at twice the cluster
# radius already has a typicality below 0.1. The PCM prototypes stay on the
# groups, whereas the outliers pull the FCM centers outward.

# [snippet:fit]
fcm = FCM(n_clusters=3, random_state=0).fit(X)
pcm = PCM(n_clusters=3, m=1.5, random_state=0).fit(X)
print(f"scales eta_i = {pcm.eta_.round(2)}")
print(f"center error: FCM {center_error(fcm.cluster_centers_):.3f}, PCM {center_error(pcm.cluster_centers_):.3f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
plot_membership_contours(fcm, X, ax=axes[0], labels=False)
axes[0].set_title(f"FCM: max membership, center error {center_error(fcm.cluster_centers_):.2f}")
plot_membership_contours(pcm, X, ax=axes[1], labels=False)
axes[1].set_title(f"PCM: max typicality, center error {center_error(pcm.cluster_centers_):.2f}")
plot_typicality(X, pcm.typicality_, ax=axes[2])
axes[2].set_title(r"PCM: $\max_i t_{ik}$ of every sample")
for ax in axes:
    ax.scatter(*true_centers.T, marker="*", s=160, color="gold", edgecolor="black", zorder=6, label="true centers")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout()

# %%
# The contour maps show a difference in kind. The maximal FCM membership
# never drops below :math:`1/c` and rises again far from the data (an
# outlier is "fully" assigned to its nearest center), whereas the PCM
# typicality is a bump of width :math:`\sqrt{\eta_i}` around every
# prototype and tends to zero everywhere else. Thresholding
# :math:`\max_i t_{ik}` therefore singles out the outliers.
tmax = pcm.typicality_.max(axis=1)
print(f"median max typicality: inliers {np.median(tmax[inliers]):.2f}, outliers {np.median(tmax[~inliers]):.2f}")
flagged = tmax < 0.2
print(f"samples with max typicality < 0.2: {flagged.sum()} of which {(~inliers & flagged).sum()} are true outliers")

# %%
# Memberships versus typicalities along a line through two clusters
# -----------------------------------------------------------------
# A probe line from the first center through the second and beyond shows
# the two concepts side by side: the FCM memberships cross at the midpoint
# and stay high beyond the prototypes (relative membership; far away they
# tend to the uniform value 1/3), whereas the PCM
# typicalities peak at the prototypes and vanish in both directions
# (absolute membership).
order = np.argsort(true_centers[:, 0])
a, b = true_centers[order[0]], true_centers[order[1]]
t = np.linspace(-0.6, 1.8, 400)
line = a + t[:, None] * (b - a)
U_line = fcm.predict_proba(line)
T_line = pcm.predict_proba(line)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, M, name in zip(axes, (U_line, T_line), ("FCM memberships $\\mu_{ik}$", "PCM typicalities $t_{ik}$"), strict=False):
    for i in range(3):
        ax.plot(t, M[:, i], color=PALETTE[i], lw=1.8, label=f"cluster {i + 1}")
    ax.axvline(0, color="0.5", ls=":")
    ax.axvline(1, color="0.5", ls=":")
    ax.set_xlabel("position along the line (0 = first center, 1 = second center)")
    ax.set_title(name)
    ax.grid(True, color="0.9")
axes[0].set_ylabel("membership / typicality")
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
# Parameter study I: the scale factor :math:`K` and the exponent :math:`m`
# -------------------------------------------------------------------------
# The scales :math:`\eta_i` estimated from FCM are inflated by the outliers
# (they are fuzzy mean squared distances, and the outliers have memberships
# of about one third in every cluster). Enlarging them further with
# :math:`K>1` widens every cluster's zone of influence, lets the outliers
# back in and eventually merges prototypes; shrinking them (:math:`K<1`)
# sharpens the clusters. The exponent :math:`m` sets the decay of the
# typicality, :math:`t\propto (\eta/D^2)^{1/(m-1)}` for large distances:
# with :math:`m=2` an outlier at three cluster radii still has a typicality
# of about 0.1, with :math:`m=1.5` one of 0.01.
Ks = (0.25, 0.5, 1.0, 2.0, 4.0)
ms = (1.5, 2.0, 3.0)
err = np.empty((len(ms), len(Ks)))
out_typ = np.empty_like(err)
for r, m in enumerate(ms):
    for c, K in enumerate(Ks):
        model = PCM(n_clusters=3, m=m, eta_scale=K, random_state=0).fit(X)
        err[r, c] = center_error(model.cluster_centers_)
        out_typ[r, c] = model.typicality_[~inliers].max(axis=1).mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for r, m in enumerate(ms):
    axes[0].plot(Ks, err[r], "o-", color=PALETTE[r], label=f"m = {m}")
    axes[1].plot(Ks, out_typ[r], "o-", color=PALETTE[r], label=f"m = {m}")
axes[0].axhline(center_error(fcm.cluster_centers_), color="0.4", ls="--", label="FCM")
axes[0].set_xscale("log")
axes[1].set_xscale("log")
axes[0].set_xlabel("scale factor $K$")
axes[1].set_xlabel("scale factor $K$")
axes[0].set_ylabel("largest center error")
axes[1].set_ylabel("mean max typicality of the outliers")
axes[0].set_title("Center error against $K$ and $m$")
axes[1].set_title("How typical the outliers remain")
for ax in axes:
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
# Parameter study II: coincident clusters
# ---------------------------------------
# Because the PCM objective is a sum of :math:`c` independent terms, nothing
# keeps two prototypes apart. On a large group next to a small, close one,
# the two prototypes stay separate only while the scales are tight; with a
# large :math:`K` both drift into the large group and the small one is
# lost. FCM does not fail in this way (its memberships compete), which is
# why PCM is best used after FCM, as a robust refinement of a partition that
# FCM has already found.
Xc, yc = make_close_blobs(sizes=(200, 40), spreads=(1.0, 0.4), distance=2.5, random_state=0)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, K in zip(axes, (0.5, 1.0, 4.0), strict=False):
    model = PCM(n_clusters=2, m=1.5, eta_scale=K, random_state=0).fit(Xc)
    sep = np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1])
    plot_clusters(Xc, model.labels_, model.cluster_centers_, ax=ax, legend=False)
    for i, (v, eta) in enumerate(zip(model.cluster_centers_, model.eta_, strict=True)):
        ax.add_patch(plt.Circle(v, np.sqrt(eta), fill=False, lw=1.8, ls="--", color=PALETTE[i]))
    ax.set_title(f"K = {K}: prototype distance {sep:.2f}")
plt.tight_layout()
print("coincident clusters: prototype distance against K")
for K in (0.25, 0.5, 1.0, 2.0, 4.0):
    model = PCM(n_clusters=2, m=1.5, eta_scale=K, random_state=0).fit(Xc)
    print(f"  K = {K:4.2f}: |v1 - v2| = {np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1]):.2f}")

# %%
# **Take-away.** PCM replaces the relative memberships of FCM by absolute
# typicalities that depend only on the distance to the cluster's own
# prototype and on a cluster scale :math:`\eta_i`; outliers become atypical
# of every cluster and the prototypes are robust. Take :math:`m` around 1.5
# (it sets the decay of the typicality, not the fuzziness) and the scales
# from an FCM run with :math:`K=1` or below. A larger :math:`K` lets the
# outliers back in and, since the clusters do not interact, makes
# prototypes coincide. Use PCM as a robust refinement of an FCM partition
# rather than as a stand-alone clusterer, and read the typicalities as a
# per-sample outlier score.
