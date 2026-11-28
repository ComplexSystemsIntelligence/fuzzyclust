r"""
Noise clustering: a virtual cluster at constant distance δ
==========================================================

Davé's noise clustering keeps the probabilistic constraint of FCM but adds a
:math:`(c+1)`-th *noise prototype* that is, by definition, at the same
distance :math:`\delta` from every sample. A sample farther than
:math:`\delta` from all real prototypes is closer to the noise prototype and
gives most of its membership to the noise cluster; the real clusters only
see the remainder,

.. math::
    \mu_{ik} = \frac{1}{\sum_{j=1}^{c}\left(D^2_{ik}/D^2_{jk}\right)^{1/(m-1)}
                      + \left(D^2_{ik}/\delta^2\right)^{1/(m-1)}},\qquad
    \mu_{*k} = 1-\sum_{i=1}^{c}\mu_{ik}.

Unlike in PCM, the real clusters still compete for every sample, so
prototypes do not coincide. The noise distance is tied to the scale of the
data, :math:`\delta^2=\lambda\,\overline{D^2}`, with the multiplier
:math:`\lambda` as the only parameter. The example compares noise
clustering with FCM on noisy blobs and shows how :math:`\lambda` (or a
fixed :math:`\delta`) decides what counts as noise.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.datasets import make_noisy_blobs
from fuzzyclust.extensions import NoiseClustering, plot_typicality
from fuzzyclust.plotting import PALETTE, plot_clusters, plot_membership_contours

X, y = make_noisy_blobs(n_clusters=3, n_samples_per_cluster=100, n_outliers=60, random_state=0)
inliers = y >= 0
true_centers = np.array([X[inliers][y[inliers] == i].mean(axis=0) for i in range(3)])


def center_error(V):
    D = np.linalg.norm(V[:, None] - true_centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max()


# %%
# FCM versus noise clustering
# ---------------------------
# With :math:`\lambda=0.3` the noise distance comes out at about 3, a few
# cluster radii. Samples beyond that distance from every prototype are
# labeled ``-1`` (gray), the prototypes stay on the groups, and the
# membership contours of the real clusters now *close* around the groups
# instead of tiling the whole plane as in FCM.

# [snippet:fit]
fcm = FCM(n_clusters=3, random_state=0).fit(X)
nc = NoiseClustering(n_clusters=3, delta_scale=0.3, random_state=0).fit(X)
n_noise = (nc.labels_ == -1).sum()
print(f"noise distance delta = {nc.delta_:.2f}; {n_noise} samples labeled noise, "
      f"{((nc.labels_ == -1) & ~inliers).sum()} of them true outliers")
print(f"center error: FCM {center_error(fcm.cluster_centers_):.3f}, noise clustering {center_error(nc.cluster_centers_):.3f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
plot_membership_contours(fcm, X, ax=axes[0], labels=False)
axes[0].set_title(f"FCM: center error {center_error(fcm.cluster_centers_):.2f}")
plot_membership_contours(nc, X, ax=axes[1], labels=False)
for v in nc.cluster_centers_:
    axes[1].add_patch(plt.Circle(v, nc.delta_, fill=False, ls="--", lw=1.2, color="0.3"))
axes[1].set_title(rf"Noise clustering, $\lambda=0.3$ ($\delta$ = {nc.delta_:.2f}): center error {center_error(nc.cluster_centers_):.2f}")
plot_typicality(X, 1.0 - nc.noise_membership_, ax=axes[2], label=r"$1-\mu_{*k}$")
axes[2].set_title(r"$1-\mu_{*k}$: membership in the real clusters")
plt.tight_layout()

# %%
# Memberships along a line through two clusters
# ---------------------------------------------
# Along a probe line from one center through another and beyond, the FCM
# memberships cross at the midpoint and stay high beyond the prototypes, whereas in
# noise clustering the real memberships decay and the noise membership
# (gray) takes over beyond :math:`\delta`; the real memberships no longer
# sum to one.
order = np.argsort(true_centers[:, 0])
a, b = true_centers[order[0]], true_centers[order[1]]
t = np.linspace(-0.8, 1.8, 400)
line = a + t[:, None] * (b - a)
U_fcm = fcm.predict_proba(line)
U_nc = nc.predict_proba(line)
noise_line = nc.predict_noise_membership(line)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for i in range(3):
    axes[0].plot(t, U_fcm[:, i], color=PALETTE[i], lw=1.8, label=f"cluster {i + 1}")
    axes[1].plot(t, U_nc[:, i], color=PALETTE[i], lw=1.8, label=f"cluster {i + 1}")
axes[1].plot(t, noise_line, color="0.4", lw=2.2, ls="--", label="noise cluster")
for ax, name in zip(axes, ("FCM memberships", r"noise clustering memberships ($\lambda=0.3$)"), strict=False):
    ax.axvline(0, color="0.5", ls=":")
    ax.axvline(1, color="0.5", ls=":")
    ax.set_xlabel("position along the line (0 = first center, 1 = second center)")
    ax.set_title(name)
    ax.grid(True, color="0.9")
axes[0].set_ylabel(r"membership $\mu_{ik}$")
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
# Parameter study: the multiplier :math:`\lambda` (and a fixed :math:`\delta`)
# ------------------------------------------------------------------------------
# :math:`\delta` acts as a radius beyond which a sample is more noise than
# member. The mean squared distance :math:`\overline{D^2}` of all samples to
# all prototypes is dominated by the distances *between* the groups, so
# :math:`\lambda` has to be well below one to catch outliers near a group;
# a small :math:`\lambda` starts to peel off the tails of the groups.
# A fixed ``delta`` can be given instead when the noise radius is known;
# with :math:`\delta\to\infty` the noise cluster is never the closest one
# and FCM is recovered.
lams = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)
rows = []
for lam in lams:
    model = NoiseClustering(n_clusters=3, delta_scale=lam, random_state=0).fit(X)
    noise = model.labels_ == -1
    rows.append((model.delta_, noise.sum(), (noise & ~inliers).sum(), (noise & inliers).sum(),
                 center_error(model.cluster_centers_), adjusted_rand_score(y[inliers], model.labels_[inliers])))
rows = np.array(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(lams, rows[:, 2], "o-", color=PALETTE[1], label="true outliers labeled noise")
axes[0].plot(lams, rows[:, 3], "s-", color=PALETTE[0], label="inliers labeled noise")
axes[0].axhline((~inliers).sum(), color="0.4", ls="--", label="number of outliers")
axes[0].set_xscale("log")
axes[0].set_xlabel(r"multiplier $\lambda$")
axes[0].set_ylabel("samples")
axes[0].set_title(r"What is declared noise as $\lambda$ varies")
axes[1].plot(lams, rows[:, 4], "o-", color=PALETTE[2], label="noise clustering")
axes[1].axhline(center_error(fcm.cluster_centers_), color="0.4", ls="--", label="FCM")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"multiplier $\lambda$")
axes[1].set_ylabel("largest center error")
axes[1].set_title(r"Center error against $\lambda$")
for ax in axes:
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
print(f"{'lambda':>7s} {'delta':>6s} {'noise':>6s} {'true':>5s} {'inl.':>5s} {'err':>6s} {'ARI':>5s}")
for lam, r in zip(lams, rows, strict=False):
    print(f"{lam:7.2f} {r[0]:6.2f} {int(r[1]):6d} {int(r[2]):5d} {int(r[3]):5d} {r[4]:6.3f} {r[5]:5.2f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, lam in zip(axes, (0.05, 0.3, 2.0), strict=False):
    model = NoiseClustering(n_clusters=3, delta_scale=lam, random_state=0).fit(X)
    plot_clusters(X, model.labels_, model.cluster_centers_, ax=ax, legend=False)
    for v in model.cluster_centers_:
        ax.add_patch(plt.Circle(v, model.delta_, fill=False, ls="--", lw=1.2, color="0.3"))
    ax.set_title(rf"$\lambda$ = {lam}, $\delta$ = {model.delta_:.2f}: {(model.labels_ == -1).sum()} noise samples")
plt.tight_layout()
fixed = NoiseClustering(n_clusters=3, delta=1e6, random_state=0).fit(X)
print(f"delta = 1e6: largest noise membership {fixed.noise_membership_.max():.1e}, "
      f"center error {center_error(fixed.cluster_centers_):.3f} (FCM: {center_error(fcm.cluster_centers_):.3f})")

# %%
# **Take-away.** One extra, position-less prototype at distance
# :math:`\delta` turns FCM into a robust clusterer: outliers migrate to the
# noise cluster, the real memberships close around the groups and the
# prototypes are protected, while the clusters keep competing for every
# sample (no coincident clusters). Everything depends on :math:`\delta`:
# tie it to the data with :math:`\lambda\approx0.1`--:math:`0.5` of the
# mean squared distance, or fix it when the noise radius is known, and read
# the noise membership as an outlier score.
