r"""
Fuzzy c-means with partial supervision: label fraction and weight
=================================================================

Unsupervised clustering finds the structure the *distance* suggests, which
need not be the structure the user means. Pedrycz and Waletzky add to the
c-means functional a term that penalizes the difference between the
memberships of a few *labeled* samples and their labels,

.. math::
    J = \sum_{i,k}\mu_{ik}^2 D_{ik}^2 + \alpha\sum_{i,k}(\mu_{ik}-f_{ik}b_k)^2 D_{ik}^2 ,

where :math:`b_k` marks the labeled samples and :math:`f_{ik}` their
label memberships. Because the prototypes are shared by labeled and
unlabeled samples, a handful of labels can rotate the whole partition.
The data set is built for this purpose: two long parallel bars whose natural
two-cluster FCM split cuts across both bars, because the variance along
the bars dominates the gap between them.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.cluster import FCM
from fuzzyclust.plotting import PALETTE
from fuzzyclust.semisupervised import (
    SemiSupervisedFCM,
    make_partial_labels,
    make_partially_labeled_pair,
    plot_label_propagation,
)

X, y, y_partial = make_partially_labeled_pair(n_samples=300, label_fraction=0.1, random_state=0)

# %%
# FCM versus partially supervised FCM
# -----------------------------------
# Plain FCM splits the bars by a tilted line (ARI around 0.4). With 10 % of
# the samples labeled (``-1`` marks the others, as in scikit-learn's
# semi-supervised module) and :math:`\alpha=1`, the prototypes move toward
# the two bars and the boundary turns toward the gap. The labeled samples
# are drawn as large black-edged markers; ``init="labels"`` starts every
# supervised prototype at the mean of its labeled samples.

# [snippet:fit]
fcm = FCM(n_clusters=2, random_state=0).fit(X)
ssfcm = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, y_partial)
print(f"FCM ARI = {adjusted_rand_score(y, fcm.labels_):.2f}, "
      f"SSFCM ARI = {adjusted_rand_score(y, ssfcm.labels_):.2f}, "
      f"accuracy of the class prediction = {ssfcm.score_labels(X, y):.2f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True, sharey=True)
plot_label_propagation(X, np.full_like(y, -1), labels=fcm.labels_, ax=axes[0], legend=False)
axes[0].scatter(*fcm.cluster_centers_.T, marker="X", s=140, color="black", edgecolor="white", zorder=5)
axes[0].set_title(f"FCM, no labels: ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
plot_label_propagation(X, y_partial, ssfcm, ax=axes[1])
axes[1].set_title(f"SSFCM, 10 % labels, $\\alpha=1$: ARI = {adjusted_rand_score(y, ssfcm.labels_):.2f}")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()

# %%
# The labeled samples are not clamped: their membership is the convex
# combination :math:`\frac{1}{1+\alpha}\mu^{\mathrm{FCM}}_{ik}+\frac{\alpha}{1+\alpha}f_{ik}`,
# so they follow the label to the degree :math:`\alpha/(1+\alpha)`. Plotted
# against the position *along* the bars, the membership in the upper-bar
# cluster is clear near the middle of the bars and drifts toward 0.5 at
# their ends, where both prototypes are almost equally far away: at the
# ends, spherical prototypes cannot tell the bars apart, and this is where
# the remaining errors occur. The labeled samples (black) lie closer to
# their label than their unlabeled neighbors.
ic = int(np.flatnonzero(ssfcm.label_mapping_ == 1)[0])  # the cluster tied to the upper bar
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for cls in (0, 1):
    mask = (y == cls) & ~ssfcm.labelled_mask_
    ax.scatter(X[mask, 0], ssfcm.membership_[mask, ic], s=10, color=PALETTE[cls], alpha=0.7,
               label=f"unlabeled, {'upper' if cls else 'lower'} bar")
lab = ssfcm.labelled_mask_
ax.scatter(X[lab, 0], ssfcm.membership_[lab, ic], color="black", s=22, zorder=4, label="labeled samples")
ax.set_xlabel("$x_1$ (along the bars)")
ax.set_ylabel("membership in the upper-bar cluster")
ax.set_title("Memberships along the bars, $\\alpha=1$")
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.grid(True, color="0.9")

# %%
# Parameter study: label fraction and weight :math:`\alpha`
# ---------------------------------------------------------
# For label fractions from 1 % to 30 % and several weights, we record the
# agreement with the true bars (each label set is drawn once with
# ``random_state=0``). Up to :math:`\alpha=3` more labels always help. The
# weight acts non-monotonically: :math:`\alpha=0` is plain FCM, a moderate weight
# turns the partition, but a very large one makes the labeled memberships
# equal to the labels. The supervision term of the labeled samples then
# vanishes, while that of the unlabeled samples, :math:`\alpha\mu_{ik}^2D_{ik}^2`,
# only rescales the FCM functional, so the labeled minority is outweighed.
# The paper's rule :math:`\alpha=N/N_b` (``alpha="auto"``, with :math:`N_b`
# labeled samples) gives such a large weight when labels are scarce.

# [snippet:scan]
fractions = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
alphas = (0.0, 0.3, 1.0, 3.0, 10.0, "auto")
ari = np.zeros((len(alphas), len(fractions)))
for j, frac in enumerate(fractions):
    yp = make_partial_labels(y, fraction=frac, random_state=0)
    for i, alpha in enumerate(alphas):
        model = SemiSupervisedFCM(n_clusters=2, alpha=alpha, random_state=0).fit(X, yp)
        ari[i, j] = adjusted_rand_score(y, model.labels_)
# [/snippet:scan]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, alpha in enumerate(alphas):
    name = r"$\alpha=N/N_b$ (auto)" if alpha == "auto" else rf"$\alpha={alpha:g}$"
    ax.plot(100 * np.asarray(fractions), ari[i], "o-", color=PALETTE[i % len(PALETTE)], label=name)
ax.set_xscale("log")
ax.set_xticks(100 * np.asarray(fractions))
ax.set_xticklabels([f"{100 * f:g}" for f in fractions])
ax.set_xlabel("labeled samples (%)")
ax.set_ylabel("ARI with the true bars")
ax.set_title("Effect of the label fraction and of the weight $\\alpha$")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(True, color="0.9")
for alpha, row in zip(alphas, ari, strict=True):
    print(f"alpha = {alpha!s:>4}: " + "  ".join(f"{v:.2f}" for v in row))

# %%
# **Take-away.** A few labels steer fuzzy c-means because they act on the
# shared prototypes: with 10 % labeled samples the ARI rises from 0.39
# to about 0.6, and with 30 % to 0.77. The boundary turns from the
# variance-driven split toward the intended one, although the bars remain
# a hard case for a spherical prototype. For the weight :math:`\alpha`,
# more is not better: the supervision term vanishes once the labeled
# memberships match their labels, so moderate weights around one turn the
# partition most, whereas the paper's rule :math:`\alpha=N/N_b` gives a large
# weight, and no better result, when labels are scarce.
# Start the prototypes from the labels (the default), because a random
# start that ties a cluster to the wrong class makes supervision fight the
# geometry.
