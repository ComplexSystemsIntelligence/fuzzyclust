r"""
Pairwise constraints: must-link, cannot-link and competitive agglomeration
==========================================================================

Class labels require an agreed set of classes. Often the user can only say
whether two objects *belong together* or *not*, for example whether two
images show the same scene or two records describe the same customer.
Grira, Crucianu and Boujemaa (PCCA) add to the c-means functional the
expected number of violated must-link and cannot-link pairs, weighted by
:math:`\alpha`, and optionally the competitive agglomeration term of
Frigui and Krishnapuram, which lets an overestimated number of clusters
shrink to the number the data support. The two parallel bars of the
previous example are used again: unconstrained FCM cuts across them, and
a few dozen pairs tell it not to.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.cluster import FCM
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.plotting import PALETTE, plot_clusters
from fuzzyclust.semisupervised import (
    PairwiseConstrainedFCM,
    make_pairwise_constraints,
    make_partially_labeled_pair,
    plot_pairwise_constraints,
)

X, y, _ = make_partially_labeled_pair(n_samples=300, random_state=0)

# %%
# Fifty pairs of each kind
# ------------------------
# The constraints are drawn at random from the true classes (in practice
# they come from a user). With :math:`\alpha=5` (the penalty is measured
# in the units of the squared distances), the constrained partition
# separates the bars and satisfies every constraint. The constraints are
# drawn on top of the partitions (green solid lines for must-link and
# vermilion dashed lines for cannot-link pairs): for FCM only the pairs it
# violates, for PCCA all of them.

# [snippet:fit]
must_link, cannot_link = make_pairwise_constraints(y, n_must_link=50, n_cannot_link=50, random_state=0)
fcm = FCM(n_clusters=2, random_state=0).fit(X)
pcca = PairwiseConstrainedFCM(n_clusters=2, alpha=5.0, random_state=0)
pcca.fit(X, must_link=must_link, cannot_link=cannot_link)
print(f"FCM  ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
print(f"PCCA ARI = {adjusted_rand_score(y, pcca.labels_):.2f}, satisfied must-link "
      f"{pcca.must_link_satisfied_:.2f}, cannot-link {pcca.cannot_link_satisfied_:.2f}")
# [/snippet:fit]

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True)
plot_pairwise_constraints(X, must_link, cannot_link, labels=fcm.labels_, centers=fcm.cluster_centers_,
                          ax=axes[0], legend=False, violated_only=True)
same = fcm.labels_[must_link[:, 0]] == fcm.labels_[must_link[:, 1]]
apart = fcm.labels_[cannot_link[:, 0]] != fcm.labels_[cannot_link[:, 1]]
n_violated = int((~same).sum() + (~apart).sum())
axes[0].set_title(f"FCM violates {n_violated} of 100 constraints: ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
plot_pairwise_constraints(X, must_link, cannot_link, labels=pcca.labels_, centers=pcca.cluster_centers_,
                          ax=axes[1])
axes[1].set_title(f"PCCA, $\\alpha=5$: ARI = {adjusted_rand_score(y, pcca.labels_):.2f}")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()

# %%
# Parameter study I: the penalty :math:`\alpha` and the number of pairs
# ----------------------------------------------------------------------
# The penalty is varied for 10, 25, 50 and 100 pairs of each kind. With
# :math:`\alpha=0` the method is FCM and violates a quarter to a third of
# the constraints. The more pairs there are, the smaller the penalty that
# turns the partition: 100 pairs need :math:`\alpha\approx2`, 25 pairs
# :math:`\alpha\approx10`, and 10 pairs reach an ARI of only 0.7 even at
# :math:`\alpha=20`, although nine tenths of them are already satisfied at
# :math:`\alpha=0.5`: a handful of pairs can be honored by bending the
# boundary locally around them. ``alpha="auto"`` resets the weight at every
# iteration to the FCM term divided by the number of constraints, following
# the paper's rule that the constraints should weigh as much as the data.

# [snippet:scan]
alphas = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
n_pairs = (10, 25, 50, 100)
ari = np.zeros((len(n_pairs), len(alphas)))
satisfied = np.zeros_like(ari)
for i, n in enumerate(n_pairs):
    ml, cl = make_pairwise_constraints(y, n_must_link=n, n_cannot_link=n, random_state=0)
    for j, alpha in enumerate(alphas):
        model = PairwiseConstrainedFCM(n_clusters=2, alpha=alpha, random_state=0)
        model.fit(X, must_link=ml, cannot_link=cl)
        ari[i, j] = adjusted_rand_score(y, model.labels_)
        satisfied[i, j] = 0.5 * (model.must_link_satisfied_ + model.cannot_link_satisfied_)
    auto = PairwiseConstrainedFCM(n_clusters=2, alpha="auto", random_state=0).fit(X, must_link=ml,
                                                                                    cannot_link=cl)
    print(f"{n:3d} pairs of each kind, alpha='auto' (final alpha = {auto.alpha_:.2f}): "
          f"ARI = {adjusted_rand_score(y, auto.labels_):.2f}")
# [/snippet:scan]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(alphas))
for i, n in enumerate(n_pairs):
    axes[0].plot(x, ari[i], "o-", color=PALETTE[i], label=f"{n} + {n} pairs")
    axes[1].plot(x, satisfied[i], "o-", color=PALETTE[i], label=f"{n} + {n} pairs")
axes[0].set_ylabel("ARI with the true bars")
axes[0].set_title("Agreement with the classes")
axes[1].set_ylabel("share of satisfied constraints")
axes[1].set_title("Constraint satisfaction")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"penalty $\alpha$")
    ax.grid(True, color="0.9")
axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
# Parameter study II: competitive agglomeration
# ---------------------------------------------
# With ``agglomeration > 0`` the term :math:`-\beta\sum_i N_i^2` rewards
# large clusters, so clusters compete for samples and those whose fuzzy
# cardinality falls below ``min_cardinality`` are discarded. Started with
# eight clusters on three Gaussian groups, a weak agglomeration
# (:math:`\eta_0\le1`) leaves the clusters in place; with the default decay
# :math:`\tau=10`, the range of :math:`\eta_0` from 5 to 20 ends with
# exactly three clusters, and a very strong agglomeration lets a single
# cluster swallow everything. The decay :math:`\tau` of the schedule
# :math:`\eta_0e^{-l/\tau}` sets how long the competition lasts: a longer
# competition (:math:`\tau=30`) shifts the window toward weaker weights.

# [snippet:agglomeration]
Xb, yb = make_fuzzy_blobs(random_state=0)
etas = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)
final = {tau: [] for tau in (5.0, 10.0, 30.0)}
for tau in final:
    for eta in etas:
        model = PairwiseConstrainedFCM(n_clusters=8, alpha=0.0, agglomeration=eta, agglomeration_decay=tau,
                                       random_state=0).fit(Xb)
        final[tau].append(model.n_clusters_)
# [/snippet:agglomeration]
best = PairwiseConstrainedFCM(n_clusters=8, alpha=0.0, agglomeration=5.0, random_state=0).fit(Xb)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for j, (tau, counts) in enumerate(final.items()):
    axes[0].plot(etas, counts, "o-", color=PALETTE[j], label=rf"$\tau={tau:g}$")
axes[0].axhline(3, color="0.5", ls=":", lw=1)
axes[0].set_xscale("log")
axes[0].set_xticks(etas)
axes[0].set_xticklabels([f"{e:g}" for e in etas])
axes[0].set_xlabel(r"agglomeration weight $\eta_0$")
axes[0].set_ylabel("final number of clusters")
axes[0].set_title("Started with eight clusters")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(True, color="0.9")
plot_clusters(Xb, best.labels_, best.cluster_centers_, ax=axes[1], legend=False)
axes[1].set_title(f"$\\eta_0=5$: {best.n_clusters_} clusters, ARI = {adjusted_rand_score(yb, best.labels_):.2f}")
axes[1].set_aspect("equal")
plt.tight_layout()

# %%
# **Take-away.** Pairwise constraints carry less information than labels
# but are cheaper to collect and need no list of classes. PCCA turns them
# into a penalty on the expected number of violated pairs, which enters
# the membership update as a correction toward the clusters that violate
# fewer constraints. The penalty must be large enough to move the
# partition, and the required value grows as the number of pairs shrinks;
# a few pairs may be satisfied locally without changing the global
# structure. Competitive agglomeration adds a way to start from too many
# clusters, with a usable window of the agglomeration weight between "no
# competition" and "one cluster takes all".
