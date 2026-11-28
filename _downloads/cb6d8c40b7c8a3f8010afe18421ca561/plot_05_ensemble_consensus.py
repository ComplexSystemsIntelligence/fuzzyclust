r"""
Consensus of soft clusterings: a library beats a single run
===========================================================

A single run of fuzzy c-means depends on its random start, on the sample
it saw and on the features it used. A *cluster ensemble* fits a library of
such runs and combines them. Soft partitions are combined through the
*co-association matrix*: the library average of the similarity of two
objects' membership vectors, which needs no matching of cluster labels
between the members and works even when the members have different
numbers of clusters. A consensus function then partitions the objects
from this matrix: fuzzy c-means on its rows, spectral partitioning,
evidence accumulation (hierarchical clustering of :math:`1-s_{kl}`), the
soft meta-clustering algorithm sMCLA or information-theoretic k-means
(Punera and Ghosh; Strehl and Ghosh; Fred and Jain).

The data are four overlapping Gaussian groups in six dimensions. Every
member sees only two random features and 30 % of the samples, so each
member is a weak, noisy clusterer, which is the situation in which
ensembles pay off.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.cluster import FCM
from fuzzyclust.ensemble import (
    FuzzyClusterEnsemble,
    make_overlapping_blobs,
    plot_coassociation,
    stability_curve,
)
from fuzzyclust.plotting import PALETTE
from fuzzyclust.validity import fuzzy_adjusted_rand_index

X, y = make_overlapping_blobs(n_clusters=4, n_features=6, separation=3.0, random_state=0)

# %%
# Twenty weak members and their consensus
# ---------------------------------------
# The members reach a mean ARI below 0.4, the best of them below 0.8; the
# consensus of the twenty reaches about 0.86. It is better than every
# single member, because the members err in different directions and the
# co-association averages out their errors. The reordered co-association
# matrix shows the four groups as dark diagonal blocks.

# [snippet:fit]
ens = FuzzyClusterEnsemble(n_clusters=4, n_estimators=20, sample_fraction=0.3, feature_fraction=0.34,
                           consensus="fcm", random_state=0).fit(X)
members = [adjusted_rand_score(y, U.argmax(axis=1)) for U in ens.base_memberships_]
print(f"members: mean ARI = {np.mean(members):.2f}, best = {np.max(members):.2f};  "
      f"consensus ARI = {adjusted_rand_score(y, ens.labels_):.2f}")
# [/snippet:fit]
full = FCM(n_clusters=4, random_state=0).fit(X)
print(f"for comparison, FCM on all samples and all six features: ARI = {adjusted_rand_score(y, full.labels_):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(members, bins=np.linspace(0, 1, 21), color="0.7", edgecolor="white", label="20 members")
axes[0].axvline(adjusted_rand_score(y, ens.labels_), color=PALETTE[1], lw=2.2, label="consensus")
axes[0].axvline(np.mean(members), color="0.3", lw=1.2, ls="--", label="member mean")
axes[0].set_xlabel("ARI with the true groups")
axes[0].set_ylabel("number of members")
axes[0].set_title("Weak members (2 features, 30 % of samples)")
axes[0].legend(frameon=False, fontsize=8)
plot_coassociation(ens, ax=axes[1])
axes[1].set_title("Co-association, ordered by the consensus")
plt.tight_layout()

# %%
# Parameter study I: library size and consensus function
# -------------------------------------------------------
# We record the ARI of the consensus for libraries of 1 to 20 members and
# the five consensus functions (mean over two random seeds). One
# member is as weak as any; the first three to ten members bring most of
# the gain, after which the curves flatten. The consensus functions behave
# alike; sMCLA, which groups the clusters of the members instead of the
# objects, stays a few points behind from ten members on.

# [snippet:library]
sizes = (1, 3, 10, 20)
functions = ("fcm", "spectral", "hierarchical", "smcla", "itk")
curve = {f: np.zeros(len(sizes)) for f in functions}
for f in functions:
    for j, Q in enumerate(sizes):
        curve[f][j] = np.mean([
            adjusted_rand_score(y, FuzzyClusterEnsemble(
                n_clusters=4, n_estimators=Q, sample_fraction=0.3, feature_fraction=0.34, consensus=f,
                random_state=seed).fit(X).labels_)
            for seed in range(2)
        ])
# [/snippet:library]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, f in enumerate(functions):
    ax.plot(sizes, curve[f], "o-", color=PALETTE[i], label=f)
ax.axhline(np.mean(members), color="0.5", ls="--", lw=1, label="mean member")
ax.set_xscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.set_xlabel("library size $Q$")
ax.set_ylabel("ARI of the consensus")
ax.set_title("Consensus quality versus library size")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(True, color="0.9")

# %%
# Parameter study II: stability of the consensus
# ----------------------------------------------
# Accuracy needs ground truth; *reproducibility* does not. On ten
# overlapping groups in the plane, FCM with ten clusters falls into
# different local optima from different starts, so two runs disagree.
# :func:`~fuzzyclust.ensemble.stability_curve` fits several independent
# ensembles per library size and compares their consensus partitions in
# pairs with the fuzzy adjusted Rand index
# (:func:`fuzzyclust.validity.fuzzy_adjusted_rand_index`). The agreement
# between repeated consensus partitions rises with the library size, so
# the ensemble turns an unstable clusterer into a reproducible one.

X10, y10 = make_overlapping_blobs(n_clusters=10, separation=2.5, box=10.0, n_samples_per_cluster=25,
                                  random_state=0)
# [snippet:stability]
table = stability_curve(X10, n_estimators_grid=(1, 3, 10, 20), n_repeats=5, n_clusters=10,
                        random_state=0, index=fuzzy_adjusted_rand_index)
print(table.round(3))
# [/snippet:stability]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.errorbar(table.index, table["mean"], yerr=table["std"], fmt="o-", color=PALETTE[0], capsize=3)
ax.set_xscale("log")
ax.set_xticks(table.index)
ax.set_xticklabels(table.index)
ax.set_xlabel("library size $Q$")
ax.set_ylabel("fuzzy ARI between repeats")
ax.set_title("Reproducibility of the consensus (ten overlapping groups)")
ax.grid(True, color="0.9")

# %%
# **Take-away.** An ensemble of weak, noisy fuzzy clusterings is combined
# through their co-association matrix, which compares only co-memberships
# and therefore needs no matching of clusters between the members. The
# consensus is better than every member here, most of the gain arrives
# with the first three to ten members, and the consensus functions agree
# closely. Beyond accuracy, the ensemble makes the result reproducible:
# repeated consensus partitions agree far more than repeated single runs
# of an algorithm with many local optima.
