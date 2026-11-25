r"""
Collaborative fuzzy clustering across three data sites
======================================================

Three sites, such as hospitals, branches of a company or laboratories,
describe the *same* 300 objects with *different* features, and the raw
data may not leave the sites. Pedrycz's collaborative clustering runs
fuzzy c-means at every site, then lets the sites exchange only their
partition matrices and re-cluster with a term that penalizes the
difference from the other sites' memberships,

.. math::
    Q[ii] = \sum_{i,k}\mu_{ik}^2[ii]D_{ik}^2[ii]
          + \sum_{jj\ne ii}\alpha[ii,jj]\sum_{i,k}(\mu_{ik}[ii]-\mu_{ik}[jj])^2D_{ik}^2[ii] .

The three sites below see the same three groups with increasingly blurred
features (standard deviations 0.6, 1.2 and 1.8), and at the third site 15 %
of the objects are moved to another group, so that this site genuinely
disagrees with the others about those objects.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.collaborative import (
    CollaborativeFCM,
    make_collaborative_sites,
    plot_collaboration_trajectory,
    plot_site_partitions,
)
from fuzzyclust.plotting import PALETTE
from fuzzyclust.validity import fuzzy_adjusted_rand_index

Xs, y = make_collaborative_sites(n_sites=3, spreads=(0.6, 1.2, 1.8), swap_fraction=0.15, random_state=0)

# %%
# Independent and collaborative partitions
# ----------------------------------------
# With :math:`\alpha=0` the sites cluster independently (the clusters are
# matched across the sites by the Hungarian algorithm on their membership
# overlap). The clear site 1 finds the groups exactly, the blurred site 3
# mixes them. With :math:`\alpha=3` every site listens to the other two.
# Site 3 is pulled toward the structure the others agree on and even
# reassigns the objects whose features were moved (its clusters now
# overlap in its own feature space), while site 1 keeps its partition and
# only becomes softer.

# [snippet:fit]
independent = CollaborativeFCM(n_clusters=3, alpha=0.0, random_state=0).fit(Xs)
collab = CollaborativeFCM(n_clusters=3, alpha=3.0, random_state=0).fit(Xs)
for name, model in (("independent", independent), ("alpha = 3", collab)):
    print(f"{name:12s}: ARI per site = "
          + ", ".join(f"{adjusted_rand_score(y, lab):.2f}" for lab in model.labels_)
          + f";  membership change = {np.round(model.collaboration_change_, 3)}")
# [/snippet:fit]

fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
plot_site_partitions(Xs, independent, axes=axes[0], independent=True)
plot_site_partitions(Xs, collab, axes=axes[1])
for row, model in zip(axes, (independent, collab), strict=True):
    for p, ax in enumerate(row):
        ax.set_title(ax.get_title() + f": ARI = {adjusted_rand_score(y, model.labels_[p]):.2f}", fontsize=9)
plt.tight_layout()

# %%
# Parameter study: the collaboration weight :math:`\alpha`
# --------------------------------------------------------
# For a range of weights, we record the agreement between the sites (fuzzy
# Rand index of their partitions, :func:`fuzzyclust.validity.fuzzy_rand_index`)
# and the agreement of every site with the true groups. The sites converge
# to each other monotonically; the partition of the clear site stays
# correct, the blurred sites improve, and site 3 follows the
# majority on its moved objects from :math:`\alpha\approx1` on. How far a
# site has to move (``collaboration_change_``, the mean absolute change of
# its memberships) measures how much it disagreed: site 3 gives in most,
# but every site moves, because a membership pulled toward the others is
# also a softer one.

# [snippet:scan]
alphas = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
models = [CollaborativeFCM(n_clusters=3, alpha=a, random_state=0).fit(Xs) for a in alphas]
# [/snippet:scan]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
plot_collaboration_trajectory(alphas, models, ax=axes[0])
axes[0].set_xscale("symlog", linthresh=0.1)
axes[0].set_title("Agreement between the sites")
for p in range(3):
    axes[1].plot(alphas, [adjusted_rand_score(y, m.labels_[p]) for m in models], "o-", color=PALETTE[p],
                 label=f"site {p + 1}")
    axes[2].plot(alphas, [m.collaboration_change_[p] for m in models], "o-", color=PALETTE[p],
                 label=f"site {p + 1}")
axes[1].set_ylabel("ARI with the true groups")
axes[1].set_title("Agreement with the truth")
axes[2].set_ylabel(r"mean $|\Delta\mu_{ik}|$ from the independent run")
axes[2].set_title("How much every site gives in")
for ax in axes[1:]:
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"collaboration weight $\alpha$")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"):
    for ax in axes[1:]:
        ax.spines[s].set_visible(False)
plt.tight_layout()

# %%
# Asymmetric collaboration
# ------------------------
# The collaboration matrix need not be symmetric. When only site 3 listens
# (row 3 of :math:`\alpha` nonzero), sites 1 and 2 keep their independent
# partitions and site 3 alone is corrected; when only site 1 listens, the
# clear site is pulled toward the noisier ones and loses some of its
# clarity. The chance-corrected fuzzy adjusted Rand index between the
# consensus of the sites (their mean membership) and the truth is highest
# when only the deviating site listens: collaboration in all directions
# makes every partition softer, which the fuzzy index penalizes although
# the hard labels of all sites are then correct.

# [snippet:asymmetric]
listen_3 = np.array([[0, 0, 0], [0, 0, 0], [3.0, 3.0, 0]])
listen_1 = np.array([[0, 3.0, 3.0], [0, 0, 0], [0, 0, 0]])
for name, A in (("only site 3 listens", listen_3), ("only site 1 listens", listen_1),
                ("all listen", 3.0)):
    m = CollaborativeFCM(n_clusters=3, alpha=A, random_state=0).fit(Xs)
    truth = np.eye(3)[y]
    print(f"{name:20s}: ARI per site = "
          + ", ".join(f"{adjusted_rand_score(y, lab):.2f}" for lab in m.labels_)
          + f";  fuzzy ARI of the consensus = {fuzzy_adjusted_rand_index(truth, m.consensus_membership_):.2f}")
# [/snippet:asymmetric]

# %%
# **Take-away.** Collaborative clustering reconciles the structures found
# at several sites without moving any raw data: only partition matrices
# travel. The collaboration weight sets how strongly a site defers to the
# others; increasing it pulls the partitions together monotonically and
# lets a weak or deviating site borrow the structure the others agree on,
# while a clear site barely moves. The amount by which every site has to
# change is itself informative, because it locates the site whose view of
# the objects differs. The collaboration matrix can be asymmetric, so trust
# can be given selectively.
