r"""
Fuzzy c-means: memberships and the fuzziness exponent
======================================================

Fuzzy c-means (FCM) is the workhorse of fuzzy clustering. Instead of assigning
every sample :math:`\mathbf{x}_k` to exactly one cluster, it assigns a membership
degree :math:`\mu_{ik}\in[0,1]` to every cluster :math:`i`, with
:math:`\sum_i \mu_{ik}=1`. The algorithm alternates two closed-form updates
derived from the c-means functional
:math:`J=\sum_i\sum_k \mu_{ik}^m D^2_{ik}`:

* the prototypes are membership-weighted means,
  :math:`\mathbf{v}_i = \sum_k \mu_{ik}^m\mathbf{x}_k / \sum_k \mu_{ik}^m`;
* the memberships follow from the distances,
  :math:`\mu_{ik} = 1/\sum_j (D_{ik}/D_{jk})^{2/(m-1)}`.

Besides the number of clusters, the only algorithmic hyperparameter is the
fuzziness exponent :math:`m>1`. This example shows what the memberships look
like and how :math:`m` controls the softness of the partition.
"""

# %%
# Generate three well-separated spherical clusters. ``make_fuzzy_blobs`` is the
# Python counterpart of the MATLAB toolbox function ``nDexample``.
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust import FCM
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.plotting import plot_clusters, plot_membership_contours

# [snippet:fit]
X, y_true = make_fuzzy_blobs(n_clusters=3, n_samples_per_cluster=100, random_state=0)

fcm = FCM(n_clusters=3, m=2.0, random_state=0).fit(X)
# [/snippet:fit]
print(f"converged in {fcm.n_iter_} iterations, J = {fcm.objective_:.2f}")
print("prototypes:\n", fcm.cluster_centers_.round(2))

# %%
# The fitted model exposes the partition as ``membership_`` (one row per sample,
# one column per cluster) and the hard labels as ``labels_``. The contour map
# shows the level curves of the maximal membership :math:`\max_i\mu_{ik}`;
# for FCM they are circles, because the norm is Euclidean.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_clusters(X, fcm.labels_, fcm.cluster_centers_, ax=axes[0])
axes[0].set_title("hard labels (argmax of memberships)")
plot_membership_contours(fcm, X, ax=axes[1])
axes[1].set_title(r"contours of $\max_i \mu_{ik}$")
plt.tight_layout()

# %%
# Memberships of the three most ambiguous and the three least ambiguous
# samples. Points near a prototype have one dominant membership; points
# between clusters share their membership among them.
idx = np.argsort(fcm.membership_.max(axis=1))[[0, 1, 2, -3, -2, -1]]
for k in idx:
    print(f"x_{k:3d} = {X[k].round(2)}  memberships = {fcm.membership_[k].round(3)}")

# %%
# Effect of the fuzziness exponent :math:`m`
# -------------------------------------------
# As :math:`m\to 1` the partition becomes crisp (k-means); for large :math:`m`
# all memberships approach :math:`1/c`. In practice, :math:`m=2` is the
# standard choice, and :math:`m\in[1.5, 2.5]` is a reasonable range to explore.
ms = [1.2, 2.0, 4.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, m in zip(axes, ms, strict=False):
    model = FCM(n_clusters=3, m=m, random_state=0).fit(X)
    plot_membership_contours(model, X, ax=ax, labels=False)
    pc = np.mean(model.membership_ ** 2) * 3
    ax.set_title(f"m = {m}   (PC = {pc:.2f})")
plt.tight_layout()

# %%
# The partition coefficient :math:`PC=\frac1N\sum_{ik}\mu_{ik}^2`, printed in
# the titles, measures this: it is 1 for a crisp partition and :math:`1/c` for
# a completely fuzzy one. On well-separated data the prototypes barely move
# with :math:`m`; on overlapping clusters a large :math:`m` pulls them toward
# the global mean.

# %%
# Parameter study: a finer scan of :math:`m`
# -------------------------------------------
# The three panels above show the extremes; a scan of the exponent on the
# same data quantifies the trend. The partition coefficient and the
# classification entropy :math:`CE=-\frac1N\sum_{ik}\mu_{ik}\log\mu_{ik}`
# change monotonically with :math:`m`, from a crisp partition (:math:`PC=1`,
# :math:`CE=0`) toward the uniform one (:math:`PC=1/c`, :math:`CE=\log c`).
# The prototypes themselves hardly move (the mean displacement from the
# :math:`m=2` solution is 0.09 for :math:`m=10`, against a cluster standard
# deviation of 0.67), but the number of iterations grows, because softer
# memberships make every update a smaller step.

# [snippet:m_scan]
from fuzzyclust.validity import validity_indices

m_values = np.array([1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0])
rows = []
for m in m_values:
    model = FCM(n_clusters=3, m=m, random_state=0).fit(X)
    v = validity_indices(X, model.membership_, model.cluster_centers_, m=m, indices=["PC", "CE"])
    d = np.linalg.norm(model.cluster_centers_[:, None] - fcm.cluster_centers_[None], axis=2)
    rows.append((v["PC"], v["CE"], d.min(axis=1).mean(), model.n_iter_))
rows = np.array(rows)
# [/snippet:m_scan]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(m_values, rows[:, 0], "o-", label="partition coefficient PC")
axes[0].plot(m_values, rows[:, 1], "s-", label="classification entropy CE")
axes[0].axhline(1 / 3, color="0.5", ls=":", lw=0.8)
axes[0].axhline(np.log(3), color="0.5", ls=":", lw=0.8)
axes[0].set_xscale("log")
axes[0].set_xlabel("fuzziness exponent $m$")
axes[0].set_title("fuzziness of the partition versus $m$ (dotted: limits $1/c$ and $\\log c$)")
axes[0].legend(frameon=False)
axes[0].grid(True, color="0.9")
axes[1].plot(m_values, rows[:, 2], "o-", label="mean prototype displacement from $m=2$")
axes[1].set_xscale("log")
axes[1].set_xlabel("fuzziness exponent $m$")
axes[1].set_ylabel("displacement")
ax2 = axes[1].twinx()
ax2.plot(m_values, rows[:, 3], "s--", color="0.4", label="iterations to converge")
ax2.set_ylabel("iterations")
axes[1].set_title("prototypes barely move, convergence slows down")
h1, l1 = axes[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axes[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
axes[1].grid(True, color="0.9")
plt.tight_layout()

print("   m      PC      CE   displacement  iterations")
for m, (pc, ce, shift, it) in zip(m_values, rows, strict=True):
    print(f"{m:5.1f}  {pc:6.3f}  {ce:6.3f}  {shift:12.3f}  {int(it):10d}")

# %%
# Take-away
# ---------
# Fuzzy c-means returns a membership matrix rather than a labeling, and the
# exponent :math:`m` is the only parameter that decides how soft that matrix
# is. On well-separated clusters the prototypes are almost independent of
# :math:`m`; what changes is the information in the memberships: the
# transition zones between the clusters widen, PC falls and CE rises
# monotonically, and the iteration needs more steps. Values near 1 reproduce
# k-means, and values above 3 wash the partition out toward :math:`1/c`, so
# :math:`m=2` is the usual compromise, to be revisited when the clusters
# overlap or when outliers are present (see the example on the fuzziness
# exponent).
