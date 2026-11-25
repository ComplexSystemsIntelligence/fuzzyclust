r"""
Hard versus fuzzy partitions: k-means, k-medoids, and FCM
=========================================================

Hard clustering assigns each sample to exactly one cluster
(:math:`\mu_{ik}\in\{0,1\}`); fuzzy clustering lets the memberships vary in
:math:`[0,1]`. All three algorithms minimize the same type of objective

.. math:: J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\,\|\mathbf{x}_k-\mathbf{v}_i\|^2

and share the alternating-optimization scheme of :mod:`fuzzyclust`; they
differ only in how the prototype is defined and how the partition is updated.

* **k-means**: the prototypes are cluster means, and the partition is crisp.
* **k-medoids**: the prototypes must be data points, which is useful when the
  mean has no meaning (discrete states, categorical codes mapped to numbers).
* **FCM**: the prototypes are weighted means, and the partition is fuzzy.

On overlapping clusters the fuzzy partition carries more information: it
shows *how* ambiguous each sample is.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM, KMeans, KMedoids
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.plotting import PALETTE, plot_clusters

# three overlapping clusters: the spread is large relative to the center distance
X, y_true = make_fuzzy_blobs(n_clusters=3, n_samples_per_cluster=120, spread=1.3, box=8.0, random_state=0)

models = {
    "k-means": KMeans(n_clusters=3, random_state=0),
    "k-medoids": KMedoids(n_clusters=3, random_state=0),
    "FCM (m=2)": FCM(n_clusters=3, m=2.0, random_state=0),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, (name, model) in zip(axes, models.items(), strict=False):
    model.fit(X)
    ari = adjusted_rand_score(y_true, model.labels_)
    plot_clusters(X, model.labels_, model.cluster_centers_, ax=ax, legend=False)
    ax.set_title(f"{name}: ARI = {ari:.2f}, {model.n_iter_} iterations")
plt.tight_layout()

# %%
# The hard labels of the three methods are nearly identical. What FCM adds is
# the *degree* of membership, which we can visualize by blending the cluster
# colors according to :math:`\mu_{ik}`; ambiguous points in the overlap
# regions appear in mixed colors.
fcm = models["FCM (m=2)"]
rgb = np.array([plt.matplotlib.colors.to_rgb(PALETTE[i]) for i in range(3)])
mixed = fcm.membership_ @ rgb

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
axes[0].scatter(X[:, 0], X[:, 1], c=mixed, s=16)
axes[0].set_title("FCM: colors blended by membership")
conf = fcm.membership_.max(axis=1)
sc = axes[1].scatter(X[:, 0], X[:, 1], c=conf, cmap="viridis", s=16, vmin=1 / 3, vmax=1)
plt.colorbar(sc, ax=axes[1], label=r"$\max_i \mu_{ik}$")
axes[1].set_title("FCM: confidence of the hard assignment")
for ax in axes:
    ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# The share of samples whose largest membership is below 0.6. For these
# samples a crisp assignment would hide real ambiguity.
print(f"ambiguous samples: {np.mean(conf < 0.6):.1%}")

# %%
# The medoids are rows of the data matrix, unlike the k-means centers:
km = models["k-medoids"]
print("medoid indices:", km.medoid_indices_)
print("medoids are data points:", np.all(np.isin(km.cluster_centers_, X)))

# %%
# Parameter study: the exponent :math:`m` interpolates between hard and fuzzy
# ----------------------------------------------------------------------------
# k-means is the limit :math:`m\to1` of FCM, so the exponent decides how much
# of the ambiguity the partition reports. For every :math:`m` we record the
# agreement of the FCM labels with the k-means labels and with the generating
# labels, the partition coefficient, and the share of ambiguous samples
# (largest membership below 0.6).

# [snippet:m_scan]
m_values = (1.2, 1.5, 2.0, 2.5, 3.0, 4.0)
km_labels = models["k-means"].labels_
rows = []
for m in m_values:
    model = FCM(n_clusters=3, m=m, random_state=0).fit(X)
    rows.append((adjusted_rand_score(km_labels, model.labels_), adjusted_rand_score(y_true, model.labels_),
                 np.mean(model.membership_ ** 2) * 3, np.mean(model.membership_.max(axis=1) < 0.6)))
rows = np.array(rows)
# [/snippet:m_scan]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(m_values, rows[:, 0], "o-", color=PALETTE[0], label="ARI(FCM, k-means)")
ax.plot(m_values, rows[:, 1], "s-", color=PALETTE[1], label="ARI(FCM, generating labels)")
ax.plot(m_values, rows[:, 2], "^-", color=PALETTE[2], label="partition coefficient")
ax.plot(m_values, rows[:, 3], "d-", color=PALETTE[3], label=r"share with $\max_i\mu_{ik}<0.6$")
ax.set_xlabel("fuzziness exponent $m$")
ax.set_ylim(-0.02, 1.05)
ax.set_title("the hard labels stay put while the reported ambiguity grows with $m$")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

print("  m   ARI vs k-means  ARI vs truth    PC   ambiguous")
for m, (a_km, a_true, pc, amb) in zip(m_values, rows, strict=True):
    print(f"{m:4.1f}  {a_km:14.3f}  {a_true:12.3f}  {pc:5.3f}  {amb:8.1%}")

# %%
# Take-away
# ---------
# The three algorithms minimize the same functional and, on these data, end
# with practically the same hard labels, so a hard partition cannot
# distinguish them. What the fuzzy algorithm adds is the membership degree:
# samples in the overlap are flagged as ambiguous instead of being assigned
# with false confidence. The exponent :math:`m` controls how much of that
# ambiguity is reported. Near 1 the memberships are as crisp as k-means, at
# :math:`m=2` a few percent of the samples are ambiguous, and above 3 the
# ambiguous share grows quickly, while the hard labels, and hence the
# agreement with the generating labels, hardly change. Choose k-medoids when
# a prototype must be an actual observation.
