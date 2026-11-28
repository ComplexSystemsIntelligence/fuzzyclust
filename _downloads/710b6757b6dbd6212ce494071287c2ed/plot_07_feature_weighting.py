r"""
Learning which features matter: SCAD and the weighted FCM
=========================================================

Fuzzy c-means treats every feature alike: after standardization a feature
of pure noise contributes as much to the distance as a feature that
separates the clusters, and enough such features drown the structure. Two
extensions learn feature weights together with the partition. The
*weighted FCM* of Keller and Klawonn uses one weight vector
:math:`\boldsymbol\alpha` for all clusters,
:math:`D^2_{ik}=\sum_j\alpha_j^t(x_{kj}-v_{ij})^2`, and *SCAD*
(simultaneous clustering and attribute discrimination) of Frigui and
Nasraoui gives every cluster its own weights :math:`w_{ij}`. In both
methods a feature receives a large weight when the members of the clusters
scatter little along it: a noise feature, along which every cluster is as
wide as the data, is down-weighted. This example adds noise features to
three groups, compares the learned weights, and shows SCAD on clusters that
live in different subspaces.
"""

# %%
# Two informative features and six noise features
# -------------------------------------------------
# Three groups in the first two features; six further features are
# independent Gaussian noise. All features are standardized, so the noise
# features have the same variance as the informative ones. In eight
# dimensions the FCM memberships with :math:`m=2` become nearly uniform
# and the prototypes collapse toward the grand mean, so :math:`m=1.3` is
# used for all methods.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from fuzzyclust import FCM
from fuzzyclust.extensions import SCAD, FeatureWeightedFCM
from fuzzyclust.plotting import PALETTE, plot_clusters

rng = np.random.RandomState(0)
group_centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.2]])
X_info = np.vstack([c + 0.9 * rng.randn(100, 2) for c in group_centers])
y = np.repeat(np.arange(3), 100)
noise = rng.randn(300, 14)
feature_names = ["x1", "x2"] + [f"noise {j + 1}" for j in range(6)]
X = StandardScaler().fit_transform(np.column_stack([X_info, noise[:, :6]]))

# [snippet:fit]
common = dict(n_clusters=3, m=1.3, init="kmeans++", n_init=5, random_state=0)
fcm = FCM(**common).fit(X)
wfcm = FeatureWeightedFCM(t=3.0, **common).fit(X)  # one weight vector for all clusters
scad = SCAD(variant="scad1", **common).fit(X)  # one weight vector per cluster
for name, model in (("FCM", fcm), ("weighted FCM", wfcm), ("SCAD-1", scad)):
    print(f"{name:13s} ARI = {adjusted_rand_score(y, model.labels_):.2f}")
print("global weights:", np.round(wfcm.feature_weights_, 2))
# [/snippet:fit]
print("SCAD weights per cluster:")
print(np.round(scad.feature_weights_, 2))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
plot_clusters(X[:, :2], wfcm.labels_, centers=wfcm.cluster_centers_[:, :2], ax=axes[0], legend=False)
axes[0].set_title(f"weighted FCM, first two features (ARI = {adjusted_rand_score(y, wfcm.labels_):.2f})")
axes[0].title.set_fontsize(10)
pos = np.arange(len(feature_names))
axes[1].bar(pos, wfcm.feature_weights_, color=[PALETTE[0]] * 2 + ["0.6"] * 6)
axes[1].axhline(1 / len(feature_names), color="0.4", ls="--", lw=1, label="uniform weight $1/n$")
axes[1].set_title(r"weighted FCM: global weights $\alpha_j$ ($t=3$)")
axes[1].legend(frameon=False)
width = 0.27
for i in range(3):
    axes[2].bar(pos + (i - 1) * width, scad.feature_weights_[i], width=width, color=PALETTE[i], label=f"cluster {i}")
axes[2].set_title(r"SCAD-1: weights $w_{ij}$ of every cluster")
axes[2].legend(frameon=False)
for ax in axes[1:]:
    ax.set_xticks(pos, feature_names, rotation=45, ha="right")
    ax.set_ylabel("weight")
plt.tight_layout()

# %%
# Both methods find the groups and put most of the weight on the two
# informative features. The global weights of the weighted FCM are about
# three times larger for :math:`x_1` and :math:`x_2` than for the noise
# features. The SCAD weights tell the same story per cluster, more mildly
# (about 0.17 against 0.11): the automatic regularization coefficient
# :math:`\delta_i` balances the weighted scatter against the quadratic
# penalty and keeps SCAD-1 weights close to uniform unless the scatters
# differ strongly. When they do, SCAD-1 can set a weight to exactly zero,
# because its update is clipped at zero and renormalized.

# %%
# Clusters in different subspaces
# --------------------------------
# Every group is tight along one feature and wide along the other: the
# first is a vertical band, the second a horizontal band, the third a
# compact blob. A single global weight vector cannot express this; the
# cluster-specific weights of SCAD can. We use SCAD-2, whose weights
# :math:`w_{ij}\propto(\text{scatter}_{ij})^{-1/(q-1)}` are never exactly zero.
n = 150
bands = np.vstack([
    np.column_stack([0.3 * rng.randn(n), 2.0 * rng.randn(n)]),
    np.column_stack([2.0 * rng.randn(n), 6.0 + 0.3 * rng.randn(n)]),
    np.column_stack([6.0 + 0.3 * rng.randn(n), 0.3 * rng.randn(n)]),
])
y_bands = np.repeat(np.arange(3), n)
X_bands = StandardScaler().fit_transform(bands)

# [snippet:subspace]
scad2 = SCAD(n_clusters=3, variant="scad2", q=2.0, n_init=5, random_state=0).fit(X_bands)
print("SCAD-2 weights (rows: clusters, columns: x1, x2):")
print(np.round(scad2.feature_weights_, 2))
# [/snippet:subspace]
fcm_bands = FCM(n_clusters=3, n_init=5, random_state=0).fit(X_bands)
print(f"ARI: FCM {adjusted_rand_score(y_bands, fcm_bands.labels_):.2f}, "
      f"SCAD-2 {adjusted_rand_score(y_bands, scad2.labels_):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
plot_clusters(X_bands, scad2.labels_, centers=scad2.cluster_centers_, ax=axes[0], legend=False)
for i, (v, w) in enumerate(zip(scad2.cluster_centers_, scad2.feature_weights_, strict=True)):
    # half-axes proportional to 1/sqrt(weight): the unit circle of the weighted distance
    a, b = 0.35 / np.sqrt(w)
    t = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(v[0] + a * np.cos(t), v[1] + b * np.sin(t), color=PALETTE[i], lw=1.5)
axes[0].set_title("SCAD-2: level sets of the weighted distance")
for i in range(3):
    axes[1].bar(np.arange(2) + (i - 1) * 0.27, scad2.feature_weights_[i], width=0.27, color=PALETTE[i],
                label=f"cluster {i}")
axes[1].set_xticks([0, 1], ["x1", "x2"])
axes[1].set_ylabel("weight")
axes[1].set_title("feature weights of every cluster")
axes[1].legend(frameon=False)
plt.tight_layout()

# %%
# Each band puts nearly all of its weight on the feature along which it is
# tight; the blob splits its weight evenly. The partition itself is no
# better than that of FCM (both ARI 0.90, the ends of the bands are
# ambiguous); what SCAD adds is the description of every cluster by its
# relevant features. The level sets of the weighted distances are ellipses
# aligned with the coordinate axes and stretched along the feature a
# cluster ignores: an axis-parallel relative of the Gustafson--Kessel norm,
# obtained without inverting covariance matrices.

# %%
# Parameter study: number of noise features and the weight exponent
# ------------------------------------------------------------------
# We add between zero and fourteen noise features and record the agreement
# with the true groups. In a second study the number of noise features is
# fixed at four and the exponent :math:`t` of the weighted FCM is varied:
# the weights are proportional to :math:`(\text{scatter}_j)^{-1/(t-1)}`, so
# a :math:`t` close to one concentrates the weight on the single feature of
# smallest scatter.
n_noise = np.arange(0, 15, 2)
scores = {"FCM": [], "weighted FCM (t = 3)": [], "SCAD-1": []}
for nn in n_noise:
    Xn = StandardScaler().fit_transform(np.column_stack([X_info, noise[:, :nn]]))
    for name, est in (("FCM", FCM(**common)), ("weighted FCM (t = 3)", FeatureWeightedFCM(t=3.0, **common)),
                      ("SCAD-1", SCAD(**common))):
        scores[name].append(adjusted_rand_score(y, est.fit(Xn).labels_))

X4 = StandardScaler().fit_transform(np.column_stack([X_info, noise[:, :4]]))
ts = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0])
t_ari, t_weight = [], []
for t in ts:
    model = FeatureWeightedFCM(t=t, **common).fit(X4)
    t_ari.append(adjusted_rand_score(y, model.labels_))
    t_weight.append(model.feature_weights_[:2].sum())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
for j, (name, s) in enumerate(scores.items()):
    axes[0].plot(n_noise, s, marker="o", color=PALETTE[j], lw=1.8, label=name)
axes[0].set_xlabel("number of noise features")
axes[0].set_ylabel("adjusted Rand index")
axes[0].set_title("agreement with the true groups")
axes[0].legend(frameon=False)
axes[1].plot(ts, t_ari, marker="o", color=PALETTE[0], lw=1.8, label="adjusted Rand index")
axes[1].plot(ts, t_weight, marker="s", color=PALETTE[1], lw=1.8, label=r"weight of $x_1, x_2$ together")
axes[1].set_xlabel("weight exponent $t$")
axes[1].set_title("weighted FCM with four noise features")
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(True, color="0.92")
plt.tight_layout()
for k, nn in enumerate(n_noise):
    print(f"{nn:2d} noise features: " + "  ".join(f"{name} {s[k]:.2f}" for name, s in scores.items()))
for t, a, w in zip(ts, t_ari, t_weight, strict=True):
    print(f"t = {t:5.1f}: ARI = {a:.2f}, weight of the informative features = {w:.2f}")

# %%
# The learned weights extend the range in which the groups are found: FCM
# loses accuracy steadily (ARI 0.92 with six, 0.81 with eight noise
# features), while the weighted FCM and SCAD stay at or near one up to
# eight. Beyond that all methods fail, because the weights are learned
# from the partition, and the partition of the first iterations is already
# dominated by noise. The exponent study shows the trap of a small
# :math:`t`: the weight concentrates on the feature along which the data
# can be sliced into the thinnest pieces, here a single informative
# feature, and the partition follows that feature alone.

# %%
# **Take-away.** Feature weighting makes the norm of FCM a learned,
# diagonal quantity: globally for all clusters (weighted FCM) or per
# cluster (SCAD). The weights improve the partition when noise features
# are present, and they report which features define which cluster, which
# is often the more valuable output. Standardize the features first,
# since a feature of small variance attracts the weight regardless of its
# relevance, and keep the weight exponent away from one (:math:`t\approx3`)
# to avoid concentrating on a single feature. Weighting cannot rescue a
# partition that the noise has already destroyed at the start: with many
# irrelevant features, select or project them first.
