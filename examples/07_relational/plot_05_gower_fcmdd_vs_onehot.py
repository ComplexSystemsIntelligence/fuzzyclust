r"""
Gower's dissimilarity versus one-hot encoding for mixed data
============================================================

There are two routes from mixed records to a fuzzy partition. The common
shortcut encodes the categories as 0/1 indicator columns and runs fuzzy
c-means on the result. The geometry is then arbitrary: a mismatch in a
categorical attribute costs :math:`\sqrt2` whatever the attribute, the
numeric columns count on their own scale, and the "centers" are
fractional indicator vectors that correspond to no record. The relational
route computes Gower's general coefficient (Gower, 1971): every attribute
contributes a score in :math:`[0,1]`, the range-scaled difference for a
numeric one and a mismatch indicator for a categorical one, and the scores
are averaged over the attributes observed in *both* records, so missing
values need no special treatment. The resulting matrix feeds fuzzy
c-medoids, whose prototypes are real records, or (since Gower showed that
:math:`\sqrt{d_G}` is Euclidean when nothing is missing) relational
c-means.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.plotting import PALETTE, plot_clusters
from fuzzyclust.relational import (
    NERFCM,
    FuzzyCMedoids,
    FuzzyKPrototypes,
    RelationalFCM,
    gower_distance_matrix,
    make_mixed_blobs,
    plot_dissimilarity_matrix,
)

mixed = make_mixed_blobs(n_samples=300, n_numeric=2, n_categorical=4, n_categories=3, n_clusters=3,
                         cluster_std=1.0, center_distance=3.0, noise=0.3, random_state=0)
X, y = mixed.data, mixed.target
num_cols, cat_cols = ["x1", "x2"], ["c1", "c2", "c3", "c4"]

# %%
# Two dissimilarity views of the same records
# -------------------------------------------
# Gower's coefficient (left) and the Euclidean distance of the one-hot
# encoded records (right), both ordered by the true group. In the Gower
# matrix every attribute counts equally; in the one-hot matrix the two
# numeric columns with ranges of about six units dominate the twelve
# indicator columns, whose contribution is at most two mismatches per
# attribute.

# [snippet:gower]
G = gower_distance_matrix(X)                          # categorical columns detected from the dtypes
onehot = pd.get_dummies(X, columns=cat_cols, dtype=float)
print(f"Gower matrix {G.shape}, entries in [{G.min():.2f}, {G.max():.2f}]; one-hot data {onehot.shape}")
# [/snippet:gower]

E = np.sqrt(((onehot.to_numpy()[:, None, :] - onehot.to_numpy()[None, :, :]) ** 2).sum(axis=2))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
plot_dissimilarity_matrix(G, y, ax=axes[0])
axes[0].set_title("Gower dissimilarity, ordered by true group")
plot_dissimilarity_matrix(E, y, ax=axes[1])
axes[1].set_title("Euclidean distance of the one-hot records")
plt.tight_layout()

# %%
# Clustering through both routes
# ------------------------------
# We compare fuzzy c-medoids and relational c-means on the Gower matrix
# (taken as squared distances), FCM on the one-hot data and, for
# reference, fuzzy k-prototypes with its default weight.

# [snippet:fit]
fcmdd = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(G)
rfcm = RelationalFCM(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(G)
fcm = FCM(n_clusters=3, n_init=5, random_state=0).fit(onehot)
kproto = FuzzyKPrototypes(n_clusters=3, n_init=5, random_state=0).fit(X)
for name, model in (("Gower + FCMdd", fcmdd), ("Gower + RFCM", rfcm), ("one-hot + FCM", fcm),
                    ("k-prototypes", kproto)):
    print(f"{name:14s}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean max membership = {model.membership_.max(axis=1).mean():.2f}")
print("medoid records of Gower + FCMdd:")
print(X.iloc[fcmdd.medoid_indices_])
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True, sharey=True)
Xn = X[num_cols].to_numpy()
plot_clusters(Xn, fcmdd.labels_, Xn[fcmdd.medoid_indices_], ax=axes[0])
axes[0].set_title(f"Gower + fuzzy c-medoids: ARI = {adjusted_rand_score(y, fcmdd.labels_):.2f}")
plot_clusters(Xn, fcm.labels_, fcm.cluster_centers_[:, :2], ax=axes[1])
axes[1].set_title(f"One-hot + FCM: ARI = {adjusted_rand_score(y, fcm.labels_):.2f}")
plot_clusters(Xn, kproto.labels_, kproto.numeric_centers_, ax=axes[2])
axes[2].set_title(f"Fuzzy k-prototypes: ARI = {adjusted_rand_score(y, kproto.labels_):.2f}")
for ax in axes:
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
plt.tight_layout()

# %%
# Parameter study I: the weight of the numeric attributes
# -------------------------------------------------------
# Gower's coefficient takes attribute weights :math:`w_j`. Scaling the
# weight of the two numeric columns from 0 (categorical only) to a large
# value (numeric only) traces the same path as :math:`\gamma` in
# k-prototypes, in reverse: equal weights already do well, a moderately
# larger weight of the numeric columns (which are only two of six) does a
# little better, and either extreme loses the information of the other half
# of the record. NERF never needs a spread: :math:`\beta=0` throughout,
# since a complete Gower matrix is Euclidean.

# [snippet:weight_scan]
weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0)
rows = []
for w in weights:
    Gw = gower_distance_matrix(X, feature_weights=[w, w, 1.0, 1.0, 1.0, 1.0])
    model = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(Gw)
    nerf = NERFCM(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(Gw)
    rows.append((adjusted_rand_score(y, model.labels_), adjusted_rand_score(y, nerf.labels_), nerf.beta_))
    print(f"numeric weight = {w:6.2f}: FCMdd ARI = {rows[-1][0]:.2f}, NERF ARI = {rows[-1][1]:.2f} "
          f"(beta = {rows[-1][2]:.3f})")
# [/snippet:weight_scan]
rows = np.asarray(rows)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
xw = np.arange(len(weights))
ax.plot(xw, rows[:, 0], "o-", color=PALETTE[0], label="Gower + fuzzy c-medoids")
ax.plot(xw, rows[:, 1], "s--", color=PALETTE[1], label="Gower + NERF c-means")
ax.set_xticks(xw)
ax.set_xticklabels([f"{w:g}" for w in weights])
ax.set_xlabel("weight of each numeric attribute (categorical weights = 1)")
ax.set_ylabel("ARI")
ax.set_ylim(0, 1.05)
ax.set_title("Attribute weights in Gower's coefficient")
ax.legend(frameon=False, fontsize=8, loc="lower center")
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study II: missing values
# ----------------------------------
# A share of the entries is deleted at random. Gower's coefficient
# averages over the attributes observed in both records, so the matrix is
# defined without any imputation, and NERF c-means on it degrades
# gracefully. Fuzzy c-medoids does not. A record with several missing
# entries is compared only on the few attributes it has, so it looks close
# to many records (often at distance zero) and is chosen as a medoid; the
# printed counts show medoids with up to four of six entries missing, and
# the partition collapses from 10 % missing entries on. The one-hot route
# has to drop the incomplete records (here before FCM), which keeps its
# accuracy on the remaining ones but quickly loses most of the sample.
# Restricting the medoid candidates to complete records, or down-weighting
# pairs compared on few attributes, are the obvious remedies.
rng = np.random.RandomState(0)
print("missing share: FCMdd ARI (missing entries of the medoids), NERF ARI | one-hot FCM ARI (complete rows)")
for share in (0.0, 0.05, 0.1, 0.2, 0.3):
    mask = rng.rand(*X.shape) < share
    Xm = X.mask(mask)
    Gm = gower_distance_matrix(Xm)
    model = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(Gm)
    nerf = NERFCM(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(Gm)
    complete = ~mask.any(axis=1)
    oh = pd.get_dummies(Xm[complete], columns=cat_cols, dtype=float)
    f = FCM(n_clusters=3, n_init=5, random_state=0).fit(oh)
    print(f"  {share:.2f}: {adjusted_rand_score(y, model.labels_):.2f} "
          f"({mask[model.medoid_indices_].sum(axis=1).tolist()}), {adjusted_rand_score(y, nerf.labels_):.2f} | "
          f"{adjusted_rand_score(y[complete], f.labels_):.2f} ({complete.sum()} of {len(X)} records)")

# %%
# **Take-away.** Gower's coefficient turns mixed, even incomplete,
# records into a dissimilarity matrix in which every attribute has the
# same say, and the relational algorithms take it from there: fuzzy
# c-medoids returns real records as prototypes, NERF c-means a fuzzy
# partition with relational prototypes (with :math:`\beta=0`, because a
# complete Gower matrix is Euclidean). One-hot encoding plus FCM is not
# wrong, but its geometry is an accident of the encoding and the scales,
# and it can only discard incomplete records. With missing values the
# relational route keeps every record, but the prototype has to be chosen
# with care: NERF stays reliable, while the medoid search is attracted by
# incomplete records that look close to everyone. The attribute weights
# play the role of :math:`\gamma` in k-prototypes; equal weights are a
# sound default.
