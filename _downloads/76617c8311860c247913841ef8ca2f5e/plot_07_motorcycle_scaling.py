r"""
Feature scaling on real data: the motorcycle data set
=====================================================

Distance-based clustering is meaningful only when the features are
commensurable. The motorcycle data set (time after impact in ms versus head
acceleration in g) is a classical case: the two variables have different units
and ranges, so the raw Euclidean distance is dominated by the variable with the
larger spread. The MATLAB toolbox offered ``clust_normalize`` for this purpose;
:mod:`fuzzyclust` delegates scaling to scikit-learn so that it composes with
``Pipeline``, and keeps :func:`fuzzyclust.preprocessing.normalize` as a
convenience function that also returns the scaler for the inverse
transformation.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fuzzyclust import GK
from fuzzyclust.datasets import load_motorcycle
from fuzzyclust.plotting import plot_clusters, plot_membership_contours
from fuzzyclust.preprocessing import normalize

mc = load_motorcycle()
X = mc.data
print(mc.feature_names, X.shape)
print("ranges:", np.ptp(X, axis=0).round(1))

# %%
# Without scaling, the time axis (range ≈ 55 ms) and the acceleration axis
# (range ≈ 210 g) receive unequal weights. We cluster the raw, the
# range-scaled, and the variance-scaled data with GK and map the prototypes
# back to the original units with the scaler's ``inverse_transform``.
variants = {
    "raw": None,
    "range [0, 1]": MinMaxScaler(),
    "unit variance": StandardScaler(),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, (name, scaler) in zip(axes, variants.items(), strict=False):
    Xs = X if scaler is None else scaler.fit_transform(X)
    gk = GK(n_clusters=4, random_state=0).fit(Xs)
    centers = gk.cluster_centers_ if scaler is None else scaler.inverse_transform(gk.cluster_centers_)
    plot_clusters(X, gk.labels_, centers, ax=ax, legend=False)
    ax.set_title(name)
    ax.set_xlabel("time [ms]")
axes[0].set_ylabel("acceleration [g]")
plt.tight_layout()

# %%
# The three partitions are identical, and so are the prototypes in physical
# units: the clusters follow the course of the signal in time, which is the
# physically meaningful segmentation. Gustafson--Kessel adapts its norm to
# the covariance of every cluster, so in exact arithmetic it is invariant to
# a rescaling of the features, and on these data the units do not change the
# optimum it reaches. A Euclidean method (k-means, FCM) would instead be
# dominated by the acceleration axis; for GK, scaling still matters when the
# covariance regularization is active or when the initialization leads to
# different local optima.
#
# A scikit-learn pipeline gives the same result and is the recommended
# pattern, because the scaler is then applied to new data automatically:
pipe = make_pipeline(StandardScaler(), GK(n_clusters=4, random_state=0)).fit(X)
new_points = np.array([[10.0, 0.0], [30.0, -100.0], [45.0, 20.0]])
print("memberships of new points:\n", pipe.predict_proba(new_points).round(2))

# %%
# The membership contours are drawn in the scaled space, where the GK
# ellipses are easy to interpret; ``normalize`` returns the fitted scaler, so
# the figure can be labeled in original units if needed.
Xs, scaler = normalize(X, "var")
gk = pipe[-1]
fig, ax = plt.subplots(figsize=(6, 4.6))
plot_membership_contours(gk, Xs, ax=ax, labels=False)
ax.set_xlabel("time (standardized)")
ax.set_ylabel("acceleration (standardized)")
ax.set_title("GK on the standardized motorcycle data")
plt.tight_layout()

# %%
# Parameter study: the number of clusters on the standardized data
# -----------------------------------------------------------------
# The motorcycle signal is one continuous curve, not a set of compact
# groups, so the number of clusters is the number of *pieces* the curve is
# cut into: a resolution, not a property of the data. The scan below fits
# GK on the standardized data for :math:`c=2,\dots,8` and reports the
# validity indices; the four panels show the partitions in the original
# units.

# [snippet:c_scan]
from fuzzyclust.validity import scan_n_clusters, suggest_n_clusters

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
table = scan_n_clusters(GK(random_state=0, n_init=3), Xs, range(2, 9), indices=["XB", "SIL", "SC", "FHV"])
# [/snippet:c_scan]
print(table.round(3))
print("suggested number of clusters:", suggest_n_clusters(table).to_dict())

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)
for ax, c in zip(axes, (2, 3, 4, 6), strict=True):
    model = GK(n_clusters=c, random_state=0, n_init=3).fit(Xs)
    plot_clusters(X, model.labels_, scaler.inverse_transform(model.cluster_centers_), ax=ax, legend=False)
    ax.set_title(f"GK, c = {c}:  XB = {table.loc[c, 'XB']:.2f},  SIL = {table.loc[c, 'SIL']:.2f}")
    ax.set_xlabel("time [ms]")
axes[0].set_ylabel("acceleration [g]")
plt.tight_layout()

# %%
# Take-away
# ---------
# Distances are meaningful only between commensurable features, so scaling
# is part of the model and belongs in the pipeline, where it is applied to
# new data automatically. On the motorcycle data the raw, range-scaled, and
# variance-scaled data give the same GK partition, a segmentation along the
# time axis, because the adaptive norm of GK absorbs the units; a Euclidean
# method would not. The number of clusters is a
# different kind of decision. The indices disagree (XB prefers three pieces,
# the silhouette two, and SC and the fuzzy hypervolume as many as allowed)
# because a curve has no "true" number of compact groups. Each extra cluster
# gives a finer local linear approximation of the signal. This is the
# viewpoint of the Takagi--Sugeno models in the modeling section, where
# :math:`c` is chosen by the prediction error rather than by a cluster
# validity index.
