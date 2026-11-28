r"""
Fuzzy k-prototypes on mixed numeric and categorical data
========================================================

Records that mix measurements and categories, such as age and income next
to occupation and region, are the rule in applications. Huang's
k-prototypes measures the numeric part by the squared Euclidean distance,
the categorical part by the number of mismatches, and adds the two with a
weight :math:`\gamma`: the prototype is the fuzzy mean of the numeric
attributes together with the mode of every categorical one. With
:math:`\gamma=0` the method is fuzzy c-means on the numeric columns, with a
sufficiently large :math:`\gamma` it is fuzzy k-modes on the categorical
columns; in between, both kinds of information shape the partition. Huang
suggests the average standard deviation of the numeric attributes as the
scale of :math:`\gamma`, and the implementation uses it as the default.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.plotting import PALETTE, plot_clusters
from fuzzyclust.relational import FuzzyKModes, FuzzyKPrototypes, make_mixed_blobs, plot_modes

mixed = make_mixed_blobs(n_samples=300, n_numeric=2, n_categorical=3, n_categories=3, n_clusters=3,
                         cluster_std=1.0, center_distance=3.0, noise=0.25, random_state=0)
X, y = mixed.data, mixed.target
print(X.head())
print("numeric centers:\n", mixed.centers, "\nplanted modes:\n", mixed.modes)

# %%
# Three groups described by two measurements and three categories
# ---------------------------------------------------------------
# The numeric centers are three units apart with unit spread, so the
# groups overlap in the plane; each group also has a typical answer
# pattern in the three categorical columns, and every categorical entry is
# replaced by a random category with probability 0.25. Neither part alone
# separates the groups cleanly.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
plot_clusters(X[["x1", "x2"]].to_numpy(), y, mixed.centers, ax=axes[0])
axes[0].set_title("Numeric part colored by the true group")
tab = X.groupby(y)[["c1", "c2", "c3"]].agg(lambda s: s.value_counts().iloc[0] / len(s))
im = axes[1].imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{mixed.modes.iloc[i, j]}\n{tab.iloc[i, j]:.0%}", ha="center", va="center",
                     color="white" if tab.iloc[i, j] > 0.6 else "black", fontsize=9)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["c1", "c2", "c3"])
axes[1].set_yticks(range(3))
axes[1].set_yticklabels([f"group {i + 1}" for i in range(3)])
axes[1].set_title("Categorical part: planted mode and its frequency")
plt.tight_layout()

# %%
# Fuzzy k-prototypes with the default weight
# ------------------------------------------
# Categorical columns are detected from the dtypes of the DataFrame; the
# default :math:`\gamma` is the mean standard deviation of the numeric
# columns. The prototypes come back as a DataFrame with means and modes.

# [snippet:fit]
kp = FuzzyKPrototypes(n_clusters=3, n_init=5, random_state=0).fit(X)
print(f"gamma = {kp.gamma_:.2f}, categorical columns: {list(X.columns[kp.categorical_features_])}")
print(kp.prototypes_.round(2))
print(f"ARI = {adjusted_rand_score(y, kp.labels_):.2f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
plot_clusters(X[["x1", "x2"]].to_numpy(), kp.labels_, kp.numeric_centers_, ax=axes[0])
axes[0].set_title(f"Fuzzy k-prototypes, $\\gamma={kp.gamma_:.2f}$: ARI = {adjusted_rand_score(y, kp.labels_):.2f}")
plot_modes(kp, X, ax=axes[1])
axes[1].set_title("Modes of the categorical part")
plt.tight_layout()

# %%
# Parameter study I: the weight gamma
# -----------------------------------
# :math:`\gamma` is scanned from zero (numeric only) to a value at which
# the three mismatches of the categorical part outweigh any numeric
# distance (categorical only). Both extremes are worse than the
# combination: the numeric part alone suffers from the overlap of the
# groups, the categorical part alone from the random answers, and a
# :math:`\gamma` of the order of the numeric spread lets each part
# correct the errors of the other. The curves also show the two limits:
# at :math:`\gamma=0` the partition equals that of FCM on the numeric
# columns, for large :math:`\gamma` that of fuzzy k-modes on the
# categorical columns.

# [snippet:gamma_scan]
fcm = FCM(n_clusters=3, n_init=5, random_state=0).fit(X[["x1", "x2"]])
kmodes = FuzzyKModes(n_clusters=3, m=2.0, n_init=5, random_state=0).fit(X[["c1", "c2", "c3"]])
gammas = (0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0)
rows = []
for g in gammas:
    model = FuzzyKPrototypes(n_clusters=3, gamma=g, n_init=5, random_state=0).fit(X)
    rows.append((adjusted_rand_score(y, model.labels_), adjusted_rand_score(fcm.labels_, model.labels_),
                 adjusted_rand_score(kmodes.labels_, model.labels_)))
    print(f"gamma = {g:6.1f}: ARI = {rows[-1][0]:.2f}, agreement with FCM(numeric) = {rows[-1][1]:.2f}, "
          f"with k-modes(categorical) = {rows[-1][2]:.2f}")
# [/snippet:gamma_scan]
rows = np.asarray(rows)
fig, ax = plt.subplots(figsize=(7, 3.8))
xg = np.arange(len(gammas))
ax.plot(xg, rows[:, 0], "o-", color=PALETTE[0], label="ARI with the true groups")
ax.plot(xg, rows[:, 1], "s--", color=PALETTE[1], label="agreement with FCM (numeric only)")
ax.plot(xg, rows[:, 2], "^:", color=PALETTE[2], label="agreement with k-modes (categorical only)")
ax.axvline(np.interp(kp.gamma_, gammas, xg), color="0.5", ls=":", label=r"default $\gamma$")
ax.set_xticks(xg)
ax.set_xticklabels([f"{g:g}" for g in gammas])
ax.set_xlabel(r"$\gamma$")
ax.set_ylim(0, 1.05)
ax.set_title(r"From fuzzy c-means ($\gamma=0$) to fuzzy k-modes ($\gamma\to\infty$)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study II: fuzziness, restarts and new records
# -------------------------------------------------------
# The exponent :math:`m` softens the partition as usual. Restarts guard
# against the local optima of random prototype starts; on these data all
# starts reach the same objective, and the restarts cost little.
# ``predict_proba`` assigns new records, including records with a category
# never seen in training, which counts as one mismatch with every mode.
for m in (1.5, 2.0, 3.0):
    model = FuzzyKPrototypes(n_clusters=3, m=m, n_init=5, random_state=0).fit(X)
    print(f"m = {m:.1f}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean max membership = {model.membership_.max(axis=1).mean():.2f}")
single = [FuzzyKPrototypes(n_clusters=3, random_state=s).fit(X).objective_ for s in range(5)]
print(f"objective of five single runs: {np.round(single, 1)}; best of five restarts: {kp.objective_:.1f}")
new = X.iloc[:3].copy()
new.iloc[0, 2] = "unseen"
print(kp.predict_proba(new).round(3))

# %%
# **Take-away.** Fuzzy k-prototypes clusters mixed records without
# forcing the categories into a numeric geometry: means for measurements,
# modes for categories, memberships from the combined dissimilarity. The
# weight :math:`\gamma` decides how much a mismatch counts against a unit
# of squared numeric distance and interpolates between fuzzy c-means on
# the numeric part and fuzzy k-modes on the categorical part. A value of
# the order of the numeric spread (Huang's rule and the default) lets the
# two kinds of attributes correct each other and beats either alone.
