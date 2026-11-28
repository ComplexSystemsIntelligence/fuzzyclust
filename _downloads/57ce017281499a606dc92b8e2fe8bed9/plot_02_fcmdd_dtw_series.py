r"""
Fuzzy c-medoids on dynamic-time-warping dissimilarities of time series
======================================================================

Time series of the same shape rarely line up in time: a peak comes a little
earlier or later, a ramp is stretched or compressed. The Euclidean distance
between two such series compares values at equal time indices and is large
even when the shapes agree, whereas dynamic time warping (Sakoe and Chiba)
aligns the series by a monotone path through their cost matrix, within a
band of width :math:`w`, and compares what belongs together. DTW is a
dissimilarity, not a metric, and its square is not Euclidean, so relational
c-means would need the :math:`\beta`-spread; fuzzy c-medoids (Krishnapuram,
Joshi, Nasraoui and Yi) needs nothing of the kind. Its prototypes are
objects (the medoid of a cluster is the series that minimizes the
membership-weighted sum of dissimilarities to the others), so it works
with any dissimilarity and returns representative series that can be
plotted. The search for a medoid among all :math:`N` objects costs
:math:`O(N^2)` per cluster and iteration; the *linear* variant of the paper
searches only among the ``n_candidates`` objects with the highest
memberships.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.plotting import PALETTE
from fuzzyclust.relational import (
    FuzzyCMedoids,
    dtw_distance_matrix,
    make_warped_series,
    pairwise_dissimilarity,
    plot_dissimilarity_matrix,
    plot_dtw_alignment,
    plot_series_clusters,
)

X, y = make_warped_series(n_series_per_class=25, length=60, warp=0.25, noise=0.1, random_state=0)
N, T = X.shape

# %%
# Three shape classes with random time warps
# ------------------------------------------
# A bell, a step and a double bump, each shifted and warped in time.
# Within a class the peaks and edges fall on different time indices, so
# two series of one class can be farther apart in the Euclidean sense
# than two series of different classes.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
names = ("bell", "step", "double bump")
for i, ax in enumerate(axes):
    for k in np.flatnonzero(y == i)[:12]:
        ax.plot(X[k], color=PALETTE[i], alpha=0.5, lw=0.9)
    ax.set_title(f"class {i + 1}: {names[i]} (12 of 25 series)")
    ax.set_xlabel("time index")
plt.tight_layout()

# %%
# Euclidean versus DTW dissimilarity
# ----------------------------------
# Both matrices are shown with the series ordered by their true class.
# The DTW matrix (band :math:`w=12` samples) has clear dark diagonal
# blocks; in the Euclidean matrix the blocks are blurred, because the
# shifted peaks of one class produce large distances inside the class.
# The alignment plot shows how DTW stretches one bell to match the other.

# [snippet:dtw]
R_euclid = pairwise_dissimilarity(X)                 # compare values at equal times
t0 = time.perf_counter()
R_dtw = dtw_distance_matrix(X, window=12)            # Sakoe-Chiba band of 12 samples
print(f"DTW matrix of {N} series of length {T}: {time.perf_counter() - t0:.2f} s")
# [/snippet:dtw]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
plot_dissimilarity_matrix(R_euclid, y, ax=axes[0])
axes[0].set_title("Euclidean distances, ordered by true class")
plot_dissimilarity_matrix(R_dtw, y, ax=axes[1])
axes[1].set_title("DTW dissimilarities ($w=12$), ordered by true class")
a, b = np.flatnonzero(y == 0)[:2]
plot_dtw_alignment(X[a], X[b], window=12, every=2, ax=axes[2])
axes[2].set_title(f"DTW alignment of two bells: {R_dtw[a, b]:.2f} vs Euclidean {R_euclid[a, b]:.2f}")
plt.tight_layout()

# %%
# Fuzzy c-medoids on the two matrices
# -----------------------------------
# The same algorithm, with five restarts, on both matrices. On the DTW
# matrix it recovers the classes and its medoids are typical,
# well-centered series of each class; on the Euclidean matrix the
# clusters cut across the classes by the position of the peaks.

# [snippet:fit]
fcmdd_dtw = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(R_dtw)
fcmdd_euc = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(R_euclid)
for name, model in (("DTW", fcmdd_dtw), ("Euclidean", fcmdd_euc)):
    print(f"{name:9s}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"medoids = {model.medoid_indices_.tolist()}, J = {model.objective_:.2f}, "
          f"{model.n_iter_} iterations")
# [/snippet:fit]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True)
plot_series_clusters(X, fcmdd_dtw.labels_, medoids=fcmdd_dtw.medoid_indices_, ax=axes[0])
plot_series_clusters(X, fcmdd_euc.labels_, medoids=fcmdd_euc.medoid_indices_, ax=axes[1])
for ax in axes[0]:
    ax.set_title("DTW: " + ax.get_title())
for ax in axes[1]:
    ax.set_title("Euclidean: " + ax.get_title())
plt.tight_layout()

# %%
# The memberships are fuzzy where shapes are ambiguous: a strongly
# warped double bump whose bumps come close resembles a bell, and a
# bell shifted toward the end resembles the rise of a step.
fig, ax = plt.subplots(figsize=(10, 3.2))
order = np.argsort(y, kind="stable")
bottom = np.zeros(N)
for i in range(3):
    ax.bar(np.arange(N), fcmdd_dtw.membership_[order, i], bottom=bottom, color=PALETTE[i], width=1.0,
           label=f"cluster {i + 1}")
    bottom += fcmdd_dtw.membership_[order, i]
for b_ in (25, 50):
    ax.axvline(b_ - 0.5, color="white", lw=1.5)
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(0, 1)
ax.set_xlabel("series (ordered by true class)")
ax.set_ylabel(r"$\mu_{ik}$")
ax.set_title("Memberships of the DTW c-medoid clustering")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
plt.tight_layout()

# %%
# Parameter study I: the number of candidates (speed against quality)
# -------------------------------------------------------------------
# The full search tries all :math:`N` objects as medoid of every cluster.
# With ``n_candidates`` :math:`=p` only the :math:`p` objects with the
# highest memberships are tried, which makes an iteration
# :math:`O(cpN)` instead of :math:`O(cN^2)`. The study records the final
# objective, the agreement with the classes and the time for thirty
# random starts per setting. One or two candidates trap the search in a
# clearly worse local optimum, five come within about 1.5 percent of the
# objective of the full search, and fifty candidates reach it in about a
# quarter of the time.

# [snippet:candidates]
R_big = dtw_distance_matrix(make_warped_series(n_series_per_class=120, random_state=1)[0], window=12)
y_big = np.repeat(np.arange(3), 120)
candidates = (1, 2, 3, 5, 10, 20, 50, None)
study = []
for p in candidates:
    t0 = time.perf_counter()
    model = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_candidates=p, init="random",
                          n_init=30, random_state=0).fit(R_big)
    study.append((1e3 * (time.perf_counter() - t0), model.objective_, adjusted_rand_score(y_big, model.labels_)))
    print(f"n_candidates = {str(p):4s}: J = {study[-1][1]:8.3f}, ARI = {study[-1][2]:.2f}, "
          f"{study[-1][0]:6.1f} ms for 30 restarts")
# [/snippet:candidates]
study = np.asarray(study)
labels = [str(p) if p is not None else "all" for p in candidates]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(labels, study[:, 1], "o-", color=PALETTE[1])
axes[0].set_xlabel("n_candidates")
axes[0].set_ylabel("final objective $J$")
axes[0].set_title(f"Objective against the candidate set size ($N={R_big.shape[0]}$)")
axes[0].grid(True, color="0.9")
axes[1].plot(labels, study[:, 0], "s-", color=PALETTE[0])
axes[1].set_xlabel("n_candidates")
axes[1].set_ylabel("milliseconds (30 restarts)")
axes[1].set_title("Time of the medoid search")
axes[1].grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study II: the band width and the fuzziness exponent
# -------------------------------------------------------------
# A band of width zero reduces DTW to the Euclidean distance. The band has
# to be about as wide as the time displacements it should absorb (here
# shifts of up to 6 samples plus the warps), and the quality rises with it
# up to unconstrained DTW. The price of a wide band is computing time and,
# on other data, over-alignment: an unconstrained path can match a bell to
# a single bump of another shape. :math:`m` controls the softness as
# usual; the medoids, being objects, hardly move with it.
print("band width w:")
for w in (0, 4, 8, 12, 20, None):
    Rw = dtw_distance_matrix(X, window=w)
    model = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(Rw)
    print(f"  w = {str(w):4s}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean within-class DTW = {np.mean([Rw[np.ix_(y == i, y == i)].mean() for i in range(3)]):.2f}")
print("fuzziness m (DTW, w = 12):")
for m in (1.3, 2.0, 3.0):
    model = FuzzyCMedoids(n_clusters=3, metric="precomputed", m=m, n_init=5, random_state=0).fit(R_dtw)
    print(f"  m = {m:.1f}: ARI = {adjusted_rand_score(y, model.labels_):.2f}, "
          f"mean max membership = {model.membership_.max(axis=1).mean():.2f}, medoids {model.medoid_indices_.tolist()}")

# %%
# **Take-away.** The dissimilarity decides what a cluster is: with the
# Euclidean distance the c-medoids group series by *when* things happen,
# with DTW by *what* happens. Fuzzy c-medoids accepts the DTW matrix as it
# is, without a Euclidean assumption or a spread, and returns actual series
# as prototypes, which is what one wants to look at. The band width is the
# only parameter of the dissimilarity and should match the displacements
# to be absorbed. A candidate set of a handful of objects turns the
# exhaustive medoid search into a linear one at almost no loss; restarts
# remain advisable, as the paper recommends.
