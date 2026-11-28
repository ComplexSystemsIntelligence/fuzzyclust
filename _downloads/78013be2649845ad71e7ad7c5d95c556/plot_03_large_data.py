r"""
Fuzzy c-means for very large data: rseFCM, spFCM and oFCM
==========================================================

Literal fuzzy c-means touches every sample in every iteration and keeps
the :math:`N\times c` partition in memory. Havens, Bezdek, Leckie, Hall
and Palaniswami (2012) compared three ways around this. *Random sample
and extend* (rseFCM) clusters a random sample and computes the
memberships of all samples in one pass from the fixed prototypes.
*Single-pass* FCM (spFCM) walks through the data in chunks and carries
the prototypes of a chunk, weighted by the membership mass they
collected, into the next chunk as :math:`c` heavy samples of weighted
FCM. *Online* FCM (oFCM) clusters every chunk independently and clusters
the weighted chunk prototypes at the end. The paper measures the
*speedup* over literal FCM and the agreement of the hardened partitions
with the literal FCM partition. This example repeats the experiment on
200 000 two-dimensional samples of five clusters of unequal size; the
smallest holds only 5 % of the data, which makes a small random sample
risky.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.large import (
    OnlineFCM,
    RandomSampleFCM,
    SinglePassFCM,
    WeightedFCM,
    make_large_blobs,
    plot_chunk_timeline,
)
from fuzzyclust.plotting import PALETTE

X, y = make_large_blobs(n_samples=200_000, n_clusters=5, weights=[35, 25, 20, 15, 5], spread=0.6,
                        random_state=0)
N = X.shape[0]


def timed_fit(model, X):
    start = time.perf_counter()
    model.fit(X)
    return model, time.perf_counter() - start


# %%
# The literal FCM and the three schemes with 1 % chunks
# ------------------------------------------------------
# The reference is FCM on all samples. It is run as weighted FCM with unit
# weights, that is, with the same algorithm and the same distance code as
# the chunked schemes, so that the timings compare algorithms and not
# implementations. Every scheme then sees 1 % of the data at a time
# (2000 samples): rseFCM one random sample of that size, spFCM and oFCM
# 100 chunks.

# [snippet:fit]
literal, t_literal = timed_fit(WeightedFCM(n_clusters=5, random_state=0), X)
schemes = {
    "rseFCM": RandomSampleFCM(n_clusters=5, sample_size=0.01, random_state=0),
    "spFCM": SinglePassFCM(n_clusters=5, chunk_size=2000, random_state=0),
    "oFCM": OnlineFCM(n_clusters=5, chunk_size=2000, random_state=0),
}
results = {}
for name, model in schemes.items():
    model, seconds = timed_fit(model, X)
    results[name] = (t_literal / seconds, adjusted_rand_score(literal.labels_, model.labels_))
    print(f"{name}: speedup {results[name][0]:6.1f}, ARI with literal FCM {results[name][1]:.4f}")
# [/snippet:fit]
print(f"literal FCM: {t_literal:.2f} s, {literal.n_iter_} iterations, ARI with truth "
      f"{adjusted_rand_score(y, literal.labels_):.3f}")

sub = np.random.RandomState(0).choice(N, size=5000, replace=False)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
for ax, (name, model) in zip(axes, [("literal FCM", literal), *schemes.items()], strict=True):
    # color by the matched literal-FCM cluster, so that a cluster has one color in all panels
    r, col = linear_sum_assignment(cdist(model.cluster_centers_, literal.cluster_centers_))
    order = col[np.argsort(r)]
    ax.scatter(X[sub, 0], X[sub, 1], s=3, c=[PALETTE[i] for i in order[model.labels_[sub]]], alpha=0.4)
    ax.scatter(*literal.cluster_centers_.T, marker="+", s=200, color="black", zorder=5)
    ax.scatter(*model.cluster_centers_.T, marker="X", s=110, color="white", edgecolor="black", zorder=6)
    title = name if model is literal else f"{name}: speedup {results[name][0]:.0f}, ARI {results[name][1]:.3f}"
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("5000 of 200 000 samples; + literal FCM prototypes")
plt.tight_layout()

# %%
# Inside the chunked schemes
# ---------------------------
# The chunk timeline of spFCM shows the carried prototypes after every
# chunk; their markers grow with the weight they carry, so after a few
# chunks the history dominates and a new chunk only nudges them. oFCM
# keeps the independent chunk prototypes, which form a small cloud around
# every merged prototype (drawn here relative to it, black cross), and
# merges them with weighted FCM at the end. The cloud is the sampling
# error of a 2000-sample chunk; the small cluster (blue) has the widest
# spread, because a chunk holds only about 100 of its members.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
plot_chunk_timeline(schemes["spFCM"], feature=0, ax=axes[0])
axes[0].set_title("spFCM: prototypes after every chunk (area ~ weight)")
ofcm = schemes["oFCM"]
Vc = ofcm.chunk_centers_.reshape(-1, 2)
lab = ofcm.centers_model_.labels_
dev = Vc - ofcm.cluster_centers_[lab]
axes[1].scatter(dev[:, 0], dev[:, 1], s=12, c=[PALETTE[i] for i in lab], edgecolor="none", alpha=0.8)
axes[1].scatter([0], [0], marker="X", s=120, color="black", edgecolor="white", zorder=5)
axes[1].set_xlabel("chunk prototype minus merged prototype, feature 1")
axes[1].set_ylabel("feature 2")
axes[1].set_title("oFCM: the 100 chunk prototypes of each cluster")
axes[1].set_aspect("equal", adjustable="datalim")
axes[1].grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study: the chunk size
# --------------------------------
# The sample or chunk size is the only parameter of all three schemes. The
# study varies it from 0.1 % to 10 % of the data and records the speedup
# over literal FCM and the agreement with its partition (ARI; 1 means
# the same hardened partition). With 600 samples or fewer, FCM on the
# sample (or on the *first chunk*) ends in a local optimum that merges
# the small cluster with a neighbor and splits the largest one. rseFCM
# keeps it by construction and spFCM carries it through all later chunks,
# because the carried prototypes soon outweigh any new chunk. oFCM merges
# many independent chunk solutions and is not misled. Restarts of the
# first chunk (``n_init=5``) repair spFCM at a small cost.

# [snippet:scan]
fractions = (0.001, 0.003, 0.01, 0.03, 0.1)
makers = {
    "rseFCM": lambda n: RandomSampleFCM(n_clusters=5, sample_size=n, random_state=0),
    "spFCM": lambda n: SinglePassFCM(n_clusters=5, chunk_size=n, random_state=0),
    "oFCM": lambda n: OnlineFCM(n_clusters=5, chunk_size=n, random_state=0),
    "spFCM, n_init=5": lambda n: SinglePassFCM(n_clusters=5, chunk_size=n, n_init=5, random_state=0),
}
speedup = {name: [] for name in makers}
agreement = {name: [] for name in makers}
for frac in fractions:
    n = int(frac * N)
    for name, make in makers.items():
        model, seconds = timed_fit(make(n), X)
        speedup[name].append(t_literal / seconds)
        agreement[name].append(adjusted_rand_score(literal.labels_, model.labels_))
# [/snippet:scan]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for j, name in enumerate(makers):
    fmt = "s--" if "n_init" in name else "o-"
    color = PALETTE[1] if "n_init" in name else PALETTE[j]
    axes[0].semilogx(np.asarray(fractions) * 100, speedup[name], fmt, color=color, label=name)
    axes[1].semilogx(np.asarray(fractions) * 100, agreement[name], fmt, color=color, label=name)
axes[0].axhline(1.0, color="0.3", ls=":", lw=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("speedup over literal FCM")
axes[0].set_title("Speedup (wall-clock time of literal FCM / scheme)")
axes[1].set_ylabel("ARI with the literal FCM partition")
axes[1].set_title("Agreement with literal FCM")
for ax in axes:
    ax.set_xlabel("sample / chunk size [% of N]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9", which="both")
plt.tight_layout()

for name in makers:
    print(f"{name:16s} ARI " + " ".join(f"{a:.3f}" for a in agreement[name])
          + " | speedup " + " ".join(f"{v:.0f}" for v in speedup[name]))

# %%
# **Take-away.** All three schemes reproduce the literal FCM partition of
# 200 000 samples while touching only a small part of the data at a time,
# at a fraction of its cost. rseFCM is the fastest and is adequate as soon
# as the sample is large enough for FCM to find the right solution on it.
# spFCM sees every sample and is still several times faster than literal
# FCM, but it inherits the solution of its first chunk: give that chunk
# restarts (``n_init``) or a good size. oFCM is the slowest of the three,
# because every chunk is clustered to convergence, and the most robust,
# because a poor chunk solution is outvoted in the final merge; its chunk
# runs are independent and could run in parallel. Choose the chunk as
# large as memory allows; the cost of small chunks is accuracy, not speed.
