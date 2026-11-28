r"""
Bottom-up segmentation with PCA models versus fuzzy segmentation
================================================================

The classical way to segment a time series is *bottom-up*: start from a
fine partition into short pieces, compute the cost of merging every pair of
adjacent segments, merge the cheapest pair, and repeat until the cheapest
merge would cost more than a tolerance or until the desired number of
segments is left. For multivariate data, the cost of a segment is the error
of one PCA model fitted to it: the mean reconstruction error :math:`Q`
(distance from the hyperplane) or the mean Hotelling :math:`T^2` (spread
inside the hyperplane). This is the baseline against which the segmentation
paper compares, and its two costs see different things: :math:`Q` reacts
to changes in the correlation structure, :math:`T^2` to shifts of the
operating point. The fuzzy segmentation combines both in one model.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust.plotting import PALETTE
from fuzzyclust.timeseries import FuzzySegmentation, bottom_up_segmentation, make_piecewise_timeseries

# %%
# Data: a correlation change, a mean shift, and a change of both
# --------------------------------------------------------------
X, true_segments = make_piecewise_timeseries(
    n_samples=2000,
    change_points=(500, 1000, 1500),
    change_kinds=("correlation", "mean", "both"),
    noise=0.15,
    random_state=0,
)
t = np.arange(X.shape[0])
true_borders = (500, 1000, 1500)

# %%
# Bottom-up segmentation with the two costs
# -----------------------------------------
# Both runs are stopped at four segments so that the results are comparable
# with the fuzzy segmentation. The merge costs recorded along the way jump
# at the moment the algorithm starts to merge pieces of *different* regimes.

# [snippet:fit]
bu_q = bottom_up_segmentation(X, n_components=2, n_segments=4, cost="q", init_length=20)
bu_t2 = bottom_up_segmentation(X, n_components=2, n_segments=4, cost="t2", init_length=20)
fuzzy = FuzzySegmentation(n_segments=4, n_components=2).fit(X)
print("bottom-up, Q   :", bu_q.boundaries)
print("bottom-up, T^2 :", bu_t2.boundaries)
print("fuzzy          :", fuzzy.boundaries_)
# [/snippet:fit]

fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True)
axes[0].plot(t, X[:, 0], color="0.3", lw=0.6)
axes[0].plot(t, X[:, 3], color="0.6", lw=0.6)
axes[0].set_ylabel("$x_1$, $x_4$")
axes[0].set_title("Two of the six variables and the segment borders found by three methods")
for ax, borders, name, col in zip(
    axes[1:],
    (bu_q.boundaries, bu_t2.boundaries, fuzzy.boundaries_),
    ("bottom-up, $Q$ cost", "bottom-up, $T^2$ cost", "fuzzy segmentation"),
    (PALETTE[0], PALETTE[2], PALETTE[1]),
    strict=True,
):
    for cp in true_borders:
        ax.axvline(cp, color="0.6", ls="--", lw=1)
    for b in borders:
        ax.axvline(b, color=col, lw=2)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center")
axes[-1].set_xlabel("time")
plt.tight_layout()

# %%
# The :math:`Q` cost finds the correlation changes (at 500 and 1500) and
# misses the pure mean shift at 1000. The data of the second and third
# regimes lie on the same hyperplane, only displaced along it, so one PCA
# model reconstructs both without error. The :math:`T^2` cost finds the
# mean shifts (1000, 1500) and misses the pure correlation change at 500,
# because the spread inside the hyperplane does not change there. The
# fuzzy segmentation detects all three.
#
# The merge costs show where to stop
# ----------------------------------
# Plotted against the number of segments left, the cost of each merge rises
# slowly while pieces of the same regime are joined and jumps when the
# algorithm has to join different regimes. The knee marks the natural
# number of segments. Here it is read off the :math:`Q` curve at three (the
# mean shift is invisible to :math:`Q`) and off the :math:`T^2` curve at
# three as well (the correlation change is invisible to :math:`T^2`).
full_q = bottom_up_segmentation(X, n_components=2, n_segments=1, cost="q", init_length=20)
full_t2 = bottom_up_segmentation(X, n_components=2, n_segments=1, cost="t2", init_length=20)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, res, name, col in zip(
    axes, (full_q, full_t2), ("$Q$ cost", "$T^2$ cost"), (PALETTE[0], PALETTE[2]), strict=True
):
    n_left = np.arange(len(res.merge_costs), 0, -1)  # segments left after each merge
    ax.semilogy(n_left, res.merge_costs, "o-", color=col, ms=3, lw=1.2)
    ax.set_xlim(12.5, 0.5)
    ax.set_xlabel("segments left after the merge")
    ax.set_ylabel("cost of the merged segment")
    ax.set_title(f"Merge costs of the bottom-up algorithm, {name}")
    ax.grid(True, color="0.9", which="both")
plt.tight_layout()

# %%
# Parameter study: number of components, cost, and initial piece length
# ---------------------------------------------------------------------
# The :math:`Q` cost depends strongly on :math:`q`. With :math:`q=2`, the
# dimension of the hidden process, a mean shift along the hyperplane is
# invisible. With :math:`q=1` the displacement between two regimes leaves
# the one-dimensional model and is detected as well, by accident rather
# than by design. With :math:`q=5` the residual is almost pure noise and the
# borders become unreliable. The :math:`T^2` cost is insensitive to
# :math:`q` but never sees the correlation change. A long initial piece (40
# samples) coarsens the borders to multiples of the piece length and cuts
# the transition at 1500 into two pieces.

# [snippet:scan]
rows = []
for cost in ("q", "t2"):
    for q in (1, 2, 3, 5):
        for init_length in (10, 40):
            res = bottom_up_segmentation(X, n_components=q, n_segments=4, cost=cost, init_length=init_length)
            found = sorted(int(b) for b in res.boundaries)
            err = [min(abs(b - cp) for b in found) for cp in true_borders]
            rows.append((cost, q, init_length, found, err))
# [/snippet:scan]
print(f"{'cost':>4} {'q':>2} {'init':>5}  borders found           error at (500, 1000, 1500)")
for cost, q, init_length, found, err in rows:
    print(f"{cost:>4} {q:>2} {init_length:>5}  {str(found):<24} {err}")

# %%
# Finally, the fuzzy segmentation is run with the same number of segments.
# Unlike the greedy method, it is not bound by an early wrong merge:
for q in (1, 2, 3, 5):
    print(f"fuzzy, q = {q}: borders {FuzzySegmentation(n_segments=4, n_components=q).fit(X).boundaries_.tolist()}")

# %%
# **Take-away.** The bottom-up algorithm is fast and simple, but each of its
# costs is blind to one kind of change. The reconstruction error does not
# see displacements along the hyperplane, the Hotelling statistic does not
# see rotations of the hyperplane, and a greedy merge made early is never
# undone. The fuzzy segmentation uses one probabilistic PCA model per
# segment that covers both subspaces and re-optimizes all segments after
# every merge, which makes it insensitive to the number of principal
# components (from :math:`q=2` on). Use the bottom-up method for a quick
# first look and to initialize or validate the fuzzy segmentation.
