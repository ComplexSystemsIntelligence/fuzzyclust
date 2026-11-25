r"""
Model-free structure selection with mixed discrete and continuous inputs
========================================================================

Girimonte, Babuška, and Abonyi select the inputs of a nonlinear model
*without fitting any model*. Every continuous candidate input is
partitioned into a few fuzzy sets by fuzzy c-means, every discrete input
into one crisp set per value, and for every subset of inputs the data are
decomposed into the product-space segments of these sets. If the subset
explains the output, the output varies little inside every segment, and
the *total fuzzy variance* :math:`\sigma_t` (the squared error of the
singleton fuzzy model that assigns each segment its membership-weighted
mean) is small. All subsets up to a maximal size are ranked by
:math:`\sigma_t`. Because an irrelevant input also lowers :math:`\sigma_t`
a little (more segments, fewer samples each), the *variance–complexity
trade-off* :math:`VCT=\sigma_t n_i` is used to compare structures of
similar variance: the first dip of the VCT curve marks the structure
with only the relevant inputs.

The first problem is the mixed one of the paper. The output switches
between two nonlinear functions according to a binary input,
:math:`y=x_1^2` if :math:`x_2=1` and :math:`y=x_4^2+x_3` if
:math:`x_2=0`, with two irrelevant inputs :math:`x_5, x_6`. The second
problem is the Friedman function with five relevant and five irrelevant
inputs.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_friedman1

from fuzzyclust.plotting import PALETTE
from fuzzyclust.timeseries import (
    MixedInputStructureSelection,
    make_mixed_input_system,
    plot_structure_ranking,
    select_structure_mixed,
)

# %%
# Mixed discrete and continuous inputs
# ------------------------------------
# [snippet:fit]
X, y, discrete = make_mixed_input_system(n_samples=1000, noise=0.1, n_dummy=2, random_state=0)
sel = MixedInputStructureSelection(n_clusters=3, max_inputs=5, random_state=0).fit(X, y, discrete=discrete)
print(sel.ranking_.head(8).to_string())
print("recommended structure:", sel.best_structure_)
# [/snippet:fit]

fig, ax = plt.subplots(figsize=(7, 3.8))
plot_structure_ranking(sel.ranking_, n_show=30, ax=ax, best=sel.best_structure_)
ax.set_title("Variance–complexity trade-off of the 30 best structures (mixed inputs)")
plt.tight_layout()

# %%
# The three best structures by :math:`\sigma_t` all contain the four true
# inputs. The two with a dummy input added have a marginally smaller
# variance but a larger VCT, and the structure :math:`\{x_1,x_2,x_3,x_4\}`
# is the first dip of the curve, exactly as in the paper. The fuzzy sets of
# a continuous input and the crisp sets of the discrete one are stored in
# ``memberships_``:
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
order = np.argsort(X[:, 0])
for i in range(3):
    axes[0].plot(X[order, 0], sel.memberships_[0][order, i], color=PALETTE[i], lw=1.6)
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("membership")
axes[0].set_title("Fuzzy sets of the continuous input $x_1$ (FCM, $c=3$)")
for v in (0, 1):
    axes[1].bar([v], [np.mean(X[:, 1] == v)], color=PALETTE[v], width=0.5)
axes[1].set_xticks([0, 1])
axes[1].set_xlabel("$x_2$")
axes[1].set_ylabel("fraction of samples")
axes[1].set_title("Crisp sets of the discrete input $x_2$")
plt.tight_layout()

# %%
# Parameter study: number of fuzzy sets, sample size and noise
# ------------------------------------------------------------
# More fuzzy sets per input resolve the nonlinearities better and lower
# :math:`\sigma_t` for every structure, but they also multiply the number of
# segments (:math:`\prod_j c_j`), which costs samples and makes dummy inputs
# look useful. With two sets, the quadratic terms are resolved poorly, and
# under strong noise the weak input :math:`x_1` is even dropped. With five
# sets, the :math:`5\cdot2\cdot5\cdot5=250` segments of the true structure
# hold only a few samples each, and a dummy input creeps in. Three sets and
# a thousand samples work best for this problem; with 300 samples a dummy
# creeps in there too. The table lists, for every setting, the structure
# recommended by the VCT and the one with the lowest :math:`\sigma_t`.

# [snippet:scan]
rows = []
for n_clusters in (2, 3, 5):
    for n_samples, noise in ((1000, 0.1), (300, 0.1), (1000, 0.5)):
        Xs, ys, disc = make_mixed_input_system(n_samples=n_samples, noise=noise, n_dummy=2, random_state=0)
        s = MixedInputStructureSelection(n_clusters=n_clusters, max_inputs=5, random_state=0).fit(Xs, ys, discrete=disc)
        rows.append((n_clusters, n_samples, noise, s.best_structure_, s.ranking_.iloc[0]["inputs"]))
# [/snippet:scan]
print(f"{'sets':>4} {'N':>5} {'noise':>5}  {'recommended (VCT)':<28} {'lowest sigma_t'}")
for n_clusters, n_samples, noise, best, lowest in rows:
    print(f"{n_clusters:>4} {n_samples:>5} {noise:>5}  {', '.join(best):<28} {', '.join(lowest)}")

# %%
# The Friedman function: five relevant inputs among ten
# -----------------------------------------------------
# The function is
# :math:`y=10\sin(\pi x_1x_2)+20(x_3-0.5)^2+10x_4+5x_5+\epsilon`, with
# :math:`x_6,\dots,x_{10}` irrelevant. This is the first example of the
# paper, with 847 candidate structures of up to six inputs. The six best
# structures by :math:`\sigma_t` are the true one and the true one plus
# each of the five dummies; the VCT singles out the true one. The weakest
# relevant input, :math:`x_3`, is the first to be dropped further down the
# list, as in the paper.
Xf, yf = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=0)
ranking = select_structure_mixed(Xf, yf, n_clusters=3, max_inputs=6, random_state=0)
print(ranking.head(8).to_string())
fig, ax = plt.subplots(figsize=(7, 3.8))
plot_structure_ranking(ranking, n_show=70, ax=ax, best=("x1", "x2", "x3", "x4", "x5"), label_top=3)
ax.set_title("Variance–complexity trade-off of the 70 best structures (Friedman function)")
plt.tight_layout()

# %%
# **Take-away.** Fuzzy clustering turns continuous inputs into fuzzy
# "values", so discrete and continuous inputs can be treated alike and the
# variance decomposition of a discrete analysis of variance carries over.
# No model is fitted, the search is exhaustive, and the result is a *ranked
# list* of structures rather than a single answer. The VCT points to the
# parsimonious structure among those of similar variance, and the final
# choice, confirmed by fitting models for a handful of candidates, is left
# to the user. The method is limited to a moderate number of candidate
# inputs (about 15) because the number of structures grows combinatorially.
