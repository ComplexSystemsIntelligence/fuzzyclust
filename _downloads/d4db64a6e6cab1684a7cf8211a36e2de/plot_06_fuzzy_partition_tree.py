r"""
Supervised fuzzy partitions as a priori discretization for decision trees
=========================================================================

Rule- and tree-induction methods of the ID3 family work on discrete
attributes, so continuous features are partitioned *before* the tree is
grown. The usual a priori partitions (equal-width bins, equal-frequency
bins, or a Ruspini partition of triangular sets) ignore the class labels, so
their cut points fall anywhere. Abonyi (2010) proposed deriving the
partition from the supervised fuzzy clustering of the previous example.
Every rule's Gaussian antecedent set, projected onto a feature, is a fuzzy
set of that feature; the Gaussians are turned into trapezoids with the
corner points :math:`v-3\sigma,\ v-\sigma/2,\ v+\sigma/2,\ v+3\sigma`;
similar sets (Jaccard similarity above a threshold :math:`\theta`) are
merged; and a feature left with a single set is a *don't care* variable.
The cut points therefore lie where the classes change. This example shows
the partitions, feeds their crisp codes and their memberships to a
scikit-learn decision tree, and compares the result with unsupervised
discretizations.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, export_text

from fuzzyclust.datasets import load_iris
from fuzzyclust.models import FuzzyPartitionDiscretizer, make_banded_classes, plot_fuzzy_partition
from fuzzyclust.plotting import PALETTE

X, y = make_banded_classes(n_samples=450, random_state=0)
feature_names = ["x1", "x2", "x3 (noise)"]

# %%
# The supervised partition
# ------------------------
# Three rules are identified; their projections give three sets per feature,
# which the merging step reduces where the rules agree.

# [snippet:fit]
disc = FuzzyPartitionDiscretizer(n_sets=3, shape="trapezoid", merge_threshold=0.8, random_state=0).fit(X, y)
print("sets per feature after merging:", disc.n_sets_)
for j, p in enumerate(disc.partitions_):
    print(f"{feature_names[j]}: centers {p['centers'].round(2)}, widths {p['sigmas'].round(2)}, merged rules {p['groups']}")
# [/snippet:fit]

ruspini = FuzzyPartitionDiscretizer(n_sets=3, method="ruspini").fit(X, y)
fig, axes = plt.subplots(2, 3, figsize=(15, 6.2))
for j in range(3):
    plot_fuzzy_partition(disc, j, X=X, y=y, ax=axes[0, j], feature_name=feature_names[j])
    axes[0, j].set_title(f"supervised partition of {feature_names[j]}: {disc.n_sets_[j]} set(s)")
    plot_fuzzy_partition(ruspini, j, X=X, y=y, ax=axes[1, j], feature_name=feature_names[j])
    axes[1, j].set_title(f"Ruspini partition of {feature_names[j]}")
plt.tight_layout()

# %%
# The rugs below the sets are the samples of the three classes. On
# :math:`x_1` two rules coincide and are merged, leaving two sets that
# separate class 0 from the others. On :math:`x_2` three sets remain, with
# the transitions exactly between the classes. On the noise feature all
# three sets are alike and merge into one, so the feature is recognized as a
# don't care variable. The Ruspini partition (bottom row) places its sets at
# equal distances regardless of the classes.
#
# A decision tree on the discretized features
# -------------------------------------------
# With ``encode="crisp"`` every feature is replaced by the index of its most
# active set, and the tree then works like ID3 on nominal attributes. The
# tree is tiny and reads as rules on linguistic terms.

# [snippet:tree]
pipe = make_pipeline(
    FuzzyPartitionDiscretizer(n_sets=3, drop_single=True, random_state=0),
    DecisionTreeClassifier(max_depth=2, random_state=0),
).fit(X, y)
codes_names = list(pipe[0].get_feature_names_out(feature_names))
print(export_text(pipe[-1], feature_names=codes_names))
# [/snippet:tree]

# %%
# Comparison with unsupervised discretizations
# --------------------------------------------
# Equal-width and equal-frequency binning (``KBinsDiscretizer``), the Ruspini
# partition, and the supervised partition (crisp codes and memberships) are
# compared as inputs to the same tree by 10-fold cross-validation; the raw
# tree, which chooses its own thresholds, serves as the reference.
cv = StratifiedKFold(10, shuffle=True, random_state=0)


def evaluate(discretizer, X, y, max_depth=None):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    est = tree if discretizer is None else make_pipeline(discretizer, tree)
    res = cross_validate(est, X, y, cv=cv, return_estimator=True)
    leaves = [(e if discretizer is None else e[-1]).get_n_leaves() for e in res["estimator"]]
    return res["test_score"].mean(), np.mean(leaves)


candidates = {
    "equal width": KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform"),
    "equal frequency": KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile"),
    "Ruspini": FuzzyPartitionDiscretizer(n_sets=3, method="ruspini"),
    "supervised, crisp": FuzzyPartitionDiscretizer(n_sets=3, random_state=0),
    "supervised, membership": FuzzyPartitionDiscretizer(n_sets=3, encode="membership", random_state=0),
    "no discretization": None,
}
rows = []
for name, d in candidates.items():
    acc, leaves = evaluate(d, X, y)
    rows.append({"partition": name, "accuracy": acc, "leaves": leaves})
table = pd.DataFrame(rows)
print(table.round(3).to_string(index=False))

# %%
# Among the a priori partitions, the supervised codes give the most accurate
# and the smallest trees, with a few leaves on two features. The Ruspini
# sets, whose cut points fall inside the classes, need many leaves and still
# lose accuracy. The tree without discretization is perfect here because the
# class boundaries are axis-parallel thresholds that CART finds exactly. The
# price of a priori discretization is this small loss; the gain is a set of
# meaningful terms ("x2 is low / medium / high") shared by all rules.
#
# Iris: the familiar case
# -----------------------
# On the iris data, the supervised partition of the petal length places its
# transitions between the species, whereas the Ruspini sets peak inside
# the versicolor group. Trees of depth three on the partitions reach about
# the accuracy of the raw tree here: equal-width bins and the supervised codes
# do slightly better, equal-frequency bins, the Ruspini sets and the
# membership encoding slightly worse (the gaps between the species are wide
# enough for equal-width bins to hit them as well), so the benefit of the
# supervised sets is their meaning rather than extra accuracy.
iris = load_iris()
disc_iris = FuzzyPartitionDiscretizer(n_sets=3, random_state=0).fit(iris.data, iris.target)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_fuzzy_partition(disc_iris, 2, X=iris.data, y=iris.target, ax=axes[0], feature_name="petal length (cm)")
axes[0].set_title("Supervised sets of the petal length (rugs: the three species)")
rusp_iris = FuzzyPartitionDiscretizer(n_sets=3, method="ruspini").fit(iris.data, iris.target)
plot_fuzzy_partition(rusp_iris, 2, X=iris.data, y=iris.target, ax=axes[1], feature_name="petal length (cm)")
axes[1].set_title("Ruspini sets of the petal length")
plt.tight_layout()
rows = []
for name, d in candidates.items():
    acc, leaves = evaluate(d, iris.data, iris.target, max_depth=3)
    rows.append({"partition": name, "accuracy": acc, "leaves": leaves})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# Parameter study: number of initial sets and merging threshold
# -------------------------------------------------------------
# The number of rules of the supervised clustering is the initial number of
# sets per feature; the merging threshold :math:`\theta` determines how many
# survive. A low threshold merges aggressively and may remove sets that a
# feature needs, whereas a threshold of one keeps every projected set.

# [snippet:scan]
rows = []
for n_sets in (2, 3, 4, 5, 7):
    for theta in (0.5, 0.8, 1.0):
        d = FuzzyPartitionDiscretizer(n_sets=n_sets, merge_threshold=theta, drop_single=True, random_state=0)
        acc, leaves = evaluate(d, X, y)
        total_sets = d.fit(X, y).n_sets_.sum()
        rows.append({"initial sets": n_sets, "theta": theta, "sets kept": total_sets, "accuracy": acc, "leaves": leaves})
scan = pd.DataFrame(rows)
# [/snippet:scan]
print(scan.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for i, (theta, sub) in enumerate(scan.groupby("theta")):
    axes[0].plot(sub["initial sets"], sub["accuracy"], "o-", color=PALETTE[i], label=f"theta = {theta}")
    axes[1].plot(sub["initial sets"], sub["sets kept"], "o-", color=PALETTE[i], label=f"theta = {theta}")
axes[0].set_ylabel("10-fold CV accuracy of the tree")
axes[1].set_ylabel("fuzzy sets kept (all features)")
for ax in axes:
    ax.set_xlabel("initial number of sets (rules)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9")
axes[0].set_title("Accuracy versus initial number of sets")
axes[1].set_title("Merging keeps the partition small")
plt.tight_layout()

# %%
# **Take-away.** A partition derived from supervised clustering places its
# fuzzy sets where the classes change and discards features that do not
# matter, so a tree grown on the resulting codes is small and accurate
# compared with trees on equal-width, equal-frequency, or Ruspini partitions.
# A tree that optimizes its own thresholds (CART) may still be more accurate;
# the a priori partition buys a common, interpretable vocabulary of sets for
# all rules. Use three to five initial sets and merge at
# :math:`\theta\approx0.8`; the membership encoding passes the fuzziness on
# to the tree.
