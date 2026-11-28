r"""
Supervised fuzzy clustering: classifier rules that carry class probabilities
============================================================================

A classical fuzzy classifier has one rule per class and a crisp class label as
its consequent, so every class must be a single compact region. Abonyi and
Szeifert proposed rules of the form

.. math::
    r_i:\ \text{If } x_1 \text{ is } A_{i1}(x_1) \text{ and } \dots \text{ then }
    \hat y = c_1 \text{ with } P(c_1\mid r_i),\ \dots,\ \hat y = c_C \text{ with } P(c_C\mid r_i)\ [w_i],

so that a rule may describe several classes and a class may be covered by
several rules. The classifier is a Bayes classifier whose class-conditional
densities are Gaussian mixtures. The rules are identified by *supervised*
Gath–Geva-type clustering, whose distance multiplies the Gaussian density of
the input by the probability that the cluster describes the class of the
sample. This example uses a two-class problem in which each class consists of
two blobs (an XOR layout); one rule per class cannot work, but four rules can.
The example shows the decision regions, the class probabilities of the rules,
and the effect of the number of rules and of the variance regularization,
and it applies the Fisher interclass-separability feature selection of the
paper to the wine data.
"""

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score

from fuzzyclust.datasets import load_wine
from fuzzyclust.models import (
    SupervisedFuzzyClassifier,
    make_two_class_overlap,
    plot_decision_regions,
    plot_membership_functions,
    select_features_by_separability,
)
from fuzzyclust.plotting import PALETTE

X, y = make_two_class_overlap(n_samples=400, spread=0.6, random_state=0)

# %%
# Two rules versus four rules
# ---------------------------

# [snippet:fit]
clf2 = SupervisedFuzzyClassifier(n_clusters=2, random_state=0).fit(X, y)
clf4 = SupervisedFuzzyClassifier(n_clusters=4, random_state=0).fit(X, y)
print(f"training accuracy: 2 rules {clf2.score(X, y):.3f}, 4 rules {clf4.score(X, y):.3f}")
print(clf4.rule_base().round(3).to_string(index=False))
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, clf, name in zip(axes, (clf2, clf4), ("two rules", "four rules"), strict=True):
    plot_decision_regions(clf, X, y, ax=ax)
    ax.set_title(f"{name}: accuracy {clf.score(X, y):.2f}")
plt.tight_layout()

# %%
# With two rules, the clustering has to cover each class with one Gaussian.
# The best it can do is one rule on a blob of class 0 and one broad rule in
# the middle whose consequent is a mixture of both classes (``P(0)`` and
# ``P(1)`` far from 0 and 1), so the accuracy is poor. With four rules,
# every blob gets its own rule with a nearly pure consequent, and the
# decision boundary follows the XOR pattern. The dashed ellipses are the
# :math:`\pm\sigma_{ij}` extents of the antecedent sets.
#
# Posterior class probabilities
# -----------------------------
# The classifier outputs :math:`P(c_j\mid\mathbf{x})=\sum_i\beta_i(\mathbf{x})P(c_j\mid r_i)/\sum_i\beta_i(\mathbf{x})`.
# The posterior of class 1 is shown together with the membership functions
# of the four rules on the two features.
fig, ax = plt.subplots(figsize=(6.2, 4.8))
plot_decision_regions(clf4, X, y, ax=ax, proba_class=1, show_rules=False)
ax.set_title("Posterior probability of class 1 and the 0.5 decision boundary")
plt.tight_layout()

fig, axes = plt.subplots(2, 1, figsize=(7, 4.6))
plot_membership_functions(clf4, X=X, axes=axes)
axes[0].set_title("Antecedent sets of the four rules (one Gaussian per rule and feature)")
plt.tight_layout()

# %%
# Two rules are centered near :math:`x_1=\pm1.5` with :math:`x_2\approx0`
# (class 0) and two near :math:`x_2=\pm1.5` with :math:`x_1\approx0` (class 1),
# so the sets on each feature come in pairs, as the XOR structure requires.
#
# Parameter study: number of rules and variance regularization
# ------------------------------------------------------------
# The number of rules :math:`R` is scanned with 5-fold cross-validation. The
# exponential distance of Gath–Geva-type algorithms tends to shrink a rule
# onto a few samples when :math:`R` is large. The variance regularization
# ``gamma`` counteracts this by shrinking every antecedent variance toward
# the variance of the feature.

# [snippet:scan]
rows = []
for R in (2, 3, 4, 5, 6, 8, 12):
    for gamma in (0.0, 0.1):
        clf = SupervisedFuzzyClassifier(n_clusters=R, gamma=gamma, random_state=0)
        acc = cross_val_score(clf, X, y, cv=5)
        rows.append({"rules": R, "gamma": gamma, "cv accuracy": acc.mean(), "std": acc.std()})
scan = pd.DataFrame(rows)
# [/snippet:scan]
print(scan.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
for i, (gamma, sub) in enumerate(scan.groupby("gamma")):
    ax.errorbar(sub["rules"], sub["cv accuracy"], yerr=sub["std"], fmt="o-", color=PALETTE[i], capsize=3,
                label=f"gamma = {gamma}")
ax.set_xlabel("number of rules $R$")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("More rules than classes are needed; regularization helps for many rules")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Feature selection by interclass separability (wine data)
# --------------------------------------------------------
# The paper selects features with a modified Fisher criterion computed from
# the clusters: :math:`J=\det\mathbf{F}_B/\det\mathbf{F}_W` with the
# between-rule scatter :math:`\mathbf{F}_B` and the within-rule scatter
# :math:`\mathbf{F}_W`. The feature whose removal lowers :math:`J` least is
# deleted, the classifier is re-identified, and so on; the training accuracy
# shows where to stop. With three rules, the wine data (178 wines, 13
# attributes, 3 cultivars) are classified almost perfectly, and five
# features suffice, as in the paper.
wine = load_wine()

# [snippet:features]
selected, eliminated, history = select_features_by_separability(
    wine.data, wine.target, n_features_to_select=3, n_clusters=3, random_state=0
)
# [/snippet:features]
history["removed feature"] = ["-"] + [wine.feature_names[j] for j in eliminated]
print(history[["n_features", "removed feature", "separability", "accuracy"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("features kept:", [wine.feature_names[j] for j in selected])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(history["n_features"], history["accuracy"], "o-", color=PALETTE[0])
for _, row in history.iterrows():
    if row["removed feature"] != "-":
        ax.annotate(row["removed feature"], (row["n_features"], row["accuracy"]), textcoords="offset points",
                    xytext=(3, 6), fontsize=7, rotation=30)
ax.set_xlabel("number of features")
ax.set_ylabel("training accuracy (3 rules)")
ax.set_title("Backward elimination of wine attributes by interclass separability")
ax.invert_xaxis()
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# **Take-away.** Letting a rule carry a probability for every class
# decouples the number of rules from the number of classes. The rule base is
# a Gaussian-mixture Bayes classifier that handles classes made of several
# groups and overlapping classes alike. Supervised clustering identifies it
# in one pass by including the class term in the distance. Choose the number
# of rules by cross-validation, regularize the variances when many rules are
# used, and prune the features with the separability criterion; a compact
# classifier with a few rules on a few attributes is usually as accurate as
# the full one.
