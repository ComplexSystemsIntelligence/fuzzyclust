r"""
Takagi–Sugeno model of a static nonlinearity: clusters become rules
===================================================================

A Takagi–Sugeno (TS) fuzzy model consists of :math:`c` rules

.. math::
    R_i:\ \text{If } x \text{ is } A_i(x) \text{ then } \hat y_i = a_i x + b_i,\quad [w_i]

whose outputs are combined with the normalized firing strengths
:math:`\beta_i(x)=w_i A_i(x)/\sum_j w_j A_j(x)`. Clustering in the product
space of inputs and outputs, :math:`\mathbf{z}_k=[x_k\ y_k]^T`, identifies both
parts at once: each cluster is a region of the data that is well described by a
local linear model, the projection of the cluster onto the input is the
antecedent set, and the local model is the consequent. The modified Gath–Geva
algorithm of Abonyi, Babuška, and Szeifert fits axis-parallel Gaussian clusters
with a local linear model inside, so the antecedents are read off directly and
no projection error arises. The classical alternative clusters the product
space with Gustafson–Kessel or Gath–Geva and projects the oblique ellipsoids
onto the input axes. This example identifies the sinc function both ways and
studies the number of rules and the consequent estimator.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuzzyclust.models import (
    TakagiSugeno,
    make_sinc,
    plot_local_models,
    plot_membership_functions,
)
from fuzzyclust.plotting import PALETTE

X, y = make_sinc(n_samples=300, noise=0.02, random_state=0)
X_test, y_test = make_sinc(n_samples=1000, noise=0.0, random_state=1)

# %%
# Four rules from modified Gath–Geva clustering
# ---------------------------------------------
# The engine behind ``TakagiSugeno`` is :class:`~fuzzyclust.models.ModifiedGG`.
# Its clusters consist of Gaussian membership functions on the input, local
# linear models, and prior probabilities; the rule weights
# :math:`w_i=\alpha_i/\sqrt{2\pi\sigma_i^2}` follow from the clustering.

# [snippet:fit]
ts = TakagiSugeno(n_clusters=4, random_state=0).fit(X, y)
print(ts.describe(digits=3))
print(f"R^2 on noise-free test data: {ts.score(X_test, y_test):.3f}")
# [/snippet:fit]

fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [2.4, 1]}, sharex=True)
plot_local_models(ts, X, y, ax=axes[0])
axes[0].set_title("Four local linear models interpolated by the firing strengths")
plot_membership_functions(ts, X=X, axes=axes[1:])
axes[1].set_title("Gaussian antecedent sets $A_i(x)$ (projection of the clusters)")
axes[1].set_xlabel("x")
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
axes[1].legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
plt.tight_layout()

# %%
# The local models (colored) are the linear laws of the clusters; the thick
# parts show where each rule fires (:math:`\beta_i>0.05`). The black curve is
# the TS output, a smooth interpolation between the local lines; their
# weights are drawn as shaded areas on the right axis. The antecedent sets
# below are exactly the Gaussians of the clusters; nothing has been
# projected or approximated.
#
# Axis-parallel clusters versus projected ellipsoids
# --------------------------------------------------
# With ``antecedent="gk_projected"`` or ``"gg_projected"`` the data are
# clustered in the :math:`(x, y)` plane with Gustafson–Kessel or Gath–Geva,
# and the marginal variance of every cluster along :math:`x` defines the
# Gaussian set. The ellipsoids are oblique (they follow the local slope), so
# the projection loses the information about where exactly the cluster ends
# along :math:`x`. This is the decomposition error that the modified
# algorithm avoids.
variants = {
    "modified GG (axis-parallel)": TakagiSugeno(n_clusters=4, random_state=0),
    "GG + projection": TakagiSugeno(n_clusters=4, antecedent="gg_projected", random_state=0),
    "GK + projection": TakagiSugeno(n_clusters=4, antecedent="gk_projected", random_state=0),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
grid = np.linspace(-10, 10, 400)[:, None]
for ax, (name, model) in zip(axes, variants.items(), strict=True):
    model.fit(X, y)
    ax.scatter(X[:, 0], y, s=8, color="0.5", alpha=0.6)
    ax.plot(grid[:, 0], np.sinc(grid[:, 0] / np.pi), "k--", lw=1, label="true")
    ax.plot(grid[:, 0], model.predict(grid), color=PALETTE[0], lw=2, label="TS model")
    for i in range(4):
        ax.axvspan(
            model.centers_[i, 0] - model.sigmas_[i, 0],
            model.centers_[i, 0] + model.sigmas_[i, 0],
            color=PALETTE[i % len(PALETTE)],
            alpha=0.12,
            lw=0,
        )
    ax.set_title(f"{name}: $R^2$ = {model.score(X_test, y_test):.3f}")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8, frameon=False)
plt.tight_layout()

# %%
# The shaded bands are the :math:`\pm\sigma_i` extents of the antecedent sets.
# The modified GG model is the most accurate. The projected variants place and
# size the sets differently because the width of an oblique ellipsoid along
# :math:`x` is not the width of its operating region, and the Gath–Geva
# clusters, whose volumes are free, suffer most from the projection (the
# central peak is cut off). The paper's remedy for the projected models is a
# global re-estimation of the consequents (``consequent="global"``), which
# compensates for part of the decomposition error at the price of the local
# interpretation.
# The next figure shows the product-space view: the GK clusters are ellipses
# in the :math:`(x, y)` plane, whereas the modified GG clusters are Gaussians
# in :math:`x` with a *line* inside.
gk_ts = variants["GK + projection"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
from fuzzyclust.plotting import plot_covariance_ellipses  # noqa: E402

axes[0].scatter(X[:, 0], y, s=8, c=[PALETTE[i % len(PALETTE)] for i in gk_ts.clusterer_.labels_])
plot_covariance_ellipses(gk_ts.clusterer_.cluster_centers_, gk_ts.clusterer_.covariances_, ax=axes[0])
axes[0].set_title("GK clusters in the product space $(x, y)$")
axes[1].scatter(X[:, 0], y, s=8, c=[PALETTE[i % len(PALETTE)] for i in ts.clusterer_.labels_])
local = ts.local_predict(grid)
beta = ts.firing_strengths(grid)
for i in range(4):
    strong = beta[:, i] > 0.05
    axes[1].plot(grid[strong, 0], local[strong, i], color=PALETTE[i % len(PALETTE)], lw=2.5)
axes[1].set_title("Modified GG clusters: Gaussian in $x$, linear model in $y$")
for ax in axes:
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
plt.tight_layout()

# %%
# Consequent estimation: local, global, or total least squares
# ------------------------------------------------------------
# With the antecedents fixed, the consequents can be estimated in three ways.
# *Local* weighted least squares (one problem per rule, with the memberships
# as weights) gives local linearizations of the function, which are
# interpretable but not the best global fit. *Global* least squares
# estimates all consequents in one problem with the normalized firing
# strengths as regressors; it gives the smallest prediction error, but the
# local models lose their meaning where rules overlap. *Total* least squares
# takes the hyperplane on which the product-space cluster lies (the
# eigenvector of its smallest eigenvalue).
rows = []
for consequent in ("local", "global", "tls"):
    model = TakagiSugeno(n_clusters=4, consequent=consequent, random_state=0).fit(X, y)
    rows.append(
        {
            "consequent": consequent,
            "train RMSE": np.sqrt(np.mean((model.predict(X) - y) ** 2)),
            "test RMSE": np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)),
            "slopes": np.round(model.coef_[:, 0], 3),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))

# %%
# Parameter study: the number of rules
# ------------------------------------
# More rules give a finer piecewise-linear approximation. Both errors fall
# with the number of rules, though not monotonically: the five-rule model is
# worse than the four-rule one, because the clustering, not the prediction
# error, places the rules. With 300 samples, the error on the noise-free test
# data keeps falling up to twelve rules (to about 0.016, below the noise
# level of 0.02), so in this range the rules do not yet fit the noise.

# [snippet:n_rules]
n_rules = [2, 3, 4, 5, 6, 8, 10, 12]
table = pd.DataFrame(
    {
        "rules": n_rules,
        "train RMSE": [
            np.sqrt(np.mean((TakagiSugeno(n_clusters=c, random_state=0).fit(X, y).predict(X) - y) ** 2))
            for c in n_rules
        ],
        "test RMSE": [
            np.sqrt(np.mean((TakagiSugeno(n_clusters=c, random_state=0).fit(X, y).predict(X_test) - y_test) ** 2))
            for c in n_rules
        ],
    }
)
# [/snippet:n_rules]
print(table.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(table["rules"], table["train RMSE"], "o-", color=PALETTE[0], label="training data (noisy)")
ax.plot(table["rules"], table["test RMSE"], "s-", color=PALETTE[1], label="test data (noise-free)")
ax.axhline(0.02, color="0.5", ls=":", label="noise level")
ax.set_xlabel("number of rules $c$")
ax.set_ylabel("RMSE")
ax.set_title("Approximation error versus number of rules")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# **Take-away.** Product-space clustering turns every cluster into a rule: the
# cluster's extent along the inputs is the antecedent, and the linear law
# hidden in the cluster is the consequent. Modified Gath–Geva clustering fits
# axis-parallel Gaussians with a local linear model inside, so the rules are
# exactly the clusters; GK or GG clustering of the product space needs a
# projection that introduces a decomposition error. Local weighted least
# squares keeps the consequents interpretable, whereas global least squares
# minimizes the prediction error. The number of rules is the main complexity
# parameter: four rules already give :math:`R^2\approx0.96`, six bring the
# test RMSE below 0.04, and up to twelve rules the model does not overfit
# these 300 samples.
