r"""
Imputation by clustering: accuracy of the completed values
==========================================================

The optimal completion strategy (OCS) of Hathaway and Bezdek treats every
missing value :math:`x_{kj}` as a variable of the c-means functional and
sets it to the membership-weighted mean of the prototype coordinates,
:math:`x_{kj}=\sum_i\mu_{ik}^m v_{ij}/\sum_i\mu_{ik}^m`. Clustering thus
delivers an imputation of the data as a by-product. The same formula can
be applied *after* a partial-distance (PDS) fit, with memberships that
were computed from the observed features only, and the nearest-prototype
strategy (NPS) copies the coordinate of the nearest prototype. This
example measures how close the completed values come to the truth. Four
Gaussian clusters in four dimensions are used because a cluster is then
identified by some features better than by others: a sample that lost
exactly its discriminating features is ambiguous, and how a strategy
treats such samples decides its imputation error. Two baselines put the
numbers in perspective: the column mean (what a clustering-free imputer
would do) and the *oracle*, the mean of the sample's true cluster, which
is the best any prototype-based imputation can achieve.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.incomplete import IncompleteFCM, make_mcar, plot_imputation
from fuzzyclust.plotting import PALETTE

X, y = make_fuzzy_blobs(n_clusters=4, n_samples_per_cluster=150, n_features=4, random_state=0)
true_means = np.array([X[y == i].mean(axis=0) for i in range(4)])


def rmse(X_completed, mask):
    """Root mean squared error of the imputed entries."""
    return float(np.sqrt(np.mean((X_completed[mask] - X[mask]) ** 2)))


def baselines(Xm, mask):
    """Column-mean imputation and the oracle (mean of the true cluster)."""
    col_mean = np.where(mask, np.nanmean(Xm, axis=0), Xm)
    oracle = np.where(mask, true_means[y], Xm)
    return rmse(col_mean, mask), rmse(oracle, mask)


# %%
# Imputing 20 % missing values
# -----------------------------
# The completed matrix is the attribute ``X_completed_``. For OCS and NPS
# it holds the values the algorithm optimized; for PDS it is the post-hoc
# estimate from the final partition. OCS is run twice: once from a random
# start and once from the prototypes of the PDS fit, as the
# ``IncompleteFCM`` documentation recommends. The scatter plots compare
# imputed with true values, one color per feature: points on the diagonal
# are perfect, horizontal bands are samples imputed with the coordinate
# of a wrong prototype.

# [snippet:impute]
Xm, mask = make_mcar(X, missing_rate=0.2, random_state=0, return_mask=True)
pds = IncompleteFCM(n_clusters=4, strategy="pds", random_state=0).fit(Xm)
models = {
    "PDS (post hoc)": pds,
    "NPS": IncompleteFCM(n_clusters=4, strategy="nps", random_state=0).fit(Xm),
    "OCS, random start": IncompleteFCM(n_clusters=4, strategy="ocs", random_state=0).fit(Xm),
    "OCS, PDS start": IncompleteFCM(n_clusters=4, strategy="ocs", init=pds.cluster_centers_).fit(Xm),
}
errors = {name: rmse(model.X_completed_, mask) for name, model in models.items()}
# [/snippet:impute]
mean_err, oracle_err = baselines(Xm, mask)
print({name: round(e, 3) for name, e in errors.items()}, "column mean:", round(mean_err, 3),
      "oracle:", round(oracle_err, 3))

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
for ax, (name, model) in zip(axes.ravel(), models.items(), strict=True):
    plot_imputation(X, model.X_completed_, mask, ax=ax)
    ax.set_title(f"{name}: RMSE {errors[name]:.2f}")
fig.suptitle(f"20 % missing; oracle RMSE {oracle_err:.2f}, column mean {mean_err:.2f}", fontsize=11)
plt.tight_layout()

# %%
# Why OCS can be trapped
# -----------------------
# OCS and the post-hoc PDS imputation use the same formula; they differ in
# the memberships. PDS computes them from the observed features only, so a
# sample whose observed features do not tell two clusters apart keeps an
# ambiguous membership. In OCS the imputed values take part in the
# distances: once a missing value leans toward one prototype, the sample
# looks closer to it, its membership rises, and the next imputation leans
# further, so the completion confirms itself, right or wrong. From a random
# start, where the first prototypes are poor, many samples are captured
# this way. The left panel compares the memberships of the incomplete
# samples in the randomly started OCS with those of PDS: OCS is far more
# confident. The right panel shows the imputation error against the PDS
# membership: the large OCS errors sit on samples that PDS considers
# ambiguous, and they disappear when OCS starts from the PDS prototypes.
incomplete = mask.any(axis=1)
ocs, warm = models["OCS, random start"], models["OCS, PDS start"]
conf_pds = pds.membership_[incomplete].max(axis=1)
conf_ocs = ocs.membership_[incomplete].max(axis=1)
max_err = {name: np.abs(model.X_completed_ - X)[incomplete].max(axis=1) for name, model in models.items()}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(conf_pds, conf_ocs, s=14, color=PALETTE[1], alpha=0.7)
axes[0].plot([0.25, 1], [0.25, 1], color="0.3", ls="--", lw=1)
axes[0].set_xlabel(r"PDS: $\max_i \mu_{ik}$")
axes[0].set_ylabel(r"OCS, random start: $\max_i \mu_{ik}$")
axes[0].set_title("Membership of the incomplete samples")
for name, col in (("OCS, random start", PALETTE[1]), ("OCS, PDS start", PALETTE[3]), ("PDS (post hoc)", PALETTE[0])):
    axes[1].scatter(conf_pds, max_err[name], s=12, color=col, alpha=0.7, label=name)
axes[1].set_xlabel(r"PDS: $\max_i \mu_{ik}$ (low = ambiguous sample)")
axes[1].set_ylabel("largest absolute imputation error")
axes[1].set_title("Imputation error against ambiguity")
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study: missing rate and fuzziness exponent
# -----------------------------------------------------
# Left: the imputation error against the missing rate, averaged over five
# random patterns of missing values. The column mean ignores the clusters
# and is far off at every rate; the oracle is flat because a cluster mean
# does not depend on how many values are missing. The clustering-based
# imputations lie in between and degrade as more samples become
# ambiguous; the randomly started OCS is clearly the worst of them, and
# the warm-started OCS is close to PDS. Right: the exponent :math:`m` at
# 30 % missing. A small :math:`m` makes the memberships nearly crisp, so
# ambiguous samples are imputed with the coordinates of one prototype,
# right or wrong; a large :math:`m` pulls all imputations toward the
# grand mean. For PDS the two effects balance in a shallow minimum near
# :math:`m=2`; the randomly started OCS gains steadily with :math:`m`,
# because softer memberships weaken the self-confirmation.

# [snippet:scan]
rates = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
seeds = range(5)
names = ("PDS (post hoc)", "NPS", "OCS, random start", "OCS, PDS start", "column mean", "oracle")


def impute_all(Xr, mr, m=2.0):
    """Imputation errors of the four clustering-based variants."""
    pds = IncompleteFCM(n_clusters=4, m=m, strategy="pds", random_state=0).fit(Xr)
    fits = {
        "PDS (post hoc)": pds,
        "NPS": IncompleteFCM(n_clusters=4, m=m, strategy="nps", random_state=0).fit(Xr),
        "OCS, random start": IncompleteFCM(n_clusters=4, m=m, strategy="ocs", random_state=0).fit(Xr),
        "OCS, PDS start": IncompleteFCM(n_clusters=4, m=m, strategy="ocs", init=pds.cluster_centers_).fit(Xr),
    }
    return {name: rmse(model.X_completed_, mr) for name, model in fits.items()}


scan = {name: np.zeros((len(rates), len(seeds))) for name in names}
for a, rate in enumerate(rates):
    for b, seed in enumerate(seeds):
        Xr, mr = make_mcar(X, missing_rate=rate, random_state=seed, return_mask=True)
        for name, err in impute_all(Xr, mr).items():
            scan[name][a, b] = err
        scan["column mean"][a, b], scan["oracle"][a, b] = baselines(Xr, mr)
# [/snippet:scan]

exponents = (1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
m_scan = {name: np.zeros((len(exponents), len(seeds))) for name in names[:4]}
for a, m in enumerate(exponents):
    for b, seed in enumerate(seeds):
        Xr, mr = make_mcar(X, missing_rate=0.3, random_state=seed, return_mask=True)
        for name, err in impute_all(Xr, mr, m=m).items():
            m_scan[name][a, b] = err

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
styles = {"PDS (post hoc)": ("o-", PALETTE[0]), "NPS": ("^-", PALETTE[2]), "OCS, random start": ("s-", PALETTE[1]),
          "OCS, PDS start": ("D-", PALETTE[3]), "column mean": ("x--", "0.45"), "oracle": (":", "0.2")}
for name in names:
    fmt, col = styles[name]
    axes[0].plot(rates, scan[name].mean(axis=1), fmt, color=col, label=name)
axes[0].set_xlabel("missing rate")
axes[0].set_title("Imputation error against the missing rate")
axes[0].legend(frameon=False, fontsize=8)
for name in m_scan:
    fmt, col = styles[name]
    axes[1].plot(exponents, m_scan[name].mean(axis=1), fmt, color=col, label=name)
axes[1].axhline(scan["oracle"][rates.index(0.3)].mean(), color="0.2", ls=":", label="oracle")
axes[1].set_xlabel("fuzziness exponent $m$")
axes[1].set_title("Effect of $m$ at 30 % missing")
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set_ylabel("RMSE of the imputed values")
    ax.grid(True, color="0.9")
plt.tight_layout()

print("RMSE at 30 % missing:", {name: round(float(scan[name][rates.index(0.3)].mean()), 3) for name in names})
for name in m_scan:
    print(f"{name:18s} RMSE over m:", m_scan[name].mean(axis=1).round(3))

# %%
# **Take-away.** Clustering is a strong imputer when the data have
# cluster structure: every strategy beats the column mean by a wide
# margin, because it fills a hole with the coordinate of the right
# group. The imputation is only as good as the memberships behind it.
# OCS lets the imputed values vote on the memberships; from a random
# start it is trapped by its own early imputations, while started from
# the PDS prototypes it is nearly as accurate as the post-hoc imputation
# of a PDS fit, which remains the most accurate and the cheapest. If the
# completed data are the goal, fit with PDS and read ``X_completed_`` (or
# warm-start OCS from the PDS fit), and keep :math:`m` moderate (about 2).
