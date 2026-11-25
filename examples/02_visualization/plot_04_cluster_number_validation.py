r"""
Visual validation of the number of clusters with FUZZSAM
========================================================

Validity indices compress a partition into one number. A faithful 2-D
picture of the partition tells more: whether the clusters are compact
groups around their centers, whether two of them should be merged, or
whether a cluster merely collects what the others leave. Since the fuzzy
Sammon mapping shows the clustering as the algorithm sees it (every
prototype becomes a point, and the recomputed memberships match the fitted
ones), it is a natural tool for such user-guided validation. This example
clusters the Wisconsin breast cancer data (569 samples, the ten standardized
"mean" cell-nucleus features, two diagnoses) with FCM for :math:`c=2,3,4`
and projects every result with FUZZSAM, as in the FUZZ-IEEE paper. It also
reports the quality measures of the maps and the convergence of
:math:`E_\text{fuzz}` for several step sizes.
"""

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fuzzyclust import FCM
from fuzzyclust.datasets import load_breast_cancer
from fuzzyclust.validity import validity_indices
from fuzzyclust.viz import FuzzySammon, plot_projection, plot_stress

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data[:, :10])  # mean radius, texture, ... (the first ten columns)

# [snippet:scan]
results = {}
for c in (2, 3, 4):
    fcm = FCM(n_clusters=c, random_state=0, tol=1e-6).fit(X)
    fsam = FuzzySammon(max_iter=100, alpha=0.4).fit(fcm, X)
    results[c] = (fcm, fsam)
# [/snippet:scan]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (c, (_fcm, fsam)) in zip(axes, results.items(), strict=True):
    plot_projection(fsam.embedding_, fsam.embedded_centers_, labels=data.target, ax=ax, levels=6,
                    legend=(c == 2), label_prefix="diagnosis")
    q = fsam.quality()
    ax.set_title(f"FCM, c = {c}:  P = {q['P']:.3f},  F = {q['F']:.3f},  F* = {q['F_star']:.3f}")
plt.tight_layout()

# %%
# With :math:`c=2` the two groups are compact and their centers are far
# apart; the diagnosis (color) coincides with the clusters except for a band
# of ambiguous cases between them. With :math:`c=3` and :math:`c=4` the
# additional prototypes split the large benign group, the new clusters
# overlap in the plane, and the recomputed memberships fit worse (:math:`P`
# grows). PC, CE, and XB give the same verdict in numbers; SC, which
# decreases with :math:`c` by design, does not:
rows = {c: validity_indices(X, fcm.membership_, fcm.cluster_centers_, m=fcm.m, indices=["PC", "CE", "XB", "SC"])
        for c, (fcm, _) in results.items()}
table = pd.DataFrame(rows).T
table.index.name = "c"
print(table.round(3))

# %%
# Step size of the fuzzy Sammon iteration
# ---------------------------------------
# FUZZSAM uses the same diagonal Newton step as Sammon mapping. Only
# :math:`N\times c` distances enter the update, so the iterations are cheap
# even for large :math:`N`; the step size :math:`\alpha` controls how fast
# :math:`E_\text{fuzz}` decreases.
fcm2 = results[2][0]
# [snippet:alpha]
alphas = (0.1, 0.2, 0.4, 0.8)
runs = [FuzzySammon(max_iter=60, alpha=a, tol=0.0).fit(fcm2, X) for a in alphas]
# [/snippet:alpha]
ax = plot_stress([r.stress_history_ for r in runs], labels=[rf"$\alpha$ = {a}" for a in alphas],
                 ylabel=r"$E_\mathrm{fuzz}$")
ax.set_title("convergence of the fuzzy Sammon stress")
plt.tight_layout()

# %%
# Parameter study: more clusters, softer partitions
# --------------------------------------------------
# The scan is extended to :math:`c=2,\dots,6` and to three exponents. Two
# quantities are read from the maps: the fidelity :math:`P` and the
# distance between the two closest projected centers, which is what the eye
# uses to decide whether two clusters should be merged. Both agree with the
# Xie–Beni index. The centers of the two-cluster partition are far apart,
# every additional prototype lands closer to an existing one, and for a
# soft partition (:math:`m=2.5`) with six clusters two prototypes collapse
# onto each other: the projected centers coincide and XB explodes. The
# exponent changes the softness (:math:`F`) but not the ranking of
# :math:`c`.

# [snippet:c_m_scan]
from scipy.spatial.distance import pdist

m_values, c_values = (1.5, 2.0, 2.5), (2, 3, 4, 5, 6)
rows = []
for m in m_values:
    for c in c_values:
        fcm_c = FCM(n_clusters=c, m=m, random_state=0, tol=1e-6).fit(X)
        fsam_c = FuzzySammon(max_iter=100, alpha=0.4).fit(fcm_c, X)
        q = fsam_c.quality()
        v = validity_indices(X, fcm_c.membership_, fcm_c.cluster_centers_, m=m, indices=["XB"])
        rows.append({"m": m, "c": c, "P": q["P"], "F": q["F"], "XB": v["XB"],
                     "closest centers": pdist(fsam_c.embedded_centers_).min()})
scan = pd.DataFrame(rows)
# [/snippet:c_m_scan]
print(scan.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for m in m_values:
    sub = scan[scan["m"] == m]
    axes[0].plot(sub["c"], sub["closest centers"], "o-", label=f"m = {m}")
    axes[1].plot(sub["c"], sub["XB"], "o-", label=f"m = {m}")
axes[0].set_xlabel("number of clusters $c$")
axes[0].set_ylabel("distance of the two closest projected centers")
axes[0].set_title("every added prototype lands closer to an existing one")
axes[1].set_xlabel("number of clusters $c$")
axes[1].set_ylabel("Xie–Beni index (log scale)")
axes[1].set_yscale("log")
axes[1].set_title("the Xie–Beni index tells the same story in one number")
for ax in axes:
    ax.set_xticks(c_values)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Take-away
# ---------
# A validity index condenses a partition into one number; a faithful map
# shows *why* that number is good or bad. Because FUZZSAM places every
# prototype as a point and reproduces the fitted memberships (:math:`P`
# small, :math:`F^*\approx F`), the picture can be trusted. Compact groups
# around distant centers support the partition; overlapping contours or
# prototypes that fall onto each other argue for fewer clusters. On the
# breast cancer data, the two-cluster solution is the only one in which the
# clusters are clearly separated, and the diagnosis labels confirm it; the
# fuzzy memberships flag the band of ambiguous cases between them. A map
# costs :math:`N\times c` distances per iteration, so this kind of visual
# validation remains affordable for large data sets, and the step size only
# trades speed against smoothness of convergence.
