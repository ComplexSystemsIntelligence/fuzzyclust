r"""
Fuzzy c-shells: circles as prototypes
=====================================

A ring has no interior points, so a point prototype placed at its center is
far from every member, and FCM cannot tell concentric rings apart at all.
Shell clustering describes cluster :math:`i` by a center
:math:`\mathbf{v}_i` *and* a radius :math:`r_i` and measures the distance of
a sample as its deviation from the circle. The fuzzy c-spherical shells
algorithm (FCSS) of Krishnapuram, Nasraoui and Frigui uses the *algebraic*
distance :math:`(\|\mathbf{x}_k-\mathbf{v}_i\|^2-r_i^2)^2`, which is quadratic
in a reparametrized prototype and therefore has a closed-form weighted
least-squares update; Davé's original fuzzy c-shells (FCS) uses the
*radial* distance :math:`(\|\mathbf{x}_k-\mathbf{v}_i\|-r_i)^2` and needs a
Newton iteration per cluster. The example recovers the radii of concentric
and of intersecting circles, compares the two algorithms, and shows why the
initialization matters.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.datasets import make_rings
from fuzzyclust.extensions import FuzzyCShells, make_circles, plot_shells
from fuzzyclust.plotting import PALETTE, plot_membership_contours

Xr, yr = make_rings(n_samples_per_ring=150, radii=(1.0, 3.0), noise=0.1, random_state=0)
Xc, yc = make_circles(centers=((0.0, 0.0), (3.0, 1.0), (1.0, 3.5)), radii=(2.0, 1.2, 1.0),
                      n_samples_per_circle=120, noise=0.06, random_state=0)
true_c = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.5]])
true_r = np.array([2.0, 1.2, 1.0])


def match(model, centers, radii):
    """Center and radius errors after matching the shells to the true circles."""
    D = np.linalg.norm(model.cluster_centers_[:, None] - centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max(), np.abs(model.radii_[r] - radii[c]).max()


# %%
# Concentric rings and intersecting circles
# -----------------------------------------
# FCM halves the rings and finds nothing on the intersecting circles either.
# FCSS recovers the radii 1 and 3 of the rings to two decimals, and on three
# circles that intersect and partly cover each other it finds the centers
# and radii to within a few hundredths; points near an intersection share
# their membership. The concentric rings need several restarts (see the
# parameter study), whereas on the intersecting circles the hardened FCM
# start is already good enough.

# [snippet:fit]
fcm_r = FCM(n_clusters=2, random_state=0).fit(Xr)
fcss_r = FuzzyCShells(n_clusters=2, algorithm="fcss", n_init=5, random_state=0).fit(Xr)
fcss_c = FuzzyCShells(n_clusters=3, algorithm="fcss", random_state=0).fit(Xc)
print(f"rings: FCM ARI = {adjusted_rand_score(yr, fcm_r.labels_):.2f}, FCSS ARI = "
      f"{adjusted_rand_score(yr, fcss_r.labels_):.2f}, radii found {np.sort(fcss_r.radii_).round(2)} (true 1, 3)")
ce, re = match(fcss_c, true_c, true_r)
print(f"circles: FCSS ARI = {adjusted_rand_score(yc, fcss_c.labels_):.2f}, "
      f"largest center error {ce:.3f}, largest radius error {re:.3f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
plot_membership_contours(fcm_r, Xr, ax=axes[0], labels=False)
axes[0].set_aspect("equal")
axes[0].set_title(f"FCM on two rings: ARI = {adjusted_rand_score(yr, fcm_r.labels_):.2f}")
plot_shells(Xr, fcss_r, ax=axes[1])
axes[1].set_title(f"FCSS on two rings: radii {np.sort(fcss_r.radii_).round(2)}")
plot_shells(Xc, fcss_c, ax=axes[2])
axes[2].set_title(f"FCSS on three intersecting circles: ARI = {adjusted_rand_score(yc, fcss_c.labels_):.2f}")
plt.tight_layout()

# %%
# The membership of a point in a shell depends only on its radial deviation
# from the circle, so the contours of the maximal membership are
# rings around the fitted circles, and the membership drops toward the
# center of a circle as well as away from it.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plot_membership_contours(fcss_r, Xr, ax=axes[0], labels=False)
for i, (v, r) in enumerate(zip(fcss_r.cluster_centers_, fcss_r.radii_, strict=True)):
    axes[0].add_patch(plt.Circle(v, r, fill=False, lw=2, color=PALETTE[i]))
axes[0].set_aspect("equal")
axes[0].set_title("FCSS: maximal membership around the two rings")
E = fcss_c.radial_deviation(Xc)
own = E[np.arange(len(Xc)), fcss_c.labels_]
for i in range(3):
    axes[1].hist(own[fcss_c.labels_ == i], bins=25, color=PALETTE[i], alpha=0.7, label=f"shell {i + 1}")
axes[1].set_xlabel(r"signed radial deviation $\|\mathbf{x}_k-\mathbf{v}_i\|-r_i$ from the own shell")
axes[1].set_ylabel("samples")
axes[1].set_title("Radial residuals of the intersecting circles (noise 0.06)")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study I: initialization and restarts
# ----------------------------------------------
# Shell clustering has a notorious local minimum: one big circle through
# everything, with the other shells fitting noise. The default start, the
# hardened FCM partition, works well when the shells are separate or
# intersect (the FCM groups then lie on different circles), but on
# *concentric* rings FCM halves every ring and both shells converge to the
# mean radius. Random restarts (``n_init``) escape this trap because the
# best run by the objective is kept: with ten restarts the rings are found
# from every seed. On concentric rings, random point prototypes need fewer
# restarts than the FCM start, which spends its first run on the halved
# rings.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, n_init in zip(axes[:2], (1, 5), strict=True):
    model = FuzzyCShells(n_clusters=2, n_init=n_init, random_state=0).fit(Xr)
    plot_shells(Xr, model, ax=ax)
    ax.set_title(f"rings, init='fcm', n_init = {n_init}: radii {np.sort(model.radii_).round(2)}, J = {model.objective_:.1f}")
seeds = range(10)
n_inits = (1, 3, 5, 10)
for j, (init, color) in enumerate((("fcm", PALETTE[0]), ("random", PALETTE[1]), ("kmeans++", PALETTE[2]))):
    means, lows, highs = [], [], []
    for n_init in n_inits:
        aris = [adjusted_rand_score(yr, FuzzyCShells(n_clusters=2, init=init, n_init=n_init, random_state=s).fit(Xr).labels_)
                for s in seeds]
        means.append(np.mean(aris))
        lows.append(np.min(aris))
        highs.append(np.max(aris))
    pos = np.array(n_inits) * (1.0 + 0.08 * (j - 1))  # small horizontal offset per initialization
    axes[2].errorbar(pos, means, yerr=[np.array(means) - lows, np.array(highs) - means], fmt="o-", color=color,
                     capsize=3, lw=1.2, label=f"init = '{init}'")
    print(f"rings, init = '{init}': mean ARI over 10 seeds for n_init = {n_inits}: {np.round(means, 2)}")
axes[2].set_xscale("log")
axes[2].set_xticks(n_inits, [str(n) for n in n_inits])
axes[2].set_xlabel("number of restarts n_init")
axes[2].set_ylabel("adjusted Rand index (mean and range over 10 seeds)")
axes[2].set_title("Rings: how often the two shells are found")
axes[2].legend(frameon=False, fontsize=8)
axes[2].grid(True, color="0.9")
plt.tight_layout()
print("intersecting circles, n_init = 1: ARI with init='fcm' "
      f"{adjusted_rand_score(yc, FuzzyCShells(n_clusters=3, init='fcm', random_state=0).fit(Xc).labels_):.2f}, "
      f"init='random' {adjusted_rand_score(yc, FuzzyCShells(n_clusters=3, init='random', random_state=0).fit(Xc).labels_):.2f}")

# %%
# Parameter study II: FCSS versus FCS, and the exponent :math:`m`
# -----------------------------------------------------------------
# Near a shell the algebraic distance is about :math:`2r_i` times the
# radial one, so FCSS weights the same deviation more on a large shell than
# on a small one, and its memberships favor small shells. On well-separated
# shells both algorithms give the same answer; with three concentric rings
# of radii 1, 1.6 and 3 the bias shows: FCS separates the two inner rings
# more reliably, at the price of the Newton steps. The exponent :math:`m`
# plays its usual role and hardly changes the difference between FCSS and
# FCS.
X3, y3 = make_rings(n_samples_per_ring=120, radii=(1.0, 1.6, 3.0), noise=0.06, random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, algorithm in zip(axes, ("fcss", "fcs"), strict=True):
    model = FuzzyCShells(n_clusters=3, algorithm=algorithm, n_init=10, random_state=0).fit(X3)
    plot_shells(X3, model, ax=ax)
    ax.set_title(f"{algorithm.upper()} on three rings: radii {np.sort(model.radii_).round(2)}, "
                 f"ARI = {adjusted_rand_score(y3, model.labels_):.2f}")
plt.tight_layout()
print("three concentric rings, ARI (mean over 5 seeds, n_init = 10):")
for algorithm in ("fcss", "fcs"):
    aris = [adjusted_rand_score(y3, FuzzyCShells(n_clusters=3, algorithm=algorithm, n_init=10, random_state=s).fit(X3).labels_)
            for s in range(5)]
    print(f"  {algorithm}: {np.mean(aris):.2f} (min {np.min(aris):.2f})")
print("intersecting circles: center / radius error and memberships against m")
for m in (1.3, 2.0, 3.0):
    for algorithm in ("fcss", "fcs"):
        model = FuzzyCShells(n_clusters=3, m=m, algorithm=algorithm, random_state=0).fit(Xc)
        ce, re = match(model, true_c, true_r)
        print(f"  m = {m}, {algorithm}: center error {ce:.3f}, radius error {re:.3f}, "
              f"mean max membership {model.membership_.max(axis=1).mean():.2f}")

# %%
# **Take-away.** Replacing the point prototype by a center and a radius
# lets the c-means functional describe hollow clusters; the membership
# formula is unchanged and depends on the radial deviation from the shell.
# FCSS with the algebraic distance has a closed-form update and is the
# default; FCS with the radial distance is unbiased with respect to the
# shell size and worth its Newton steps when nested shells of similar
# radius must be separated. Start from the hardened FCM partition for
# separate or intersecting shells, and use restarts for concentric ones.
