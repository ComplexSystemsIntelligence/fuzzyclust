r"""
PCA versus Sammon mapping: preserving variance or distances
===========================================================

A clustering of :math:`n`-dimensional data cannot be inspected directly; the
classical remedy is to project the data onto the plane. Principal component
analysis (PCA) chooses the plane that retains as much *variance* as possible,
which says nothing about the distances the clustering algorithm works with.
Sammon's mapping instead places the :math:`N` points in the plane so that their
pairwise distances :math:`d^*_{kl}` reproduce the original distances
:math:`d_{kl}`; it minimizes the stress

.. math::
    E = \frac{1}{\lambda}\sum_{k<l}\frac{(d_{kl}-d^*_{kl})^2}{d_{kl}},
    \qquad \lambda=\sum_{k<l}d_{kl}.

This example projects five-dimensional Gaussian clusters with both methods,
compares the stress, and shows how the step size :math:`\alpha` and the
initialization affect the convergence of the Sammon iteration.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fuzzyclust import FCM
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.viz import Sammon, pca_projection, plot_projection, plot_stress, sammon_stress

# five clusters in five dimensions; they are not far apart, so the 2-D
# picture must be chosen with care
X, y = make_fuzzy_blobs(n_clusters=5, n_samples_per_cluster=60, n_features=5, spread=0.9, box=8.0, random_state=0)
fcm = FCM(n_clusters=5, random_state=0).fit(X)

# [snippet:pca_sammon]
Y_pca, Z_pca, pca = pca_projection(X, n_components=2, centers=fcm.cluster_centers_)
sammon = Sammon(n_components=2, max_iter=100, alpha=0.3, init="pca").fit(X)
Y_sam = sammon.embedding_
# the projected prototypes of a Euclidean clustering are the fuzzy means of the projected data
Um = fcm.membership_**fcm.m
Z_sam = (Um.T @ Y_sam) / Um.sum(axis=0)[:, None]
# [/snippet:pca_sammon]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plot_projection(Y_pca, Z_pca, labels=fcm.labels_, contours=False, ax=axes[0])
axes[0].set_title(f"PCA: {100 * pca.explained_variance_ratio_.sum():.0f}% of the variance, "
                  f"stress E = {sammon_stress(X, Y_pca):.3f}")
plot_projection(Y_sam, Z_sam, labels=fcm.labels_, contours=False, ax=axes[1])
axes[1].set_title(f"Sammon mapping: stress E = {sammon.stress_:.3f} after {sammon.n_iter_} iterations")
plt.tight_layout()

# %%
# PCA keeps the two directions of largest spread; clusters that differ mainly
# in the discarded directions fall on top of each other. Sammon mapping starts
# from the PCA picture and bends it until the inter-point distances match, so
# the stress drops and the five groups separate. The price is a nonlinear,
# data-dependent map that cannot be applied to new samples.
#
# Step size :math:`\alpha`
# ------------------------
# The update :math:`y_{kl}\leftarrow y_{kl}-\alpha\,
# (\partial E/\partial y_{kl})/|\partial^2E/\partial y_{kl}^2|` is a diagonal
# Newton step scaled by the "magic factor" :math:`\alpha`. Sammon recommended
# 0.3–0.4; smaller values converge slowly and larger ones overshoot.
# [snippet:alpha]
alphas = (0.1, 0.3, 0.5, 0.9)
runs = [Sammon(alpha=a, max_iter=60, init="pca", tol=0.0).fit(X) for a in alphas]
# [/snippet:alpha]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
plot_stress([r.stress_history_ for r in runs], labels=[rf"$\alpha$ = {a}" for a in alphas], ax=axes[0])
axes[0].set_title("convergence of the stress for different step sizes")

# %%
# Initialization
# --------------
# The stress surface has local minima. Starting from the PCA scores
# (``init="pca"``) is deterministic and usually close to a good minimum;
# random starts need more iterations and end in different configurations.
rand_runs = [Sammon(alpha=0.3, max_iter=60, init="random", tol=0.0, random_state=s).fit(X) for s in range(3)]
plot_stress([runs[1].stress_history_] + [r.stress_history_ for r in rand_runs],
            labels=["PCA start"] + [f"random start {s + 1}" for s in range(3)], ax=axes[1])
axes[1].set_title("PCA versus random initialization")
plt.tight_layout()

for name, r in zip(["pca"] + [f"random {s}" for s in range(3)], [runs[1]] + rand_runs, strict=True):
    print(f"{name:10s}: final stress {r.stress_:.4f}")

# %%
# The random starts reach stresses of the same order but not the same
# minimum, and their maps are rotated and reflected versions of each other.
# Since only distances matter, rotations and reflections do not affect the
# interpretation; what distinguishes the initializations is the number of
# iterations they need.
print("stress of the data in the first 2 PCA directions:", round(float(sammon_stress(X, Y_pca)), 4))
print("number of distances per Sammon iteration: N(N-1)/2 =", X.shape[0] * (X.shape[0] - 1) // 2)
print("per fuzzy-Sammon iteration:                N*c      =", X.shape[0] * 5)

# %%
# Parameter study: step size with and without backtracking, and the dimension
# ----------------------------------------------------------------------------
# Sammon's recommendation of :math:`\alpha\approx0.3` applies to the plain
# iteration. :class:`~fuzzyclust.viz.Sammon` halves the step whenever the
# stress would increase (``backtracking=True``, the default). This makes the
# iteration monotone for every :math:`\alpha` and lets large steps converge
# fastest. With ``backtracking=False`` the classical behavior returns: slow
# convergence for small :math:`\alpha` and divergence for :math:`\alpha\ge0.9`.
# The other parameter is the number of components: one dimension cannot
# reproduce the distances of five clusters, but three dimensions almost can.

# [snippet:alpha_scan]
print("alpha   backtracking   final stress   iterations to within 1% of final   monotone")
for a in (0.05, 0.1, 0.3, 0.5, 0.9, 1.2):
    for backtracking in (True, False):
        run = Sammon(alpha=a, max_iter=60, init="pca", tol=0.0, backtracking=backtracking).fit(X)
        h = np.asarray(run.stress_history_)
        n_within = "diverged" if h[-1] > 2 * h.min() else str(int(np.argmax(h <= h[-1] * 1.01)))
        print(f"{a:5.2f}   {str(backtracking):12s}   {run.stress_:12.4g}   {n_within:>32s}   {np.all(np.diff(h) <= 1e-12)}")

print("\ncomponents   PCA stress   Sammon stress   variance kept by PCA")
for n_components in (1, 2, 3):
    Y_p, _, pca_n = pca_projection(X, n_components=n_components)
    run = Sammon(n_components=n_components, alpha=0.3, max_iter=100, init="pca").fit(X)
    print(f"{n_components:10d}   {sammon_stress(X, Y_p):10.4f}   {run.stress_:13.4f}   {pca_n.explained_variance_ratio_.sum():20.2f}")
# [/snippet:alpha_scan]

# %%
# Take-away
# ---------
# PCA and Sammon mapping answer different questions. PCA keeps the
# directions of largest variance and is a linear map that can be applied to
# new data. Sammon mapping keeps the pairwise distances and yields a
# data-dependent configuration of :math:`N` points, faithful to what a
# distance-based clustering algorithm "sees" but not transferable to new
# samples. The Sammon stress is a genuine quality measure that can score any
# projection, including the PCA scores; here the Sammon map halves the
# stress of the PCA picture. The iteration is a diagonal Newton method whose
# step size matters only without the safeguard: with backtracking any
# :math:`\alpha` works, and the PCA start removes the dependence on a random
# initialization.
