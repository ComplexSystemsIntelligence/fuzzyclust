r"""
Gustafson–Kessel: clusters with their own shape
===============================================

FCM measures all distances with one fixed norm, so it can only find clusters of
one shape (spheres for the Euclidean norm). The Gustafson–Kessel (GK) algorithm
gives every cluster its own norm-inducing matrix

.. math::
    \mathbf{A}_i = \left[\rho_i\det(\mathbf{F}_i)\right]^{1/n}\mathbf{F}_i^{-1},

the scaled inverse of the fuzzy covariance matrix :math:`\mathbf{F}_i`. Each
cluster therefore adapts its orientation and elongation to the local data,
while the determinant constraint :math:`\det\mathbf{A}_i=\rho_i` fixes its
volume. This example compares FCM and GK on elongated clusters and shows the
effect of the volume parameter :math:`\rho_i` and of the covariance
regularization introduced by Babuška, van der Veen, and Kaymak.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM, GK
from fuzzyclust.datasets import make_ellipses
from fuzzyclust.plotting import plot_covariance_ellipses, plot_membership_contours

X, y_true = make_ellipses(n_clusters=3, n_samples_per_cluster=120, axes=(3.0, 0.35), random_state=0)

fcm = FCM(n_clusters=3, random_state=0).fit(X)
gk = GK(n_clusters=3, random_state=0).fit(X)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, model, name in zip(axes, (fcm, gk), ("FCM", "GK"), strict=False):
    plot_membership_contours(model, X, ax=ax, labels=False)
    plot_covariance_ellipses(model.cluster_centers_, model.covariances_, ax=ax)
    ari = adjusted_rand_score(y_true, model.labels_)
    ax.set_title(f"{name}: ARI = {ari:.2f}")
plt.tight_layout()

# %%
# The contour lines of GK are ellipses aligned with the clusters; those of FCM
# are circles that cut across the elongated groups. The 2σ ellipses of the
# fuzzy covariance matrices (colored) summarize the shape each algorithm
# assigns to the clusters. The eigendecomposition of :math:`\mathbf{F}_i` is
# available for further analysis: the eigenvector of the smallest eigenvalue
# is the normal direction of the cluster, the basis of clustering-based local
# linear modeling.
for i in range(3):
    lam = gk.eigenvalues_[i]
    print(f"cluster {i + 1}: eigenvalues {lam.round(3)}, elongation ratio {np.sqrt(lam[1] / lam[0]):.1f}")

# %%
# Cluster volumes :math:`\rho_i`
# ------------------------------
# GK assumes equal volumes (:math:`\rho_i=1`). If one cluster is known to be
# larger, its :math:`\rho_i` can be increased, and the membership contours of
# that cluster expand. Here the volume of the first cluster is varied; the
# effect is visible, but the partition of well-separated clusters hardly
# changes.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
for ax, rho1 in zip(axes, (0.2, 1.0, 5.0), strict=False):
    model = GK(n_clusters=3, rho=[rho1, 1.0, 1.0], init=gk.cluster_centers_).fit(X)
    plot_membership_contours(model, X, ax=ax, labels=False)
    ax.set_title(rf"$\rho_1$ = {rho1}")
plt.tight_layout()

# %%
# Covariance regularization (:math:`\gamma`, :math:`\beta`)
# ------------------------------------------------------------
# When a cluster has few points or the data lie almost on a line, the fuzzy
# covariance matrix becomes (nearly) singular and :math:`\mathbf{F}_i^{-1}`
# blows up. Two safeguards are available: shrinkage toward the scaled
# identity, :math:`\mathbf{F}_i\leftarrow(1-\gamma)\mathbf{F}_i+\gamma\det(\mathbf{F}_0)^{1/n}\mathbf{I}`,
# and a bound :math:`\beta` on the condition number. We create an extreme
# case, a perfectly collinear cluster, and show that the regularized GK
# still produces a sensible partition.
rng = np.random.RandomState(1)
line = np.c_[np.linspace(-4, 4, 80), 0.5 * np.linspace(-4, 4, 80) + 9]  # exactly collinear
X2 = np.vstack([X, line])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
settings = [dict(gamma=0.0, beta=1e15), dict(gamma=0.1, beta=1e4)]
for ax, kw in zip(axes, settings, strict=False):
    model = GK(n_clusters=4, random_state=0, **kw).fit(X2)
    plot_membership_contours(model, X2, ax=ax, labels=False)
    cond = [np.linalg.cond(F) for F in model.covariances_]
    ax.set_title(f"gamma={kw['gamma']}, beta={kw['beta']:.0e}\nmax cond(F_i) = {max(cond):.1e}")
plt.tight_layout()

# %%
# Without regularization the collinear cluster yields a covariance matrix
# whose condition number (about 6000) is two orders of magnitude above that
# of the other clusters (50 to 80), and a razor-thin membership region.
# With :math:`\gamma=0.1` the cluster gets a finite width, but the shrinkage
# also rounds the other ellipses. The bound :math:`\beta=10^4` is not
# active here, because no condition number reaches it; the parameter study
# below shows that a tighter bound repairs the collinear cluster alone.

# %%
# Parameter study: the amount of regularization
# ---------------------------------------------
# Both safeguards trade adaptivity for numerical stability, but in different
# ways. Shrinkage with :math:`\gamma` acts on *every* cluster: it shortens
# all elongation ratios :math:`\sqrt{\lambda_{\max}/\lambda_{\min}}` of the
# fuzzy covariance matrices, and for :math:`\gamma\to1` GK degenerates into
# FCM with round clusters. The condition-number bound :math:`\beta` only
# clips the eigenvalues of the clusters that violate it, so the three
# well-conditioned ellipses keep their shape while the collinear cluster
# gets a finite width. The agreement with the four generating groups (the
# three ellipses and the line) shows the price of over-regularization.

# [snippet:regularisation_scan]
y2 = np.r_[y_true, np.full(len(line), 3)]
settings = [dict(gamma=g, beta=1e15) for g in (0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.0)]
settings += [dict(gamma=0.0, beta=b) for b in (1e1, 1e2, 1e3, 1e4)]
print(" gamma     beta   max cond(F_i)   elongation min..max    ARI")
for kw in settings:
    model = GK(n_clusters=4, random_state=0, **kw).fit(X2)
    cond = max(np.linalg.cond(F) for F in model.covariances_)
    elong = np.sqrt(model.eigenvalues_[:, 1] / model.eigenvalues_[:, 0])
    print(f"{kw['gamma']:6.3f}  {kw['beta']:7.0e}  {cond:13.1e}   {elong.min():6.1f} .. {elong.max():5.1f}"
          f"  {adjusted_rand_score(y2, model.labels_):6.3f}")
# [/snippet:regularisation_scan]

# %%
# Take-away
# ---------
# Gustafson--Kessel keeps the c-means functional and replaces the single
# global norm by one adaptive norm per cluster, so every cluster can be an
# ellipsoid with its own orientation and elongation while the determinant
# constraint keeps the volumes fixed. The volume parameters :math:`\rho_i`
# are rarely worth tuning, because the partition of separated clusters is
# insensitive to them, but the regularization matters as soon as a cluster
# is thin or small. A little shrinkage (:math:`\gamma\le 0.01`) or a bound on
# the condition number (:math:`\beta` around :math:`10^2`--:math:`10^3`) is
# enough to give a degenerate cluster a finite width. A large :math:`\gamma`
# rounds off *all* clusters, and GK loses exactly the shape adaptivity that
# distinguishes it from FCM; this is why :math:`\beta` is the gentler of the
# two safeguards.
