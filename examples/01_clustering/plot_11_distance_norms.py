r"""
Distance norms: Euclidean, diagonal, and Mahalanobis fuzzy c-means
==================================================================

Fuzzy c-means measures every distance with one inner-product norm

.. math:: D^2_{ik\mathbf{A}} = (\mathbf{x}_k-\mathbf{v}_i)^T\mathbf{A}(\mathbf{x}_k-\mathbf{v}_i),

and the choice of the norm-inducing matrix :math:`\mathbf{A}` decides which
shape a cluster can have: :math:`\mathbf{A}=\mathbf{I}` gives spheres, a
diagonal :math:`\mathbf{A}` with the inverse feature variances gives
axis-parallel ellipsoids, and :math:`\mathbf{A}=\mathbf{F}^{-1}` with the
covariance of the whole data set gives ellipsoids with the orientation of the
data (the Mahalanobis norm). The norm is a *global* choice shared by all
clusters, and this is exactly what separates FCM from the Gustafson--Kessel
algorithm. This example applies the three norms to three elongated clusters
of the same orientation and then shows that a diagonal norm is equivalent to
scaling the features beforehand.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import FCM
from fuzzyclust.plotting import plot_covariance_ellipses, plot_membership_contours

rng = np.random.RandomState(0)
# three clusters with the same elongated, rotated covariance
theta = np.deg2rad(35.0)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = R @ np.diag([2.2, 0.35]) @ R.T          # square root of the common covariance
along, perp = R[:, 0], R[:, 1]              # major and minor axis directions
centers = np.array([0.0 * perp, 2.0 * perp + 3.0 * along, 4.0 * perp - 3.0 * along])
X = np.vstack([rng.randn(120, 2) @ S.T + c for c in centers])
y_true = np.repeat(np.arange(3), 120)

# %%
# The same data are clustered with the three built-in norms and with an
# explicit matrix: the inverse of the covariance from which the clusters were
# generated. The contour maps show the level curves of the maximal
# membership; their shape is the shape of the unit ball of the norm.
# [snippet:norms]
norms = {
    "euclidean": "euclidean",
    "diagonal": "diagonal",
    "mahalanobis": "mahalanobis",
    r"explicit $\mathbf{A}=\mathbf{\Sigma}^{-1}$": np.linalg.inv(S @ S.T),
}
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharex=True, sharey=True)
for ax, (name, norm) in zip(axes, norms.items(), strict=True):
    fcm = FCM(n_clusters=3, norm=norm, random_state=0, n_init=5).fit(X)
    plot_membership_contours(fcm, X, ax=ax, labels=False)
    plot_covariance_ellipses(fcm.cluster_centers_, fcm.covariances_, ax=ax)
    ax.set_title(f"{name}: ARI = {adjusted_rand_score(y_true, fcm.labels_):.2f}")
# [/snippet:norms]
plt.tight_layout()

# %%
# The Euclidean norm cuts the elongated groups with circles and mixes
# neighboring clusters. The diagonal norm helps only insofar as the
# variances differ along the axes. The Mahalanobis norm, whose unit ball has
# the orientation of the covariance of the *whole* data set, does better but
# is not perfect: the covariance of the pooled data also contains the spread
# of the cluster centers and is therefore rounder than the clusters
# themselves. The explicit matrix with the true within-cluster covariance
# recovers the groups. The fitted model stores the matrix it used as
# ``norm_matrix_``:
fcm = FCM(n_clusters=3, norm="mahalanobis", random_state=0, n_init=5).fit(X)
print("A = F^-1 (pooled data) =\n", fcm.norm_matrix_.round(3))
print("eigenvalue ratio of A (pooled):", (np.linalg.eigvalsh(fcm.norm_matrix_)[1] / np.linalg.eigvalsh(fcm.norm_matrix_)[0]).round(2))
print("eigenvalue ratio of the true within-cluster inverse covariance:", (2.2 / 0.35) ** 2)

# %%
# A norm is a change of coordinates
# ---------------------------------
# Writing :math:`\mathbf{A}=\mathbf{L}^T\mathbf{L}` (Cholesky factor
# :math:`\mathbf{L}`), the distance :math:`D_{ik\mathbf{A}}` is the Euclidean
# distance between :math:`\mathbf{L}\mathbf{x}_k` and :math:`\mathbf{L}\mathbf{v}_i`:
# clustering with the norm :math:`\mathbf{A}` is Euclidean clustering of the
# linearly transformed data. For the diagonal norm the transformation is
# ordinary feature scaling, which is why scaling matters so much for
# distance-based methods. We stretch the second feature by a factor of 20 and
# compare the Euclidean norm on raw data, the diagonal norm on raw data, and
# the Euclidean norm on standardized data.
X_stretched = X * np.array([1.0, 20.0])
variants = {
    "Euclidean, raw": (FCM(n_clusters=3, random_state=0, n_init=5), X_stretched),
    "diagonal, raw": (FCM(n_clusters=3, norm="diagonal", random_state=0, n_init=5), X_stretched),
    "Euclidean, standardized": (FCM(n_clusters=3, random_state=0, n_init=5),
                                (X_stretched - X_stretched.mean(0)) / X_stretched.std(0)),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, (model, data)) in zip(axes, variants.items(), strict=True):
    model.fit(data)
    plot_membership_contours(model, data, ax=ax, labels=False)
    ax.set_title(f"{name}: ARI = {adjusted_rand_score(y_true, model.labels_):.2f}")
plt.tight_layout()

# %%
# Parameter study: the stretch factor and the invariance of the norms
# --------------------------------------------------------------------
# The factor by which the second feature is stretched is varied from 1 to
# 100, and the three built-in norms are compared. The diagonal and the
# Mahalanobis norms are computed from the data, so rescaling a feature
# rescales the norm matrix accordingly and the partition does not change at
# all: these norms are invariant to the units of the features. The
# Euclidean norm is not. Its result first improves slightly (a factor of two
# happens to make the clusters rounder) and then collapses to horizontal
# bands as soon as one feature dominates the distance.

# [snippet:stretch_scan]
print("stretch   Euclidean   diagonal   Mahalanobis   (ARI with the generating labels)")
for factor in (1, 2, 5, 10, 20, 100):
    data = X * np.array([1.0, factor])
    aris = [adjusted_rand_score(y_true, FCM(n_clusters=3, norm=norm, random_state=0, n_init=5).fit(data).labels_)
            for norm in ("euclidean", "diagonal", "mahalanobis")]
    print(f"{factor:7d}   {aris[0]:9.3f}   {aris[1]:8.3f}   {aris[2]:11.3f}")
# [/snippet:stretch_scan]

# %%
# **Take-away.** The diagonal norm and standardization give identical
# partitions (both remove the arbitrary units of the features); with the raw
# Euclidean norm the stretched feature dominates, and the clusters become
# horizontal bands. A global Mahalanobis norm goes one step further and
# removes a common correlation, but it estimates that correlation from the
# pooled data, where the arrangement of the clusters blurs the shape of the
# individual clusters. None of the global norms can give *every* cluster its
# own shape, estimated from its own members; for that the norm must be
# adapted per cluster, which is the idea of the Gustafson--Kessel algorithm.
