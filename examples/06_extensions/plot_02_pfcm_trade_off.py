r"""
Possibilistic fuzzy c-means: memberships and typicalities together
==================================================================

FCM memberships are *relative*: they say to which cluster a sample belongs
more than to the others, and they keep the prototypes apart because every
sample has to be shared. PCM typicalities are *absolute*: they say how
representative a sample is of a cluster and make the prototypes robust, but
nothing keeps the prototypes apart. The possibilistic fuzzy c-means (PFCM)
of Pal, Pal, Keller and Bezdek computes both and combines them in the
prototype update,

.. math::
    \mathbf{v}_i = \frac{\sum_k\left(a\,\mu_{ik}^m + b\,t_{ik}^{\eta}\right)\mathbf{x}_k}
                        {\sum_k\left(a\,\mu_{ik}^m + b\,t_{ik}^{\eta}\right)},

with weights :math:`a` for the memberships and :math:`b` for the
typicalities. This example shows the two outputs, the :math:`a/b` trade-off
between robustness and the coincident-cluster failure, and the role of the
typicality exponent :math:`\eta`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from fuzzyclust import FCM
from fuzzyclust.datasets import make_noisy_blobs
from fuzzyclust.extensions import PCM, PFCM, make_close_blobs, plot_typicality
from fuzzyclust.plotting import PALETTE, plot_clusters, plot_membership_contours

X, y = make_noisy_blobs(n_clusters=3, n_samples_per_cluster=100, n_outliers=60, random_state=0)
inliers = y >= 0
true_centers = np.array([X[inliers][y[inliers] == i].mean(axis=0) for i in range(3)])


def center_error(V):
    D = np.linalg.norm(V[:, None] - true_centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max()


# %%
# Two outputs of one model
# ------------------------
# The memberships (left) sum to one and partition the plane into the usual
# FCM cells; the typicalities (right) are bumps around the prototypes and
# flag the outliers. With :math:`b=4>a=1` the prototypes follow the
# typicalities and sit on the groups.

# [snippet:fit]
pfcm = PFCM(n_clusters=3, a=1.0, b=4.0, m=2.0, eta=2.0, random_state=0).fit(X)
U, T = pfcm.membership_, pfcm.typicality_
print(f"rows of U sum to one: {np.allclose(U.sum(axis=1), 1.0)};  typicality mass per sample: "
      f"inliers {T[inliers].sum(axis=1).mean():.2f}, outliers {T[~inliers].sum(axis=1).mean():.2f}")
# [/snippet:fit]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
plot_membership_contours(pfcm, X, ax=axes[0], labels=False)
axes[0].set_title(r"PFCM memberships $\mu_{ik}$ (rows sum to one)")
plot_typicality(X, T, ax=axes[1])
axes[1].scatter(*pfcm.cluster_centers_.T, marker="X", s=140, color="black", edgecolor="white", zorder=5)
axes[1].set_title(r"PFCM typicalities: $\max_i t_{ik}$")
plt.tight_layout()

# %%
# Parameter study I: the weights :math:`a` and :math:`b`
# -------------------------------------------------------
# With :math:`b=0` PFCM is FCM, with :math:`a=0` it is PCM. In between, a
# larger share of the typicalities makes the prototypes less sensitive to
# the outliers: on the noisy blobs the center error drops as soon as the
# typicalities enter the update (left). The membership term keeps the
# prototypes apart. On a large group next to a small, close one (the
# coincident-cluster test case of the PCM example), PCM merges the two
# prototypes as soon as the scale factor :math:`K` reaches one; PFCM with
# :math:`a=b` keeps them apart at :math:`K=1` and lets them approach each
# other from :math:`K=2` on, and the larger the
# membership weight :math:`a`, the wider the range of scales over which the
# prototypes stay separated (right). The weight :math:`b` also enters the
# typicality formula, :math:`t_{ik}=1/(1+(b D^2_{ik}/\gamma_i)^{1/(\eta-1)})`,
# so raising :math:`b` alone tightens the zone of influence rather than
# bringing PCM back; what matters is the weight of the membership term
# relative to the *effective* scale :math:`\gamma_i/b`.
fcm = FCM(n_clusters=3, random_state=0).fit(X)
Xc, yc = make_close_blobs(sizes=(200, 40), spreads=(1.0, 0.4), distance=2.5, random_state=0)
bs = (0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0)
errors = [center_error(PFCM(n_clusters=3, a=1.0, b=b, random_state=0).fit(X).cluster_centers_) for b in bs]
Ks = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
weights = {"PCM ($a=0$)": None, "PFCM $a=1, b=1$": (1.0, 1.0), "PFCM $a=4, b=1$": (4.0, 1.0), "PFCM $a=20, b=1$": (20.0, 1.0)}
separation = {name: [] for name in weights}
for K in Ks:
    for name, ab in weights.items():
        if ab is None:
            model = PCM(n_clusters=2, m=1.5, eta_scale=K, random_state=0).fit(Xc)
        else:
            model = PFCM(n_clusters=2, a=ab[0], b=ab[1], eta=1.5, gamma_scale=K, random_state=0).fit(Xc)
        separation[name].append(np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(bs, errors, "o-", color=PALETTE[0], label="PFCM")
axes[0].axhline(center_error(fcm.cluster_centers_), color="0.4", ls="--", label="FCM ($b=0$)")
axes[0].set_xscale("symlog", linthresh=0.5)
axes[0].set_xlabel("typicality weight $b$ (with $a=1$)")
axes[0].set_ylabel("largest center error")
axes[0].set_title("Noisy blobs: the typicalities protect the prototypes")
for i, (name, sep) in enumerate(separation.items()):
    axes[1].plot(Ks, sep, "o-", color=PALETTE[i], label=name)
axes[1].set_xscale("log")
axes[1].set_xlabel("scale factor $K$")
axes[1].set_ylabel("distance between the two prototypes")
axes[1].set_title("Close groups: the memberships keep the prototypes apart")
for ax in axes:
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
settings = (("PCM", PCM(n_clusters=2, m=1.5, eta_scale=4.0, random_state=0)),
            ("PFCM a = 1, b = 1", PFCM(n_clusters=2, a=1.0, b=1.0, eta=1.5, gamma_scale=4.0, random_state=0)),
            ("PFCM a = 20, b = 1", PFCM(n_clusters=2, a=20.0, b=1.0, eta=1.5, gamma_scale=4.0, random_state=0)))
for ax, (name, model) in zip(axes, settings, strict=False):
    model.fit(Xc)
    plot_clusters(Xc, model.labels_, model.cluster_centers_, ax=ax, legend=False)
    sep = np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1])
    ax.set_title(f"{name}, K = 4: prototype distance {sep:.2f}")
plt.tight_layout()

# %%
# Parameter study II: the typicality exponent :math:`\eta` and the scales
# --------------------------------------------------------------------------
# :math:`\eta` plays the role of :math:`m` in PCM: it sets how fast the
# typicality decays with the distance, and a small :math:`\eta` makes the
# outliers atypical whatever the scales. The scale factor ``gamma_scale``
# (:math:`K`) widens or narrows the zone of influence of every cluster as
# :math:`K` does in PCM. In PFCM it also shifts the balance between the two
# terms of the prototype update: with tight scales even the core samples
# have typicalities well below one, their weight :math:`b\,t_{ik}^{\eta}`
# shrinks and the membership term (which the outliers still contaminate)
# regains influence, so the center error *grows* when the scales are made
# too small. The printed table shows both effects.
print("mean max typicality of the outliers / center error:")
for eta in (1.5, 2.0, 3.0):
    row = []
    for K in (0.5, 1.0, 2.0):
        model = PFCM(n_clusters=3, a=1.0, b=4.0, eta=eta, gamma_scale=K, random_state=0).fit(X)
        row.append(f"K={K}: {model.typicality_[~inliers].max(axis=1).mean():.2f} / {center_error(model.cluster_centers_):.3f}")
    print(f"  eta = {eta}:  " + "   ".join(row))

# %%
# **Take-away.** PFCM gives every sample a relative membership *and* an
# absolute typicality. The typicalities keep the outliers away from the
# prototypes and the memberships keep the prototypes apart, so PFCM
# tolerates scales several times wider than PCM before clusters coincide;
# the more weight :math:`a` the memberships carry, the wider the safe range.
# A ratio :math:`b/a` of 2--5 with scales from FCM is a sound choice for
# noisy data. :math:`\eta` controls the decay of the typicality as
# :math:`m` does in PCM. The scale factor sets the width of the clusters'
# zone of influence and, through the typicality weights of the core
# samples, how much the robust term weighs; do not make the scales tighter
# than the clusters.
