r"""
Spatial FCM: the neighborhood weight and the window size
========================================================

FCM_S, FCM_S1/S2 and EnFCM mix every pixel with its neighborhood, and two
parameters set how much. The weight :math:`\alpha` balances the pixel
against its neighbors (:math:`\alpha=0` is pixel-wise FCM, a large
:math:`\alpha` lets the neighborhood decide), and the window size sets
how far the neighborhood reaches. Both suppress noise and both erase
detail: a structure narrower than the window is outvoted by its
surroundings. To make this trade-off visible, the test image of the
previous example gets two thin bars, one and two pixels wide, and the
accuracy is measured separately on the *interior* of the regions and on
the *detail*, the pixels within one pixel of a region boundary, which
include the bars. The study also shows why :math:`\alpha` must exceed
one for the median variant FCM_S2 to remove impulse noise.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from fuzzyclust.image import (
    SpatialFCM,
    add_image_noise,
    make_synthetic_image,
    match_labels,
    plot_image,
    plot_label_image,
)
from fuzzyclust.plotting import PALETTE

base = make_synthetic_image(shape=(128, 128), n_classes=4, noise="none", random_state=0)
truth = base.truth.copy()
truth[20:108, 64] = 3  # a bar one pixel wide
truth[100:102, 12:116] = 2  # a bar two pixels wide
clean = base.levels[truth]

# detail = pixels with a differently labeled pixel in their 3x3 neighborhood
detail = ndimage.maximum_filter(truth, size=3) != ndimage.minimum_filter(truth, size=3)
interior = ~detail


def scores(labels):
    """Accuracy on the interior and on the detail pixels after matching the labels."""
    mapped = match_labels(truth, labels)[labels]
    ok = mapped == truth
    return float(ok[interior].mean()), float(ok[detail].mean())


gauss = add_image_noise(clean, noise="gaussian", noise_level=0.2, random_state=0)
impulse = add_image_noise(clean, noise="salt_pepper", noise_level=0.15, random_state=0, value_range=(0.0, 1.0))
print(f"{detail.mean():.1%} of the pixels are detail pixels")

# %%
# Segmentations for growing alpha and window size
# -----------------------------------------------
# The figure shows EnFCM on the Gaussian-noise image. Top row:
# :math:`\alpha` grows with a 3x3 window; the speckle disappears, and from
# :math:`\alpha=4` on the one-pixel bar breaks up. Bottom row: the window
# grows with :math:`\alpha=4`; larger windows round the corners of the
# regions, shrink the discs and dissolve both bars.

# [snippet:enfcm]
model = SpatialFCM(n_clusters=4, method="enfcm", alpha=4.0, window_size=5, init="kmeans++", random_state=0)
labels = model.fit(gauss).labels_
interior_acc, detail_acc = scores(labels)
# [/snippet:enfcm]

settings = [(0.0, 3), (1.0, 3), (4.0, 3), (16.0, 3), (4.0, 5), (4.0, 7), (4.0, 9)]
fig, axes = plt.subplots(2, 4, figsize=(11, 6.0))
axes = axes.ravel()
plot_image(gauss, ax=axes[0], title="noisy image ($\\sigma=0.2$)", vmin=0, vmax=1)
for ax, (alpha, w) in zip(axes[1:], settings, strict=True):
    lab = SpatialFCM(n_clusters=4, method="enfcm", alpha=alpha, window_size=w, init="kmeans++",
                     random_state=0).fit(gauss).labels_
    acc_in, acc_det = scores(lab)
    plot_label_image(lab, ax=ax, n_clusters=4, order=match_labels(truth, lab),
                     title=f"$\\alpha$={alpha:g}, {w}x{w}: {acc_in:.2f} / {acc_det:.2f}")
fig.suptitle("EnFCM; titles: accuracy on interior / detail pixels", fontsize=11)
plt.tight_layout()

# %%
# Parameter study: alpha for three methods and two kinds of noise
# ----------------------------------------------------------------
# The plots show the accuracy on the interior (solid) and on the detail
# (dashed) against :math:`\alpha` with the 3x3 window. Under Gaussian
# noise, the interior improves with :math:`\alpha` for all three methods,
# while the detail peaks at a moderate weight and then decays. Under
# salt-and-pepper noise, the mean-based FCM_S and EnFCM gain at most a
# few points on the interior over pixel-wise FCM (:math:`\alpha=0`) and
# lose much of the detail, because an impulse pulls its neighbors'
# averages along. The median-based FCM_S2 removes impulses,
# but only for :math:`\alpha>1`: for a white pixel in a gray region, the
# pixel term favors white and the median term favors gray with equal
# strength when :math:`\alpha=1`, so the median must be given the larger
# weight.

# [snippet:scan]
alphas = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
methods = ("fcm_s", "enfcm", "fcm_s2")
study = {}
for noise, image in (("Gaussian", gauss), ("salt-and-pepper", impulse)):
    for method in methods:
        study[noise, method] = np.array([
            scores(SpatialFCM(n_clusters=4, method=method, alpha=a, init="kmeans++", random_state=0)
                   .fit(image).labels_)
            for a in alphas
        ])
# [/snippet:scan]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
x = np.array(alphas) + 0.1  # shifted for the logarithmic axis
names = {"fcm_s": "FCM_S", "enfcm": "EnFCM", "fcm_s2": "FCM_S2"}
for ax, noise in zip(axes, ("Gaussian", "salt-and-pepper"), strict=True):
    for j, method in enumerate(methods):
        acc = study[noise, method]
        ax.semilogx(x, acc[:, 0], "o-", color=PALETTE[j], label=f"{names[method]} interior")
        ax.semilogx(x, acc[:, 1], "s--", color=PALETTE[j], label=f"{names[method]} detail")
    ax.axvline(1.1, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x, [f"{a:g}" for a in alphas])
    ax.set_xlabel(r"$\alpha$ (3x3 window)")
    ax.set_title(f"{noise} noise")
    ax.grid(True, color="0.9", which="both")
axes[0].set_ylabel("accuracy")
axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
plt.tight_layout()

# %%
# Parameter study: the window size
# ----------------------------------
# With :math:`\alpha=4` and Gaussian noise, a larger window costs detail
# for all three methods. For the mean-based FCM_S it soon costs the
# interior as well (the discs shrink and the corners round off), while
# the median of FCM_S2 keeps region boundaries sharp and profits from the
# larger sample. EnFCM computes its smoothed image once, so its cost does
# not grow with the window; FCM_S recomputes the neighborhood sums of all
# cluster distances in every iteration.
windows = (3, 5, 7, 9, 11)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for j, method in enumerate(methods):
    acc = np.array([
        scores(SpatialFCM(n_clusters=4, method=method, alpha=4.0, window_size=w, init="kmeans++",
                          random_state=0).fit(gauss).labels_)
        for w in windows
    ])
    ax.plot(windows, acc[:, 0], "o-", color=PALETTE[j], label=f"{names[method]} interior")
    ax.plot(windows, acc[:, 1], "s--", color=PALETTE[j], label=f"{names[method]} detail")
    print(names[method], "interior", acc[:, 0].round(3), "detail", acc[:, 1].round(3))
ax.set_xticks(windows, [f"{w}x{w}" for w in windows])
ax.set_xlabel("window size")
ax.set_ylabel("accuracy")
ax.set_title(r"Window size with $\alpha=4$, Gaussian noise")
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.grid(True, color="0.9")
plt.tight_layout()

for key, acc in study.items():
    print(key, "interior", acc[:, 0].round(3), "detail", acc[:, 1].round(3))

# %%
# **Take-away.** The neighborhood weight and the window size buy noise
# suppression at the cost of detail: the interior of the regions always
# profits, but structures as narrow as the window do not survive. A 3x3
# window with :math:`\alpha` between 1 and 4 is a good default for
# Gaussian noise; go larger only when the regions are large and the noise
# is heavy. Under impulse noise the choice of the method matters more than
# its parameters: averaging neighborhoods do not help, and the median
# variant FCM_S2 works only when the median outweighs the pixel
# (:math:`\alpha>1`).
