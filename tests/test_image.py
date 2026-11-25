import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import ndimage  # noqa: E402

from fuzzyclust.image import (  # noqa: E402
    METHODS,
    SpatialFCM,
    add_image_noise,
    local_similarity_image,
    make_synthetic_image,
    match_labels,
    neighborhood_sum,
    plot_image,
    plot_label_image,
    plot_membership_maps,
    plot_segmentation,
    segmentation_accuracy,
    weighted_sum_image,
)
from fuzzyclust.large import WeightedFCM  # noqa: E402

SPATIAL = tuple(m for m in METHODS if m != "fcm")
HISTOGRAM = ("enfcm", "fgfcm")

# SpatialFCM.fit takes an image (H, W) or (H, W, C), not an (n_samples, n_features) matrix, so
# sklearn's generic estimator checks (which feed 2-D sample matrices) do not apply; its API
# behavior (fit/predict/predict_proba/fit_predict, validation, reproducibility) is tested below.


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def noisy():
    return make_synthetic_image(shape=(96, 96), n_classes=4, noise_level=0.2, random_state=0)


@pytest.fixture(scope="module")
def salt():
    return make_synthetic_image(
        shape=(96, 96), n_classes=4, noise="salt_pepper", noise_level=0.15, random_state=0
    )


# ------------------------------------------------------------------ data and metrics
def test_make_synthetic_image_properties():
    data = make_synthetic_image(shape=(64, 80), n_classes=4, noise_level=0.1, random_state=0)
    assert data.image.shape == data.clean.shape == data.truth.shape == (64, 80)
    assert np.array_equal(np.unique(data.truth), [0, 1, 2, 3])
    np.testing.assert_allclose(data.levels, [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(data.clean, data.levels[data.truth])
    assert np.std(data.image - data.clean) == pytest.approx(0.1, rel=0.1)
    # regions have comparable size: no class below 8 % of the pixels
    assert np.bincount(data.truth.ravel()).min() > 0.08 * data.truth.size
    # reproducible, and different seeds give different layouts
    again = make_synthetic_image(shape=(64, 80), n_classes=4, noise_level=0.1, random_state=0)
    np.testing.assert_array_equal(again.truth, data.truth)
    np.testing.assert_allclose(again.image, data.image)
    other = make_synthetic_image(shape=(64, 80), n_classes=4, noise_level=0.1, random_state=1)
    assert not np.array_equal(other.truth, data.truth)
    # salt and pepper hits the requested share of pixels with the extreme values
    sp = make_synthetic_image((64, 80), 3, noise="salt_pepper", noise_level=0.2, random_state=1)
    changed = sp.image != sp.clean
    assert 0.1 < changed.mean() < 0.2  # some hits land on pixels that already had the extreme value
    assert set(np.unique(sp.image[changed])) <= {0.0, 1.0}
    both = make_synthetic_image(shape=(32, 32), noise="both", noise_level=0.1, random_state=0)
    assert (both.image != both.clean).mean() > 0.9
    none = make_synthetic_image(shape=(32, 32), noise="none", random_state=0)
    np.testing.assert_array_equal(none.image, none.clean)
    # multichannel
    rgb = make_synthetic_image(shape=(32, 32), n_classes=3, n_channels=3, random_state=0)
    assert rgb.image.shape == (32, 32, 3) and rgb.levels.shape == (3, 3)
    for kw in (dict(n_classes=1), dict(n_channels=0), dict(noise="bogus"), dict(noise_level=-1.0)):
        with pytest.raises(ValueError):
            make_synthetic_image(shape=(16, 16), **kw)
    with pytest.raises(ValueError):
        add_image_noise(np.zeros((4, 4)), noise="bogus")


def test_segmentation_accuracy_and_label_matching():
    truth = np.array([[0, 0, 1, 1], [2, 2, 1, 1]])
    assert segmentation_accuracy(truth, truth) == 1.0
    perm = np.array([2, 0, 1])[truth]
    assert segmentation_accuracy(truth, perm) == 1.0
    np.testing.assert_array_equal(match_labels(truth, perm), [1, 2, 0])
    wrong = perm.copy()
    wrong[0, 0] = 1
    assert segmentation_accuracy(truth, wrong) == pytest.approx(7 / 8)
    # more predicted than true labels: unmatched labels count as errors
    assert segmentation_accuracy(np.zeros(4, int), np.array([0, 0, 1, 1])) == 0.5
    with pytest.raises(ValueError):
        match_labels(truth, truth.ravel()[:5])


# ------------------------------------------------------------------ neighborhood operations
def test_neighborhood_operations():
    img = np.arange(25.0).reshape(5, 5)
    S = neighborhood_sum(img, 3)
    # interior pixel: sum of the eight neighbors
    k = 12
    assert S[2, 2] == pytest.approx(img[1:4, 1:4].sum() - img[2, 2])
    assert k == 12
    # reflected borders: equal to scipy's convolution with a hollow box
    K = np.ones((3, 3))
    K[1, 1] = 0
    np.testing.assert_allclose(S, ndimage.convolve(img, K, mode="reflect"))
    # custom kernel and a 5x5 window
    K5 = np.ones((5, 5))
    hollow = K5.copy()
    hollow[2, 2] = 0.0
    S5 = neighborhood_sum(img, 5, kernel=K5)
    np.testing.assert_allclose(S5, ndimage.convolve(img, hollow, mode="reflect"))
    # multichannel: channel-wise
    img3 = np.stack([img, 2 * img], axis=2)
    S3 = neighborhood_sum(img3, 3)
    np.testing.assert_allclose(S3[..., 1], 2 * S)
    # the weighted sum image: constant image stays constant, alpha=0 gives the image back
    const = np.full((6, 6), 0.7)
    np.testing.assert_allclose(weighted_sum_image(const, alpha=2.0), const)
    np.testing.assert_allclose(weighted_sum_image(img, alpha=0.0), img)
    xi = weighted_sum_image(img, alpha=1.0)
    np.testing.assert_allclose(xi, (img + S / 8.0) / 2.0)
    # local similarity image: a lone noise pixel in a flat region is replaced by its surroundings
    flat = np.zeros((9, 9))
    flat[4, 4] = 1.0
    xi = local_similarity_image(flat)
    assert xi[4, 4] == pytest.approx(0.0, abs=1e-12)  # center excluded, all neighbors are 0
    # neighbors of the noise pixel: the noise pixel differs strongly from them and gets a tiny weight
    assert xi[4, 3] < 0.05
    # an edge is kept: on a step image the summed image stays a step away from the edge pixels
    step = np.zeros((8, 8))
    step[:, 4:] = 1.0
    xs = local_similarity_image(step)
    assert xs[0, 0] == pytest.approx(0.0, abs=1e-12) and xs[0, 7] == pytest.approx(1.0, abs=1e-12)
    for kw in (dict(window_size=2), dict(window_size=1), dict(lambda_s=0.0)):
        with pytest.raises(ValueError):
            local_similarity_image(step, **kw)
    with pytest.raises(ValueError):
        neighborhood_sum(step, 4)
    with pytest.raises(ValueError):
        neighborhood_sum(step, 3, kernel=np.ones((5, 5)))


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("method", METHODS)
def test_fit_shapes_and_api(method, noisy):
    seg = SpatialFCM(n_clusters=4, method=method, random_state=0).fit(noisy.image)
    H, W = noisy.image.shape
    assert seg.cluster_centers_.shape == (4, 1)
    assert seg.membership_.shape == (H, W, 4) and seg.labels_.shape == (H, W)
    np.testing.assert_allclose(seg.membership_.sum(axis=2), 1.0, atol=1e-9)
    np.testing.assert_array_equal(seg.labels_, seg.membership_.argmax(axis=2))
    assert seg.n_iter_ == len(seg.objective_history_) >= 1 and np.isfinite(seg.objective_)
    assert seg.image_shape_ == (H, W) and seg.n_channels_ == 1
    # the prototypes recover the four gray levels
    assert np.abs(np.sort(seg.cluster_centers_.ravel()) - noisy.levels).max() < 0.15
    # predict on the training image reproduces the fit
    U = seg.predict_proba(noisy.image)
    assert U.shape == (H, W, 4)
    agreement = (seg.predict(noisy.image) == seg.labels_).mean()
    assert agreement > 0.97
    if method == "flicm":  # fixed-point iteration of the membership update with fixed prototypes
        np.testing.assert_allclose(U, seg.membership_, atol=1e-2)
    elif method not in HISTOGRAM:
        np.testing.assert_allclose(U, seg.membership_, atol=1e-6)
    labels = SpatialFCM(n_clusters=4, method=method, random_state=0).fit_predict(noisy.image)
    np.testing.assert_array_equal(labels, seg.labels_)
    # a new image of another size, same number of channels
    other = make_synthetic_image(shape=(40, 50), n_classes=4, noise_level=0.2, random_state=1)
    assert seg.predict(other.image).shape == (40, 50)
    assert segmentation_accuracy(other.truth, seg.predict(other.image)) > 0.6
    # method specific attributes
    if method in ("fcm_s1", "fcm_s2", "enfcm", "fgfcm"):
        assert seg.filtered_image_.shape == (H, W)
    else:
        assert not hasattr(seg, "filtered_image_")
    if method in HISTOGRAM:
        assert seg.histogram_.shape == (256,) and seg.histogram_.sum() == H * W
        assert seg.level_values_.shape == (256,) and seg.level_membership_.shape == (256, 4)
        assert isinstance(seg.level_model_, WeightedFCM)


def test_spatial_variants_beat_plain_fcm_under_gaussian_noise(noisy):
    plain = SpatialFCM(n_clusters=4, method="fcm", random_state=0).fit(noisy.image)
    acc_plain = segmentation_accuracy(noisy.truth, plain.labels_)
    assert acc_plain < 0.8
    for method in SPATIAL:
        seg = SpatialFCM(n_clusters=4, method=method, random_state=0).fit(noisy.image)
        acc = segmentation_accuracy(noisy.truth, seg.labels_)
        assert acc > acc_plain + 0.1, (method, acc, acc_plain)
        assert acc > 0.8, method
    # the adaptive methods do best
    for method in ("fgfcm", "flicm"):
        seg = SpatialFCM(n_clusters=4, method=method, random_state=0).fit(noisy.image)
        assert segmentation_accuracy(noisy.truth, seg.labels_) > 0.95, method


def test_median_variant_and_fgfcm_handle_salt_and_pepper(salt):
    acc = {}
    for method in ("fcm", "fcm_s1", "fcm_s2", "fgfcm"):
        seg = SpatialFCM(n_clusters=4, method=method, alpha=4.0, random_state=0).fit(salt.image)
        acc[method] = segmentation_accuracy(salt.truth, seg.labels_)
    assert acc["fcm_s2"] > acc["fcm_s1"]  # the median ignores the impulses, the mean smears them
    assert acc["fcm_s2"] > acc["fcm"] + 0.05
    assert acc["fgfcm"] > acc["fcm"] + 0.05


def test_histogram_methods_are_fast_and_pixel_independent():
    data = make_synthetic_image(shape=(256, 256), n_classes=4, noise_level=0.2, random_state=0)
    for method in HISTOGRAM:
        t = time.perf_counter()
        seg = SpatialFCM(n_clusters=4, method=method, random_state=0).fit(data.image)
        dt = time.perf_counter() - t
        assert dt < 2.0, (method, dt)
        assert segmentation_accuracy(data.truth, seg.labels_) > 0.85
        # the pixel memberships are those of their gray-level bins
        xi = seg.filtered_image_
        lo, hi = xi.min(), xi.max()
        idx = np.clip(((xi - lo) / (hi - lo) * 256).astype(int), 0, 255)
        np.testing.assert_allclose(seg.membership_, seg.level_membership_[idx])
        # fewer levels: coarser but still fine
        coarse = SpatialFCM(n_clusters=4, method=method, n_levels=32, random_state=0).fit(data.image)
        assert coarse.histogram_.shape == (32,)
        assert segmentation_accuracy(data.truth, coarse.labels_) > 0.8
    # the histogram clustering is weighted FCM on the levels: same prototypes as a direct run
    seg = SpatialFCM(n_clusters=4, method="enfcm", random_state=0).fit(data.image)
    keep = seg.histogram_ > 0
    direct = WeightedFCM(n_clusters=4, init=seg.cluster_centers_).fit(
        seg.level_values_[keep][:, None], sample_weight=seg.histogram_[keep]
    )
    np.testing.assert_allclose(
        np.sort(direct.cluster_centers_.ravel()), np.sort(seg.cluster_centers_.ravel()), atol=1e-3
    )


def test_alpha_effect(noisy):
    """alpha = 0 is plain FCM; larger alpha smooths more (fewer isolated pixels)."""
    plain = SpatialFCM(n_clusters=4, method="fcm", random_state=0).fit(noisy.image)
    zero = SpatialFCM(n_clusters=4, method="fcm_s", alpha=0.0, random_state=0).fit(noisy.image)
    np.testing.assert_allclose(zero.cluster_centers_, plain.cluster_centers_, atol=1e-6)
    assert segmentation_accuracy(plain.labels_, zero.labels_) > 0.999

    def isolated(labels):
        # share of pixels whose label differs from all four axis neighbors
        same = np.zeros(labels.shape, dtype=int)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            same += np.roll(labels, shift, axis=axis) == labels
        return (same == 0).mean()

    for method in ("fcm_s", "fcm_s1", "enfcm"):
        low = SpatialFCM(n_clusters=4, method=method, alpha=0.5, random_state=0).fit(noisy.image)
        high = SpatialFCM(n_clusters=4, method=method, alpha=6.0, random_state=0).fit(noisy.image)
        assert isolated(high.labels_) < isolated(low.labels_), method
        assert isolated(low.labels_) < isolated(plain.labels_), method


def test_window_size_and_fgfcm_scales(noisy):
    small = SpatialFCM(n_clusters=4, method="flicm", window_size=3, random_state=0).fit(noisy.image)
    large = SpatialFCM(n_clusters=4, method="flicm", window_size=5, random_state=0).fit(noisy.image)
    acc_small = segmentation_accuracy(noisy.truth, small.labels_)
    acc_large = segmentation_accuracy(noisy.truth, large.labels_)
    assert acc_small > 0.9 and acc_large > 0.9
    # a larger window changes the fuzzy factor: the partitions differ but both are good
    assert 0.9 < segmentation_accuracy(small.labels_, large.labels_) < 1.0
    # FGFCM scales: a very large lambda_g makes all neighbors equally similar (EnFCM-like mean),
    # a tiny one trusts only near-identical neighbors
    wide = SpatialFCM(n_clusters=4, method="fgfcm", lambda_g=1e9, random_state=0).fit(noisy.image)
    nb_mean = neighborhood_sum(noisy.image, 3) / 8.0
    np.testing.assert_allclose(wide.filtered_image_, nb_mean, atol=1e-5)
    narrow = SpatialFCM(n_clusters=4, method="fgfcm", lambda_g=0.05, random_state=0).fit(noisy.image)
    assert np.std(narrow.filtered_image_ - noisy.clean) > np.std(wide.filtered_image_ - noisy.clean)
    # in a 3x3 window the spatial factor is constant and lambda_s has no effect
    other = SpatialFCM(n_clusters=4, method="fgfcm", lambda_s=0.5, random_state=0).fit(noisy.image)
    ref = SpatialFCM(n_clusters=4, method="fgfcm", random_state=0).fit(noisy.image)
    np.testing.assert_allclose(other.filtered_image_, ref.filtered_image_, atol=1e-10)
    # in a 5x5 window it matters
    w5 = SpatialFCM(n_clusters=4, method="fgfcm", window_size=5, lambda_s=0.5).fit(noisy.image)
    w5b = SpatialFCM(n_clusters=4, method="fgfcm", window_size=5, lambda_s=50.0).fit(noisy.image)
    assert np.abs(w5.filtered_image_ - w5b.filtered_image_).max() > 1e-3


def test_fuzziness_init_restarts_tolerance(noisy):
    base = SpatialFCM(n_clusters=4, method="flicm", random_state=0).fit(noisy.image)
    again = SpatialFCM(n_clusters=4, method="flicm", random_state=0).fit(noisy.image)
    np.testing.assert_allclose(base.membership_, again.membership_)
    soft = SpatialFCM(n_clusters=4, method="fcm_s", m=3.0, random_state=0).fit(noisy.image)
    crisp = SpatialFCM(n_clusters=4, method="fcm_s", m=1.5, random_state=0).fit(noisy.image)
    assert soft.membership_.max(axis=2).mean() < crisp.membership_.max(axis=2).mean()
    pp = SpatialFCM(n_clusters=4, method="fcm_s1", init="kmeans++", random_state=0).fit(noisy.image)
    assert segmentation_accuracy(noisy.truth, pp.labels_) > 0.85
    given = SpatialFCM(n_clusters=4, method="fcm_s1", init=noisy.levels[:, None]).fit(noisy.image)
    np.testing.assert_allclose(np.sort(given.cluster_centers_.ravel()), noisy.levels, atol=0.1)
    # an initial partition as image or as matrix
    U0 = given.membership_
    from_img = SpatialFCM(n_clusters=4, method="fcm_s1", init=U0).fit(noisy.image)
    from_mat = SpatialFCM(n_clusters=4, method="fcm_s1", init=U0.reshape(-1, 4)).fit(noisy.image)
    np.testing.assert_allclose(from_img.cluster_centers_, from_mat.cluster_centers_)
    assert from_img.n_iter_ <= given.n_iter_
    multi = SpatialFCM(n_clusters=4, method="fcm_s", n_init=3, random_state=0).fit(noisy.image)
    single = SpatialFCM(n_clusters=4, method="fcm_s", n_init=1, random_state=0).fit(noisy.image)
    assert multi.objective_ <= single.objective_ + 1e-9
    few = SpatialFCM(n_clusters=4, method="fcm_s", max_iter=3, random_state=0).fit(noisy.image)
    assert few.n_iter_ == 3
    loose = SpatialFCM(n_clusters=4, method="fcm_s", tol=1e-1, random_state=0).fit(noisy.image)
    assert loose.n_iter_ <= single.n_iter_
    # the FCM_S family descends monotonically
    for method in ("fcm", "fcm_s", "fcm_s1", "fcm_s2", "enfcm"):
        J = SpatialFCM(n_clusters=4, method=method, random_state=0).fit(noisy.image).objective_history_
        assert np.all(np.diff(J) <= 1e-6 * abs(J[0])), method


def test_multichannel_images():
    rgb = make_synthetic_image(shape=(48, 48), n_classes=3, n_channels=3, noise_level=0.15, random_state=0)
    for method in ("fcm", "fcm_s", "fcm_s1", "fcm_s2", "flicm"):
        seg = SpatialFCM(n_clusters=3, method=method, random_state=0).fit(rgb.image)
        assert seg.cluster_centers_.shape == (3, 3) and seg.labels_.shape == (48, 48)
        assert seg.n_channels_ == 3
        assert segmentation_accuracy(rgb.truth, seg.labels_) > 0.85, method
        assert seg.predict(rgb.image).shape == (48, 48)
    for method in HISTOGRAM:
        with pytest.raises(ValueError):
            SpatialFCM(n_clusters=3, method=method).fit(rgb.image)
    gray = SpatialFCM(n_clusters=3, random_state=0).fit(rgb.image[..., 0])
    with pytest.raises(ValueError):
        gray.predict(rgb.image)


def test_parameter_validation(noisy):
    img = noisy.image
    bad = (
        dict(method="bogus"), dict(n_clusters=0), dict(m=1.0), dict(alpha=-1.0), dict(window_size=2),
        dict(window_size=1), dict(lambda_s=0.0), dict(lambda_g=-1.0), dict(n_levels=1), dict(max_iter=0),
        dict(n_init=0), dict(init="bogus"), dict(init=np.zeros((2, 5))),
    )
    for kw in bad:
        with pytest.raises(ValueError):
            SpatialFCM(**{"n_clusters": 3, **kw}).fit(img)
    with pytest.raises(ValueError):
        SpatialFCM(n_clusters=3).fit(np.zeros(10))
    with pytest.raises(ValueError):
        SpatialFCM(n_clusters=3).fit(np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        SpatialFCM(n_clusters=3, method="enfcm", init=np.zeros((8, 8, 3))).fit(img)
    with pytest.raises(ValueError):
        SpatialFCM(n_clusters=3, method="enfcm").fit(np.zeros((8, 8)))  # a flat image has one gray level


# ------------------------------------------------------------------ plots
def test_plot_helpers(noisy):
    seg = SpatialFCM(n_clusters=4, method="flicm", random_state=0).fit(noisy.image)
    axes = plot_segmentation(seg, noisy.image, clean=noisy.clean, truth=noisy.truth)
    assert len(axes) == 2 + 2 + 4
    assert "acc." in axes[3].get_title()
    axes = plot_segmentation(seg, noisy.image, membership=False, title="FLICM")
    assert len(axes) == 2 and axes[1].get_title() == "FLICM"
    ax = plot_image(noisy.image, title="x")
    assert ax.get_title() == "x" and len(ax.get_xticks()) == 0
    rgb = make_synthetic_image(shape=(16, 16), n_classes=2, n_channels=3, random_state=0)
    assert plot_image(rgb.image) is not None
    ax = plot_label_image(seg.labels_, order=match_labels(noisy.truth, seg.labels_))
    assert len(ax.images) == 1
    axes = plot_membership_maps(seg.membership_)
    assert len(axes) == 4
    plt.close("all")
