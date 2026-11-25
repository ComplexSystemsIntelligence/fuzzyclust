import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.cluster import FCM, GK  # noqa: E402
from fuzzyclust.datasets import make_fuzzy_blobs, make_noisy_blobs, make_rings  # noqa: E402
from fuzzyclust.extensions import (  # noqa: E402
    KERNELS,
    PCM,
    PFCM,
    FuzzyCShells,
    FuzzyCVarieties,
    KernelFCM,
    NoiseClustering,
    make_circles,
    make_close_blobs,
    make_collinear_segments,
    make_crossing_lines,
    plot_shells,
    plot_typicality,
    plot_varieties,
    rbf_kernel,
)

ALL = [PCM, PFCM, NoiseClustering, KernelFCM, FuzzyCVarieties, FuzzyCShells]


# ------------------------------------------------------------------ helpers and fixtures
def center_error(V, true_centers):
    """Largest distance between matched estimated and true centers."""
    D = np.linalg.norm(np.asarray(V)[:, None] - true_centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max()


@pytest.fixture(scope="module")
def noisy():
    X, y = make_noisy_blobs(n_outliers=60, random_state=0)
    inl = y >= 0
    true_centers = np.array([X[inl][y[inl] == i].mean(axis=0) for i in range(3)])
    return X, y, true_centers


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


@pytest.fixture(scope="module")
def rings():
    return make_rings(radii=(1.0, 3.0), random_state=0)


@pytest.fixture(scope="module")
def crossing():
    return make_crossing_lines(random_state=0)


@pytest.fixture(scope="module")
def collinear():
    return make_collinear_segments(random_state=0)


@pytest.fixture(scope="module")
def circles():
    return make_circles(random_state=0)


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    if isinstance(estimator, FuzzyCVarieties) and estimator.alpha == 0.0:
        return {
            "check_clustering": (
                "Pure fuzzy c-lines group blobs by the line they lie on, not by position: "
                "three Gaussian blobs are legitimately covered by lines through two of them, "
                "so the ARI assertion of the check does not apply (it passes with alpha > 0)."
            )
        }
    return {}


@parametrize_with_checks(
    [
        PCM(n_clusters=2, random_state=0),
        PFCM(n_clusters=2, random_state=0),
        NoiseClustering(n_clusters=2, random_state=0),
        KernelFCM(n_clusters=2, random_state=0),
        KernelFCM(n_clusters=2, prototypes="input", random_state=0),
        FuzzyCVarieties(n_clusters=2, random_state=0),
        FuzzyCVarieties(n_clusters=2, alpha=0.5, random_state=0),
        FuzzyCShells(n_clusters=2, random_state=0),
        FuzzyCShells(n_clusters=2, algorithm="fcs", random_state=0),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("cls", ALL)
def test_n_init_is_accepted_and_keeps_the_best_objective(cls, blobs):
    X, _ = blobs
    single = [cls(n_clusters=3, n_init=1, random_state=s).fit(X).objective_ for s in range(3)]
    multi = cls(n_clusters=3, n_init=3, random_state=0).fit(X)
    assert multi.objective_ <= max(single) + 1e-9
    with pytest.raises(ValueError):
        cls(n_clusters=3, n_init=0).fit(X)


@pytest.mark.parametrize("cls", ALL)
def test_pandas_input_and_feature_names(cls, blobs):
    X, _ = blobs
    df = pd.DataFrame(X, columns=["a", "b"])
    est = cls(n_clusters=3, random_state=0).fit(df)
    assert list(est.feature_names_in_) == ["a", "b"]
    assert est.predict_proba(df).shape == (X.shape[0], 3)
    assert est.predict(df).shape == (X.shape[0],)
    D = est.set_output(transform="pandas").transform(df)
    assert isinstance(D, pd.DataFrame) and list(D.columns) == [f"dist_cluster_{i}" for i in range(3)]
    with pytest.raises(ValueError):
        est.predict(df.rename(columns={"a": "zz"}))


@pytest.mark.parametrize("cls", ALL)
def test_reproducible_with_random_state(cls, blobs):
    X, _ = blobs
    a = cls(n_clusters=3, random_state=3).fit(X)
    b = cls(n_clusters=3, random_state=3).fit(X)
    np.testing.assert_allclose(a.membership_, b.membership_)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)


# ------------------------------------------------------------------ data generators
def test_generators_shapes_and_structure():
    X, y = make_circles(n_samples_per_circle=50, n_outliers=7, random_state=0)
    assert X.shape == (157, 2) and (y == -1).sum() == 7 and set(np.unique(y)) == {-1, 0, 1, 2}
    X, y = make_circles(arc=0.5, n_samples_per_circle=40, noise=0.0, random_state=0)
    r = np.linalg.norm(X[y == 0] - np.array([0.0, 0.0]), axis=1)
    np.testing.assert_allclose(r, 2.0, atol=1e-9)
    X, y = make_close_blobs(sizes=(30, 10), random_state=0)
    assert X.shape == (40, 2) and np.bincount(y).tolist() == [30, 10]
    X, y = make_crossing_lines(n_lines=4, n_samples_per_line=20, noise=0.0, random_state=0)
    assert X.shape == (80, 2)
    # noise-free lines through the origin: every point is collinear with the origin along its line
    for i, phi in enumerate(np.deg2rad(np.arange(4) * 45.0)):
        nrm = np.array([-np.sin(phi), np.cos(phi)])
        np.testing.assert_allclose(X[y == i] @ nrm, 0.0, atol=1e-9)
    X, y = make_collinear_segments(n_samples_per_segment=15, extra_line=False, random_state=0)
    assert X.shape == (30, 2) and set(np.unique(y)) == {0, 1}
    assert X[y == 0, 0].max() < X[y == 1, 0].min()
    X, y = make_collinear_segments(n_samples_per_segment=15, random_state=0)
    assert X.shape == (45, 2) and set(np.unique(y)) == {0, 1, 2}


# ------------------------------------------------------------------ PCM
def test_pcm_typicalities_are_absolute_and_robust(noisy):
    X, y, true_centers = noisy
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    pcm = PCM(n_clusters=3, m=1.5, random_state=0).fit(X)
    assert pcm.typicality_.shape == (X.shape[0], 3)
    assert pcm.typicality_ is pcm.membership_
    assert pcm.typicality_.min() >= 0 and pcm.typicality_.max() <= 1
    assert not np.allclose(pcm.typicality_.sum(axis=1), 1.0)  # rows do not sum to one
    assert pcm.eta_.shape == (3,) and np.all(pcm.eta_ > 0)
    # outliers are atypical of every cluster, inliers typical of one
    tmax = pcm.typicality_.max(axis=1)
    assert np.median(tmax[y == -1]) < 0.1 < 0.6 < np.median(tmax[y >= 0])
    # the prototypes are pulled less by the outliers than those of FCM
    fcm_err = center_error(fcm.cluster_centers_, true_centers)
    assert center_error(pcm.cluster_centers_, true_centers) < 0.5 * fcm_err
    # predict_proba returns typicalities, identical to predict_typicality
    np.testing.assert_allclose(pcm.predict_proba(X), pcm.typicality_, atol=1e-8)
    np.testing.assert_allclose(pcm.predict_typicality(X), pcm.predict_proba(X))
    assert pcm.predict(X[:5]).shape == (5,)


def test_pcm_m_controls_the_decay_of_typicality(noisy):
    X, y, _ = noisy
    sharp = PCM(n_clusters=3, m=1.5, random_state=0).fit(X)
    soft = PCM(n_clusters=3, m=3.0, random_state=0).fit(X)
    assert sharp.typicality_[y == -1].max(axis=1).mean() < soft.typicality_[y == -1].max(axis=1).mean()


def test_pcm_eta_scale_and_explicit_eta(noisy):
    X, y, true_centers = noisy
    base = PCM(n_clusters=3, m=1.5, random_state=0).fit(X)
    wide = PCM(n_clusters=3, m=1.5, eta_scale=4.0, random_state=0).fit(X)
    np.testing.assert_allclose(wide.eta_, 4.0 * base.eta_, rtol=1e-6)
    # a larger zone of influence gives every sample more typicality mass
    assert wide.typicality_.sum() > base.typicality_.sum()
    fixed = PCM(n_clusters=3, m=1.5, eta=0.8, random_state=0).fit(X)
    np.testing.assert_allclose(fixed.eta_, 0.8)
    per_cluster = PCM(n_clusters=3, m=1.5, eta=[0.5, 1.0, 1.5], random_state=0).fit(X)
    np.testing.assert_allclose(np.sort(per_cluster.eta_), [0.5, 1.0, 1.5])
    # typicality 0.5 exactly at squared distance eta
    d2 = fixed.transform(X) ** 2
    t = 1.0 / (1.0 + (d2 / 0.8) ** (1.0 / 0.5))
    np.testing.assert_allclose(fixed.predict_proba(X), t, atol=1e-6)


def test_pcm_coincident_clusters_with_large_scales():
    X, y = make_close_blobs(random_state=0)
    good = PCM(n_clusters=2, m=1.5, eta_scale=0.5, random_state=0).fit(X)
    bad = PCM(n_clusters=2, m=1.5, eta_scale=6.0, random_state=0).fit(X)
    sep_good = np.linalg.norm(good.cluster_centers_[0] - good.cluster_centers_[1])
    sep_bad = np.linalg.norm(bad.cluster_centers_[0] - bad.cluster_centers_[1])
    assert sep_good > 1.5
    assert sep_bad < 0.5 * sep_good  # the two prototypes collapse onto the same dense region


def test_pcm_init_from_partition_or_centers(noisy):
    X, _, _ = noisy
    gk = GK(n_clusters=3, random_state=0).fit(X)
    from_u = PCM(n_clusters=3, init=gk.membership_).fit(X)
    from_v = PCM(n_clusters=3, init=gk.cluster_centers_).fit(X)
    assert center_error(from_u.cluster_centers_, from_v.cluster_centers_) < 0.5
    with pytest.raises(ValueError):
        PCM(n_clusters=3, init=np.ones((4, 4))).fit(X)


def test_pcm_parameter_validation(blobs):
    X, _ = blobs
    for kw in (dict(m=1.0), dict(eta_scale=0.0), dict(eta=-1.0), dict(eta=[1.0, 2.0]), dict(init="nope")):
        with pytest.raises(ValueError):
            PCM(n_clusters=3, **kw).fit(X)


# ------------------------------------------------------------------ PFCM
def test_pfcm_memberships_and_typicalities(noisy):
    X, y, true_centers = noisy
    pfcm = PFCM(n_clusters=3, a=1.0, b=4.0, random_state=0).fit(X)
    np.testing.assert_allclose(pfcm.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert pfcm.typicality_.shape == (X.shape[0], 3)
    assert pfcm.typicality_.min() >= 0 and pfcm.typicality_.max() <= 1
    assert pfcm.gamma_.shape == (3,)
    assert adjusted_rand_score(y[y >= 0], pfcm.labels_[y >= 0]) > 0.95
    assert np.median(pfcm.typicality_[y == -1].max(axis=1)) < np.median(pfcm.typicality_[y >= 0].max(axis=1))
    np.testing.assert_allclose(pfcm.predict_proba(X), pfcm.membership_, atol=1e-8)
    np.testing.assert_allclose(pfcm.predict_typicality(X), pfcm.typicality_, atol=1e-8)
    # the prototypes do not collapse (memberships keep them apart)
    assert np.linalg.norm(pfcm.cluster_centers_[0] - pfcm.cluster_centers_[1]) > 3.0


def test_pfcm_a_b_trade_off(noisy):
    X, y, true_centers = noisy
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    membership_only = PFCM(n_clusters=3, a=1.0, b=0.0, random_state=0).fit(X)
    typicality_heavy = PFCM(n_clusters=3, a=1.0, b=4.0, random_state=0).fit(X)
    # b = 0 is FCM
    assert center_error(membership_only.cluster_centers_, fcm.cluster_centers_) < 1e-3
    # more weight on the typicalities -> prototypes less affected by outliers
    fcm_err = center_error(fcm.cluster_centers_, true_centers)
    assert center_error(typicality_heavy.cluster_centers_, true_centers) < fcm_err


def test_pfcm_eta_exponent_and_gamma(noisy):
    X, y, _ = noisy
    sharp = PFCM(n_clusters=3, eta=1.5, random_state=0).fit(X)
    soft = PFCM(n_clusters=3, eta=3.0, random_state=0).fit(X)
    assert sharp.typicality_[y == -1].max(axis=1).mean() < soft.typicality_[y == -1].max(axis=1).mean()
    base = PFCM(n_clusters=3, random_state=0).fit(X)
    scaled = PFCM(n_clusters=3, gamma_scale=2.0, random_state=0).fit(X)
    np.testing.assert_allclose(scaled.gamma_, 2.0 * base.gamma_, rtol=1e-6)
    assert scaled.typicality_.sum() > base.typicality_.sum()
    fixed = PFCM(n_clusters=3, gamma=1.5, b=2.0, random_state=0).fit(X)
    np.testing.assert_allclose(fixed.gamma_, 1.5)
    d2 = fixed.transform(X) ** 2
    np.testing.assert_allclose(fixed.predict_typicality(X), 1.0 / (1.0 + 2.0 * d2 / 1.5), atol=1e-6)


def test_pfcm_m_and_validation(noisy):
    X, _, _ = noisy
    crisp = PFCM(n_clusters=3, m=1.3, random_state=0).fit(X)
    fuzzy = PFCM(n_clusters=3, m=3.0, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()
    bad = (dict(m=1.0), dict(eta=1.0), dict(a=0.0, b=0.0), dict(a=-1.0), dict(gamma_scale=0.0))
    for kw in (*bad, dict(gamma=[1.0])):
        with pytest.raises(ValueError):
            PFCM(n_clusters=3, **kw).fit(X)


# ------------------------------------------------------------------ noise clustering
def test_noise_clustering_finds_outliers_and_protects_prototypes(noisy):
    X, y, true_centers = noisy
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    nc = NoiseClustering(n_clusters=3, delta_scale=0.3, random_state=0).fit(X)
    assert nc.membership_.shape == (X.shape[0], 3) and nc.noise_membership_.shape == (X.shape[0],)
    np.testing.assert_allclose(nc.membership_.sum(axis=1) + nc.noise_membership_, 1.0, atol=1e-9)
    assert nc.delta_ > 0
    # delta acts as a radius: outliers farther than delta from every group are labeled -1,
    # (almost) no inlier is
    dmin = np.linalg.norm(X[:, None] - true_centers[None], axis=2).min(axis=1)
    far = (y == -1) & (dmin > nc.delta_)
    assert far.sum() > 20 and (nc.labels_[far] == -1).mean() > 0.95
    assert (nc.labels_[y >= 0] == -1).mean() < 0.02
    assert adjusted_rand_score(y[y >= 0], nc.labels_[y >= 0]) > 0.95
    fcm_err = center_error(fcm.cluster_centers_, true_centers)
    assert center_error(nc.cluster_centers_, true_centers) < 0.5 * fcm_err
    # prediction on new data: proba rows + noise membership sum to one, labels agree
    Xn = X[::7]
    U = nc.predict_proba(Xn)
    np.testing.assert_allclose(U.sum(axis=1) + nc.predict_noise_membership(Xn), 1.0, atol=1e-9)
    np.testing.assert_array_equal(nc.predict(Xn), nc.labels_[::7])


def test_noise_clustering_delta_scale_effect(noisy):
    X, y, _ = noisy
    n_noise = []
    for lam in (0.1, 0.5, 2.0):
        nc = NoiseClustering(n_clusters=3, delta_scale=lam, random_state=0).fit(X)
        n_noise.append((nc.labels_ == -1).sum())
    assert n_noise[0] > n_noise[1] > n_noise[2]  # smaller lambda -> more samples declared noise


def test_noise_clustering_fixed_delta_and_limit_to_fcm(noisy):
    X, y, _ = noisy
    nc = NoiseClustering(n_clusters=3, delta=2.0, random_state=0).fit(X)
    assert nc.delta_ == 2.0
    # a huge delta switches the noise cluster off: FCM memberships are recovered
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    huge = NoiseClustering(n_clusters=3, delta=1e6, random_state=0).fit(X)
    assert huge.noise_membership_.max() < 1e-6
    assert center_error(huge.cluster_centers_, fcm.cluster_centers_) < 1e-3
    assert (huge.labels_ == -1).sum() == 0


def test_noise_clustering_m_and_validation(noisy):
    X, _, _ = noisy
    crisp = NoiseClustering(n_clusters=3, m=1.3, random_state=0).fit(X)
    fuzzy = NoiseClustering(n_clusters=3, m=3.0, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()
    for kw in (dict(m=1.0), dict(delta=0.0), dict(delta_scale=0.0)):
        with pytest.raises(ValueError):
            NoiseClustering(n_clusters=3, **kw).fit(X)


# ------------------------------------------------------------------ kernel FCM
def test_kernel_fcm_separates_rings_where_fcm_fails(rings):
    X, y = rings
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    kfcm = KernelFCM(n_clusters=2, m=1.2, gamma=0.3, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, fcm.labels_) < 0.1
    assert adjusted_rand_score(y, kfcm.labels_) > 0.95
    assert kfcm.membership_.shape == (X.shape[0], 2)
    np.testing.assert_allclose(kfcm.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert kfcm.cluster_centers_.shape == (2, 2)
    # kernelised distances for new data agree with the training ones
    np.testing.assert_allclose(kfcm.predict_proba(X), kfcm.membership_, atol=1e-8)
    np.testing.assert_allclose(kfcm.transform(X), kfcm.distances_, atol=1e-8)
    assert kfcm.predict(X[:10]).shape == (10,)


def test_kernel_fcm_gamma_effect(rings):
    X, y = rings
    scores = {}
    for g in (0.3, 5.0):
        est = KernelFCM(n_clusters=2, m=1.2, gamma=g, n_init=5, random_state=0).fit(X)
        scores[g] = adjusted_rand_score(y, est.labels_)
    assert scores[0.3] > 0.95 > scores[5.0]  # a too local kernel loses the ring structure
    default = KernelFCM(n_clusters=2, random_state=0).fit(X)
    np.testing.assert_allclose(default.gamma_, 1.0 / (2 * X.var()))


def test_kernel_fcm_linear_kernel_is_fcm(blobs):
    X, _ = blobs
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    lin = KernelFCM(n_clusters=3, kernel="linear", init=fcm.cluster_centers_).fit(X)
    np.testing.assert_allclose(lin.membership_, fcm.membership_, atol=1e-4)
    np.testing.assert_allclose(lin.cluster_centers_, fcm.cluster_centers_, atol=1e-3)
    np.testing.assert_allclose(lin.transform(X), fcm.transform(X), atol=1e-3)


def test_kernel_fcm_kernels_degree_and_coef0(blobs):
    X, _ = blobs
    for kernel in KERNELS:
        est = KernelFCM(n_clusters=3, kernel=kernel, random_state=0).fit(X)
        np.testing.assert_allclose(est.predict_proba(X), est.membership_, atol=1e-8)
    p2 = KernelFCM(n_clusters=3, kernel="poly", degree=2, coef0=1.0, init=X[:3]).fit(X)
    p3 = KernelFCM(n_clusters=3, kernel="poly", degree=3, coef0=1.0, init=X[:3]).fit(X)
    c0 = KernelFCM(n_clusters=3, kernel="poly", degree=2, coef0=5.0, init=X[:3]).fit(X)
    assert not np.allclose(p2.distances_, p3.distances_)
    assert not np.allclose(p2.distances_, c0.distances_)
    assert np.all(p2.distances_ >= 0) and np.all(c0.distances_ >= 0)


def test_kernel_fcm_input_prototypes_are_robust(noisy):
    X, y, true_centers = noisy
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    kf = KernelFCM(n_clusters=3, prototypes="input", gamma=0.1, n_init=3, random_state=0).fit(X)
    fcm_err = center_error(fcm.cluster_centers_, true_centers)
    assert center_error(kf.cluster_centers_, true_centers) < 0.5 * fcm_err
    assert adjusted_rand_score(y[y >= 0], kf.labels_[y >= 0]) > 0.95
    # the feature-space distance of the Gaussian kernel is bounded by 2
    assert kf.distances_.max() <= np.sqrt(2.0) + 1e-9
    np.testing.assert_allclose(kf.predict_proba(X), kf.membership_, atol=1e-8)


def test_kernel_fcm_m_and_validation(rings):
    X, _ = rings
    crisp = KernelFCM(n_clusters=2, m=1.2, gamma=0.3, random_state=0).fit(X)
    fuzzy = KernelFCM(n_clusters=2, m=2.0, gamma=0.3, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()
    for kw in (dict(m=1.0), dict(kernel="chi2"), dict(gamma=0.0), dict(prototypes="hilbert"),
               dict(prototypes="input", kernel="poly")):
        with pytest.raises(ValueError):
            KernelFCM(n_clusters=2, **kw).fit(X)
    K = rbf_kernel(X[:5], X[:3], gamma=0.7)
    np.testing.assert_allclose(K, np.exp(-0.7 * ((X[:5, None] - X[None, :3]) ** 2).sum(-1)))


# ------------------------------------------------------------------ fuzzy c-varieties
def test_fcv_recovers_crossing_lines_where_gk_cuts_them(crossing):
    X, y = crossing
    fcv = FuzzyCVarieties(n_clusters=3, n_dims=1, n_init=5, random_state=0).fit(X)
    gk = GK(n_clusters=3, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, fcv.labels_) > 0.9
    assert adjusted_rand_score(y, gk.labels_) < 0.6
    assert fcv.directions_.shape == (3, 1, 2)
    # orthonormal directions that match the generating angles 0, 60, 120 degrees
    norms = np.linalg.norm(fcv.directions_[:, 0], axis=1)
    np.testing.assert_allclose(norms, 1.0)
    angles = np.degrees(np.arctan2(fcv.directions_[:, 0, 1], fcv.directions_[:, 0, 0]))
    for true in (0.0, 60.0, 120.0):  # orientation is defined modulo 180 degrees
        diff = np.abs(((angles - true) + 90.0) % 180.0 - 90.0)
        assert diff.min() < 3.0
    # eigenvalues descending; covariances are the normalized scatter matrices
    assert np.all(np.diff(fcv.eigenvalues_, axis=1) <= 1e-9)
    s = (fcv.membership_**2).sum(axis=0)
    np.testing.assert_allclose(fcv.covariances_, fcv.scatter_matrices_ / s[:, None, None], rtol=1e-4)
    np.testing.assert_allclose(fcv.predict_proba(X), fcv.membership_, atol=1e-8)
    assert fcv.predict(X[:4]).shape == (4,)


def test_fcv_n_dims_zero_is_fcm_and_planes_in_3d(blobs):
    X, _ = blobs
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    fcv0 = FuzzyCVarieties(n_clusters=3, n_dims=0, random_state=0).fit(X)
    np.testing.assert_allclose(fcv0.membership_, fcm.membership_, atol=1e-6)
    assert fcv0.directions_.shape == (3, 0, 2)
    # two planes in 3-D (n_dims = 2)
    rng = np.random.RandomState(0)
    P1 = np.c_[rng.uniform(-3, 3, (120, 2)), rng.randn(120) * 0.05]
    P2 = np.c_[rng.uniform(-3, 3, 120), rng.randn(120) * 0.05 + 2.0, rng.uniform(-3, 3, 120)]
    X3 = np.vstack([P1, P2])
    y3 = np.repeat([0, 1], 120)
    fcv2 = FuzzyCVarieties(n_clusters=2, n_dims=2, n_init=5, random_state=0).fit(X3)
    assert adjusted_rand_score(y3, fcv2.labels_) > 0.95
    assert fcv2.directions_.shape == (2, 2, 3)
    # the normal of each plane is the discarded eigenvector: directions span the plane
    for S in fcv2.directions_:
        np.testing.assert_allclose(S @ S.T, np.eye(2), atol=1e-8)


def test_fcv_alpha_elliptotypes_separate_collinear_segments(collinear):
    X, y = collinear
    pure = FuzzyCVarieties(n_clusters=3, alpha=0.0, n_init=5, random_state=0).fit(X)
    ell = FuzzyCVarieties(n_clusters=3, alpha=0.3, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, pure.labels_) < 0.7  # collinear segments are one line
    assert adjusted_rand_score(y, ell.labels_) > 0.95
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    one = FuzzyCVarieties(n_clusters=3, alpha=1.0, init=fcm.cluster_centers_).fit(X)
    np.testing.assert_allclose(one.membership_, fcm.membership_, atol=1e-4)


def test_fcv_m_and_validation(crossing):
    X, _ = crossing
    crisp = FuzzyCVarieties(n_clusters=3, m=1.3, n_init=3, random_state=0).fit(X)
    fuzzy = FuzzyCVarieties(n_clusters=3, m=3.0, n_init=3, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()
    for kw in (dict(m=1.0), dict(n_dims=-1), dict(n_dims=2), dict(alpha=1.5)):
        with pytest.raises(ValueError):
            FuzzyCVarieties(n_clusters=3, **kw).fit(X)


# ------------------------------------------------------------------ fuzzy c-shells
@pytest.mark.parametrize("algorithm", ["fcss", "fcs"])
def test_shells_recover_concentric_rings(rings, algorithm):
    X, y = rings
    fcs = FuzzyCShells(n_clusters=2, algorithm=algorithm, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, fcs.labels_) > 0.99
    np.testing.assert_allclose(np.sort(fcs.radii_), [1.0, 3.0], atol=0.05)
    np.testing.assert_allclose(fcs.cluster_centers_, 0.0, atol=0.05)
    np.testing.assert_allclose(fcs.predict_proba(X), fcs.membership_, atol=1e-8)
    assert fcs.predict(X[:3]).shape == (3,)
    # signed radial deviation: near zero on the own shell
    E = fcs.radial_deviation(X)
    assert np.abs(E[np.arange(len(X)), fcs.labels_]).mean() < 0.15


def test_shells_n_init_escapes_the_fcm_start_on_concentric_rings(rings):
    X, y = rings
    one = FuzzyCShells(n_clusters=2, n_init=1, random_state=0).fit(X)
    many = FuzzyCShells(n_clusters=2, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, one.labels_) < 0.5  # FCM halves the rings
    assert adjusted_rand_score(y, many.labels_) > 0.99
    assert many.objective_ < one.objective_


def test_shells_intersecting_circles_and_algorithm_distances(circles):
    X, y = circles
    true_c = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.5]])
    true_r = np.array([2.0, 1.2, 1.0])
    for algorithm in ("fcss", "fcs"):
        fcs = FuzzyCShells(n_clusters=3, algorithm=algorithm, random_state=0).fit(X)
        assert adjusted_rand_score(y, fcs.labels_) > 0.9
        D = np.linalg.norm(fcs.cluster_centers_[:, None] - true_c[None], axis=2)
        r, c = linear_sum_assignment(D)
        assert D[r, c].max() < 0.1
        np.testing.assert_allclose(fcs.radii_[r], true_r[c], atol=0.1)
    fcss = FuzzyCShells(n_clusters=3, algorithm="fcss", random_state=0).fit(X)
    fcs = FuzzyCShells(n_clusters=3, algorithm="fcs", random_state=0).fit(X)
    d = np.linalg.norm(X[:, None] - fcss.cluster_centers_[None], axis=2)
    np.testing.assert_allclose(fcss.transform(X), np.abs(d**2 - fcss.radii_**2), atol=1e-8)
    d = np.linalg.norm(X[:, None] - fcs.cluster_centers_[None], axis=2)
    np.testing.assert_allclose(fcs.transform(X), np.abs(d - fcs.radii_), atol=1e-8)
    np.testing.assert_allclose(fcs.radial_deviation(X), d - fcs.radii_, atol=1e-8)


def test_shells_newton_steps_and_random_init(circles):
    X, y = circles
    a = FuzzyCShells(n_clusters=3, algorithm="fcs", newton_steps=1, random_state=0).fit(X)
    b = FuzzyCShells(n_clusters=3, algorithm="fcs", newton_steps=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, a.labels_) > 0.9 and adjusted_rand_score(y, b.labels_) > 0.9
    assert b.objective_ <= a.objective_ + 1e-3  # more Newton steps -> at least as good a fit
    rnd = FuzzyCShells(n_clusters=3, init="kmeans++", n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, rnd.labels_) > 0.9
    # objective of a pure algorithm decreases monotonically
    assert np.all(np.diff(a.objective_history_) <= 1e-8)


def test_shells_m_and_validation(rings):
    X, _ = rings
    crisp = FuzzyCShells(n_clusters=2, m=1.3, n_init=5, random_state=0).fit(X)
    fuzzy = FuzzyCShells(n_clusters=2, m=3.0, n_init=5, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()
    for kw in (dict(m=1.0), dict(algorithm="fcqs"), dict(newton_steps=0)):
        with pytest.raises(ValueError):
            FuzzyCShells(n_clusters=2, **kw).fit(X)


# ------------------------------------------------------------------ robustness overview
def test_robust_variants_beat_fcm_on_noisy_blobs(noisy):
    X, y, true_centers = noisy
    fcm_err = center_error(FCM(n_clusters=3, random_state=0).fit(X).cluster_centers_, true_centers)
    robust = {
        "PCM": PCM(n_clusters=3, m=1.5, random_state=0),
        "PFCM": PFCM(n_clusters=3, b=4.0, random_state=0),
        "NC": NoiseClustering(n_clusters=3, delta_scale=0.3, random_state=0),
        "KFCM": KernelFCM(n_clusters=3, prototypes="input", gamma=0.1, random_state=0),
    }
    for est in robust.values():
        assert center_error(est.fit(X).cluster_centers_, true_centers) < fcm_err


# ------------------------------------------------------------------ plot helpers
def test_plot_helpers(rings, crossing, noisy):
    X, _ = rings
    fcs = FuzzyCShells(n_clusters=2, n_init=5, random_state=0).fit(X)
    ax = plot_shells(X, fcs)
    assert len(ax.patches) == 2
    _, ax2 = plt.subplots()
    assert plot_shells(X, fcs, ax=ax2, labels=np.zeros(len(X), int)) is ax2
    Xl, _ = crossing
    fcv = FuzzyCVarieties(n_clusters=3, n_init=3, random_state=0).fit(Xl)
    ax = plot_varieties(Xl, fcv)
    assert len(ax.lines) >= 3
    _, ax2 = plt.subplots()
    assert plot_varieties(Xl, fcv, ax=ax2, length=2.0) is ax2
    Xn, _, _ = noisy
    pcm = PCM(n_clusters=3, m=1.5, random_state=0).fit(Xn)
    ax = plot_typicality(Xn, pcm.typicality_)
    assert ax.collections
    nc = NoiseClustering(n_clusters=3, random_state=0).fit(Xn)
    _, ax2 = plt.subplots()
    assert plot_typicality(Xn, 1.0 - nc.noise_membership_, ax=ax2) is ax2
    plt.close("all")
