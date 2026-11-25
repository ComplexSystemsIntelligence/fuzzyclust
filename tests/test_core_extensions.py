"""Tests of the release-0.2 core extensions: fuzzifiers, density-based cluster estimation, merging,
feature weighting and the validity additions (fuzzy Rand indices, stability, Kwon, PCAES)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import adjusted_rand_score, rand_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust._utils import labels_to_membership  # noqa: E402
from fuzzyclust.cluster import (  # noqa: E402
    FCM,
    GK,
    CompatibleClusterMerging,
    KMeans,
    MountainClustering,
    PolynomialFuzzifierFCM,
    RegularizedFCM,
    SubtractiveClustering,
    cluster_similarity,
    entropy_memberships,
    geometric_compatibility,
    polynomial_fuzzifier,
    polynomial_memberships,
    quadratic_memberships,
    reference_fuzziness,
)
from fuzzyclust.datasets import make_different_densities, make_fuzzy_blobs, make_lines  # noqa: E402
from fuzzyclust.extensions import SCAD, FeatureWeightedFCM, VolumePrototypeFCM  # noqa: E402
from fuzzyclust.models import TakagiSugeno, make_sinc  # noqa: E402
from fuzzyclust.plotting import plot_validity_scan  # noqa: E402
from fuzzyclust.validity import (  # noqa: E402
    INDEX_DIRECTION,
    fuzzy_adjusted_rand_index,
    fuzzy_jaccard_index,
    fuzzy_rand_index,
    kwon_index,
    pcaes_index,
    scan_n_clusters,
    stability_selection,
    suggest_n_clusters,
    validity_indices,
)

AO_ESTIMATORS = [RegularizedFCM, PolynomialFuzzifierFCM, VolumePrototypeFCM, SCAD, FeatureWeightedFCM]


def center_error(V, true_centers):
    """Largest distance between matched estimated and true centers."""
    D = np.linalg.norm(np.asarray(V)[:, None] - true_centers[None], axis=2)
    r, c = linear_sum_assignment(D)
    return D[r, c].max()


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


@pytest.fixture(scope="module")
def blobs4():
    return make_fuzzy_blobs(n_clusters=4, random_state=0)


@pytest.fixture(scope="module")
def lines():
    return make_lines(n_lines=3, random_state=0)


@pytest.fixture(scope="module")
def noisy_feature_blobs(blobs):
    X, y = blobs
    rng = np.random.RandomState(0)
    Xn = np.c_[X, 3.0 * rng.randn(X.shape[0])]
    return StandardScaler().fit_transform(Xn), y


# ------------------------------------------------------------------ sklearn compatibility
# GK with the default (unregularized) covariances can hit a degenerate cluster on the tiny random
# data sets of the checks, so the geometric variant is checked with a regularized GK and a small start.
@parametrize_with_checks(
    [
        RegularizedFCM(n_clusters=2, random_state=0),
        RegularizedFCM(n_clusters=2, regularization="quadratic", random_state=0),
        RegularizedFCM(n_clusters=2, annealing=True, lam=0.5, random_state=0),
        PolynomialFuzzifierFCM(n_clusters=2, random_state=0),
        SubtractiveClustering(),
        MountainClustering(),
        CompatibleClusterMerging(FCM(random_state=0), start_n_clusters=4, criterion="similarity"),
        CompatibleClusterMerging(GK(gamma=0.1, beta=1e4, random_state=0), start_n_clusters=3),
        VolumePrototypeFCM(n_clusters=2, random_state=0),
        VolumePrototypeFCM(n_clusters=4, merge=True, random_state=0),
        VolumePrototypeFCM(n_clusters=2, norm="adaptive", gamma=0.1, random_state=0),
        SCAD(n_clusters=2, random_state=0),
        SCAD(n_clusters=2, variant="scad2", random_state=0),
        FeatureWeightedFCM(n_clusters=2, random_state=0),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("cls", AO_ESTIMATORS)
def test_pandas_input_and_feature_names(cls, blobs):
    X, _ = blobs
    df = pd.DataFrame(X, columns=["a", "b"])
    est = cls(n_clusters=3, random_state=0).fit(df)
    assert list(est.feature_names_in_) == ["a", "b"]
    assert est.predict_proba(df).shape == (X.shape[0], 3)
    D = est.set_output(transform="pandas").transform(df)
    assert isinstance(D, pd.DataFrame) and list(D.columns) == [f"dist_cluster_{i}" for i in range(3)]
    with pytest.raises(ValueError):
        est.predict(df.rename(columns={"a": "zz"}))


@pytest.mark.parametrize("cls", AO_ESTIMATORS)
def test_reproducible_and_n_init(cls, blobs):
    X, _ = blobs
    a = cls(n_clusters=3, random_state=3).fit(X)
    b = cls(n_clusters=3, random_state=3).fit(X)
    np.testing.assert_allclose(a.membership_, b.membership_)
    single = [cls(n_clusters=3, n_init=1, random_state=s).fit(X).objective_ for s in range(3)]
    multi = cls(n_clusters=3, n_init=3, random_state=0).fit(X)
    assert multi.objective_ <= max(single) + 1e-9
    with pytest.raises(ValueError):
        cls(n_clusters=3, n_init=0).fit(X)


# ------------------------------------------------------------------ fuzzifiers
def test_membership_formulas_are_partitions():
    d2 = np.array([[0.1, 1.0, 5.0], [1.0, 1.0, 1.0], [0.0, 0.2, 0.3]])
    for U in (entropy_memberships(d2, 0.7), quadratic_memberships(d2, 0.7), polynomial_memberships(d2, 0.5)):
        np.testing.assert_allclose(U.sum(axis=1), 1.0)
        assert (U >= 0).all()
        np.testing.assert_allclose(U[1], 1 / 3)  # equidistant sample
    assert quadratic_memberships(d2, 0.7)[0, 2] == 0.0
    assert polynomial_memberships(d2, 0.5)[0, 2] == 0.0
    # beta = 0 is FCM with m = 2
    U_fcm = (1 / (d2 + 1e-10)) / (1 / (d2 + 1e-10)).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(polynomial_memberships(d2, 0.0), U_fcm, atol=1e-8)
    np.testing.assert_allclose(polynomial_fuzzifier(np.array([0.0, 1.0]), 0.5), [0.0, 1.0])


def test_entropy_fcm_small_lambda_is_hard(blobs):
    X, y = blobs
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    efcm = RegularizedFCM(n_clusters=3, lam=1e-4, init=km.cluster_centers_).fit(X)
    assert efcm.membership_.max(axis=1).min() > 0.999
    assert adjusted_rand_score(km.labels_, efcm.labels_) == 1.0
    # large lambda: nearly uniform memberships and the prototypes collapse to the grand mean
    flat = RegularizedFCM(n_clusters=3, lam=1e4, random_state=0).fit(X)
    assert flat.membership_.max() < 0.34
    assert np.linalg.norm(flat.cluster_centers_ - X.mean(axis=0), axis=1).max() < 0.1


def test_entropy_fcm_lambda_controls_fuzziness_and_objective(blobs):
    X, _ = blobs
    pcs = [
        validity_indices(X, RegularizedFCM(n_clusters=3, lam=lam, random_state=0).fit(X).membership_, None, 1,
                         ["PC"])["PC"]
        for lam in (0.1, 1.0, 5.0)
    ]
    assert pcs[0] > pcs[1] > pcs[2]
    est = RegularizedFCM(n_clusters=3, lam=1.0, random_state=0).fit(X)
    assert est.lam_ == 1.0 and est.covariances_.shape == (3, 2, 2)
    assert np.all(np.diff(est.objective_history_) <= 1e-6)


def test_quadratic_fcm_gives_exact_zeros(blobs):
    X, _ = blobs
    q = RegularizedFCM(n_clusters=3, regularization="quadratic", lam=2.0, random_state=0).fit(X)
    assert (q.membership_ == 0.0).mean() > 0.3
    np.testing.assert_allclose(q.membership_.sum(axis=1), 1.0)
    few = RegularizedFCM(n_clusters=3, regularization="quadratic", lam=0.2, random_state=0).fit(X)
    many = RegularizedFCM(n_clusters=3, regularization="quadratic", lam=20.0, random_state=0).fit(X)
    assert (few.membership_ == 0.0).mean() > (many.membership_ == 0.0).mean()


def test_polynomial_fuzzifier_zero_memberships_and_beta(blobs):
    X, y = blobs
    pf = PolynomialFuzzifierFCM(n_clusters=3, beta=0.5, random_state=0).fit(X)
    assert (pf.membership_ == 0.0).mean() > 0.3
    np.testing.assert_allclose(pf.membership_.sum(axis=1), 1.0)
    assert pf.n_active_.min() >= 1 and pf.n_active_.max() <= 3
    assert adjusted_rand_score(y, pf.labels_) > 0.95
    fcm = FCM(n_clusters=3, m=2.0, init=pf.cluster_centers_).fit(X)
    pf0 = PolynomialFuzzifierFCM(n_clusters=3, beta=0.0, init=pf.cluster_centers_).fit(X)
    np.testing.assert_allclose(pf0.membership_, fcm.membership_, atol=1e-6)
    zeros = [
        (PolynomialFuzzifierFCM(n_clusters=3, beta=b, random_state=0).fit(X).membership_ == 0).mean()
        for b in (0.1, 0.5, 0.9)
    ]
    assert zeros[0] <= zeros[1] <= zeros[2]
    with pytest.raises(ValueError):
        PolynomialFuzzifierFCM(n_clusters=3, beta=1.0).fit(X)


def test_deterministic_annealing_phase_transitions(blobs):
    X, y = blobs
    da = RegularizedFCM(n_clusters=5, lam=0.3, annealing=True, anneal_rate=0.8, random_state=0).fit(X)
    hist = da.cluster_history_
    assert list(hist.columns) == ["lam", "n_effective", "objective"]
    assert hist["lam"].iloc[0] > hist["lam"].iloc[-1] == pytest.approx(0.3)
    assert hist["n_effective"].iloc[0] == 1  # all prototypes coincide at high temperature
    assert np.all(np.diff(hist["n_effective"]) >= 0)  # clusters only split while cooling
    assert da.n_effective_clusters_ == hist["n_effective"].iloc[-1] >= 3
    assert da.critical_temperatures_.shape == (5,)
    # at an intermediate final temperature exactly the three groups are resolved
    da3 = RegularizedFCM(n_clusters=5, lam=1.5, annealing=True, random_state=0).fit(X)
    assert da3.n_effective_clusters_ == 3
    for kw in (dict(regularization="quadratic", annealing=True), dict(annealing=True, anneal_rate=1.0),
               dict(annealing=True, lam_start=0.1, lam=0.5), dict(lam=0.0)):
        with pytest.raises(ValueError):
            RegularizedFCM(n_clusters=3, **kw).fit(X)


# ------------------------------------------------------------------ subtractive and mountain
@pytest.mark.parametrize("c", [2, 3, 4, 5])
def test_subtractive_finds_planted_number_of_clusters(c):
    X, y = make_fuzzy_blobs(n_clusters=c, random_state=1)
    sub = SubtractiveClustering(radii=0.3).fit(X)
    assert sub.n_clusters_ == c
    assert sub.cluster_centers_.shape == (c, 2) and sub.potentials_[0] == 1.0
    assert np.all(np.diff(sub.potentials_) <= 0)
    assert adjusted_rand_score(y, sub.labels_) > 0.9
    np.testing.assert_array_equal(sub.cluster_centers_, X[sub.center_indices_])


def test_subtractive_radii_and_ratios(blobs4):
    X, _ = blobs4
    counts = [SubtractiveClustering(radii=r).fit(X).n_clusters_ for r in (0.15, 0.3, 0.8)]
    assert counts[0] >= counts[1] >= counts[2]
    lenient = SubtractiveClustering(radii=0.3, reject_ratio=0.01, accept_ratio=0.05).fit(X).n_clusters_
    strict = SubtractiveClustering(radii=0.3, reject_ratio=0.5, accept_ratio=0.9).fit(X).n_clusters_
    assert lenient >= strict
    assert SubtractiveClustering(radii=0.3, squash=3.0).fit(X).n_clusters_ <= 4
    assert SubtractiveClustering(radii=0.3, max_clusters=2).fit(X).n_clusters_ == 2
    per_feature = SubtractiveClustering(radii=[0.3, 0.3]).fit(X)
    assert per_feature.n_clusters_ == SubtractiveClustering(radii=0.3).fit(X).n_clusters_
    bad = (dict(radii=-1), dict(radii=[0.3] * 3), dict(accept_ratio=0.1, reject_ratio=0.5), dict(squash=0))
    for kw in bad:
        with pytest.raises(ValueError):
            SubtractiveClustering(**kw).fit(X)


def test_subtractive_as_initializer_and_rule_number_estimator(blobs4):
    X, y = blobs4
    sub = SubtractiveClustering(radii=0.3).fit(X)
    fcm = FCM(n_clusters=sub.n_clusters_, init=sub.cluster_centers_).fit(X)
    assert fcm.n_iter_ < FCM(n_clusters=4, random_state=0).fit(X).n_iter_ + 5
    assert adjusted_rand_score(y, fcm.labels_) > 0.95
    gk = GK(n_clusters=sub.n_clusters_, init=sub.cluster_centers_).fit(X)
    assert gk.membership_.shape == (X.shape[0], 4)
    # product-space clustering of a regression problem gives a rule number for a TS model
    Xs, ys = make_sinc(n_samples=200, random_state=0)
    Z = np.c_[Xs, ys]
    sub_z = SubtractiveClustering(radii=0.3).fit(Z)
    assert 4 <= sub_z.n_clusters_ <= 8
    ts = TakagiSugeno(n_clusters=sub_z.n_clusters_, random_state=0).fit(Xs, ys)
    assert ts.score(Xs, ys) > 0.9
    # predict_proba / transform on new data
    P = sub.predict_proba(X[:5])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert sub.transform(X[:5]).shape == (5, 4)
    assert sub.predict(X[:5]).shape == (5,)


def test_mountain_method(blobs):
    X, y = blobs
    mc = MountainClustering(grid_size=15, radii=0.3).fit(X)
    assert mc.n_clusters_ == 3 and mc.grid_.shape == (225, 2) and mc.mountain_.shape == (225,)
    assert mc.grid_size_ == 15
    assert adjusted_rand_score(y, mc.labels_) > 0.9
    assert center_error(mc.cluster_centers_, np.array([X[y == i].mean(axis=0) for i in range(3)])) < 1.0
    auto = MountainClustering(radii=0.3).fit(X)
    assert auto.grid_size_ == 100 and auto.grid_.shape == (10000, 2)
    assert MountainClustering(grid_size=15, radii=0.3, stop_ratio=0.9).fit(X).n_clusters_ <= 3
    assert MountainClustering(grid_size=15, radii=0.3, max_clusters=2).fit(X).n_clusters_ == 2
    coarse = MountainClustering(grid_size=5, radii=0.3).fit(X)
    assert coarse.grid_.shape == (25, 2)
    with pytest.raises(ValueError):
        MountainClustering(grid_size=200).fit(X)  # 40000 nodes > max_nodes
    with pytest.raises(ValueError):
        MountainClustering(grid_size=1).fit(X)
    rng = np.random.RandomState(0)
    high = MountainClustering().fit(rng.rand(30, 8))  # grid adapts to 3 ** 8 = 6561 nodes
    assert high.grid_size_ == 3


# ------------------------------------------------------------------ merging
def test_cluster_similarity_and_reference_fuzziness():
    U = labels_to_membership(np.array([0, 0, 1, 1, 2, 2]))
    S = cluster_similarity(U)
    np.testing.assert_allclose(S, np.eye(3))
    Uf = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert cluster_similarity(Uf)[0, 1] == pytest.approx(1.0)
    # rescaling a partition from m = 3 to m = 2 reproduces the FCM memberships for m = 2
    d2 = np.random.RandomState(0).rand(10, 3)
    U3 = (d2 ** (-1 / 2)) / (d2 ** (-1 / 2)).sum(axis=1, keepdims=True)
    U2 = (1 / d2) / (1 / d2).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(reference_fuzziness(U3, 3.0), U2, atol=1e-6)
    np.testing.assert_allclose(reference_fuzziness(U, 1.0), U)


def test_geometric_compatibility_criteria():
    V = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [10.0, 10.0]])
    thin = np.diag([1.0, 0.01])  # elongated along x, normal along y
    F = np.array([thin, thin, thin, np.eye(2)])
    C = geometric_compatibility(V, F)
    assert C[0, 1] > 0  # collinear, close
    assert C[0, 2] == 0  # parallel normals but the center is off the common line
    assert C[0, 3] == 0 and C[1, 3] == 0  # the round cluster is not thin
    assert C[0, 1] == C[1, 0] and np.all(np.diag(C) == 0)
    assert geometric_compatibility(V, F, distance_threshold=0.5)[0, 1] == 0


def test_merging_recovers_three_lines_from_eight(lines):
    X, y = lines
    ccm = CompatibleClusterMerging(GK(random_state=0, n_init=3), start_n_clusters=8).fit(X)
    assert ccm.n_clusters_ == 3
    assert adjusted_rand_score(y, ccm.labels_) > 0.85
    assert [h["n_clusters"] for h in ccm.merge_history_] == [8, 7, 6, 5, 4]
    assert ccm.compatibility_.max() == 0.0
    assert ccm.membership_.shape == (X.shape[0], 3) and ccm.estimator_.n_clusters == 3
    assert ccm.predict(X).shape == (X.shape[0],) and ccm.transform(X).shape == (X.shape[0], 3)
    assert list(ccm.get_feature_names_out()) == [f"dist_cluster_{i}" for i in range(3)]


@pytest.mark.parametrize("c", [2, 3, 4, 5])
def test_merging_recovers_blobs_by_similarity(c):
    X, y = make_fuzzy_blobs(n_clusters=c, random_state=2)
    ccm = CompatibleClusterMerging(FCM(random_state=0), start_n_clusters=8, criterion="similarity").fit(X)
    assert ccm.n_clusters_ == c
    assert adjusted_rand_score(y, ccm.labels_) > 0.9


def test_merging_parameters(blobs, lines):
    X, y = blobs
    # the exponent of the base estimator does not change the result thanks to the rescaling
    for m in (1.5, 2.5):
        ccm_m = CompatibleClusterMerging(FCM(m=m, random_state=0), 8, criterion="similarity").fit(X)
        assert ccm_m.n_clusters_ == 3
    floor = CompatibleClusterMerging(FCM(random_state=0), 8, min_n_clusters=5, criterion="similarity").fit(X)
    assert floor.n_clusters_ == 5
    strict = CompatibleClusterMerging(
        FCM(random_state=0), 8, criterion="similarity", similarity_threshold=0.95
    ).fit(X)
    assert strict.n_clusters_ == 8 and strict.merge_history_ == []
    # geometric criterion refuses to merge round clusters
    geo = CompatibleClusterMerging(GK(random_state=0), 6).fit(X)
    assert geo.n_clusters_ == 6
    # loose thresholds merge more on the lines, strict ones less
    Xl, _ = lines
    loose = CompatibleClusterMerging(
        GK(random_state=0, n_init=3),
        8,
        parallel_threshold=0.5,
        hyperplane_threshold=5.0,
        distance_threshold=20.0,
        thin_threshold=1.0,
    ).fit(Xl)
    tight = CompatibleClusterMerging(GK(random_state=0, n_init=3), 8, parallel_threshold=0.9999).fit(Xl)
    assert loose.n_clusters_ < 3 <= tight.n_clusters_
    assert CompatibleClusterMerging(start_n_clusters=4).fit(X).estimator_.__class__ is GK
    for kw in (dict(criterion="foo"), dict(start_n_clusters=0), dict(min_n_clusters=9)):
        with pytest.raises(ValueError):
            CompatibleClusterMerging(FCM(random_state=0), **{"start_n_clusters": 8, **kw}).fit(X)


# ------------------------------------------------------------------ volume prototypes
def test_volume_prototypes_reduce_size_equalization():
    X, y = make_different_densities(random_state=0)
    fcm = FCM(n_clusters=3, random_state=0, n_init=3).fit(X)
    vp = VolumePrototypeFCM(n_clusters=3, random_state=0, n_init=3).fit(X)
    assert adjusted_rand_score(y, vp.labels_) > adjusted_rand_score(y, fcm.labels_)
    assert vp.radii_.shape == (3,) and (vp.radii_ > 0).all()
    assert vp.n_clusters_ == 3 and vp.merge_history_ == []
    # inside a core the distance is zero and the membership one
    inside = vp.distances_ == 0.0
    assert inside.any()
    assert vp.membership_[inside.any(axis=1)].max(axis=1).min() > 0.99
    # zero radius recovers FCM
    vp0 = VolumePrototypeFCM(n_clusters=3, radius=0.0, init=fcm.cluster_centers_).fit(X)
    fcm0 = FCM(n_clusters=3, init=fcm.cluster_centers_).fit(X)
    np.testing.assert_allclose(vp0.membership_, fcm0.membership_, atol=1e-6)
    big = VolumePrototypeFCM(n_clusters=3, radius_scale=1.5, random_state=0).fit(X)
    small = VolumePrototypeFCM(n_clusters=3, radius_scale=0.5, random_state=0).fit(X)
    assert big.radii_.sum() > small.radii_.sum()
    fixed = VolumePrototypeFCM(n_clusters=3, radius=[0.5, 0.5, 0.5], random_state=0).fit(X)
    np.testing.assert_allclose(fixed.radii_, 0.5)
    adaptive = VolumePrototypeFCM(n_clusters=3, norm="adaptive", random_state=0).fit(X)
    assert adaptive.norm_matrices_.shape == (3, 2, 2)
    assert not np.allclose(adaptive.norm_matrices_[0], np.eye(2))
    for kw in (dict(norm="foo"), dict(radius="x"), dict(radius=[1, 2]), dict(radius_scale=-1), dict(m=1.0)):
        with pytest.raises(ValueError):
            VolumePrototypeFCM(n_clusters=3, **kw).fit(X)


@pytest.mark.parametrize("c", [3, 4, 5])
def test_volume_prototype_merging_finds_number_of_clusters(c):
    X, y = make_fuzzy_blobs(n_clusters=c, random_state=1)
    vpm = VolumePrototypeFCM(n_clusters=9, merge=True, random_state=0).fit(X)
    assert vpm.n_clusters_ == c
    assert adjusted_rand_score(y, vpm.labels_) > 0.9
    assert len(vpm.merge_history_) == 9 - c
    assert vpm.membership_.shape == (X.shape[0], c) and vpm.predict_proba(X[:4]).shape == (4, c)
    assert list(vpm.get_feature_names_out()) == [f"dist_cluster_{i}" for i in range(c)]
    assert vpm.transform(X[:4]).shape == (4, c)
    strict = VolumePrototypeFCM(n_clusters=9, merge=True, similarity_threshold=0.99, random_state=0).fit(X)
    assert strict.n_clusters_ == 9


def test_volume_prototype_merging_max_iter_is_per_stage():
    # a long merge sequence needs more than max_iter iterations in total; the budget applies per c
    X, y = make_different_densities(random_state=0)
    vpm = VolumePrototypeFCM(n_clusters=8, merge=True, random_state=0, max_iter=150).fit(X)
    assert vpm.n_clusters_ == 3 and vpm.n_iter_ > 150
    assert adjusted_rand_score(y, vpm.labels_) > 0.8
    capped = VolumePrototypeFCM(n_clusters=8, merge=True, random_state=0, max_iter=5).fit(X)
    assert capped.n_iter_ <= 5 * 8


# ------------------------------------------------------------------ feature weighting
def test_scad_down_weights_irrelevant_feature(noisy_feature_blobs):
    X, y = noisy_feature_blobs
    for variant in ("scad1", "scad2"):
        scad = SCAD(n_clusters=3, variant=variant, random_state=0, n_init=3).fit(X)
        W = scad.feature_weights_
        assert W.shape == (3, 3)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        assert (W[:, 2] < 0.1).all() and (W[:, :2] > 0.25).all()
        assert adjusted_rand_score(y, scad.labels_) > 0.95
        assert scad.distances_.shape == (X.shape[0], 3)
    s1 = SCAD(n_clusters=3, variant="scad1", random_state=0, n_init=3).fit(X)
    assert (s1.feature_weights_ == 0.0).any()  # SCAD-1 gives exact zeros
    assert (s1.delta_ > 0).all()
    s2 = SCAD(n_clusters=3, variant="scad2", random_state=0, n_init=3).fit(X)
    assert (s2.feature_weights_ > 0.0).all()  # SCAD-2 never does
    # a smaller q sharpens the weights, a fixed delta is honored
    sharp = SCAD(n_clusters=3, variant="scad2", q=1.2, random_state=0, n_init=3).fit(X)
    assert sharp.feature_weights_.max(axis=1).mean() > s2.feature_weights_.max(axis=1).mean()
    fixed = SCAD(n_clusters=3, delta=1e6, random_state=0).fit(X)
    np.testing.assert_allclose(fixed.delta_, 1e6)
    np.testing.assert_allclose(fixed.feature_weights_, 1 / 3, atol=1e-3)  # huge delta: uniform weights
    scaled = SCAD(n_clusters=3, delta_scale=0.2, random_state=0, n_init=3).fit(X)
    assert (scaled.feature_weights_ == 0.0).sum() >= (s1.feature_weights_ == 0.0).sum()
    for kw in (dict(variant="foo"), dict(q=1.0), dict(delta=-1.0), dict(delta=[1, 2]), dict(delta_scale=0)):
        with pytest.raises(ValueError):
            SCAD(n_clusters=3, **kw).fit(X)


def test_weighted_fcm_global_weights(noisy_feature_blobs):
    X, y = noisy_feature_blobs
    w = FeatureWeightedFCM(n_clusters=3, random_state=0, n_init=3).fit(X)
    assert w.feature_weights_.shape == (3,)
    assert w.feature_weights_.sum() == pytest.approx(1.0)
    assert w.feature_weights_.argmin() == 2 and w.feature_weights_[2] < 0.1
    assert adjusted_rand_score(y, w.labels_) > 0.95
    # the weights sharpen as t -> 1 and flatten for large t
    sharp = FeatureWeightedFCM(n_clusters=3, t=1.3, random_state=0, n_init=3).fit(X)
    assert sharp.feature_weights_.max() > w.feature_weights_.max()
    flat = FeatureWeightedFCM(n_clusters=3, t=6.0, random_state=0, n_init=3).fit(X)
    assert flat.feature_weights_.max() < w.feature_weights_.max()
    np.testing.assert_allclose(w.transform(X[:3]) ** 2, w.predict_proba(X[:3]) * 0 + w._distances(X[:3]))
    with pytest.raises(ValueError):
        FeatureWeightedFCM(n_clusters=3, t=1.0).fit(X)


# ------------------------------------------------------------------ validity: comparing partitions
def test_fuzzy_rand_equals_crisp_rand_on_hard_partitions():
    rng = np.random.RandomState(0)
    l1, l2 = rng.randint(0, 3, 80), rng.randint(0, 4, 80)
    U1, U2 = labels_to_membership(l1), labels_to_membership(l2)
    for method in ("huellermeier", "campello"):
        for tnorm in (None, "minimum", "product", "lukasiewicz"):
            assert fuzzy_rand_index(U1, U2, method, tnorm) == pytest.approx(rand_score(l1, l2))
            ari = fuzzy_adjusted_rand_index(U1, U2, method, tnorm)
            assert ari == pytest.approx(adjusted_rand_score(l1, l2))
    # Jaccard on hard partitions: a / (a + b + c) from the pair counts
    same = (l1[:, None] == l1[None, :]) & (l2[:, None] == l2[None, :])
    either = (l1[:, None] == l1[None, :]) | (l2[:, None] == l2[None, :])
    tri = np.triu(np.ones_like(same), 1).astype(bool)
    assert fuzzy_jaccard_index(U1, U2) == pytest.approx(same[tri].sum() / either[tri].sum())


def test_fuzzy_rand_properties():
    rng = np.random.RandomState(1)
    U = rng.dirichlet(np.ones(3), 50)
    assert fuzzy_rand_index(U, U) == pytest.approx(1.0)  # Hüllermeier's index is reflexive
    assert fuzzy_adjusted_rand_index(U, U) == pytest.approx(1.0)
    assert fuzzy_jaccard_index(U, U) == pytest.approx(1.0)
    assert fuzzy_rand_index(U, U, method="campello") < 1.0  # Campello's is not
    perm = U[:, [2, 0, 1]]
    assert fuzzy_rand_index(U, perm) == pytest.approx(1.0)  # invariant to cluster order
    self_c = fuzzy_rand_index(U, U, method="campello")
    assert fuzzy_rand_index(U, perm, method="campello") == pytest.approx(self_c)
    V = rng.dirichlet(np.ones(4), 50)
    assert 0.0 <= fuzzy_rand_index(U, V) <= 1.0
    assert fuzzy_rand_index(U, V) == pytest.approx(fuzzy_rand_index(V, U))
    assert abs(fuzzy_adjusted_rand_index(U, V)) < 0.3  # independent partitions
    with pytest.raises(ValueError):
        fuzzy_rand_index(U, V[:10])
    with pytest.raises(ValueError):
        fuzzy_rand_index(U, V, method="foo")


def test_fuzzy_rand_between_algorithms(blobs):
    X, y = blobs
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    gk = GK(n_clusters=3, random_state=0).fit(X)
    assert fuzzy_rand_index(fcm.membership_, gk.membership_) > 0.9
    assert fuzzy_adjusted_rand_index(fcm.membership_, labels_to_membership(y)) > 0.8
    # a fuzzier partition of the same data is less in agreement with the crisp truth
    fuzzy = FCM(n_clusters=3, m=4.0, random_state=0).fit(X)
    assert fuzzy_rand_index(fuzzy.membership_, labels_to_membership(y)) < fuzzy_rand_index(
        fcm.membership_, labels_to_membership(y)
    )


# ------------------------------------------------------------------ validity: stability and new indices
def test_stability_peaks_at_true_c(blobs4):
    X, _ = blobs4
    for method in ("subsample", "split"):
        table = stability_selection(
            FCM(random_state=0), X, range(2, 7), n_splits=6, method=method, random_state=0
        )
        assert table.shape == (5, 6) and table.index.name == "n_clusters"
        assert list(table.columns) == [f"split_{s}" for s in range(6)]
        assert table.mean(axis=1).idxmax() == 4
        assert table.loc[4].min() > 0.9
    rand = stability_selection(FCM(random_state=0), X, [3, 4], n_splits=3, index="fuzzy_rand", random_state=0)
    assert (rand.to_numpy() <= 1.0).all()
    same = stability_selection(FCM(random_state=0), X, [4], n_splits=3, random_state=1)
    again = stability_selection(FCM(random_state=0), X, [4], 3, random_state=1)
    np.testing.assert_allclose(same.to_numpy(), again.to_numpy())
    for kw in (dict(method="foo"), dict(index="foo"), dict(subsample=1.0)):
        with pytest.raises(ValueError):
            stability_selection(FCM(random_state=0), X, [3], n_splits=2, **kw)


def test_kwon_and_pcaes_registered_and_select_true_c():
    X, y = make_fuzzy_blobs(n_clusters=5, n_samples_per_cluster=80, spread=0.55, box=12.0, random_state=2)
    assert INDEX_DIRECTION["KWON"] == "min" and INDEX_DIRECTION["PCAES"] == "max"
    table = scan_n_clusters(FCM(random_state=0, n_init=3), X, range(2, 9), indices=["KWON", "PCAES", "XB"])
    suggestion = suggest_n_clusters(table)
    assert suggestion["KWON"] == 5
    # PCAES normalizes by the smallest cluster mass (as published), so a tiny spurious cluster
    # inflates it for c > 5; among the partitions without such a cluster it still picks c = 5
    assert table.loc[2:5, "PCAES"].idxmax() == 5
    fcm = FCM(n_clusters=5, random_state=0).fit(X)
    res = validity_indices(X, fcm.membership_, fcm.cluster_centers_)
    assert set(res) == set(INDEX_DIRECTION) and "KWON" in res and "PCAES" in res
    assert res["KWON"] == pytest.approx(kwon_index(X, fcm.membership_, fcm.cluster_centers_))
    assert res["PCAES"] == pytest.approx(pcaes_index(X, fcm.membership_, fcm.cluster_centers_))
    # Kwon adds a punishing term to the Xie-Beni ratio, so it is never below XB (for m = 2)
    assert res["KWON"] >= res["XB"]
    assert res["PCAES"] >= 0.0  # first sum >= c, penalty <= c
    one = labels_to_membership(np.zeros(X.shape[0], dtype=int), 1)
    assert np.isnan(kwon_index(X, one, X.mean(axis=0, keepdims=True)))
    assert np.isnan(pcaes_index(X, one, X.mean(axis=0, keepdims=True)))
    ax = plot_validity_scan(table)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "KWON (min)" in labels and "PCAES (max)" in labels
    matplotlib.pyplot.close("all")
