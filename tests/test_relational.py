import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.cluster import FCM  # noqa: E402
from fuzzyclust.datasets import make_fuzzy_blobs  # noqa: E402
from fuzzyclust.relational import (  # noqa: E402
    NERFCM,
    FuzzyCMedoids,
    FuzzyKModes,
    FuzzyKPrototypes,
    RelationalFCM,
    autocorrelation_distance_matrix,
    autocorrelation_features,
    check_dissimilarity_matrix,
    dtw_alignment,
    dtw_distance_matrix,
    gower_distance_matrix,
    make_categorical_blobs,
    make_mixed_blobs,
    make_warped_series,
    matching_dissimilarity,
    pairwise_dissimilarity,
    plot_dissimilarity_matrix,
    plot_dtw_alignment,
    plot_modes,
    plot_series_clusters,
    relational_distances,
)

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


@pytest.fixture(scope="module")
def categorical():
    return make_categorical_blobs(random_state=0)


@pytest.fixture(scope="module")
def mixed():
    return make_mixed_blobs(random_state=0)


@pytest.fixture(scope="module")
def series():
    return make_warped_series(random_state=0)


@pytest.fixture(scope="module")
def series_dtw(series):
    X, _ = series
    return dtw_distance_matrix(X, window=12)


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    if getattr(estimator, "metric", None) == "precomputed":
        return {
            "check_clustering": (
                "The check fits raw (50, 2) blob coordinates although the estimator declares the pairwise "
                "tag; a precomputed-dissimilarity estimator must reject a non-square input, as "
                "sklearn's own precomputed clusterers do."
            )
        }
    if isinstance(estimator, FuzzyKModes):
        return {
            "check_clustering": (
                "Fuzzy k-modes treats every distinct value as a category: on continuous blob coordinates "
                "all values are unique, the matching dissimilarity carries no structure and the ARI "
                "assertion of the check cannot hold. The algorithm is for categorical data."
            )
        }
    return {}


@parametrize_with_checks(
    [
        RelationalFCM(n_clusters=2, random_state=0),
        RelationalFCM(n_clusters=2, metric="precomputed", random_state=0),
        NERFCM(n_clusters=2, random_state=0),
        NERFCM(n_clusters=2, metric="precomputed", random_state=0),
        FuzzyCMedoids(n_clusters=2, random_state=0),
        FuzzyCMedoids(n_clusters=2, n_candidates=3, trim=0.1, random_state=0),
        FuzzyCMedoids(n_clusters=2, metric="precomputed", random_state=0),
        FuzzyKModes(n_clusters=2, random_state=0),
        FuzzyKPrototypes(n_clusters=2, random_state=0),
        FuzzyKPrototypes(n_clusters=2, categorical_features=[0], random_state=0),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ dissimilarity helpers
def test_pairwise_dissimilarity_and_validation(blobs):
    X, _ = blobs
    R = pairwise_dissimilarity(X)
    np.testing.assert_allclose(R, cdist(X, X))
    np.testing.assert_allclose(pairwise_dissimilarity(X, metric="sqeuclidean"), R**2, rtol=1e-10, atol=1e-8)
    C = pairwise_dissimilarity(X[:5], X[:7], metric="cityblock")
    assert C.shape == (5, 7)
    np.testing.assert_allclose(pairwise_dissimilarity(R, metric="precomputed"), R)
    assert pairwise_dissimilarity(X[:1]).shape == (1, 1)
    with pytest.raises(ValueError):
        pairwise_dissimilarity(R, Y=R, metric="precomputed")
    with pytest.raises(ValueError, match="symmetric"):
        check_dissimilarity_matrix(R + np.triu(np.ones_like(R), 1))
    with pytest.raises(ValueError, match="diagonal"):
        check_dissimilarity_matrix(R + np.eye(R.shape[0]))
    with pytest.raises(ValueError, match="Negative"):
        check_dissimilarity_matrix(-R)
    with pytest.raises(ValueError, match="square"):
        check_dissimilarity_matrix(R[:5])
    with pytest.raises(ValueError, match="training objects"):
        check_dissimilarity_matrix(R[:5, :10], n_objects=20, square=False)
    with pytest.raises(ValueError):
        check_dissimilarity_matrix(np.array([[0.0, np.nan], [np.nan, 0.0]]))


def test_relational_distances_match_euclidean(blobs):
    X, _ = blobs
    R = pairwise_dissimilarity(X, metric="sqeuclidean")
    rng = np.random.RandomState(0)
    V = rng.rand(3, X.shape[0])
    V /= V.sum(axis=1, keepdims=True)
    d2 = relational_distances(R, V)
    np.testing.assert_allclose(d2, cdist(X, V @ X) ** 2, atol=1e-8)
    # the beta-spread adds beta/2 * ||e_k - v_i||^2
    shift = 0.5 * 3.0 * (1.0 - 2.0 * V.T + (V**2).sum(axis=1)[None, :])
    np.testing.assert_allclose(relational_distances(R, V, beta=3.0), d2 + shift, atol=1e-8)
    with pytest.raises(ValueError):
        relational_distances(R[:5], V)


def test_dtw_distance_properties(series):
    X, y = series
    D0 = dtw_distance_matrix(X, window=0)
    np.testing.assert_allclose(D0, cdist(X, X), atol=1e-8)  # band 0 = Euclidean
    D = dtw_distance_matrix(X, window=12)
    Dfull = dtw_distance_matrix(X)
    assert D.shape == (X.shape[0], X.shape[0])
    np.testing.assert_allclose(D, D.T)
    np.testing.assert_allclose(np.diag(D), 0.0)
    assert np.all(D <= D0 + 1e-9)  # warping can only lower the cost
    assert np.all(Dfull <= D + 1e-9)  # a wider band can only lower it further
    # a time-shifted copy is far in Euclidean terms but close under DTW
    t = np.linspace(0, 1, 60)
    pair = np.vstack([np.exp(-0.5 * ((t - 0.4) / 0.08) ** 2), np.exp(-0.5 * ((t - 0.55) / 0.08) ** 2)])
    e = dtw_distance_matrix(pair, window=0)[0, 1]
    d = dtw_distance_matrix(pair, window=12)[0, 1]
    assert d < 0.2 * e
    # fraction window and rectangular output
    assert dtw_distance_matrix(X[:4], X[:6], window=0.1).shape == (4, 6)
    # the alignment reproduces the matrix entry and is a valid monotone path
    dist, path = dtw_alignment(X[0], X[1], window=12)
    assert dist == pytest.approx(D[0, 1], rel=1e-9)
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (59, 59)
    steps = np.diff(path, axis=0)
    assert np.all(steps >= 0) and np.all(steps <= 1) and np.all(steps.sum(axis=1) >= 1)
    assert np.all(np.abs(path[:, 0] - path[:, 1]) <= 12)
    # multichannel series
    X3 = np.stack([X[:5], X[:5] * 0.5], axis=2)
    assert dtw_distance_matrix(X3, window=3).shape == (5, 5)
    with pytest.raises(ValueError):
        dtw_distance_matrix(X, window=-1)
    with pytest.raises(ValueError):
        dtw_distance_matrix(X[:3], X3)
    with pytest.raises(ValueError):
        dtw_distance_matrix(np.ones((2, 2, 2, 2)))


def test_dtw_beats_euclidean_on_warped_series(series, series_dtw):
    X, y = series
    De = pairwise_dissimilarity(X)
    fe = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(De)
    fd = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(series_dtw)
    assert adjusted_rand_score(y, fd.labels_) > 0.9
    assert adjusted_rand_score(y, fe.labels_) < 0.7


def test_autocorrelation_distance():
    rng = np.random.RandomState(0)
    e = rng.randn(30, 300)
    ar = np.empty_like(e)
    ar[:, 0] = e[:, 0]
    for t in range(1, 300):
        ar[:, t] = 0.9 * ar[:, t - 1] + e[:, t]
    X = np.vstack([ar[:15], e[15:]])
    y = np.r_[np.zeros(15, int), np.ones(15, int)]
    A = autocorrelation_features(X, n_lags=5)
    assert A.shape == (30, 5)
    assert A[:15, 0].mean() > 0.8 and abs(A[15:, 0].mean()) < 0.2
    D = autocorrelation_distance_matrix(X, n_lags=5)
    assert D.shape == (30, 30) and np.allclose(np.diag(D), 0.0)
    # white-noise and AR series separate; a constant series does not break the estimate
    model = RelationalFCM(n_clusters=2, metric="precomputed", random_state=0).fit(D**2)
    assert adjusted_rand_score(y, model.labels_) == 1.0
    assert np.all(np.isfinite(autocorrelation_features(np.ones((1, 20)), 3)))
    W = autocorrelation_distance_matrix(X, X[:3], n_lags=5, weights=0.5 ** np.arange(5))
    assert W.shape == (30, 3)
    with pytest.raises(ValueError):
        autocorrelation_features(X, n_lags=300)
    with pytest.raises(ValueError):
        autocorrelation_distance_matrix(X, n_lags=5, weights=[1, 2])


def test_gower_distance(mixed):
    df = mixed.data
    D = gower_distance_matrix(df)
    assert D.shape == (len(df), len(df))
    assert np.all(D >= 0) and np.all(D <= 1 + 1e-12)
    np.testing.assert_allclose(D, D.T)
    # numeric only: scaled city-block mean; categorical only: matching share
    small = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": ["x", "x", "y"]})
    G = gower_distance_matrix(small)
    np.testing.assert_allclose(G[0, 1], 0.25)  # (0.5 + 0) / 2
    np.testing.assert_allclose(G[0, 2], 1.0)  # (1 + 1) / 2
    # missing values are left out of the average
    miss = pd.DataFrame({"a": [0.0, np.nan, 2.0], "b": ["x", "x", None]})
    Gm = gower_distance_matrix(miss)
    np.testing.assert_allclose(Gm[0, 1], 0.0)  # only "b" compared: equal
    np.testing.assert_allclose(Gm[0, 2], 1.0)  # only "a" compared: full range
    # explicit column specification and weights, new data against the training ranges
    Gs = gower_distance_matrix(small, categorical_features=["b"], feature_weights=[1.0, 0.0])
    np.testing.assert_allclose(Gs[0, 2], 1.0)
    new = gower_distance_matrix(small.iloc[:1], small, ranges=[2.0])
    np.testing.assert_allclose(new[0], G[0])
    arr = gower_distance_matrix(np.array([[0.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(arr[0, 1], 0.5)
    with pytest.raises(ValueError):
        gower_distance_matrix(small, feature_weights=[1.0])
    with pytest.raises(ValueError):
        gower_distance_matrix(small, categorical_features=["zzz"])
    with pytest.raises(ValueError):
        gower_distance_matrix(small, ranges=[1.0, 2.0])


def test_matching_dissimilarity():
    D = matching_dissimilarity([["a", "x"], ["b", "x"]], [["a", "x"], ["b", "y"]])
    np.testing.assert_array_equal(D, [[0, 2], [1, 1]])
    with pytest.raises(ValueError):
        matching_dissimilarity([["a"]], [["a", "b"]])


# ------------------------------------------------------------------ data generators
def test_generators(categorical, mixed, series):
    d = categorical
    assert d.data.shape == (300, 6) and d.modes.shape == (3, 6)
    assert np.bincount(d.target).tolist() == [100] * 3
    # modes differ in every attribute and the records follow them
    M = d.modes.to_numpy()
    assert all(len(set(M[:, j])) == 3 for j in range(6))
    agree = (d.data.to_numpy() == M[d.target]).mean()
    assert 0.7 < agree < 0.9  # noise 0.25 (a random replacement may hit the mode)
    arr = make_categorical_blobs(n_samples=40, n_features=3, as_frame=False, random_state=1)
    assert isinstance(arr.data, np.ndarray) and arr.data.shape == (40, 3)
    with pytest.raises(ValueError):
        make_categorical_blobs(n_categories=2, n_clusters=3)
    with pytest.raises(ValueError):
        make_categorical_blobs(noise=2.0)
    m = mixed
    assert m.data.shape == (300, 6) and m.centers.shape == (3, 3) and m.modes.shape == (3, 3)
    assert list(m.data.columns) == ["x1", "x2", "x3", "c1", "c2", "c3"]
    assert m.data.select_dtypes(include="number").shape[1] == 3
    numeric_only = make_mixed_blobs(n_categorical=0, random_state=0)
    assert numeric_only.data.shape[1] == 3 and numeric_only.modes.shape[1] == 0
    cat_only = make_mixed_blobs(n_numeric=0, random_state=0)
    assert cat_only.data.shape[1] == 3
    many = make_mixed_blobs(n_numeric=1, n_clusters=4, n_categories=4, random_state=0)
    assert many.centers.shape == (4, 1) and len(np.unique(many.centers)) == 4
    with pytest.raises(ValueError):
        make_mixed_blobs(n_numeric=0, n_categorical=0)
    X, y = series
    assert X.shape == (60, 60) and np.bincount(y).tolist() == [20, 20, 20]
    np.testing.assert_allclose(make_warped_series(random_state=3)[0], make_warped_series(random_state=3)[0])
    with pytest.raises(ValueError):
        make_warped_series(warp=0.5)


# ------------------------------------------------------------------ RFCM
def test_rfcm_is_the_dual_of_fcm(blobs):
    """Hathaway, Davenport, Bezdek: on squared Euclidean distances RFCM reproduces FCM exactly."""
    X, y = blobs
    R = pairwise_dissimilarity(X, metric="sqeuclidean")
    U0 = np.random.RandomState(0).rand(X.shape[0], 3)
    for m in (1.5, 2.0, 3.0):
        fcm = FCM(n_clusters=3, m=m, init=U0).fit(X)
        rfcm = RelationalFCM(n_clusters=3, m=m, metric="precomputed", init=U0).fit(R)
        np.testing.assert_allclose(rfcm.membership_, fcm.membership_, atol=1e-8)
        np.testing.assert_allclose(rfcm.objective_history_, fcm.objective_history_, rtol=1e-8)
        assert rfcm.n_iter_ == fcm.n_iter_
        np.testing.assert_allclose(rfcm.distances_, fcm.distances_, atol=1e-6)
    # computed from features: the dual centers are the FCM centers
    fcm = FCM(n_clusters=3, init=U0).fit(X)
    rf = RelationalFCM(n_clusters=3, init=U0).fit(X)
    np.testing.assert_allclose(rf.cluster_centers_, fcm.cluster_centers_, atol=1e-8)
    np.testing.assert_allclose(rf.relational_centers_.sum(axis=1), 1.0)
    assert rf.relational_centers_.shape == (3, X.shape[0])
    np.testing.assert_allclose(rf.dissimilarity_matrix_, R, atol=1e-8)
    assert rf.beta_ == 0.0 and rf.n_spread_updates_ == 0
    assert adjusted_rand_score(y, rf.labels_) > 0.95
    # with the precomputed metric the centers *are* the relational prototypes
    rp = RelationalFCM(n_clusters=3, metric="precomputed", init=U0).fit(R)
    np.testing.assert_array_equal(rp.cluster_centers_, rp.relational_centers_)


def test_rfcm_predicts_new_objects_and_pandas(blobs):
    X, _ = blobs
    R = pairwise_dissimilarity(X, metric="sqeuclidean")
    U0 = np.random.RandomState(0).rand(X.shape[0], 3)
    fcm = FCM(n_clusters=3, init=U0).fit(X)
    rp = RelationalFCM(n_clusters=3, metric="precomputed", init=U0).fit(R)
    rf = RelationalFCM(n_clusters=3, init=U0).fit(X)
    # training objects through their own rows
    np.testing.assert_allclose(rp.predict_proba(R[:10]), rp.membership_[:10], atol=1e-8)
    assert (rp.predict(R[:10]) == rp.labels_[:10]).all()
    # genuinely new objects: squared distances to the training objects reproduce FCM's predict_proba
    new = X[:10] + 0.3
    Rnew = cdist(new, X) ** 2
    np.testing.assert_allclose(rp.predict_proba(Rnew), fcm.predict_proba(new), atol=1e-8)
    np.testing.assert_allclose(rf.predict_proba(new), fcm.predict_proba(new), atol=1e-8)
    np.testing.assert_allclose(rp.transform(Rnew), fcm.transform(new), atol=1e-6)
    assert rp.score(R) == pytest.approx(-rp.objective_, rel=1e-6)
    assert rf.score(X) == pytest.approx(fcm.score(X), rel=1e-6)
    assert list(rp.get_feature_names_out()) == ["dist_cluster_0", "dist_cluster_1", "dist_cluster_2"]
    with pytest.raises(ValueError):
        rp.predict_proba(Rnew[:, :5])
    with pytest.raises(ValueError):
        rp.predict_proba(-Rnew)
    df = pd.DataFrame(X, columns=["a", "b"])
    pdm = RelationalFCM(n_clusters=3, init=U0).fit(df)
    assert list(pdm.feature_names_in_) == ["a", "b"]
    np.testing.assert_allclose(pdm.membership_, rf.membership_)
    out = pdm.set_output(transform="pandas").transform(df)
    assert isinstance(out, pd.DataFrame)
    Rdf = pd.DataFrame(R)
    assert RelationalFCM(n_clusters=3, metric="precomputed", init=U0).fit(Rdf).n_features_in_ == X.shape[0]


def test_rfcm_init_restarts_metric_and_validation(blobs):
    X, y = blobs
    R = pairwise_dissimilarity(X, metric="sqeuclidean")
    base = RelationalFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    a = RelationalFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    np.testing.assert_allclose(a.membership_, base.membership_)
    pp = RelationalFCM(n_clusters=3, metric="precomputed", init="kmeans++", random_state=0).fit(R)
    assert adjusted_rand_score(y, pp.labels_) > 0.95
    idx = RelationalFCM(n_clusters=3, metric="precomputed", init=np.array([0, 100, 200])).fit(R)
    assert adjusted_rand_score(y, idx.labels_) > 0.95
    multi = RelationalFCM(n_clusters=3, metric="precomputed", n_init=4, random_state=0).fit(R)
    assert multi.objective_ <= base.objective_ + 1e-9
    # other feature metrics are squared; a non-Euclidean one is refused by RFCM
    cb = RelationalFCM(n_clusters=3, metric="cityblock", random_state=0)
    with pytest.raises(ValueError, match="NERFCM"):
        cb.fit(X)
    # larger m -> softer partition
    soft = RelationalFCM(n_clusters=3, metric="precomputed", m=3.0, random_state=0).fit(R)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    for kw in (dict(m=1.0), dict(metric=3), dict(init="bogus"), dict(init=np.ones((5, 5))), dict(n_init=0)):
        with pytest.raises(ValueError):
            RelationalFCM(n_clusters=3, metric=kw.pop("metric", "precomputed"), **kw).fit(R)
    with pytest.raises(ValueError):
        RelationalFCM(n_clusters=3, metric="precomputed", init=np.array([0, 1, 10_000])).fit(R)
    with pytest.raises(ValueError):
        RelationalFCM(n_clusters=3, metric="precomputed").fit(R[:, :10])


# ------------------------------------------------------------------ NERF
def test_nerf_reduces_to_rfcm_on_euclidean_data(blobs):
    X, _ = blobs
    R = pairwise_dissimilarity(X, metric="sqeuclidean")
    U0 = np.random.RandomState(0).rand(X.shape[0], 3)
    rfcm = RelationalFCM(n_clusters=3, metric="precomputed", init=U0).fit(R)
    nerf = NERFCM(n_clusters=3, metric="precomputed", init=U0).fit(R)
    assert nerf.beta_ == 0.0 and nerf.n_spread_updates_ == 0
    np.testing.assert_allclose(nerf.membership_, rfcm.membership_, atol=1e-10)


def test_nerf_spreads_non_euclidean_matrices(blobs):
    """A squared city-block matrix is not Euclidean: RFCM fails, NERF raises beta once and clusters."""
    X, y = blobs
    R = pairwise_dissimilarity(X, metric="cityblock") ** 2
    with pytest.raises(ValueError, match="NERFCM"):
        RelationalFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    nerf = NERFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    assert nerf.beta_ > 0 and nerf.n_spread_updates_ >= 1
    assert adjusted_rand_score(y, nerf.labels_) > 0.95
    assert np.all(nerf.distances_ >= 0)
    # the spread corresponds to the relational distances of R_beta = R + beta (11^T - I)
    V = nerf.relational_centers_
    Rb = R + nerf.beta_ * (1.0 - np.eye(R.shape[0]))
    shifted = relational_distances(R, V, beta=nerf.beta_)
    np.testing.assert_allclose(relational_distances(Rb, V), shifted, atol=1e-8)
    np.testing.assert_allclose(nerf.distances_**2, shifted, atol=1e-8)
    # the same from features with the metric argument
    nf = NERFCM(n_clusters=3, metric="cityblock", random_state=0).fit(X)
    assert nf.beta_ == pytest.approx(nerf.beta_)
    assert nf.cluster_centers_.shape == (3, 2)
    # strongly non-Euclidean (cubed distances): larger beta, flatter memberships, still the right groups
    strong = NERFCM(n_clusters=3, metric="precomputed", random_state=0).fit(pairwise_dissimilarity(X) ** 3)
    assert strong.beta_ > nerf.beta_
    assert adjusted_rand_score(y, strong.labels_) > 0.9
    # predictions: objects passed to predict_proba are treated as new (shift with 1 + ||v||^2, not
    # 1 - 2 v_k + ||v||^2), so training rows are reproduced only approximately when beta > 0
    np.testing.assert_allclose(nerf.predict_proba(R[:5]), nerf.membership_[:5], atol=2e-3)
    assert (nerf.predict(R) == nerf.labels_).all()
    new = cdist(X[:5] + 0.2, X, metric="cityblock") ** 2
    U = nerf.predict_proba(new)
    assert U.shape == (5, 3) and np.allclose(U.sum(axis=1), 1.0)
    assert (nerf.predict(new) == nerf.labels_[:5]).all()
    # restarts and reproducibility
    multi = NERFCM(n_clusters=3, metric="precomputed", n_init=3, random_state=0).fit(R)
    assert multi.objective_ <= nerf.objective_ + 1e-9
    again = NERFCM(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    np.testing.assert_allclose(again.membership_, nerf.membership_)
    assert again.beta_ == nerf.beta_


# ------------------------------------------------------------------ FCMdd
def test_fcmdd_recovers_planted_medoids(blobs):
    X, y = blobs
    R = pairwise_dissimilarity(X)
    fcmdd = FuzzyCMedoids(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    assert adjusted_rand_score(y, fcmdd.labels_) > 0.95
    # the medoids are the objects closest to the class means, up to a small radius
    means = np.vstack([X[y == i].mean(axis=0) for i in range(3)])
    nearest = cdist(means, X).argmin(axis=1)
    med = X[fcmdd.medoid_indices_]
    assert cdist(med, X[nearest]).min(axis=1).max() < 0.5
    np.testing.assert_array_equal(fcmdd.cluster_centers_, R[fcmdd.medoid_indices_])
    assert len(set(fcmdd.medoid_indices_.tolist())) == 3  # distinct medoids
    np.testing.assert_allclose(fcmdd.membership_.sum(axis=1), 1.0)
    # an object that is a medoid has membership one in its cluster
    for i, q in enumerate(fcmdd.medoid_indices_):
        assert fcmdd.membership_[q, i] > 0.999
    # distances_ and transform are the dissimilarities themselves (not square-rooted)
    np.testing.assert_allclose(fcmdd.distances_, R[:, fcmdd.medoid_indices_], atol=1e-8)
    np.testing.assert_allclose(fcmdd.transform(R), fcmdd.distances_, atol=1e-6)
    # objective = sum mu^m r and it is non-increasing
    J = fcmdd.objective_history_
    np.testing.assert_allclose(
        fcmdd.objective_, np.sum(fcmdd.membership_**2 * fcmdd.distances_), rtol=1e-6
    )
    assert J[-1] <= J[0] + 1e-9
    # feature input: medoids are rows of X
    ff = FuzzyCMedoids(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(ff.cluster_centers_, X[ff.medoid_indices_])
    np.testing.assert_allclose(ff.membership_, fcmdd.membership_, atol=1e-8)


def test_fcmdd_candidates_trim_and_init(blobs):
    X, y = blobs
    R = pairwise_dissimilarity(X)
    full = FuzzyCMedoids(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    # the linear variant restricts the search: never a lower objective than the full search (same start)
    lin = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_candidates=5, random_state=0).fit(R)
    assert lin.objective_ >= full.objective_ - 1e-9
    assert adjusted_rand_score(y, lin.labels_) > 0.7
    # with enough candidates it coincides with the full search
    big = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_candidates=X.shape[0], random_state=0).fit(R)
    np.testing.assert_array_equal(big.medoid_indices_, full.medoid_indices_)
    # trimming: the objective counts only the best (1 - trim) N weighted terms, medoids resist outliers
    Xo = np.vstack([X, [[40.0, 40.0], [41.0, 40.0], [40.0, 41.0]]])
    Ro = pairwise_dissimilarity(Xo)
    plain = FuzzyCMedoids(n_clusters=3, metric="precomputed", init=full.medoid_indices_).fit(Ro)
    trimmed = FuzzyCMedoids(n_clusters=3, metric="precomputed", trim=0.05, init=full.medoid_indices_).fit(Ro)
    assert trimmed.objective_ < plain.objective_
    assert adjusted_rand_score(y, trimmed.labels_[: X.shape[0]]) > 0.95
    keep = int(np.ceil(0.95 * Xo.shape[0]))
    W = np.sort(trimmed.membership_**2 * trimmed.distances_, axis=0)[:keep]
    np.testing.assert_allclose(trimmed.objective_, W.sum(), rtol=1e-6)
    # initializations and restarts
    rnd = FuzzyCMedoids(n_clusters=3, metric="precomputed", init="random", n_init=4, random_state=0).fit(R)
    single = FuzzyCMedoids(n_clusters=3, metric="precomputed", init="random", random_state=0).fit(R)
    assert rnd.objective_ <= single.objective_ + 1e-9
    given = FuzzyCMedoids(n_clusters=3, metric="precomputed", init=full.medoid_indices_).fit(R)
    np.testing.assert_array_equal(given.medoid_indices_, full.medoid_indices_)
    assert given.n_iter_ == 1
    # fuzziness
    soft = FuzzyCMedoids(n_clusters=3, metric="precomputed", m=3.0, random_state=0).fit(R)
    assert soft.membership_.max(axis=1).mean() < full.membership_.max(axis=1).mean()
    # duplicated objects cannot capture two medoids
    Xd = np.vstack([X[:50], X[:50]])
    dup = FuzzyCMedoids(n_clusters=2, random_state=0).fit(Xd)
    assert dup.medoid_indices_[0] != dup.medoid_indices_[1]
    for kw in (
        dict(m=1.0),
        dict(n_candidates=0),
        dict(trim=1.0),
        dict(trim=-0.1),
        dict(init="bogus"),
        dict(init=np.array([0.5, 1.5, 2.5])),
        dict(init=np.array([0, 1, 10_000])),
        dict(n_init=0),
    ):
        with pytest.raises(ValueError):
            FuzzyCMedoids(n_clusters=3, metric="precomputed", **kw).fit(R)
    with pytest.raises(ValueError):
        FuzzyCMedoids(n_clusters=3, metric=5).fit(X)


def test_fcmdd_predicts_new_objects_and_pandas(blobs):
    X, _ = blobs
    R = pairwise_dissimilarity(X)
    fp = FuzzyCMedoids(n_clusters=3, metric="precomputed", random_state=0).fit(R)
    ff = FuzzyCMedoids(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(fp.predict_proba(R[:7]), fp.membership_[:7], atol=1e-8)
    new = X[:7] + 0.25
    Rnew = cdist(new, X)
    np.testing.assert_allclose(fp.predict_proba(Rnew), ff.predict_proba(new), atol=1e-8)
    np.testing.assert_allclose(fp.transform(Rnew), Rnew[:, fp.medoid_indices_], atol=1e-6)
    assert (fp.predict(Rnew) == ff.predict(new)).all()
    assert fp.score(R) == pytest.approx(-fp.objective_, rel=1e-6)
    with pytest.raises(ValueError):
        fp.predict_proba(Rnew[:, :10])
    df = pd.DataFrame(X, columns=["a", "b"])
    pdm = FuzzyCMedoids(n_clusters=3, random_state=0).fit(df)
    assert list(pdm.feature_names_in_) == ["a", "b"]
    np.testing.assert_array_equal(pdm.medoid_indices_, ff.medoid_indices_)
    assert isinstance(pdm.set_output(transform="pandas").transform(df), pd.DataFrame)


# ------------------------------------------------------------------ fuzzy k-modes
def test_kmodes_recovers_planted_modes(categorical):
    d = categorical
    km = FuzzyKModes(n_clusters=3, n_init=3, random_state=0).fit(d.data)
    assert adjusted_rand_score(d.target, km.labels_) > 0.9
    assert isinstance(km.modes_, pd.DataFrame) and list(km.modes_.columns) == list(d.data.columns)
    # every planted mode is found (as some row of modes_)
    found = {tuple(r) for r in km.modes_.to_numpy()}
    planted = {tuple(r) for r in d.modes.to_numpy()}
    assert found == planted
    assert km.cluster_centers_.shape == (3, 6) and km.prototypes_.equals(km.modes_)
    assert km.gamma_ == 1.0 and len(km.categorical_features_) == 6 and len(km.numeric_features_) == 0
    assert len(km.categories_) == 6
    # distances are matching counts (transform returns their square root), memberships sum to one
    D = km.transform(d.data) ** 2
    np.testing.assert_allclose(D, matching_dissimilarity(d.data.to_numpy(), km.modes_.to_numpy()), atol=1e-8)
    np.testing.assert_allclose(km.membership_.sum(axis=1), 1.0)
    # a record identical to a mode has membership one there
    U = km.predict_proba(km.modes_)
    np.testing.assert_allclose(U, np.eye(3), atol=1e-6)
    # the objective is sum mu^m * matching dissimilarity
    np.testing.assert_allclose(km.objective_, np.sum(km.membership_**km.m * km.distances_**2), rtol=1e-6)


def test_kmodes_theorem_mode_is_largest_membership_mass(categorical):
    """Huang & Ng's theorem: per attribute the mode carries the largest sum of mu^m."""
    d = categorical
    km = FuzzyKModes(n_clusters=3, random_state=0).fit(d.data)
    Um = km.membership_**km.m
    X = d.data.to_numpy()
    for i in range(3):
        for j in range(X.shape[1]):
            mass = {cat: Um[X[:, j] == cat, i].sum() for cat in np.unique(X[:, j])}
            assert mass[km.modes_.iloc[i, j]] == pytest.approx(max(mass.values()))


def test_kmodes_parameters_inputs_and_prediction(categorical):
    d = categorical
    base = FuzzyKModes(n_clusters=3, random_state=0).fit(d.data)
    # m: larger -> fuzzier
    soft = FuzzyKModes(n_clusters=3, m=3.0, random_state=0).fit(d.data)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    # kmeans++ seeding, index init, partition init, encoded-mode init
    pp = FuzzyKModes(n_clusters=3, init="kmeans++", random_state=0).fit(d.data)
    assert adjusted_rand_score(d.target, pp.labels_) > 0.8
    idx = FuzzyKModes(n_clusters=3, init=np.array([0, 1, 2])).fit(d.data)
    assert idx.membership_.shape == (300, 3)
    part = FuzzyKModes(n_clusters=3, init=base.membership_).fit(d.data)
    assert adjusted_rand_score(part.labels_, base.labels_) > 0.95
    enc = FuzzyKModes(n_clusters=3, init=base.cluster_centers_).fit(d.data)
    np.testing.assert_array_equal(enc.cluster_centers_, base.cluster_centers_)
    # restarts keep the best objective; reproducible
    multi = FuzzyKModes(n_clusters=3, n_init=4, random_state=0).fit(d.data)
    assert multi.objective_ <= base.objective_ + 1e-9
    again = FuzzyKModes(n_clusters=3, random_state=0).fit(d.data)
    np.testing.assert_allclose(again.membership_, base.membership_)
    # numpy object input (no column names) and integer-coded input
    arr = FuzzyKModes(n_clusters=3, random_state=0).fit(d.data.to_numpy())
    assert isinstance(arr.modes_, np.ndarray) and arr.modes_.shape == (3, 6)
    assert not hasattr(arr, "feature_names_in_")
    codes = d.data.apply(lambda s: s.astype("category").cat.codes).to_numpy()
    num = FuzzyKModes(n_clusters=3, random_state=0).fit(codes)
    assert adjusted_rand_score(d.target, num.labels_) > 0.9
    assert num.modes_.dtype == object
    # pandas categorical dtype round trip and prediction on new records with unknown categories
    cat_df = d.data.astype("category")
    pc = FuzzyKModes(n_clusters=3, random_state=0).fit(cat_df)
    assert list(pc.feature_names_in_) == list(d.data.columns)
    new = d.data.iloc[:5].copy()
    new.iloc[0, 0] = "ZZZ"  # unseen category: one mismatch everywhere
    U = pc.predict_proba(new)
    assert U.shape == (5, 3) and np.allclose(U.sum(axis=1), 1.0)
    assert (pc.predict(new[1:]) == pc.labels_[1:5]).all()
    assert pc.score(d.data) == pytest.approx(-pc.objective_, rel=1e-6)
    out = pc.set_output(transform="pandas").transform(d.data.iloc[:3])
    assert isinstance(out, pd.DataFrame) and out.shape == (3, 3)
    with pytest.raises(ValueError):
        pc.predict_proba(d.data.iloc[:3, :4])
    miss = d.data.iloc[:10].copy()
    miss.iloc[0, 0] = None
    with pytest.raises(ValueError):
        FuzzyKModes(n_clusters=2).fit(miss)
    bad = (dict(m=1.0), dict(init="bogus"), dict(init=np.ones((4, 4))), dict(n_init=0), dict(n_clusters=0))
    for kw in bad:
        with pytest.raises(ValueError):
            FuzzyKModes(**{"n_clusters": 3, **kw}).fit(d.data)
    with pytest.raises(ValueError):
        FuzzyKModes(n_clusters=3, init=np.array([0, 1, 10_000])).fit(d.data)


# ------------------------------------------------------------------ fuzzy k-prototypes
def test_kprototypes_on_mixed_data(mixed):
    d = mixed
    kp = FuzzyKPrototypes(n_clusters=3, n_init=3, random_state=0).fit(d.data)
    assert adjusted_rand_score(d.target, kp.labels_) > 0.75
    assert list(kp.categorical_features_) == [3, 4, 5] and list(kp.numeric_features_) == [0, 1, 2]
    assert kp.gamma_ == pytest.approx(d.data.iloc[:, :3].std(ddof=0).mean())
    assert isinstance(kp.prototypes_, pd.DataFrame) and kp.prototypes_.shape == (3, 6)
    assert kp.modes_.shape == (3, 3) and kp.numeric_centers_.shape == (3, 3)
    # the planted modes and centers are recovered (up to permutation)
    found = {tuple(r) for r in kp.modes_.to_numpy()}
    assert found == {tuple(r) for r in d.modes.to_numpy()}
    assert cdist(kp.numeric_centers_, d.centers).min(axis=1).max() < 0.5
    # the dissimilarity: squared Euclidean + gamma * mismatches
    Z_num = d.data.iloc[:, :3].to_numpy()
    mism = matching_dissimilarity(d.data.iloc[:, 3:].to_numpy(), kp.modes_.to_numpy())
    expected = cdist(Z_num, kp.numeric_centers_) ** 2 + kp.gamma_ * mism
    np.testing.assert_allclose(kp.transform(d.data) ** 2, expected, atol=1e-8)
    np.testing.assert_allclose(kp.predict_proba(d.data), kp.membership_, atol=1e-8)


def test_kprototypes_gamma_and_feature_specification(mixed):
    d = mixed
    # gamma = 0 ignores the categorical part: FCM on the numeric columns
    g0 = FuzzyKPrototypes(n_clusters=3, gamma=0.0, init=np.array([0, 1, 2])).fit(d.data)
    fcm = FCM(n_clusters=3, init=d.data.iloc[:3, :3].to_numpy()).fit(d.data.iloc[:, :3])
    np.testing.assert_allclose(g0.membership_, fcm.membership_, atol=1e-6)
    # a huge gamma lets the categorical part dominate: k-modes partition on the categorical columns
    big = FuzzyKPrototypes(n_clusters=3, gamma=1e6, n_init=3, random_state=0).fit(d.data)
    km = FuzzyKModes(n_clusters=3, m=2.0, n_init=3, random_state=0).fit(d.data.iloc[:, 3:])
    assert adjusted_rand_score(big.labels_, km.labels_) > 0.9
    # columns by name, by position, by mask; numeric columns declared categorical become categories
    names = ["c1", "c2", "c3"]
    by_name = FuzzyKPrototypes(n_clusters=3, categorical_features=names, random_state=0).fit(d.data)
    by_pos = FuzzyKPrototypes(n_clusters=3, categorical_features=[3, 4, 5], random_state=0).fit(d.data)
    by_mask = FuzzyKPrototypes(
        n_clusters=3, categorical_features=np.array([0, 0, 0, 1, 1, 1], bool), random_state=0
    ).fit(d.data)
    for est in (by_pos, by_mask):
        np.testing.assert_allclose(est.membership_, by_name.membership_)
    extra = FuzzyKPrototypes(n_clusters=3, categorical_features=["x1", *names], random_state=0).fit(d.data)
    assert list(extra.categorical_features_) == [0, 3, 4, 5]
    assert extra.modes_.shape == (3, 4) and extra.modes_["x1"].dtype == object
    # numpy input: all numeric unless told otherwise; object input: all categorical
    Xn = d.data.iloc[:, :3].to_numpy()
    kn = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(Xn)
    assert len(kn.categorical_features_) == 0 and kn.modes_.shape == (3, 0)
    ko = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(d.data.iloc[:, 3:].to_numpy())
    assert len(ko.categorical_features_) == 3 and ko.gamma_ == 1.0
    # new records, score and pandas output
    new = d.data.iloc[:4]
    assert kn.predict_proba(Xn[:4]).shape == (4, 3)
    assert by_name.predict(new).shape == (4,)
    assert by_name.score(d.data) == pytest.approx(-by_name.objective_, rel=1e-6)
    # restarts and reproducibility
    base = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(d.data)
    multi = FuzzyKPrototypes(n_clusters=3, n_init=4, random_state=0).fit(d.data)
    assert multi.objective_ <= base.objective_ + 1e-9
    again = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(d.data)
    np.testing.assert_allclose(again.membership_, base.membership_)
    soft = FuzzyKPrototypes(n_clusters=3, m=3.0, random_state=0).fit(d.data)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    pp = FuzzyKPrototypes(n_clusters=3, init="kmeans++", random_state=0).fit(d.data)
    assert adjusted_rand_score(d.target, pp.labels_) > 0.6
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3, gamma=-1.0).fit(d.data)
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3, categorical_features=["nope"]).fit(d.data)
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3, categorical_features=np.array([True, False])).fit(d.data)
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3, categorical_features="some").fit(d.data)
    with pytest.raises(ValueError):  # strings in an undeclared (numeric) column
        FuzzyKPrototypes(n_clusters=3, categorical_features=[3, 4]).fit(d.data)
    bad = d.data.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3).fit(bad)
    with pytest.raises(ValueError):
        FuzzyKPrototypes(n_clusters=3, m=1.0).fit(d.data)


def test_gower_fcmdd_versus_one_hot_fcm(mixed):
    """Gower + FCMdd uses both attribute kinds on their own scale; it recovers the planted groups."""
    d = mixed
    G = gower_distance_matrix(d.data)
    fcmdd = FuzzyCMedoids(n_clusters=3, metric="precomputed", n_init=5, random_state=0).fit(G)
    assert adjusted_rand_score(d.target, fcmdd.labels_) > 0.7
    nerf = NERFCM(n_clusters=3, metric="precomputed", random_state=0).fit(G)
    assert adjusted_rand_score(d.target, nerf.labels_) > 0.7
    onehot = pd.get_dummies(d.data, dtype=float).to_numpy()
    fcm = FCM(n_clusters=3, n_init=5, random_state=0).fit(onehot)
    assert adjusted_rand_score(d.target, fcm.labels_) > 0.3  # works too, but the scales are arbitrary


# ------------------------------------------------------------------ plots
def test_plot_helpers(series, series_dtw, categorical, mixed):
    X, y = series
    fcmdd = FuzzyCMedoids(n_clusters=3, metric="precomputed", random_state=0).fit(series_dtw)
    ax = plot_dissimilarity_matrix(series_dtw, fcmdd.labels_, medoids=fcmdd.medoid_indices_)
    assert ax.get_xlabel().startswith("object")
    assert len(ax.images) == 1
    ax = plot_dissimilarity_matrix(series_dtw, colorbar=False)
    assert ax is not None
    ax = plot_dtw_alignment(X[0], X[1], window=12, every=3)
    assert ax.get_title().startswith("DTW alignment")
    ax = plot_series_clusters(X, fcmdd.labels_, medoids=fcmdd.medoid_indices_)
    assert ax.get_legend() is not None
    fig, axes = plt.subplots(1, 3)
    out = plot_series_clusters(X, fcmdd.labels_, medoids=fcmdd.medoid_indices_, ax=axes, max_per_cluster=5)
    assert len(out) == 3 and axes[1].get_title().startswith("cluster 2")
    km = FuzzyKModes(n_clusters=3, random_state=0).fit(categorical.data)
    ax = plot_modes(km, categorical.data)
    assert len(ax.texts) == 18
    ax = plot_modes(km)  # no shares without data
    assert len(ax.texts) == 18 and "%" not in ax.texts[0].get_text()
    kp = FuzzyKPrototypes(n_clusters=3, random_state=0).fit(mixed.data.to_numpy()[:, 3:])
    ax = plot_modes(kp, mixed.data.to_numpy()[:, 3:])
    assert ax.get_xticklabels()[0].get_text() == "attribute 1"
    plt.close("all")
