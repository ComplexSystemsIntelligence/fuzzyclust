import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.cluster import FCM  # noqa: E402
from fuzzyclust.large import (  # noqa: E402
    OnlineFCM,
    RandomSampleFCM,
    SinglePassFCM,
    WeightedFCM,
    extend_partition,
    make_drifting_stream,
    make_large_blobs,
    plot_center_trajectories,
    plot_chunk_timeline,
)

CHUNKED = (SinglePassFCM, OnlineFCM)
ALL = (RandomSampleFCM, SinglePassFCM, OnlineFCM)


def center_error(V, V_ref):
    D = cdist(V, V_ref)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].max())


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def big():
    """100k samples, 5 clusters of unequal size."""
    return make_large_blobs(n_samples=100_000, n_clusters=5, weights=(3, 2, 2, 1, 1), random_state=0)


@pytest.fixture(scope="module")
def big_fcm(big):
    X, _ = big
    t = time.perf_counter()
    fcm = FCM(n_clusters=5, random_state=0).fit(X)
    fcm.fit_time_ = time.perf_counter() - t
    return fcm


@pytest.fixture(scope="module")
def small():
    return make_large_blobs(n_samples=1500, n_clusters=3, random_state=1)


# ------------------------------------------------------------------ sklearn compatibility
@parametrize_with_checks(
    [
        WeightedFCM(n_clusters=2, random_state=0),
        RandomSampleFCM(n_clusters=2, sample_size=0.5, random_state=0),
        SinglePassFCM(n_clusters=2, chunk_size=20, random_state=0),
        OnlineFCM(n_clusters=2, chunk_size=20, random_state=0),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ data generators
def test_make_large_blobs_properties():
    X, y = make_large_blobs(n_samples=1000, n_clusters=4, n_features=3, weights=(4, 3, 2, 1), random_state=0)
    assert X.shape == (1000, 3) and y.shape == (1000,)
    assert np.bincount(y).tolist() == [400, 300, 200, 100]
    # shuffled: the first chunk contains every cluster
    assert len(np.unique(y[:100])) == 4
    Xo, yo = make_large_blobs(n_samples=1000, n_clusters=4, shuffle=False, random_state=0)
    assert np.all(np.diff(yo) >= 0)
    a = make_large_blobs(n_samples=50, random_state=2)[0]
    np.testing.assert_allclose(a, make_large_blobs(n_samples=50, random_state=2)[0])
    with pytest.raises(ValueError):
        make_large_blobs(n_clusters=3, weights=(1, 1))


def test_make_drifting_stream_properties():
    chunks = list(make_drifting_stream(n_chunks=5, chunk_size=200, n_clusters=3, drift=0.5, random_state=0))
    assert len(chunks) == 5
    X, y, C = chunks[0]
    assert X.shape == (200, 2) and y.shape == (200,) and C.shape == (3, 2)
    # every center moves by exactly `drift` per chunk
    for (_, _, C0), (_, _, C1) in zip(chunks[:-1], chunks[1:], strict=True):
        np.testing.assert_allclose(np.linalg.norm(C1 - C0, axis=1), 0.5)


# ------------------------------------------------------------------ wFCM
def test_weighted_fcm_duplicates_equal_weight_two(small):
    X, _ = small
    dup = np.vstack([X, X[:300]])
    a = WeightedFCM(n_clusters=3, random_state=0).fit(dup)
    w = np.r_[np.full(300, 2.0), np.ones(X.shape[0] - 300)]
    b = WeightedFCM(n_clusters=3, random_state=0).fit(X, sample_weight=w)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_, atol=1e-8)
    np.testing.assert_allclose(a.membership_[: X.shape[0]], b.membership_, atol=1e-8)
    assert a.objective_ == pytest.approx(b.objective_, rel=1e-8)
    assert a.n_iter_ == b.n_iter_
    np.testing.assert_array_equal(b.sample_weight_, w)
    # weights of one reproduce FCM exactly; zero weight means absence
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    ones = WeightedFCM(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(ones.cluster_centers_, fcm.cluster_centers_, atol=1e-10)
    w0 = np.r_[np.zeros(300), np.ones(X.shape[0] - 300)]
    zero = WeightedFCM(n_clusters=3, random_state=0).fit(X, sample_weight=w0)
    rest = WeightedFCM(n_clusters=3, random_state=0).fit(X[300:])
    np.testing.assert_allclose(zero.cluster_centers_, rest.cluster_centers_, atol=1e-8)


def test_weighted_fcm_weight_effect_and_formulas(small):
    X, y = small
    # moving all the weight onto one cluster pulls the prototypes toward it when c is too small
    w = np.where(y == 0, 50.0, 1.0)
    heavy = WeightedFCM(n_clusters=2, random_state=0).fit(X, sample_weight=w)
    light = WeightedFCM(n_clusters=2, random_state=0).fit(X)
    c0 = X[y == 0].mean(axis=0)
    assert cdist(heavy.cluster_centers_, [c0]).min() < cdist(light.cluster_centers_, [c0]).min()
    # prototype formula with the final partition
    Um = heavy.membership_**2
    V = (Um * w[:, None]).T @ X / (Um * w[:, None]).sum(axis=0)[:, None]
    np.testing.assert_allclose(heavy.cluster_centers_, V, atol=1e-3)
    # objective is the weighted functional
    J = float(np.sum(w[:, None] * Um * heavy.distances_**2))
    assert heavy.objective_ == pytest.approx(J, rel=1e-3)
    assert heavy.score(X, sample_weight=w) == pytest.approx(-J, rel=1e-3)
    assert heavy.score(X) > heavy.score(X, sample_weight=w)  # unweighted score of new data
    # kmeans++ seeding honours the weights
    pp = WeightedFCM(n_clusters=3, init="kmeans++", random_state=0).fit(X, sample_weight=np.ones(X.shape[0]))
    assert adjusted_rand_score(y, pp.labels_) > 0.95
    soft = WeightedFCM(n_clusters=3, m=3.0, random_state=0).fit(X)
    assert soft.membership_.max(axis=1).mean() < light.membership_.max(axis=1).mean()


def test_weighted_fcm_validation_and_pandas(small):
    X, _ = small
    for bad in (np.ones(5), -np.ones(X.shape[0]), np.zeros(X.shape[0]), np.full(X.shape[0], np.nan)):
        with pytest.raises(ValueError):
            WeightedFCM(n_clusters=3).fit(X, sample_weight=bad)
    with pytest.raises(ValueError):
        WeightedFCM(n_clusters=3, m=1.0).fit(X)
    df = pd.DataFrame(X, columns=["a", "b"])
    est = WeightedFCM(n_clusters=3, random_state=0).fit(df, sample_weight=pd.Series(np.ones(X.shape[0])))
    assert list(est.feature_names_in_) == ["a", "b"]
    assert est.predict_proba(df).shape == (X.shape[0], 3)


def test_extend_partition_chunked_equals_full(small):
    X, _ = small
    V = X[:3]
    U, d2 = extend_partition(X, V, 2.0)
    Uc, d2c = extend_partition(X, V, 2.0, chunk_size=100)
    np.testing.assert_allclose(U, Uc)
    np.testing.assert_allclose(d2, d2c)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    np.testing.assert_allclose(d2, cdist(X, V) ** 2, atol=1e-8)


# ------------------------------------------------------------------ the three schemes on 100k samples
def test_schemes_approach_fcm_on_100k_samples(big, big_fcm):
    X, y = big
    ref = big_fcm
    models = {
        "rseFCM": RandomSampleFCM(n_clusters=5, sample_size=0.05, random_state=0),
        "spFCM": SinglePassFCM(n_clusters=5, chunk_size=10_000, random_state=0),
        "oFCM": OnlineFCM(n_clusters=5, chunk_size=10_000, random_state=0),
    }
    for name, model in models.items():
        t = time.perf_counter()
        model.fit(X)
        dt = time.perf_counter() - t
        assert model.cluster_centers_.shape == (5, 2)
        assert model.membership_.shape == (X.shape[0], 5) and model.labels_.shape == (X.shape[0],)
        np.testing.assert_allclose(model.membership_.sum(axis=1), 1.0, atol=1e-9)
        assert adjusted_rand_score(ref.labels_, model.labels_) > 0.95, name
        assert center_error(model.cluster_centers_, ref.cluster_centers_) < 0.1, name
        assert model.objective_ < 1.02 * ref.objective_, name
        assert dt < 5.0 * max(ref.fit_time_, 0.5), name  # modest runtime
        # the extension is one membership step from the final prototypes
        np.testing.assert_allclose(model.predict_proba(X[:1000]), model.membership_[:1000], atol=1e-9)
        assert (model.predict(X[:1000]) == model.labels_[:1000]).all()


def test_rsefcm_sample_and_parameters(small):
    X, y = small
    rse = RandomSampleFCM(n_clusters=3, sample_size=200, random_state=0).fit(X)
    assert rse.sample_indices_.shape == (200,) and len(np.unique(rse.sample_indices_)) == 200
    assert isinstance(rse.fcm_, FCM) and rse.fcm_.membership_.shape == (200, 3)
    np.testing.assert_allclose(rse.cluster_centers_, rse.fcm_.cluster_centers_)
    assert rse.n_iter_ == rse.fcm_.n_iter_ == len(rse.objective_history_)
    # a fraction
    frac = RandomSampleFCM(n_clusters=3, sample_size=0.2, random_state=0).fit(X)
    assert frac.sample_indices_.shape == (300,)
    # larger samples approach FCM more closely (on average; here checked for a tiny sample)
    ref = FCM(n_clusters=3, random_state=0).fit(X)
    tiny = RandomSampleFCM(n_clusters=3, sample_size=6, random_state=3).fit(X)
    large = RandomSampleFCM(n_clusters=3, sample_size=600, random_state=3).fit(X)
    err_large = center_error(large.cluster_centers_, ref.cluster_centers_)
    assert err_large <= center_error(tiny.cluster_centers_, ref.cluster_centers_)
    # chunked extension is identical
    chunked = RandomSampleFCM(n_clusters=3, sample_size=200, chunk_size=128, random_state=0).fit(X)
    np.testing.assert_allclose(chunked.membership_, rse.membership_)
    # reproducible, restarts, fuzziness
    again = RandomSampleFCM(n_clusters=3, sample_size=200, random_state=0).fit(X)
    np.testing.assert_array_equal(again.sample_indices_, rse.sample_indices_)
    multi = RandomSampleFCM(n_clusters=3, sample_size=200, n_init=3, random_state=0).fit(X)
    assert multi.fcm_.objective_ <= rse.fcm_.objective_ + 1e-9
    soft = RandomSampleFCM(n_clusters=3, sample_size=200, m=3.0, random_state=0).fit(X)
    assert soft.membership_.max(axis=1).mean() < rse.membership_.max(axis=1).mean()
    bad = (dict(sample_size=2), dict(sample_size=0.0), dict(sample_size="x"), dict(m=1.0), dict(chunk_size=0))
    for kw in bad:
        with pytest.raises(ValueError):
            RandomSampleFCM(n_clusters=3, **kw).fit(X)


def test_spfcm_carries_weighted_prototypes(small):
    X, y = small
    sp = SinglePassFCM(n_clusters=3, chunk_size=500, shuffle=False, random_state=0).fit(X)
    assert sp.n_chunks_ == 3
    assert sp.chunk_centers_.shape == (3, 3, 2) and sp.chunk_weights_.shape == (3, 3)
    assert sp.chunk_objective_.shape == (3,) and sp.chunk_n_iter_.shape == (3,)
    assert sp.n_iter_ == sp.chunk_n_iter_.sum()
    np.testing.assert_allclose(sp.cluster_centers_, sp.chunk_centers_[-1])
    np.testing.assert_allclose(sp.center_weights_, sp.chunk_weights_[-1])
    # the weights add up to the number of observations seen so far
    np.testing.assert_allclose(sp.chunk_weights_.sum(axis=1), [500, 1000, 1500], rtol=1e-6)
    # the first chunk is a plain FCM run on the first 500 samples
    first = FCM(n_clusters=3, random_state=0).fit(X[:500])
    np.testing.assert_allclose(sp.chunk_centers_[0], first.cluster_centers_, atol=1e-8)
    # the second chunk clusters the samples plus the carried prototypes
    Xa = np.vstack([X[500:1000], sp.chunk_centers_[0]])
    wa = np.r_[np.ones(500), sp.chunk_weights_[0]]
    second = WeightedFCM(n_clusters=3, init=sp.chunk_centers_[0]).fit(Xa, sample_weight=wa)
    np.testing.assert_allclose(sp.chunk_centers_[1], second.cluster_centers_, atol=1e-8)
    assert adjusted_rand_score(y, sp.labels_) > 0.95
    # one chunk holding everything is FCM
    one = SinglePassFCM(n_clusters=3, chunk_size=5000, random_state=0).fit(X)
    assert one.n_chunks_ == 1
    full = FCM(n_clusters=3, random_state=0).fit(X)
    assert center_error(one.cluster_centers_, full.cluster_centers_) < 1e-4


def test_ofcm_clusters_the_chunk_prototypes(small):
    X, y = small
    of = OnlineFCM(n_clusters=3, chunk_size=300, shuffle=False, random_state=0).fit(X)
    assert of.n_chunks_ == 5
    assert of.chunk_centers_.shape == (5, 3, 2) and of.chunk_weights_.shape == (5, 3)
    np.testing.assert_allclose(of.chunk_weights_.sum(axis=1), 300.0, rtol=1e-6)
    assert isinstance(of.centers_model_, WeightedFCM)
    assert of.centers_model_.membership_.shape == (15, 3)
    np.testing.assert_allclose(of.cluster_centers_, of.centers_model_.cluster_centers_)
    assert of.center_weights_.sum() == pytest.approx(1500.0, rel=1e-6)
    # independent chunk clusterings reach the same partition
    indep = OnlineFCM(
        n_clusters=3, chunk_size=300, chunk_init="independent", shuffle=False, random_state=0
    ).fit(X)
    assert adjusted_rand_score(of.labels_, indep.labels_) > 0.95
    assert indep.chunk_n_iter_.sum() >= of.chunk_n_iter_.sum()  # warm starts save iterations
    assert adjusted_rand_score(y, of.labels_) > 0.95


@pytest.mark.parametrize("cls", CHUNKED)
def test_chunk_size_study(cls, big, big_fcm):
    """Smaller chunks mean more chunk runs; all sizes with enough samples per cluster stay close to FCM."""
    X, _ = big
    ref = big_fcm
    errors = {}
    for size in (1000, 5000, 25_000):
        model = cls(n_clusters=5, chunk_size=size, random_state=0).fit(X)
        assert model.n_chunks_ == int(np.ceil(X.shape[0] / size))
        errors[size] = center_error(model.cluster_centers_, ref.cluster_centers_)
        assert adjusted_rand_score(ref.labels_, model.labels_) > 0.95
    assert max(errors.values()) < 0.15


@pytest.mark.parametrize("cls", CHUNKED)
def test_partial_fit_streams(cls):
    stream = list(make_drifting_stream(n_chunks=8, chunk_size=400, n_clusters=3, drift=0.0, random_state=0))
    model = cls(n_clusters=3, chunk_size=400, random_state=0)
    for l, (Xc, yc, _C) in enumerate(stream, start=1):
        model.partial_fit(Xc)
        assert model.n_chunks_ == l
        assert model.chunk_centers_.shape == (l, 3, 2)
        assert model.membership_.shape == (400, 3) and model.labels_.shape == (400,)
        assert adjusted_rand_score(yc, model.labels_) > 0.9
    # the whole stream in one fit (unshuffled) is the same sequence of chunks
    X_all = np.vstack([Xc for Xc, _, _ in stream])
    batch = cls(n_clusters=3, chunk_size=400, shuffle=False, random_state=0).fit(X_all)
    np.testing.assert_allclose(batch.chunk_centers_, model.chunk_centers_, atol=1e-6)
    np.testing.assert_allclose(batch.cluster_centers_, model.cluster_centers_, atol=1e-6)
    assert center_error(model.cluster_centers_, stream[-1][2]) < 0.15
    # predictions for new data use the current prototypes
    Xn, yn, _ = stream[-1]
    assert adjusted_rand_score(yn, model.predict(Xn)) > 0.9
    # a drifting stream: the chunked schemes summarize *all* data seen so far, so their
    # prototypes sit near the time average of the drifting centers, not at the current ones
    drift = list(make_drifting_stream(n_chunks=10, chunk_size=400, n_clusters=3, drift=0.2, random_state=1))
    model = cls(n_clusters=3, chunk_size=400, random_state=0)
    for Xc, _, _ in drift:
        model.partial_fit(Xc)
    mean_centers = np.mean([C for _, _, C in drift], axis=0)
    err_mean = center_error(model.cluster_centers_, mean_centers)
    assert err_mean < 0.3
    assert err_mean < center_error(model.cluster_centers_, drift[-1][2])
    assert err_mean < center_error(model.cluster_centers_, drift[0][2])
    # a fit after partial_fit starts afresh
    model.fit(X_all)
    assert model.n_chunks_ == 8


@pytest.mark.parametrize("cls", CHUNKED)
def test_chunked_parameters_and_validation(cls, small):
    X, y = small
    base = cls(n_clusters=3, chunk_size=500, random_state=0).fit(X)
    again = cls(n_clusters=3, chunk_size=500, random_state=0).fit(X)
    np.testing.assert_allclose(base.cluster_centers_, again.cluster_centers_)
    soft = cls(n_clusters=3, chunk_size=500, m=3.0, random_state=0).fit(X)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    unshuffled = cls(n_clusters=3, chunk_size=500, shuffle=False, random_state=0).fit(X)
    assert adjusted_rand_score(unshuffled.labels_, base.labels_) > 0.95
    pp = cls(n_clusters=3, chunk_size=500, init="kmeans++", random_state=0).fit(X)
    assert adjusted_rand_score(pp.labels_, base.labels_) > 0.95
    multi = cls(n_clusters=3, chunk_size=500, n_init=3, random_state=0).fit(X)
    assert multi.chunk_objective_[0] <= base.chunk_objective_[0] + 1e-9
    few = cls(n_clusters=3, chunk_size=500, max_iter=1, random_state=0).fit(X)
    assert few.chunk_n_iter_.max() == 1
    assert base.score(X) == pytest.approx(-base.objective_, rel=1e-6)
    np.testing.assert_allclose(base.transform(X), base.distances_, rtol=1e-6)
    assert list(base.get_feature_names_out()) == ["dist_cluster_0", "dist_cluster_1", "dist_cluster_2"]
    df = pd.DataFrame(X, columns=["a", "b"])
    est = cls(n_clusters=3, chunk_size=500, random_state=0).fit(df)
    assert list(est.feature_names_in_) == ["a", "b"]
    assert isinstance(est.set_output(transform="pandas").transform(df), pd.DataFrame)
    for kw in (dict(chunk_size=0), dict(chunk_size=2), dict(chunk_size=2.5), dict(m=1.0), dict(n_init=0)):
        with pytest.raises(ValueError):
            cls(n_clusters=3, **kw).fit(X)
    if cls is OnlineFCM:
        with pytest.raises(ValueError):
            OnlineFCM(n_clusters=3, chunk_init="bogus").fit(X)


def test_spfcm_forgetting_factor_tracks_drift():
    stream = list(make_drifting_stream(n_chunks=12, chunk_size=400, n_clusters=3, drift=0.2, random_state=1))
    errors = {}
    for lam in (1.0, 0.5, 0.0):
        model = SinglePassFCM(n_clusters=3, chunk_size=400, forgetting_factor=lam, random_state=0)
        for Xc, _, _ in stream:
            model.partial_fit(Xc)
        errors[lam] = center_error(model.cluster_centers_, stream[-1][2])
    # the published algorithm lags behind the drift, forgetting lets it follow
    assert errors[0.0] < errors[0.5] < errors[1.0]
    assert errors[0.0] < 0.25 and errors[1.0] > 0.5
    # lambda = 1 is the default (the paper); fit honors the factor as well
    X = np.vstack([Xc for Xc, _, _ in stream])
    kw = dict(n_clusters=3, chunk_size=400, shuffle=False, random_state=0)
    a = SinglePassFCM(**kw).fit(X)
    b = SinglePassFCM(forgetting_factor=1.0, **kw).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    c = SinglePassFCM(forgetting_factor=0.0, **kw).fit(X)
    assert center_error(c.cluster_centers_, stream[-1][2]) < center_error(a.cluster_centers_, stream[-1][2])
    for bad in (-0.1, 1.5, "x", True):
        with pytest.raises(ValueError):
            SinglePassFCM(n_clusters=3, chunk_size=400, forgetting_factor=bad).fit(X)


# ------------------------------------------------------------------ plots
def test_plot_helpers(small):
    X, _ = small
    sp = SinglePassFCM(n_clusters=3, chunk_size=300, random_state=0).fit(X)
    ax = plot_chunk_timeline(sp)
    assert ax.get_xlabel() == "chunk" and len(ax.collections) == 3
    truth = np.repeat(sp.chunk_centers_[-1:], 5, axis=0)
    ax = plot_chunk_timeline(sp, feature=1, weights=False, true_centers=truth)
    assert "true centers" in ax.get_legend_handles_labels()[1]
    ax = plot_center_trajectories(X, sp)
    assert "final prototypes" in ax.get_legend_handles_labels()[1]
    ax = plot_center_trajectories(X, sp, max_points=100, labels=None)
    assert ax is not None
    plt.close("all")
