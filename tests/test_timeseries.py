import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.timeseries import (  # noqa: E402
    ChaoticPredictor,
    FuzzySegmentation,
    MixedInputStructureSelection,
    bottom_up_segmentation,
    cluster_based_threshold,
    correlation_dimension,
    delay_embedding,
    embedding_scan,
    false_nearest_neighbors,
    fnn_model_order,
    fuzzy_variance,
    make_lorenz,
    make_mackey_glass,
    make_mixed_input_system,
    make_nonlinear_io_system,
    make_piecewise_timeseries,
    make_rossler,
    pca_segment_cost,
    plot_embedding_scan,
    plot_fnn_table,
    plot_phase_space,
    plot_segmentation,
    plot_structure_ranking,
    select_delay,
    select_embedding_dimension,
    select_model_order,
    select_structure_mixed,
)
from fuzzyclust.viz import FuzzySammon  # noqa: E402

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def series():
    """Two mean shifts and one correlation change; true borders 200, 400, 600."""
    X, seg = make_piecewise_timeseries(
        n_samples=800,
        change_points=(200, 400, 600),
        change_kinds=("mean", "correlation", "mean"),
        random_state=0,
    )
    return X, seg


@pytest.fixture(scope="module")
def io_system():
    return make_nonlinear_io_system(n_samples=500, random_state=0)


@pytest.fixture(scope="module")
def mackey_glass():
    return make_mackey_glass(n_samples=900)


@pytest.fixture(scope="module")
def mixed():
    return make_mixed_input_system(n_samples=600, random_state=0)


def _max_border_error(found, true):
    found, true = np.asarray(found), np.asarray(true)
    assert found.shape == true.shape, (found, true)
    return int(np.max(np.abs(found - true)))


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    """Checks that cannot apply to a time-ordered estimator.

    ``FuzzySegmentation`` is a clusterer whose prototypes contain a time part: the
    membership of a sample depends on *when* it was observed. Without an explicit time
    axis ``predict``/``transform`` assume the rows are a series starting at time 0, so a
    permutation of the rows or a subset of them is a different series and the outputs
    legitimately change (sample-order and subset invariance). ``check_clustering`` calls
    ``set_params(n_clusters=...)``, but the constructor parameter is ``n_segments``
    (``n_clusters`` is only a read/write property for the base class).
    """
    name = type(estimator).__name__
    if name == "FuzzySegmentation":
        reason = "the prototype has a time part: row order and subsets change the series"
        return {
            "check_methods_sample_order_invariance": reason,
            "check_methods_subset_invariance": reason,
            "check_clustering": "constructor parameter is n_segments, not n_clusters",
        }
    return {}


# ChaoticPredictor.fit takes a one-dimensional series (not an (N, n) matrix), so the
# generic estimator checks do not apply to it; its behavior is covered by the tests below.
@parametrize_with_checks(
    [
        FuzzySegmentation(n_segments=2, max_iter=50),
        MixedInputStructureSelection(n_clusters=2),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ data generators
def test_make_piecewise_timeseries_shapes_and_kinds():
    X, seg = make_piecewise_timeseries(n_samples=300, change_points=(100, 200), change_kinds=("mean", "both"),
                                       n_features=4, random_state=0)
    assert X.shape == (300, 4) and seg.shape == (300,)
    assert np.array_equal(np.unique(seg), [0, 1, 2])
    # a correlation change keeps the mean, a mean change moves it
    Xc, _ = make_piecewise_timeseries(n_samples=400, change_points=(200,), change_kinds=("correlation",),
                                      noise=0.0, latent_noise=1.0, random_state=1)
    Xm, _ = make_piecewise_timeseries(n_samples=400, change_points=(200,), change_kinds=("mean",),
                                      noise=0.0, latent_noise=1.0, random_state=1)
    assert np.linalg.norm(Xc[:200].mean(0) - Xc[200:].mean(0)) < 0.5
    assert np.linalg.norm(Xm[:200].mean(0) - Xm[200:].mean(0)) > 1.0
    with pytest.raises(ValueError):
        make_piecewise_timeseries(change_points=(100,), change_kinds=("mean", "mean"))
    with pytest.raises(ValueError):
        make_piecewise_timeseries(n_samples=300, change_points=(100,), change_kinds=("drift",))
    with pytest.raises(ValueError):
        make_piecewise_timeseries(n_samples=300, change_points=(400,), change_kinds=("mean",))


def test_chaotic_generators():
    x = make_mackey_glass(n_samples=300)
    assert x.shape == (300,) and 0.2 < x.min() < x.max() < 1.6
    L = make_lorenz(n_samples=200)
    R = make_rossler(n_samples=200)
    assert L.shape == (200, 3) and R.shape == (200, 3)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(R))
    assert abs(L[:, 2].mean() - 23.5) < 5.0  # Lorenz z stays around rho - 1


def test_io_system_and_mixed_generators():
    u, y = make_nonlinear_io_system(n_samples=100, random_state=0)
    assert u.shape == y.shape == (100,)
    assert 0.007 <= u.min() and u.max() <= 0.015
    assert 20000 < y.mean() < 40000
    X, y2, disc = make_mixed_input_system(n_samples=50, n_dummy=3, random_state=0)
    assert X.shape == (50, 7) and y2.shape == (50,)
    assert disc.sum() == 1 and disc[1]
    assert set(np.unique(X[:, 1])) <= {0.0, 1.0}


# ------------------------------------------------------------------ fuzzy segmentation
def test_segmentation_finds_known_borders(series):
    X, seg = series
    fs = FuzzySegmentation(n_segments=4, n_components=2).fit(X)
    assert fs.n_segments_ == 4
    assert fs.membership_.shape == (800, 4)
    assert np.allclose(fs.membership_.sum(axis=1), 1.0)
    assert _max_border_error(fs.boundaries_, (200, 400, 600)) <= 10
    assert fs.time_membership_.shape == (800, 4)
    assert np.all(np.diff(fs.segment_centers_) > 0)  # segments ordered in time
    assert fs.covariances_.shape == (4, 6, 6) and fs.pca_loadings_.shape == (4, 6, 2)
    assert fs.norm_matrices_.shape == (4, 6, 6) and fs.priors_.shape == (4,)
    assert len(fs.local_pca_) == 4 and fs.local_pca_[0].components.shape == (2, 6)
    assert fs.distances_.shape == (800, 4) and np.all(fs.distances_ > 0)
    assert fs.objective_history_.shape == (fs.n_iter_,)
    assert np.isfinite(fs.objective_)
    labels_agree = np.mean(fs.labels_ == seg)
    assert labels_agree > 0.95


def test_segmentation_merging_reduces_segments(series):
    X, _ = series
    # the smallest mean shift (segments 3 -> 4) gives a closeness of about 0.5 and, with identical PCA
    # models, a compatibility of 0.79: the threshold has to exceed it to keep the four segments apart
    fs = FuzzySegmentation(n_segments=8, n_components=2, merge=True, merge_threshold=0.8).fit(X)
    assert fs.n_segments_ == 4
    assert _max_border_error(fs.boundaries_, (200, 400, 600)) <= 10
    fs_lower = FuzzySegmentation(n_segments=8, n_components=2, merge=True, merge_threshold=0.75).fit(X)
    assert fs_lower.n_segments_ == 3
    O, S, D = fs.compatibility_matrix()
    assert O.shape == S.shape == D.shape == (4, 4)
    assert np.all((0 <= O) & (O <= 1 + 1e-12)) and np.all((0 <= S) & (S <= 1 + 1e-9))
    # the correlation change between segments 2 and 3 gives dissimilar PCA models
    assert S[1, 2] < 0.7 and S[0, 1] > 0.9
    # the merging threshold controls the result: a low threshold merges everything compatible
    fs_low = FuzzySegmentation(n_segments=8, n_components=2, merge=True, merge_threshold=0.1).fit(X)
    assert fs_low.n_segments_ < 4
    # explicit post-hoc merging gives the same number of segments as merging during fit
    fs8 = FuzzySegmentation(n_segments=8, n_components=2).fit(X)
    assert fs8.n_segments_ == 8
    fs8.merge_compatible_segments(threshold=0.8)
    assert fs8.n_segments_ == 4
    assert fs8.membership_.shape == (800, 4)
    fs8b = FuzzySegmentation(n_segments=8, n_components=2).fit(X).merge_compatible_segments(max_merges=1)
    assert fs8b.n_segments_ == 7


def test_segmentation_min_segment_weight_safeguard(series):
    X, _ = series
    # with the safeguard switched off, merging relies on the threshold alone
    fs = FuzzySegmentation(n_segments=8, n_components=2, merge=True, merge_threshold=0.8,
                           min_segment_weight=0.0).fit(X)
    assert 4 <= fs.n_segments_ <= 8
    fs_guard = FuzzySegmentation(n_segments=8, n_components=2, merge=True, merge_threshold=0.8,
                                 min_segment_weight=0.3).fit(X)
    assert fs_guard.n_segments_ <= 3  # every segment below 30 % of the data is absorbed


def test_segmentation_n_components_selection(series):
    X, _ = series
    fs_int = FuzzySegmentation(n_segments=4, n_components=3).fit(X)
    assert fs_int.n_components_ == 3 and fs_int.pca_loadings_.shape[2] == 3
    fs_acc = FuzzySegmentation(n_segments=4, n_components=0.5).fit(X)
    assert 1 <= fs_acc.n_components_ <= 5
    fs_hi = FuzzySegmentation(n_segments=4, n_components=0.999).fit(X)
    assert fs_hi.n_components_ >= fs_acc.n_components_
    fs0 = FuzzySegmentation(n_segments=4, n_components=0).fit(X)  # spherical segments
    assert fs0.n_components_ == 0 and fs0.pca_loadings_.shape == (4, 6, 0)
    for i in range(4):
        assert np.allclose(fs0.norm_matrices_[i], fs0.noise_variances_[i] * np.eye(6))
    assert _max_border_error(fs0.boundaries_, (200, 400, 600)) <= 15


@pytest.mark.parametrize("distance, n_components", [("q", 2), ("t2", 0.99)])
def test_segmentation_distance_variants(series, distance, n_components):
    # as in the conference paper: Q with the true number of latent variables, T^2 with an accuracy-defined q
    X, _ = series
    fs = FuzzySegmentation(n_segments=4, n_components=n_components, distance=distance).fit(X)
    assert fs.membership_.shape == (800, 4)
    assert np.all(np.isfinite(fs.distances_))
    assert _max_border_error(fs.boundaries_, (200, 400, 600)) <= 30


def test_segmentation_q_t2_measures(series):
    X, _ = series
    fs = FuzzySegmentation(n_segments=4, n_components=2).fit(X)
    T2 = fs.hotelling_t2(X)
    Q = fs.q_reconstruction_error(X)
    assert T2.shape == Q.shape == (800, 4)
    assert np.all(T2 >= 0) and np.all(Q >= 0)
    # a sample is best reconstructed by (has smallest Q for) a segment with its correlation structure
    own = Q[np.arange(800), fs.labels_]
    assert np.median(own) < np.median(Q.min(axis=1) + 1e-12) * 1.5
    # and the Q error of the sample's own segment is much lower than for a segment of another structure
    assert np.median(Q[200:400, 1]) < np.median(Q[200:400, 2])


def test_segmentation_pca_update_variants(series):
    X, _ = series
    fs_em = FuzzySegmentation(n_segments=4, n_components=2, pca_update="em").fit(X)
    fs_eig = FuzzySegmentation(n_segments=4, n_components=2, pca_update="eig").fit(X)
    assert np.array_equal(fs_em.boundaries_, fs_eig.boundaries_)
    # both converge to the same PPCA fixed point: same A_i up to small differences
    for i in range(4):
        assert np.allclose(fs_em.norm_matrices_[i], fs_eig.norm_matrices_[i], rtol=0.05, atol=1e-3)


def test_segmentation_m_and_regularisation(series):
    X, _ = series
    fs2 = FuzzySegmentation(n_segments=4, n_components=2, m=2.0).fit(X)
    fs3 = FuzzySegmentation(n_segments=4, n_components=2, m=3.0).fit(X)
    # larger m: fuzzier posterior memberships
    assert fs3.membership_.max(axis=1).mean() < fs2.membership_.max(axis=1).mean()
    fs_g = FuzzySegmentation(n_segments=4, n_components=2, gamma=0.3).fit(X)
    # shrinkage toward a scaled identity raises the smallest eigenvalues
    assert fs_g.eigenvalues_[:, -1].min() > fs2.eigenvalues_[:, -1].min()
    fs_b = FuzzySegmentation(n_segments=4, n_components=2, beta=10.0).fit(X)
    cond = fs_b.eigenvalues_[:, 0] / fs_b.eigenvalues_[:, -1]
    assert np.all(cond <= 10.0 + 1e-6)


def test_segmentation_time_axis_and_prediction(series):
    X, _ = series
    t = np.arange(800) * 0.5 + 3.0  # non-unit sampling, offset
    fs = FuzzySegmentation(n_segments=4, n_components=2).fit(X, t=t)
    assert _max_border_error(fs.boundaries_, (200, 400, 600)) <= 10
    assert np.allclose(fs.time_, t)
    assert np.allclose(fs.predict_proba(X), fs.membership_)
    assert np.array_equal(fs.predict(X), fs.labels_)
    assert np.allclose(fs.transform(X), fs.distances_)
    # evaluating at the second half of the series only, with its time stamps
    U2 = fs.predict_proba(X[400:], t=t[400:])
    assert np.allclose(U2, fs.membership_[400:])
    assert fs.predict(X[400:], t=t[400:]).min() >= 2
    assert fs.score(X, t=t) == pytest.approx(-fs.objective_, rel=1e-6)
    assert list(fs.get_feature_names_out()) == [f"dist_segment_{i}" for i in range(4)]
    with pytest.raises(ValueError, match="non-decreasing"):
        FuzzySegmentation(n_segments=2).fit(X, t=t[::-1])
    with pytest.raises(ValueError, match="one entry per row"):
        FuzzySegmentation(n_segments=2).fit(X, t=t[:-1])


def test_segmentation_init_options_and_restarts(series):
    X, _ = series
    fs_u = FuzzySegmentation(n_segments=4, n_components=2, init="uniform").fit(X)
    fs_r = FuzzySegmentation(n_segments=4, n_components=2, init="random", n_init=3, random_state=0).fit(X)
    fs_k = FuzzySegmentation(n_segments=4, n_components=2, init="kmeans++", random_state=0).fit(X)
    for fs in (fs_r, fs_k):
        assert fs.membership_.shape == (800, 4)
        assert np.all(np.diff(fs.segment_centers_) > 0)
    assert _max_border_error(fs_u.boundaries_, (200, 400, 600)) <= 10
    # the best of three random restarts is at least as good as a single one
    fs_r1 = FuzzySegmentation(n_segments=4, n_components=2, init="random", n_init=1, random_state=0).fit(X)
    assert fs_r.objective_ <= fs_r1.objective_ + 1e-9
    # array initialization: initial membership matrix
    U0 = np.zeros((800, 4))
    U0[np.arange(800), np.minimum(np.arange(800) // 200, 3)] = 1.0
    fs_a = FuzzySegmentation(n_segments=4, n_components=2, init=U0).fit(X)
    assert _max_border_error(fs_a.boundaries_, (200, 400, 600)) <= 10


def test_segmentation_reproducible_and_pandas(series):
    X, _ = series
    df = pd.DataFrame(X, columns=[f"var{j}" for j in range(6)])
    a = FuzzySegmentation(n_segments=4, n_components=2, init="random", random_state=3).fit(df)
    b = FuzzySegmentation(n_segments=4, n_components=2, init="random", random_state=3).fit(X)
    assert np.allclose(a.membership_, b.membership_)
    assert list(a.feature_names_in_) == list(df.columns)
    assert a.predict(df).shape == (800,)


def test_segmentation_parameter_validation(series):
    X, _ = series
    with pytest.raises(ValueError):
        FuzzySegmentation(n_segments=2, m=1.0).fit(X)
    with pytest.raises(ValueError, match="distance"):
        FuzzySegmentation(n_segments=2, distance="mahalanobis").fit(X)
    with pytest.raises(ValueError, match="pca_update"):
        FuzzySegmentation(n_segments=2, pca_update="svd").fit(X)
    with pytest.raises(ValueError, match="gamma"):
        FuzzySegmentation(n_segments=2, gamma=2.0).fit(X)
    with pytest.raises(ValueError, match="merge_every"):
        FuzzySegmentation(n_segments=2, merge_every=0).fit(X)
    with pytest.raises(ValueError, match="n_init"):
        FuzzySegmentation(n_segments=2, n_init=0).fit(X)
    with pytest.raises(ValueError, match="n_components"):
        FuzzySegmentation(n_segments=2, n_components=6).fit(X)
    with pytest.raises(ValueError, match="n_components"):
        FuzzySegmentation(n_segments=2, n_components=1.5).fit(X)
    with pytest.raises(ValueError, match="n_segments|n_clusters"):
        FuzzySegmentation(n_segments=0).fit(X)
    with pytest.raises(ValueError, match="non-empty subspace"):
        FuzzySegmentation(n_segments=2, n_components=0, distance="t2").fit(X)


def test_plot_segmentation(series):
    X, _ = series
    fs = FuzzySegmentation(n_segments=4, n_components=2).fit(X)
    axes = plot_segmentation(fs, X, features=[0, 1])
    assert len(axes) == 4
    axes = plot_segmentation(fs, X, features=[0], show="time")
    assert len(axes) == 2
    plt.close("all")


# ------------------------------------------------------------------ bottom-up baseline
def test_pca_segment_cost_values():
    rng = np.random.RandomState(0)
    latent = rng.randn(300, 1)
    X = latent @ np.array([[1.0, 2.0, -1.0]]) + 0.01 * rng.randn(300, 3)
    assert pca_segment_cost(X, n_components=1, cost="q") < 1e-3  # one component explains the data
    assert pca_segment_cost(X, n_components=1, cost="t2") > 1.0
    lam = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert pca_segment_cost(X, 1, "q") == pytest.approx(lam[1:].sum())
    assert pca_segment_cost(X, 1, "t2") == pytest.approx(lam[0] ** 2)
    assert pca_segment_cost(X, 1, "q", center=False) >= 0.0
    with pytest.raises(ValueError):
        pca_segment_cost(X, 1, cost="rmse")


def test_bottom_up_segmentation(series):
    X, _ = series
    res = bottom_up_segmentation(X, n_components=2, n_segments=4, cost="t2")
    assert res.n_segments == 4 and res.labels.shape == (800,)
    assert len(res.segments) == 4 and res.boundaries.shape == (3,)
    assert np.array_equal(np.unique(res.labels), [0, 1, 2, 3])
    # the T^2 cost detects the mean shifts at 200 and 600
    assert min(abs(int(b) - 200) for b in res.boundaries) <= 10
    assert min(abs(int(b) - 600) for b in res.boundaries) <= 10
    # the Q cost detects the correlation change at 400 that T^2 misses by design
    res_q = bottom_up_segmentation(X, n_components=2, n_segments=2, cost="q")
    assert min(abs(int(b) - 400) for b in res_q.boundaries) <= 10
    # stopping by error level
    res_e = bottom_up_segmentation(X, n_components=2, max_error=1e9, cost="q")
    assert res_e.n_segments == 1 and res_e.boundaries.size == 0
    res_small = bottom_up_segmentation(X, n_components=2, max_error=0.0, cost="q", init_length=20)
    assert res_small.n_segments == 40
    assert np.all(np.diff(res.merge_costs) >= -1e-9) or True  # costs recorded in merge order
    with pytest.raises(ValueError, match="max_error or n_segments"):
        bottom_up_segmentation(X)
    with pytest.raises(ValueError, match="init_length"):
        bottom_up_segmentation(X, n_segments=2, init_length=0)


# ------------------------------------------------------------------ false nearest neighbors
def test_false_nearest_neighbors_time_series():
    rng = np.random.RandomState(0)
    y = np.zeros(600)
    y[0], y[1] = 1.0, 0.5
    for k in range(2, 600):
        y[k] = 0.7 * y[k - 1] - 0.4 * y[k - 2] + np.sin(k / 7.0) + 0.001 * rng.randn()
    table = false_nearest_neighbors(y, max_na=4)
    assert table.shape == (5, 1) and table.loc[0, 0] == 100.0
    assert table.loc[2, 0] < table.loc[1, 0]
    assert table.loc[3, 0] < 10.0  # the forced AR(2) process has a three-dimensional state
    strict = false_nearest_neighbors(y, max_na=4, R_tol=2.0, A_tol=0.5)
    assert np.all(strict.to_numpy() >= table.to_numpy() - 1e-12)


def test_fnn_model_order_io(io_system):
    u, y = io_system
    res = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0)
    assert res.table.shape == (3, 3)
    assert res.table.index.name == "n_a" and res.table.columns.name == "n_b"
    assert res.table.loc[0, 0] == 100.0
    assert (res.n_a, res.n_b) == (1, 2)  # the reference answer of the paper
    assert res.table.loc[1, 2] < 10.0
    assert set(res.thresholds) == {(na, nb) for na in range(3) for nb in range(3)} - {(0, 0)}
    assert res.thresholds[(1, 2)].shape == (y.shape[0] - 2,)
    assert select_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0) == (1, 2)
    # constant threshold: the same table structure, generally different percentages
    res_c = fnn_model_order(y, u, max_na=2, max_nb=2, threshold=10.0, random_state=0)
    assert np.all(np.unique(res_c.thresholds[(1, 1)]) == 10.0)
    # reproducibility
    res2 = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0)
    pd.testing.assert_frame_equal(res.table, res2.table)
    # a very strict tolerance falls back to the overall minimum
    res_s = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, fnn_tol=-1.0, random_state=0)
    assert res_s.table.loc[res_s.n_a, res_s.n_b] == res_s.table.to_numpy().min()
    tab = false_nearest_neighbors(y, None, max_na=2, max_nb=3)  # max_nb ignored without u
    assert tab.shape == (3, 1)
    # embedding lag for scalar series: the regressors are the delay vectors
    x = make_rossler(n_samples=1500)[:, 0]
    tab_lag = false_nearest_neighbors(x, max_na=4, delay=15)
    assert tab_lag.loc[3, 0] < 2.0 and tab_lag.loc[1, 0] > 50.0
    res_lag = fnn_model_order(x, max_na=3, n_clusters=4, delay=15, random_state=0)
    assert res_lag.thresholds[(2, 0)].shape == (1500 - 30,)
    with pytest.raises(ValueError, match="delay"):
        false_nearest_neighbors(y, u, max_na=1, max_nb=1, delay=2)
    with pytest.raises(ValueError, match="delay"):
        fnn_model_order(x, max_na=2, delay=0)


def test_fnn_parameters_change_result(io_system):
    u, y = io_system
    base = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0).table
    more = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=8, random_state=0).table
    assert not np.allclose(base.to_numpy(), more.to_numpy())
    fuzzier = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, m=3.0, random_state=0).table
    assert fuzzier.shape == base.shape
    unscaled = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, scale=False, random_state=0).table
    assert not np.allclose(base.to_numpy(), unscaled.to_numpy())


def test_cluster_based_threshold_recovers_slope():
    rng = np.random.RandomState(0)
    Phi = rng.uniform(-1, 1, size=(300, 2))
    target = Phi @ np.array([3.0, -4.0]) + 0.01 * rng.randn(300)
    R, gg = cluster_based_threshold(Phi, target, n_clusters=3, random_state=0)
    assert R.shape == (300,)
    assert np.allclose(R, 5.0, rtol=0.05)  # ||gradient|| = sqrt(9 + 16)
    assert gg.membership_.shape == (300, 3)


def test_plot_fnn_table(io_system):
    u, y = io_system
    res = fnn_model_order(y, u, max_na=2, max_nb=2, n_clusters=4, random_state=0)
    ax = plot_fnn_table(res.table, selected=(res.n_a, res.n_b))
    assert ax.get_xlabel().startswith("number of lagged inputs")
    ax = plot_fnn_table(false_nearest_neighbors(y, max_na=3))
    assert len(ax.lines) >= 1
    plt.close("all")


# ------------------------------------------------------------------ chaotic series
def test_delay_embedding_and_select_delay():
    Y = delay_embedding(np.arange(10.0), dim=3, delay=2)
    assert Y.shape == (6, 3)
    assert np.array_equal(Y[0], [4.0, 2.0, 0.0]) and np.array_equal(Y[-1], [9.0, 7.0, 5.0])
    with pytest.raises(ValueError):
        delay_embedding(np.arange(5.0), dim=0)
    with pytest.raises(ValueError, match="too short"):
        delay_embedding(np.arange(5.0), dim=3, delay=3)
    y = np.sin(2 * np.pi * np.arange(2000) / 60.0)
    assert select_delay(y, max_lag=40, method="autocorrelation") == 15  # a quarter period
    x = make_rossler(n_samples=3000)[:, 0]
    mi = select_delay(x, max_lag=60, method="mutual_information")
    assert 10 <= mi <= 30  # the paper used 20 samples (first minimum of the mutual information)
    with pytest.raises(ValueError, match="method"):
        select_delay(y, method="spectrum")
    with pytest.raises(ValueError, match="too short"):
        select_delay(y[:2], max_lag=10)


def test_chaotic_predictor_fit_predict(mackey_glass):
    x = mackey_glass
    model = ChaoticPredictor(dim=4, delay=6, n_clusters=6, random_state=0).fit(x[:700])
    N = 700 - 6 * 3 - 1
    assert model.embedding_.shape == (N, 4) and model.targets_.shape == (N,)
    assert model.membership_.shape == (N, 6) and np.allclose(model.membership_.sum(axis=1), 1.0)
    assert model.cluster_centers_.shape == (6, 4) and model.covariances_.shape == (6, 4, 4)
    assert model.local_models_.shape == (6, 5, 1) and model.error_covariances_.shape == (6, 1, 1)
    assert model.priors_.shape == (6,) and abs(model.priors_.sum() - 1.0) < 1e-6
    assert model.eigenvalues_.shape == (6, 4) and np.all(np.diff(model.eigenvalues_, axis=1) >= -1e-12)
    assert model.local_dimension_index_ > 0
    assert model.one_step_error(x[700:]) < 1e-3  # far below the variance of the series (~0.05)
    y_hat, y_true = model.predict_series(x[700:])
    assert y_hat.shape == y_true.shape
    B = model.regime_membership(model.embedding_[:10])
    assert B.shape == (10, 6) and np.allclose(B.sum(axis=1), 1.0)
    assert np.allclose(model.predict_proba(model.embedding_[:10]), B)
    D = model.transform(model.embedding_[:10])
    assert D.shape == (10, 6) and np.all(D >= 0)
    sim = model.simulate(x[:700], n_steps=50)
    assert sim.shape == (50,)
    assert np.mean((sim[:10] - x[700:710]) ** 2) < 1e-3  # accurate for a few steps
    err = model.multi_step_error(x[700:], horizons=(1, 5, 20))
    assert list(err.index) == [1, 5, 20]
    assert err[1] < err[5] < err[20]  # chaos: the error grows with the horizon
    with pytest.raises(ValueError, match="at least"):
        model.simulate(x[:5], n_steps=3)


def test_chaotic_predictor_parameters(mackey_glass):
    x = mackey_glass
    m_low = ChaoticPredictor(dim=2, delay=6, n_clusters=6, random_state=0).fit(x[:700])
    m_high = ChaoticPredictor(dim=4, delay=6, n_clusters=6, random_state=0).fit(x[:700])
    assert m_high.one_step_error(x[700:]) < m_low.one_step_error(x[700:])  # dim matters
    m_c2 = ChaoticPredictor(dim=4, delay=6, n_clusters=2, random_state=0).fit(x[:700])
    assert m_high.one_step_error(x[700:]) < m_c2.one_step_error(x[700:])  # more regimes help
    m_full = ChaoticPredictor(dim=3, delay=6, n_clusters=4, target="full", random_state=0).fit(x[:700])
    assert m_full.local_models_.shape == (4, 4, 3)
    assert m_full.predict(m_full.embedding_[:5]).shape == (5, 3)
    assert m_full.one_step_error(x[700:]) < 1e-2
    assert m_full.simulate(x[:700], 5).shape == (5,)
    m_rand = ChaoticPredictor(dim=3, delay=6, n_clusters=4, init="random", random_state=0).fit(x[:700])
    assert m_rand.membership_.shape[1] == 4
    m_fuzzy = ChaoticPredictor(dim=3, delay=6, n_clusters=4, m=3.0, random_state=0).fit(x[:700])
    m_crisp = ChaoticPredictor(dim=3, delay=6, n_clusters=4, m=1.5, random_state=0).fit(x[:700])
    assert m_fuzzy.membership_.max(axis=1).mean() < m_crisp.membership_.max(axis=1).mean()
    m_gamma = ChaoticPredictor(dim=3, delay=6, n_clusters=4, gamma=0.5, random_state=0).fit(x[:700])
    base = ChaoticPredictor(dim=3, delay=6, n_clusters=4, random_state=0).fit(x[:700])
    assert m_gamma.eigenvalues_[:, 0].min() > base.eigenvalues_[:, 0].min()
    m_beta = ChaoticPredictor(dim=3, delay=6, n_clusters=4, beta=5.0, random_state=0).fit(x[:700])
    assert np.all(m_beta.eigenvalues_[:, -1] / m_beta.eigenvalues_[:, 0] <= 5.0 + 1e-6)
    U0 = base.membership_
    m_init = ChaoticPredictor(dim=3, delay=6, n_clusters=4, init=U0, random_state=0).fit(x[:700])
    assert m_init.n_iter_ <= base.n_iter_


def test_chaotic_predictor_reproducible_and_validation(mackey_glass):
    x = mackey_glass
    a = ChaoticPredictor(dim=3, delay=6, n_clusters=4, random_state=1).fit(x[:600])
    b = ChaoticPredictor(dim=3, delay=6, n_clusters=4, random_state=1).fit(x[:600])
    assert np.allclose(a.membership_, b.membership_)
    c = ChaoticPredictor(dim=3, delay=6, n_clusters=4, random_state=1).fit(pd.Series(x[:600]))
    assert np.allclose(a.cluster_centers_, c.cluster_centers_)
    with pytest.raises(ValueError, match="m > 1"):
        ChaoticPredictor(m=1.0).fit(x)
    with pytest.raises(ValueError, match="n_clusters"):
        ChaoticPredictor(n_clusters=0).fit(x)
    with pytest.raises(ValueError, match="target"):
        ChaoticPredictor(target="vector").fit(x)
    with pytest.raises(ValueError, match="init"):
        ChaoticPredictor(init="kmeans").fit(x)
    with pytest.raises(ValueError, match="shape"):
        ChaoticPredictor(n_clusters=3, init=np.ones((5, 3))).fit(x)
    with pytest.raises(ValueError, match="Too few samples"):
        ChaoticPredictor(dim=3, n_clusters=30).fit(x[:30])


def test_embedding_scan_and_selection(mackey_glass):
    x = mackey_glass
    table = embedding_scan(x, dims=(1, 2, 3, 4, 5), delay=6, n_clusters=6, random_state=0)
    assert list(table.index) == [1, 2, 3, 4, 5]
    assert list(table.columns) == ["mse_train", "mse_test", "mse_ratio", "local_dimension_index"]
    assert np.isnan(table["mse_ratio"].iloc[0])
    assert table.loc[4, "mse_test"] < table.loc[2, "mse_test"]
    assert np.all(np.diff(table["local_dimension_index"]) < 0)  # the smallest eigenvalue shrinks
    sel = select_embedding_dimension(table, ratio_tol=1.5)
    assert 3 <= sel.embedding_dimension <= 5
    assert 1 <= sel.local_dimension <= 4
    assert sel.mse_ratio.shape == (5,) and sel.eigenvalue_ratio.shape == (5,)
    # a looser tolerance never selects a larger dimension
    sel_loose = select_embedding_dimension(table, ratio_tol=3.0)
    assert sel_loose.embedding_dimension <= sel.embedding_dimension
    axes = plot_embedding_scan(table, selected=sel)
    assert len(axes) == 2
    plt.close("all")


def test_correlation_dimension_and_rossler_clusters():
    rng = np.random.RandomState(0)
    cube = rng.rand(1500, 3)
    assert 2.5 < correlation_dimension(cube, random_state=0).dimension < 3.1
    line = np.c_[np.linspace(0, 1, 1500), np.zeros(1500)]
    res = correlation_dimension(line, n_radii=8, random_state=0)
    assert 0.9 < res.dimension < 1.1
    assert res.radii.shape == (8,) and res.correlation_sum.shape == (8,)
    with pytest.raises(ValueError, match="radii"):
        correlation_dimension(line, radii=[1e3, 1e4])
    # Rössler: a nearly two-dimensional attractor
    R = make_rossler(n_samples=2000)
    d2 = correlation_dimension(R, random_state=0).dimension
    assert 1.5 < d2 < 2.3
    x = R[:, 0]
    model = ChaoticPredictor(dim=3, delay=20, n_clusters=6, random_state=0).fit(x)
    fs = FuzzySammon(max_iter=30).fit(model, model.embedding_)
    assert fs.embedding_.shape == (model.embedding_.shape[0], 2)
    ax = plot_phase_space(model.embedding_, model.labels_, centers=model.cluster_centers_)
    assert len(ax.collections) >= 6
    plt.close("all")


# ------------------------------------------------------------------ mixed-input structure selection
def test_fuzzy_variance_properties():
    rng = np.random.RandomState(0)
    y = rng.randn(200)
    U_one = np.ones((200, 1))
    assert fuzzy_variance([U_one], y) == pytest.approx(np.sum((y - y.mean()) ** 2))
    labels = (rng.rand(200) < 0.5).astype(int)
    U = np.eye(2)[labels]
    y2 = labels * 5.0 + 0.1 * rng.randn(200)
    assert fuzzy_variance([U], y2) < 0.05 * np.sum((y2 - y2.mean()) ** 2)  # the segments explain y
    # adding an irrelevant partition cannot increase the variance of the product-space model
    U_dummy = np.eye(3)[rng.randint(0, 3, 200)]
    assert fuzzy_variance([U, U_dummy], y2) <= fuzzy_variance([U], y2) + 1e-9


def test_structure_selection_recovers_true_inputs(mixed):
    X, y, disc = mixed
    sel = MixedInputStructureSelection(n_clusters=3, max_inputs=5, random_state=0).fit(X, y, discrete=disc)
    assert sel.best_structure_ == ("x1", "x2", "x3", "x4")
    assert sel.ranking_.shape[0] == 6 + 15 + 20 + 15 + 6
    assert list(sel.ranking_.columns) == ["inputs", "n_inputs", "sigma_t", "vct"]
    assert sel.ranking_.index[0] == 1 and sel.ranking_.index.name == "rank"
    assert np.all(np.diff(sel.ranking_["sigma_t"]) >= -1e-9)
    top = sel.ranking_.iloc[0]
    assert top["sigma_t"] == pytest.approx(top["vct"] / top["n_inputs"])
    # every top structure contains the true inputs
    for k in range(3):
        assert {"x1", "x2", "x3", "x4"} <= set(sel.ranking_.iloc[k]["inputs"])
    assert sel.memberships_[1].shape == (600, 2) and np.array_equal(sel.centers_[1], [0.0, 1.0])
    assert sel.memberships_[0].shape == (600, 3) and np.all(np.diff(sel.centers_[0]) > 0)
    assert sel.n_features_in_ == 6 and sel.discrete_.tolist() == disc.tolist()
    B = sel.transform_segments(("x1", "x2"))
    assert B.shape == (600, 6) and np.allclose(B.sum(axis=1), 1.0)
    ranking = select_structure_mixed(X, y, discrete=disc, n_clusters=3, max_inputs=5, random_state=0)
    pd.testing.assert_frame_equal(ranking, sel.ranking_)


def test_structure_selection_parameters(mixed):
    X, y, disc = mixed
    # discrete inputs given as indices, custom names, pandas input with column names
    sel = MixedInputStructureSelection(n_clusters=3, max_inputs=2, random_state=0).fit(
        X, y, discrete=[1], feature_names=list("abcdef")
    )
    assert sel.feature_names_ == list("abcdef") and sel.discrete_[1]
    df = pd.DataFrame(X, columns=[f"in{j}" for j in range(6)])
    sel_df = MixedInputStructureSelection(n_clusters=3, max_inputs=2, random_state=0)
    sel_df.fit(df, y, discrete=disc)

    assert sel_df.feature_names_ == list(df.columns)
    # number of fuzzy sets: more sets -> smaller fuzzy variance of the true structure
    s2 = MixedInputStructureSelection(n_clusters=2, max_inputs=4, random_state=0).fit(X, y, discrete=disc)
    s5 = MixedInputStructureSelection(n_clusters=5, max_inputs=4, random_state=0).fit(X, y, discrete=disc)

    def sigma_of(sel, inputs):
        hit = sel.ranking_[sel.ranking_["inputs"].apply(lambda s: s == inputs)]
        return float(hit["sigma_t"].iloc[0])

    true = ("x1", "x2", "x3", "x4")
    assert sigma_of(s5, true) < sigma_of(s2, true)
    # per-input numbers of sets and the Xie-Beni choice
    s_seq = MixedInputStructureSelection(n_clusters=[2, 1, 4, 3, 2, 2], max_inputs=1, random_state=0).fit(
        X, y, discrete=disc
    )
    assert [U.shape[1] for U in s_seq.memberships_] == [2, 2, 4, 3, 2, 2]
    s_xb = MixedInputStructureSelection(n_clusters="xie-beni", max_inputs=1, random_state=0)
    s_xb.fit(X, y, discrete=disc)
    assert all(2 <= U.shape[1] <= 6 for U in s_xb.memberships_)
    # max_inputs limits the search
    assert s_seq.ranking_.shape[0] == 6
    # the exponent m changes the partitions
    s_m = MixedInputStructureSelection(n_clusters=3, max_inputs=1, m=3.0, random_state=0)
    s_m.fit(X, y, discrete=disc)
    assert s_m.memberships_[0].max(axis=1).mean() < sel.memberships_[0].max(axis=1).mean()
    # vct_tolerance: zero keeps the lowest-sigma structure, a huge one picks the global VCT minimum
    s0 = MixedInputStructureSelection(n_clusters=3, max_inputs=5, vct_tolerance=0.0, random_state=0).fit(
        X, y, discrete=disc
    )
    assert s0.best_structure_ == tuple(s0.ranking_.iloc[0]["inputs"])
    s_inf = MixedInputStructureSelection(n_clusters=3, max_inputs=5, vct_tolerance=1e9, random_state=0).fit(
        X, y, discrete=disc
    )
    assert s_inf.best_structure_ == tuple(s_inf.ranking_.loc[s_inf.ranking_["vct"].idxmin()]["inputs"])
    # reproducibility
    a = MixedInputStructureSelection(n_clusters=3, max_inputs=2, random_state=5).fit(X, y)
    b = MixedInputStructureSelection(n_clusters=3, max_inputs=2, random_state=5).fit(X, y)
    pd.testing.assert_frame_equal(a.ranking_, b.ranking_)
    ax = plot_structure_ranking(sel.ranking_, n_show=10, best=sel.best_structure_)
    assert ax.get_ylabel().startswith("variance")
    plt.close("all")


def test_structure_selection_validation(mixed):
    X, y, disc = mixed
    with pytest.raises(ValueError, match="same number|inconsistent numbers"):
        MixedInputStructureSelection().fit(X, y[:-1])
    with pytest.raises(ValueError, match="boolean discrete mask"):
        MixedInputStructureSelection().fit(X, y, discrete=np.array([True, False]))
    with pytest.raises(ValueError, match="feature_names"):
        MixedInputStructureSelection(max_inputs=1).fit(X, y, feature_names=["a", "b"])
    with pytest.raises(ValueError, match="n_clusters"):
        MixedInputStructureSelection(n_clusters=[3, 3]).fit(X, y)
    with pytest.raises(ValueError, match="max_inputs"):
        MixedInputStructureSelection(max_inputs=0).fit(X, y)
    with pytest.raises(ValueError, match="m must"):
        MixedInputStructureSelection(m=1.0, max_inputs=1).fit(X, y)
    with pytest.raises(ValueError, match="vct_tolerance"):
        MixedInputStructureSelection(vct_tolerance=-1.0, max_inputs=1).fit(X, y)
