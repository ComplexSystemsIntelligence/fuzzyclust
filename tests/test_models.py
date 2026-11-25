import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.datasets import load_iris  # noqa: E402
from fuzzyclust.models import (  # noqa: E402
    FuzzyCRegression,
    FuzzyPartitionDiscretizer,
    HingingHyperplaneTree,
    ModifiedGG,
    SupervisedFuzzyClassifier,
    TakagiSugeno,
    make_mimo_plant,
    make_piecewise_linear,
    make_second_order_plant,
    make_sinc,
    make_switching_lines,
    make_two_class_overlap,
    membership_1d,
    piecewise_linear_function,
    plot_decision_regions,
    plot_fuzzy_partition,
    plot_hinge_tree_surface,
    plot_local_models,
    plot_membership_functions,
    ruspini_partition,
    second_order_plant_function,
    select_features_by_separability,
    set_similarity,
    simulate_free_run,
    supervised_fuzzy_partition,
    weighted_least_squares,
)
from fuzzyclust.preprocessing import make_regressors  # noqa: E402

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def sinc():
    return make_sinc(n_samples=300, random_state=0)


@pytest.fixture(scope="module")
def lines():
    return make_switching_lines(n_samples=300, random_state=0)


@pytest.fixture(scope="module")
def two_class():
    return make_two_class_overlap(n_samples=400, random_state=0)


@pytest.fixture(scope="module")
def piecewise():
    return make_piecewise_linear(n_samples=400, random_state=0)


@pytest.fixture(scope="module")
def iris():
    data = load_iris()
    return data.data, data.target


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    """Checks that cannot apply to supervised estimators built on the clustering base class.

    ``check_clustering`` is selected because the estimators inherit ``ClusterMixin`` from
    ``BaseFuzzyClusterer`` (they *are* alternating-optimization clusterers), but it calls
    ``fit(X)`` without ``y``; all three raise the scikit-learn style "requires y" error.
    ``FuzzyCRegression`` measures the distance of a sample to a cluster by its regression
    residual, so ``transform`` needs ``(X, y)``; the checks that call ``transform(X)`` fail
    with exactly this informative ``ValueError``.
    """
    name = type(estimator).__name__
    needs_y = f"{name} is supervised: fit(X) without y raises the 'requires y' ValueError."
    out = {}
    if name in ("ModifiedGG", "FuzzyCRegression", "SupervisedFuzzyClassifier"):
        out["check_clustering"] = needs_y
    if name == "FuzzyCRegression":
        reason = "FCRM residual distances need y: transform(X) raises an informative ValueError."
        for check in (
            "check_estimators_dtypes",
            "check_dtype_object",
            "check_estimators_pickle",
            "check_f_contiguous_array_estimator",
            "check_transformer_data_not_an_array",
            "check_transformer_general",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_dict_unchanged",
            "check_fit_idempotent",
        ):
            out[check] = reason
    return out


@parametrize_with_checks(
    [
        ModifiedGG(n_clusters=2, n_init=1, max_iter=50, random_state=0),
        TakagiSugeno(n_clusters=2, max_iter=50, random_state=0),
        FuzzyCRegression(n_clusters=2, max_iter=50, random_state=0),
        SupervisedFuzzyClassifier(n_clusters=3, n_init=1, max_iter=100, random_state=0),
        FuzzyPartitionDiscretizer(n_sets=2, max_iter=50, random_state=0),
        HingingHyperplaneTree(max_depth=1, n_init=1, random_state=0),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_requires_y_messages(sinc):
    X, _ = sinc
    for est in (
        ModifiedGG(),
        TakagiSugeno(),
        FuzzyCRegression(),
        SupervisedFuzzyClassifier(),
        FuzzyPartitionDiscretizer(),
        HingingHyperplaneTree(),
    ):
        with pytest.raises(ValueError, match="requires y to be passed"):
            est.fit(X)


# ------------------------------------------------------------------ weighted least squares
def test_weighted_least_squares_recovers_line():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(50, 2))
    Xe = np.hstack([X, np.ones((50, 1))])
    theta_true = np.array([[2.0], [-1.0], [0.5]])
    y = Xe @ theta_true
    np.testing.assert_allclose(weighted_least_squares(Xe, y, np.ones(50)), theta_true, atol=1e-6)
    # weights select the samples: zero weight on half the data, still exact
    w = (X[:, 0] > 0).astype(float)
    np.testing.assert_allclose(weighted_least_squares(Xe, y, w), theta_true, atol=1e-6)


# ------------------------------------------------------------------ ModifiedGG
def test_modified_gg_shapes_and_attributes(sinc):
    X, y = sinc
    mgg = ModifiedGG(n_clusters=4, random_state=0).fit(X, y)
    N, c = X.shape[0], 4
    assert mgg.cluster_centers_.shape == (c, 1)
    assert mgg.input_variances_.shape == (c, 1)
    assert mgg.local_models_.shape == (c, 2, 1)
    assert mgg.coef_.shape == (c, 1) and mgg.intercept_.shape == (c,)
    assert mgg.error_variances_.shape == (c,) and mgg.error_covariances_.shape == (c, 1, 1)
    assert mgg.priors_.shape == (c,) and mgg.rule_weights_.shape == (c,)
    assert mgg.membership_.shape == (N, c) and mgg.labels_.shape == (N,)
    assert mgg.distances_.shape == (N, c)
    assert mgg.n_iter_ == mgg.objective_history_.size
    assert mgg.restart_scores_.shape == (3,)
    np.testing.assert_allclose(mgg.priors_.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(mgg.membership_.sum(axis=1), 1.0)
    assert mgg.predict(X).shape == (N,)
    assert mgg.local_predict(X).shape == (N, c)
    assert mgg.firing_strengths(X).shape == (N, c)
    np.testing.assert_allclose(mgg.firing_strengths(X).sum(axis=1), 1.0)
    assert mgg.firing_strengths(X, normalize=False).min() >= 0
    assert mgg.transform(X).shape == (N, c)
    assert mgg.transform(X, y).shape == (N, c)
    assert list(mgg.get_feature_names_out()) == [f"dist_rule_{i}" for i in range(c)]
    assert np.isfinite(mgg.score(X, y)) and np.isfinite(mgg.log_likelihood_)
    assert mgg.fit_predict(X, y).shape == (N,)
    np.testing.assert_allclose(mgg.fit_transform(X, y), mgg.transform(X))


def test_modified_gg_recovers_two_local_linear_models():
    # two well separated operating regions with different linear laws
    rng = np.random.RandomState(1)
    x = np.concatenate([rng.uniform(-3, -1, 150), rng.uniform(1, 3, 150)])
    y = np.where(x < 0, 2.0 * x + 1.0, -1.0 * x + 3.0) + 0.01 * rng.randn(300)
    mgg = ModifiedGG(n_clusters=2, random_state=0).fit(x[:, None], y)
    order = np.argsort(mgg.cluster_centers_[:, 0])
    np.testing.assert_allclose(mgg.coef_[order, 0], [2.0, -1.0], atol=0.05)
    np.testing.assert_allclose(mgg.intercept_[order], [1.0, 3.0], atol=0.1)
    np.testing.assert_allclose(mgg.cluster_centers_[order, 0], [-2.0, 2.0], atol=0.2)
    assert mgg.error_variances_.max() < 1e-3
    # memberships with y are the posterior cluster probabilities (EM for m = 2)
    U = mgg.predict_proba(x[:, None], y)
    logp = -np.log(mgg.transform(x[:, None], y) ** 2)
    post = np.exp(logp - logp.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(U, post, atol=1e-6)
    # memberships without y are the normalized firing strengths
    np.testing.assert_allclose(mgg.predict_proba(x[:, None]), mgg.firing_strengths(x[:, None]))


def test_modified_gg_mimo_and_antecedent_features():
    u, y = make_mimo_plant(n_samples=300, random_state=0)
    Phi, T, _ = make_regressors(y, u, n_a=1, n_b=1, n_k=1)
    mgg = ModifiedGG(n_clusters=3, antecedent_features=[0, 1], random_state=0).fit(Phi, T)
    assert mgg.n_outputs_ == 2
    assert mgg.cluster_centers_.shape == (3, 2)  # antecedents on the two lagged outputs only
    assert mgg.local_models_.shape == (3, 5, 2)  # consequents use all four regressors + intercept
    assert mgg.coef_.shape == (3, 2, 4) and mgg.intercept_.shape == (3, 2)
    assert mgg.error_covariances_.shape == (3, 2, 2) and mgg.error_variances_.shape == (3, 2)
    assert mgg.predict(Phi).shape == T.shape
    assert mgg.predict_proba(Phi, T).shape == (Phi.shape[0], 3)
    table = mgg.describe_rules()
    assert table.shape[0] == 6 and "output" in table.columns
    with pytest.raises(ValueError, match="outputs"):
        mgg.predict_proba(Phi, T[:, :1])
    with pytest.raises(ValueError, match="antecedent_features"):
        ModifiedGG(n_clusters=2, antecedent_features=[7]).fit(Phi, T)


def test_modified_gg_parameters_gamma_var_floor_m(sinc):
    X, y = sinc
    base = ModifiedGG(n_clusters=4, random_state=0).fit(X, y)
    feat_var = X.var(axis=0)
    # gamma shrinks the variances toward the feature variance
    reg = ModifiedGG(n_clusters=4, gamma=0.5, random_state=0).fit(X, y)
    assert np.abs(reg.input_variances_ - feat_var).mean() < np.abs(base.input_variances_ - feat_var).mean()
    # var_floor bounds the variances from below (relative to the feature variance)
    floored = ModifiedGG(n_clusters=4, var_floor=0.5, random_state=0).fit(X, y)
    assert floored.input_variances_.min() >= 0.5 * feat_var[0] * (1 - 1e-9)
    assert base.input_variances_.min() < 0.5 * feat_var[0]
    # a larger m gives a fuzzier partition
    fuzzy = ModifiedGG(n_clusters=4, m=3.0, random_state=0).fit(X, y)
    assert fuzzy.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    # describe_rules columns
    cols = list(base.describe_rules().columns)
    assert cols[:3] == ["rule", "weight", "prior"] and cols[-1] == "error sigma"
    assert base.rules_.shape == (4, len(cols))


def test_modified_gg_restart_criteria(sinc):
    X, y = sinc

    def sse(est):
        return float(np.sum((est.predict(X) - y) ** 2))

    ll = ModifiedGG(n_clusters=3, n_init=5, restart_criterion="likelihood", random_state=0).fit(X, y)
    err = ModifiedGG(n_clusters=3, n_init=5, restart_criterion="error", random_state=0).fit(X, y)
    assert ll.restart_scores_.shape == (5,) and err.restart_scores_.shape == (5,)
    assert ll.log_likelihood_ == pytest.approx(ll.restart_scores_.max())
    assert sse(err) == pytest.approx(-err.restart_scores_.max())
    assert sse(err) <= sse(ll) + 1e-9
    # every restart is a different local optimum of the same data: scores differ
    assert np.ptp(ll.restart_scores_) > 0
    # an array init disables the restarts
    one = ModifiedGG(n_clusters=3, n_init=5, init=ll.membership_, random_state=0).fit(X, y)
    assert one.restart_scores_.shape == (1,)
    with pytest.raises(ValueError, match="restart_criterion"):
        ModifiedGG(restart_criterion="bogus").fit(X, y)


def test_modified_gg_reproducible_and_pandas(sinc):
    X, y = sinc
    a = ModifiedGG(n_clusters=3, random_state=4).fit(X, y)
    b = ModifiedGG(n_clusters=3, random_state=4).fit(X, y)
    np.testing.assert_array_equal(a.membership_, b.membership_)
    df = pd.DataFrame(X, columns=["speed"])
    c = ModifiedGG(n_clusters=3, random_state=4).fit(df, y)
    assert list(c.feature_names_in_) == ["speed"]
    assert "speed center" in c.describe_rules().columns
    np.testing.assert_allclose(c.predict(df), a.predict(X))


def test_modified_gg_validation(sinc):
    X, y = sinc
    with pytest.raises(ValueError, match="m > 1"):
        ModifiedGG(m=1.0).fit(X, y)
    with pytest.raises(ValueError, match="gamma"):
        ModifiedGG(gamma=2.0).fit(X, y)
    with pytest.raises(ValueError, match="var_floor"):
        ModifiedGG(var_floor=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="n_init"):
        ModifiedGG(n_init=0).fit(X, y)
    with pytest.raises(ValueError, match="feature_names"):
        ModifiedGG(n_clusters=2, random_state=0).fit(X, y).describe_rules(feature_names=["a", "b"])


# ------------------------------------------------------------------ TakagiSugeno
@pytest.mark.parametrize("antecedent", ["gaussian_axis_parallel", "gg_projected", "gk_projected"])
@pytest.mark.parametrize("consequent", ["local", "global", "tls"])
def test_takagi_sugeno_fits_sinc(sinc, antecedent, consequent):
    X, y = sinc
    ts = TakagiSugeno(n_clusters=5, antecedent=antecedent, consequent=consequent, random_state=0).fit(X, y)
    assert ts.score(X, y) > 0.85
    assert ts.centers_.shape == (5, 1) and ts.sigmas_.shape == (5, 1)
    assert ts.local_models_.shape == (5, 2, 1)
    assert ts.membership_.shape == (X.shape[0], 5)
    assert ts.n_iter_ >= 1
    if antecedent == "gk_projected":
        np.testing.assert_array_equal(ts.rule_weights_, 1.0)
    else:
        assert np.all(ts.rule_weights_ > 0) and np.ptp(ts.rule_weights_) > 0


def test_takagi_sugeno_api(sinc):
    X, y = sinc
    N = X.shape[0]
    ts = TakagiSugeno(n_clusters=4, random_state=0).fit(X, y)
    A = ts.antecedent_membership(X)
    assert A.shape == (N, 4, 1) and A.max() <= 1.0 and A.min() >= 0.0
    B = ts.firing_strengths(X)
    np.testing.assert_allclose(B.sum(axis=1), 1.0)
    Bu = ts.firing_strengths(X, normalize=False)
    np.testing.assert_allclose(Bu / Bu.sum(axis=1, keepdims=True), B)
    local = ts.local_predict(X)
    assert local.shape == (N, 4)
    np.testing.assert_allclose(np.sum(B * local, axis=1), ts.predict(X))
    rb = ts.rule_base()
    expected = {"rule", "weight", "x1 center", "x1 sigma", "coef x1", "intercept"}
    assert rb.shape[0] == 4 and expected <= set(rb.columns)
    text = ts.describe()
    assert text.count("\n") == 3 and text.startswith("R1: If x1 is N(")
    # the engine is exposed
    assert isinstance(ts.clusterer_, ModifiedGG)
    gk = TakagiSugeno(n_clusters=2, antecedent="gk_projected", random_state=0).fit(X, y)
    assert isinstance(gk.clusterer_.covariances_, np.ndarray)


def test_takagi_sugeno_consequent_estimators_differ(sinc):
    X, y = sinc
    local = TakagiSugeno(n_clusters=4, consequent="local", random_state=0).fit(X, y)
    glob = TakagiSugeno(n_clusters=4, consequent="global", random_state=0).fit(X, y)
    tls = TakagiSugeno(n_clusters=4, consequent="tls", random_state=0).fit(X, y)
    # same antecedents (same clustering), different consequents
    np.testing.assert_allclose(local.centers_, glob.centers_)
    assert not np.allclose(local.local_models_, glob.local_models_)
    assert not np.allclose(local.local_models_, tls.local_models_)
    # global least squares minimizes the prediction error of the whole model
    assert glob.score(X, y) >= local.score(X, y) - 1e-9


def test_takagi_sugeno_mimo_and_antecedent_features():
    u, y = make_mimo_plant(n_samples=400, random_state=0)
    Phi, T, _ = make_regressors(y, u, n_a=1, n_b=1, n_k=1)
    ts = TakagiSugeno(n_clusters=3, antecedent_features=[0, 1], random_state=0).fit(Phi, T)
    assert ts.n_outputs_ == 2
    np.testing.assert_array_equal(ts.antecedent_features_, [0, 1])
    assert ts.centers_.shape == (3, 2) and ts.local_models_.shape == (3, 5, 2)
    assert ts.coef_.shape == (3, 2, 4) and ts.intercept_.shape == (3, 2)
    assert ts.predict(Phi).shape == T.shape and ts.score(Phi, T) > 0.95
    assert ts.local_predict(Phi).shape == (Phi.shape[0], 3, 2)
    rb = ts.rule_base(output_names=["top", "bottom"])
    assert rb.shape[0] == 6 and set(rb["output"]) == {"top", "bottom"}
    assert "y1 =" in ts.describe() and "y2 =" in ts.describe()
    for ant in ("gg_projected", "gk_projected"):
        proj = TakagiSugeno(n_clusters=2, antecedent=ant, antecedent_features=[1], random_state=0).fit(Phi, T)
        assert proj.centers_.shape == (2, 1) and proj.predict(Phi).shape == T.shape
    with pytest.raises(ValueError, match="single output"):
        TakagiSugeno(n_clusters=2, consequent="tls").fit(Phi, T)
    with pytest.raises(ValueError, match="antecedent_features"):
        TakagiSugeno(n_clusters=2, antecedent_features=[9]).fit(Phi, T)


def test_takagi_sugeno_parameters(sinc):
    X, y = sinc
    with pytest.raises(ValueError, match="antecedent must be"):
        TakagiSugeno(antecedent="bogus").fit(X, y)
    with pytest.raises(ValueError, match="consequent must be"):
        TakagiSugeno(consequent="bogus").fit(X, y)
    # more rules -> better fit on the training data
    scores = [TakagiSugeno(n_clusters=c, random_state=0).fit(X, y).score(X, y) for c in (2, 4, 8)]
    assert scores[0] < scores[1] < scores[2]
    # gamma is forwarded to the engine
    reg = TakagiSugeno(n_clusters=4, gamma=0.3, random_state=0).fit(X, y)
    assert reg.clusterer_.gamma == 0.3
    # n_init and restart_criterion are forwarded
    multi = TakagiSugeno(n_clusters=4, n_init=4, restart_criterion="error", random_state=0).fit(X, y)
    assert multi.clusterer_.restart_scores_.shape == (4,)
    single = TakagiSugeno(n_clusters=4, n_init=1, random_state=0).fit(X, y)
    assert multi.score(X, y) >= single.score(X, y) - 1e-9
    # reproducibility and pandas
    a = TakagiSugeno(n_clusters=3, random_state=1).fit(X, y).predict(X)
    b = TakagiSugeno(n_clusters=3, random_state=1).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
    df = pd.DataFrame(X, columns=["temp"])
    ts = TakagiSugeno(n_clusters=3, random_state=1).fit(df, y)
    assert "temp is N(" in ts.describe() and "coef temp" in ts.rule_base().columns


def test_simulate_free_run_linear_system_is_exact():
    rng = np.random.RandomState(0)
    T = 120
    u = rng.uniform(-1, 1, T)
    y = np.zeros(T)
    for k in range(2, T):
        y[k] = 0.6 * y[k - 1] - 0.2 * y[k - 2] + 0.5 * u[k - 1]
    Phi, target, _ = make_regressors(y, u, n_a=2, n_b=1, n_k=1)
    ts = TakagiSugeno(n_clusters=1, random_state=0).fit(Phi, target)
    sim = simulate_free_run(ts, u, y[:2], n_a=2, n_b=1, n_k=1)
    assert sim.shape == (T,)
    np.testing.assert_array_equal(sim[:2], y[:2])
    np.testing.assert_allclose(sim, y, atol=1e-6)
    with pytest.raises(ValueError, match="y_init"):
        simulate_free_run(ts, u, y[:1], n_a=2, n_b=1, n_k=1)


def test_simulate_free_run_nonlinear_plant_and_mimo():
    u, y = make_second_order_plant(n_samples=200, random_state=0)
    Phi, target, _ = make_regressors(y, u, n_a=2, n_b=1, n_k=0)
    ts = TakagiSugeno(n_clusters=4, antecedent_features=[0, 1], random_state=0).fit(Phi, target)
    us, ys = make_second_order_plant(n_samples=200, excitation="sine", random_state=1)
    sim = simulate_free_run(ts, us, ys[:2], n_a=2, n_b=1, n_k=0)
    assert sim.shape == ys.shape
    assert np.mean((sim[2:] - ys[2:]) ** 2) < 0.1 * np.var(ys)
    # MIMO
    u2, y2 = make_mimo_plant(n_samples=300, random_state=0)
    Phi2, T2, _ = make_regressors(y2, u2, n_a=1, n_b=1, n_k=1)
    ts2 = TakagiSugeno(n_clusters=3, antecedent_features=[0, 1], random_state=0).fit(Phi2, T2)
    sim2 = simulate_free_run(ts2, u2, y2[:1], n_a=1, n_b=1, n_k=1)
    assert sim2.shape == y2.shape
    np.testing.assert_array_equal(sim2[0], y2[0])
    assert np.mean((sim2[1:] - y2[1:]) ** 2) < 0.2 * np.var(y2)
    # autonomous model (no input)
    Phi3, T3, _ = make_regressors(y, n_a=2)
    ts3 = TakagiSugeno(n_clusters=2, random_state=0).fit(Phi3, T3)
    sim3 = simulate_free_run(ts3, None, y[:2], n_a=2)
    assert sim3.shape == (2,)  # without an input sequence the horizon is the length of y_init


# ------------------------------------------------------------------ FuzzyCRegression
def test_fcrm_recovers_crossing_lines(lines):
    X, y, lab = lines
    fcrm = FuzzyCRegression(n_clusters=2, n_init=3, random_state=0).fit(X, y)
    order = np.argsort(fcrm.coef_[:, 0])
    np.testing.assert_allclose(fcrm.coef_[order, 0], [-1.0, 1.0], atol=0.05)
    np.testing.assert_allclose(fcrm.intercept_[order], [1.0, 0.0], atol=0.05)
    assert adjusted_rand_score(lab, fcrm.labels_) > 0.9
    assert fcrm.local_models_.shape == (2, 2) and fcrm.cluster_centers_.shape == (2, 1)
    assert fcrm.membership_.shape == (300, 2) and fcrm.distances_.shape == (300, 2)
    np.testing.assert_allclose(fcrm.distances_, np.abs(fcrm.transform(X, y)))
    assert fcrm.n_iter_ == fcrm.objective_history_.size
    assert np.all(np.diff(fcrm.objective_history_) <= 1e-8)  # alternating optimization is monotone
    # predictions of every local model, memberships of pairs, hard labels of pairs
    P = fcrm.predict(X)
    assert P.shape == (300, 2)
    np.testing.assert_allclose(P, X @ fcrm.coef_.T + fcrm.intercept_)
    U = fcrm.predict_proba(X, y)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    np.testing.assert_array_equal(fcrm.predict_labels(X, y), U.argmax(axis=1))
    np.testing.assert_array_equal(fcrm.fit_predict(X, y), fcrm.labels_)
    assert fcrm.fit_transform(X, y).shape == (300, 2)
    assert fcrm.score(X, y) == pytest.approx(-fcrm.objective_, rel=1e-6)
    assert list(fcrm.get_feature_names_out()) == ["residual_model_0", "residual_model_1"]
    with pytest.raises(ValueError, match="needs the output y"):
        fcrm.transform(X)
    with pytest.raises(ValueError, match="needs the output y"):
        fcrm.predict_proba(X)


def test_fcrm_parameters(lines):
    X, y, _ = lines
    base = FuzzyCRegression(n_clusters=2, random_state=0).fit(X, y)
    # fit_intercept=False forces the lines through the origin
    no_b = FuzzyCRegression(n_clusters=2, fit_intercept=False, random_state=0).fit(X, y)
    np.testing.assert_array_equal(no_b.intercept_, 0.0)
    assert no_b.local_models_.shape == (2, 2) and np.all(no_b.local_models_[:, -1] == 0)
    assert no_b.objective_ > base.objective_
    # larger m -> fuzzier partition
    fuzzy = FuzzyCRegression(n_clusters=2, m=3.0, random_state=0).fit(X, y)
    assert fuzzy.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    # n_init keeps the lowest objective among the restarts
    seeds = [FuzzyCRegression(n_clusters=2, random_state=s).fit(X, y).objective_ for s in range(4)]
    multi = FuzzyCRegression(n_clusters=2, n_init=4, random_state=0).fit(X, y)
    assert multi.objective_ <= min(seeds) + 1e-6 or multi.objective_ <= base.objective_ + 1e-6
    # initializations: kmeans++ centers, an initial partition, initial centers
    km = FuzzyCRegression(n_clusters=2, init="kmeans++", random_state=0).fit(X, y)
    assert km.coef_.shape == (2, 1)
    part = FuzzyCRegression(n_clusters=2, init=base.membership_).fit(X, y)
    np.testing.assert_allclose(np.sort(part.coef_[:, 0]), np.sort(base.coef_[:, 0]), atol=1e-3)
    cen = FuzzyCRegression(n_clusters=2, init=np.array([[-1.0], [1.0]])).fit(X, y)
    assert cen.coef_.shape == (2, 1)
    with pytest.raises(ValueError, match="m > 1"):
        FuzzyCRegression(m=1.0).fit(X, y)
    with pytest.raises(ValueError, match="init"):
        FuzzyCRegression(n_clusters=2, init=np.zeros((5, 5))).fit(X, y)


def test_fcrm_three_lines_reproducible_pandas():
    X, y, lab = make_switching_lines(
        n_samples=450, slopes=(1.0, -1.0, 0.0), intercepts=(0.0, 1.0, -1.0), noise=0.05, random_state=3
    )
    a = FuzzyCRegression(n_clusters=3, n_init=5, random_state=0).fit(X, y)
    b = FuzzyCRegression(n_clusters=3, n_init=5, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.membership_, b.membership_)
    assert adjusted_rand_score(lab, a.labels_) > 0.8
    df = pd.DataFrame(X, columns=["x"])
    c = FuzzyCRegression(n_clusters=3, n_init=5, random_state=0).fit(df, y)
    assert list(c.feature_names_in_) == ["x"]
    np.testing.assert_allclose(c.predict(df), a.predict(X))


# ------------------------------------------------------------------ SupervisedFuzzyClassifier
def test_supervised_classifier_needs_more_rules_than_classes(two_class):
    X, y = two_class
    two = SupervisedFuzzyClassifier(n_clusters=2, random_state=0).fit(X, y)
    four = SupervisedFuzzyClassifier(n_clusters=4, random_state=0).fit(X, y)
    assert two.score(X, y) < 0.8 < 0.9 < four.score(X, y)
    # every rule of the four-rule model is (nearly) pure, each class gets two rules
    pure = four.class_probs_.max(axis=1)
    assert np.all(pure > 0.9)
    assert sorted(four.class_probs_.argmax(axis=1)) == [0, 0, 1, 1]
    # rule centers sit on the four blobs
    centres = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 1.5], [0.0, -1.5]])
    d = np.linalg.norm(four.cluster_centers_[:, None, :] - centres[None], axis=2)
    assert d.min(axis=1).max() < 0.4


def test_supervised_classifier_api(iris):
    X, y = iris
    N = X.shape[0]
    clf = SupervisedFuzzyClassifier(n_clusters=3, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.9
    np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
    P = clf.predict_proba(X)
    assert P.shape == (N, 3) and P.min() >= 0
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[P.argmax(axis=1)])
    assert clf.cluster_centers_.shape == (3, 4) and clf.input_variances_.shape == (3, 4)
    assert clf.class_probs_.shape == (3, 3) and clf.priors_.shape == (3,)
    np.testing.assert_allclose(clf.class_probs_.sum(axis=1), 1.0)
    np.testing.assert_allclose(clf.priors_.sum(), 1.0, atol=1e-8)
    assert clf.rule_weights_.shape == (3,) and clf.restart_scores_.shape == (3,)
    assert clf.labels_.shape == (N,) and set(np.unique(clf.labels_)) <= {0, 1, 2}
    # cluster memberships with and without the labels
    M = clf.cluster_membership(X)
    np.testing.assert_allclose(M.sum(axis=1), 1.0)
    My = clf.cluster_membership(X, y)
    np.testing.assert_allclose(My.sum(axis=1), 1.0)
    assert not np.allclose(M, My)
    with pytest.raises(ValueError, match="unseen"):
        clf.cluster_membership(X, np.full(N, 7))
    assert clf.transform(X).shape == (N, 3) and clf.fit_transform(X, y).shape == (N, 3)
    assert list(clf.get_feature_names_out()) == ["dist_rule_0", "dist_rule_1", "dist_rule_2"]
    np.testing.assert_array_equal(clf.fit_predict(X, y), clf.predict(X))
    rb = clf.rule_base(class_names=["setosa", "versicolor", "virginica"])
    assert rb.shape == (3, 3 + 2 * 4 + 3) and "P(setosa)" in rb.columns
    assert "x1 center" in rb.columns and "x4 sigma" in rb.columns


def test_supervised_classifier_string_labels_and_pandas(two_class):
    X, y = two_class
    names = np.array(["cat", "dog"])[y]
    df = pd.DataFrame(X, columns=["length", "weight"])
    clf = SupervisedFuzzyClassifier(n_clusters=4, random_state=0).fit(df, names)
    np.testing.assert_array_equal(clf.classes_, ["cat", "dog"])
    assert set(clf.predict(df)) <= {"cat", "dog"}
    assert "length center" in clf.rule_base().columns and "P(dog)" in clf.rule_base().columns
    a = SupervisedFuzzyClassifier(n_clusters=4, random_state=2).fit(X, y).predict_proba(X)
    b = SupervisedFuzzyClassifier(n_clusters=4, random_state=2).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_supervised_classifier_parameters(two_class):
    X, y = two_class
    base = SupervisedFuzzyClassifier(n_clusters=4, random_state=0).fit(X, y)
    feat_var = X.var(axis=0)
    reg = SupervisedFuzzyClassifier(n_clusters=4, gamma=0.5, random_state=0).fit(X, y)
    assert np.abs(reg.input_variances_ - feat_var).mean() < np.abs(base.input_variances_ - feat_var).mean()
    floored = SupervisedFuzzyClassifier(n_clusters=4, var_floor=0.9, random_state=0).fit(X, y)
    assert floored.input_variances_.min() >= 0.9 * feat_var.min() * (1 - 1e-9)
    fuzzy = SupervisedFuzzyClassifier(n_clusters=4, m=3.0, random_state=0).fit(X, y)
    assert fuzzy.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    # literal eq. (20): no normalizing constants in the clustering distance
    plain = SupervisedFuzzyClassifier(n_clusters=4, normalize_density=False, random_state=0).fit(X, y)
    assert plain.score(X, y) > 0.9
    assert not np.allclose(plain.distances_, base.distances_)
    # the firing strengths of both carry the eq. (18) weights
    log_w = np.log(plain.priors_) - 0.5 * np.sum(np.log(2 * np.pi * plain.input_variances_), axis=1)
    np.testing.assert_allclose(np.log(plain.rule_weights_), log_w)
    # restarts
    multi = SupervisedFuzzyClassifier(n_clusters=4, n_init=5, random_state=0).fit(X, y)
    assert multi.restart_scores_.shape == (5,)
    assert multi.log_likelihood_ == pytest.approx(multi.restart_scores_.max())
    one = SupervisedFuzzyClassifier(n_clusters=4, n_init=1, random_state=0).fit(X, y)
    assert multi.log_likelihood_ >= one.log_likelihood_ - 1e-9
    with pytest.raises(ValueError, match="m > 1"):
        SupervisedFuzzyClassifier(m=1.0).fit(X, y)
    with pytest.raises(ValueError, match="gamma"):
        SupervisedFuzzyClassifier(gamma=-0.1).fit(X, y)
    with pytest.raises(ValueError, match="var_floor"):
        SupervisedFuzzyClassifier(var_floor=-1).fit(X, y)


def test_interclass_separability_and_feature_selection(iris):
    X, y = iris
    clf = SupervisedFuzzyClassifier(n_clusters=3, random_state=0).fit(X, y)
    J = clf.interclass_separability()
    assert J > 0 and np.isfinite(J)
    assert clf.interclass_separability(criterion="trace") > 0
    assert np.isnan(clf.interclass_separability(features=[]))
    imp = clf.feature_importance()
    assert imp.shape == (4,) and np.all(imp > 0)
    # sepal length is the least needed feature: leaving it out keeps the largest separability
    assert imp.argmax() == 0
    assert clf.feature_importance(criterion="trace").shape == (4,)
    # separated classes are more separable than overlapping ones
    rng = np.random.RandomState(0)
    Xa = np.vstack([rng.randn(100, 2), rng.randn(100, 2) + 6.0])
    Xb = np.vstack([rng.randn(100, 2), rng.randn(100, 2) + 1.0])
    ya = np.repeat([0, 1], 100)
    Ja = SupervisedFuzzyClassifier(n_clusters=2, random_state=0).fit(Xa, ya).interclass_separability()
    Jb = SupervisedFuzzyClassifier(n_clusters=2, random_state=0).fit(Xb, ya).interclass_separability()
    assert Ja > Jb
    with pytest.raises(ValueError, match="criterion"):
        clf.interclass_separability(criterion="bogus")
    sel, order, hist = select_features_by_separability(
        X, y, n_features_to_select=2, n_clusters=3, random_state=0
    )
    assert len(sel) == 2 and len(order) == 2 and set(sel) | set(order) == {0, 1, 2, 3}
    assert list(hist["n_features"]) == [4, 3, 2] and hist["removed"].iloc[0] == -1
    assert hist["accuracy"].min() > 0.85
    assert 3 in sel or 2 in sel  # a petal feature survives
    # with a single feature left it is a petal feature, for both criteria
    for crit in ("det", "trace"):
        sel1, order1, hist1 = select_features_by_separability(
            X, y, n_features_to_select=1, criterion=crit, n_clusters=3, random_state=0
        )
        assert sel1[0] in (2, 3) and len(order1) == 3 and hist1.shape[0] == 4
    with pytest.raises(ValueError, match="n_features_to_select"):
        select_features_by_separability(X, y, n_features_to_select=0)


# ------------------------------------------------------------------ fuzzy partitions
def test_membership_1d_shapes_and_set_similarity():
    grid = np.linspace(-5, 5, 201)
    for shape in ("gaussian", "trapezoid", "triangle"):
        A = membership_1d(grid, [-2.0, 2.0], [1.0, 1.0], shape)
        assert A.shape == (201, 2) and A.min() >= 0 and A.max() <= 1.0 + 1e-12
        assert A[np.argmin(np.abs(grid + 2)), 0] == pytest.approx(1.0, abs=1e-6)
    trap = membership_1d(grid, [0.0], [1.0], "trapezoid")[:, 0]
    assert np.all(trap[np.abs(grid) <= 0.5] == 1.0) and np.all(trap[np.abs(grid) >= 3.0] == 0.0)
    with pytest.raises(ValueError, match="shape"):
        membership_1d(grid, [0.0], [1.0], "bogus")
    A = membership_1d(grid, [-3.0, -3.0, 3.0], [0.5, 0.5, 0.5], "gaussian")
    assert set_similarity(A[:, 0], A[:, 1]) == pytest.approx(1.0)
    assert set_similarity(A[:, 0], A[:, 2]) < 1e-6


def test_supervised_fuzzy_partition_and_merging(iris):
    X, y = iris
    parts, clf = supervised_fuzzy_partition(X, y, n_clusters=3, random_state=0)
    assert isinstance(clf, SupervisedFuzzyClassifier) and len(parts) == 4
    for p in parts:
        assert set(p) == {"centers", "sigmas", "weights", "groups", "range", "shape"}
        assert np.all(np.diff(p["centers"]) > 0)  # sorted
        assert len(p["centers"]) == len(p["sigmas"]) == len(p["weights"]) == len(p["groups"])
        assert sum(len(g) for g in p["groups"]) == 3  # every rule ends up in exactly one set
    # the petal sets are far apart, the sepal-width sets overlap: a low threshold merges the latter first
    parts_merge, _ = supervised_fuzzy_partition(X, y, n_clusters=3, merge_threshold=0.3, random_state=0)
    n_sets = [len(p["centers"]) for p in parts_merge]
    assert n_sets[1] < 3 and n_sets[2] == 3
    parts_none, _ = supervised_fuzzy_partition(X, y, n_clusters=3, merge_threshold=1.0, random_state=0)
    assert all(len(p["centers"]) == 3 for p in parts_none)
    with pytest.raises(ValueError, match="shape"):
        supervised_fuzzy_partition(X, y, shape="bogus")
    with pytest.raises(ValueError, match="merge_threshold"):
        supervised_fuzzy_partition(X, y, merge_threshold=2.0)


def test_ruspini_partition_sums_to_one():
    X = np.linspace(0, 10, 50)[:, None]
    parts = ruspini_partition(X, n_sets=4)
    p = parts[0]
    np.testing.assert_allclose(p["centers"], [0.0, 10 / 3, 20 / 3, 10.0])
    A = membership_1d(X[:, 0], p["centers"], p["sigmas"], p["shape"])
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-9)
    with pytest.raises(ValueError, match="n_sets"):
        ruspini_partition(X, n_sets=1)


def test_fuzzy_partition_discretizer(iris):
    X, y = iris
    disc = FuzzyPartitionDiscretizer(n_sets=3, random_state=0).fit(X, y)
    codes = disc.transform(X)
    assert codes.shape == (150, 4) and codes.dtype == float
    assert set(np.unique(codes)) <= {0.0, 1.0, 2.0}
    assert disc.n_sets_.shape == (4,) and disc.n_iter_ >= 1
    assert list(disc.get_feature_names_out()) == ["x0_set", "x1_set", "x2_set", "x3_set"]
    M = disc.membership(X)
    assert len(M) == 4 and all(A.shape == (150, disc.n_sets_[j]) for j, A in enumerate(M))
    soft = FuzzyPartitionDiscretizer(n_sets=3, encode="membership", random_state=0).fit(X, y)
    S = soft.transform(X)
    assert S.shape == (150, soft.n_sets_.sum()) and S.min() >= 0 and S.max() <= 1
    assert len(soft.get_feature_names_out(["a", "b", "c", "d"])) == S.shape[1]
    assert disc.evaluate(2, np.linspace(1, 7, 50)).shape == (50, disc.n_sets_[2])
    # the supervised cut points separate the species on the petal length
    petal = codes[:, 2]
    assert adjusted_rand_score(y, petal) > 0.8
    # a tree on the discretized features is accurate
    pipe = make_pipeline(
        FuzzyPartitionDiscretizer(n_sets=3, random_state=0), DecisionTreeClassifier(random_state=0)
    )
    assert pipe.fit(X, y).score(X, y) > 0.9
    # parameters: merge_threshold / drop_single / shape / method
    merged = FuzzyPartitionDiscretizer(n_sets=3, merge_threshold=0.3, drop_single=True, random_state=0)
    merged.fit(X, y)
    assert merged.n_sets_.sum() < 12 and merged.transform(X).shape[1] == len(merged.active_features_)
    gauss = FuzzyPartitionDiscretizer(n_sets=3, shape="gaussian", random_state=0).fit(X, y)
    assert gauss.partitions_[0]["shape"] == "gaussian"
    rusp = FuzzyPartitionDiscretizer(n_sets=3, method="ruspini").fit(X, y)
    assert rusp.classifier_ is None and rusp.n_iter_ == 0
    assert all(p["shape"] == "triangle" for p in rusp.partitions_)
    assert rusp.transform(X).shape == (150, 4)
    assert adjusted_rand_score(y, rusp.transform(X)[:, 2]) < adjusted_rand_score(y, petal)
    # pandas
    df = pd.DataFrame(X, columns=["sl", "sw", "pl", "pw"])
    d = FuzzyPartitionDiscretizer(n_sets=3, random_state=0).fit(df, y)
    assert list(d.get_feature_names_out()) == ["sl_set", "sw_set", "pl_set", "pw_set"]
    with pytest.raises(ValueError, match="encode"):
        FuzzyPartitionDiscretizer(encode="bogus").fit(X, y)
    with pytest.raises(ValueError, match="method"):
        FuzzyPartitionDiscretizer(method="bogus").fit(X, y)


# ------------------------------------------------------------------ HingingHyperplaneTree
def test_hinge_tree_fits_piecewise_linear_surface(piecewise):
    X, y = piecewise
    tree = HingingHyperplaneTree(max_depth=3, random_state=0).fit(X, y)
    assert tree.score(X, y) > 0.99
    assert tree.n_leaves_ >= 2 and tree.n_nodes_ == 2 * tree.n_leaves_ - 1
    assert tree.depth_ <= 3 and len(tree.leaves_) == tree.n_leaves_
    assert tree.predict(X).shape == (400,)
    leaves = tree.apply(X)
    assert leaves.shape == (400,) and leaves.min() >= 0 and leaves.max() < tree.n_leaves_
    # prediction is the hinge value of the leaf the sample falls into
    Xe = np.hstack([X, np.ones((400, 1))])
    manual = np.array([tree.leaves_[leaf].hinge_value(Xe[k : k + 1])[0] for k, leaf in enumerate(leaves)])
    np.testing.assert_allclose(manual, tree.predict(X))
    table = tree.node_table()
    assert list(table.columns) == ["node", "depth", "kind", "n_samples", "rmse", "condition", "leaf"]
    assert table["leaf"].sum() == tree.n_leaves_ and table["n_samples"].iloc[0] == 400
    assert np.all((table["condition"] >= 0) & (table["condition"] <= 1))
    text = tree.describe()
    assert text.startswith("node 0 (depth 0") and "h1 =" in text and "plane 1 active" in text


def test_hinge_tree_single_hinge_is_exact():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = np.maximum(X[:, 0] + 0.5 * X[:, 1], 1.0 - X[:, 0])
    tree = HingingHyperplaneTree(max_depth=0, random_state=0).fit(X, y)
    assert tree.n_leaves_ == 1 and tree.tree_.kind == "max"
    assert tree.tree_.rmse < 1e-6 and tree.tree_.condition > 0.99
    planes = tree.tree_.theta[np.argsort(tree.tree_.theta[:, 0])]
    np.testing.assert_allclose(planes, [[-1.0, 0.0, 1.0], [1.0, 0.5, 0.0]], atol=1e-6)
    # a min-type hinge is recognized as well
    tree_min = HingingHyperplaneTree(max_depth=0, random_state=0).fit(X, -y)
    assert tree_min.tree_.kind == "min" and tree_min.tree_.rmse < 1e-6


def test_hinge_tree_parameters(piecewise):
    X, y = piecewise
    scores = [HingingHyperplaneTree(max_depth=d, random_state=0).fit(X, y).score(X, y) for d in (0, 1, 2)]
    assert scores[0] < scores[1] < scores[2]
    big_leaf = HingingHyperplaneTree(max_depth=3, min_samples_leaf=150, random_state=0).fit(X, y)
    small_leaf = HingingHyperplaneTree(max_depth=3, min_samples_leaf=10, random_state=0).fit(X, y)
    assert big_leaf.n_leaves_ < small_leaf.n_leaves_
    assert min(nd.n_samples for nd in big_leaf.leaves_) >= 150
    greedy = HingingHyperplaneTree(max_depth=4, growth="greedy", max_leaves=3, random_state=0).fit(X, y)
    assert greedy.n_leaves_ == 3
    stop = HingingHyperplaneTree(max_depth=3, tol_rmse=1.0, random_state=0).fit(X, y)
    assert stop.n_leaves_ == 1  # the root is already below the tolerance
    linear = HingingHyperplaneTree(max_depth=2, leaf_model="linear", random_state=0).fit(X, y)
    assert all(leaf.kind == "linear" and leaf.theta.shape == (1, 3) for leaf in linear.leaves_)
    hinge = HingingHyperplaneTree(max_depth=2, leaf_model="hinge", random_state=0).fit(X, y)
    assert hinge.score(X, y) > linear.score(X, y)  # two pieces per leaf fit better than one
    assert "): y =" in linear.describe()
    free = HingingHyperplaneTree(max_depth=1, constrain_hinge=False, random_state=0).fit(X, y)
    assert free.score(X, y) > 0.9
    fuzzy = HingingHyperplaneTree(max_depth=1, m=3.0, n_init=1, random_state=0).fit(X, y)
    assert fuzzy.score(X, y) > 0.9
    a = HingingHyperplaneTree(max_depth=2, random_state=5).fit(X, y).predict(X)
    b = HingingHyperplaneTree(max_depth=2, random_state=5).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
    df = pd.DataFrame(X, columns=["p", "q"])
    tr = HingingHyperplaneTree(max_depth=1, random_state=0).fit(df, y)
    assert "*p" in tr.describe() and "*q" in tr.describe()
    with pytest.raises(ValueError, match="growth"):
        HingingHyperplaneTree(growth="bogus").fit(X, y)
    with pytest.raises(ValueError, match="leaf_model"):
        HingingHyperplaneTree(leaf_model="bogus").fit(X, y)
    with pytest.raises(ValueError, match="max_depth"):
        HingingHyperplaneTree(max_depth=-1).fit(X, y)
    with pytest.raises(ValueError, match="m must"):
        HingingHyperplaneTree(m=1.0).fit(X, y)


def test_hinge_tree_small_nodes_fall_back_to_linear():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(5, 2))
    y = X @ [1.0, 2.0] + 0.5
    tree = HingingHyperplaneTree(max_depth=2, random_state=0).fit(X, y)
    assert tree.tree_.kind == "linear" and tree.n_leaves_ == 1
    np.testing.assert_allclose(tree.predict(X), y, atol=1e-8)


# ------------------------------------------------------------------ data generators
def test_data_generators():
    X, y = make_sinc(n_samples=50, random_state=0)
    assert X.shape == (50, 1) and y.shape == (50,)
    X2, y2 = make_sinc(n_samples=50, n_features=2, noise=0.0, random_state=0)
    np.testing.assert_allclose(y2, np.prod(np.sinc(X2 / np.pi), axis=1))
    with pytest.raises(ValueError):
        make_sinc(n_features=3)
    X, y = make_piecewise_linear(n_samples=30, noise=0.0, random_state=0)
    np.testing.assert_allclose(y, piecewise_linear_function(X))
    X, y, lab = make_switching_lines(n_samples=40, random_state=0)
    assert X.shape == (40, 1) and set(np.unique(lab)) <= {0, 1}
    with pytest.raises(ValueError):
        make_switching_lines(slopes=(1, 2), intercepts=(0,))
    u, y = make_second_order_plant(n_samples=30, random_state=0)
    assert u.shape == y.shape == (30,)
    np.testing.assert_allclose(y[5], second_order_plant_function(y[4], y[3]) + u[5])
    us, _ = make_second_order_plant(n_samples=30, excitation="sine")
    np.testing.assert_allclose(us, np.sin(2 * np.pi * np.arange(1, 31) / 25))
    _, yn = make_second_order_plant(n_samples=30, noise=0.1, random_state=0)
    assert not np.allclose(yn, y)
    with pytest.raises(ValueError):
        make_second_order_plant(excitation="bogus")
    u, y = make_mimo_plant(n_samples=40, hold=5, random_state=0)
    assert u.shape == (40, 2) and y.shape == (40, 2)
    assert np.all(u[0] == u[4]) and not np.all(u[0] == u[5])  # steps held for 5 samples
    X, y = make_two_class_overlap(n_samples=60, random_state=0)
    assert X.shape == (60, 2) and set(np.unique(y)) == {0, 1}
    a = make_two_class_overlap(random_state=1)[0]
    b = make_two_class_overlap(random_state=1)[0]
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ plots
def test_plot_helpers(sinc, two_class, piecewise, iris):
    X, y = sinc
    ts = TakagiSugeno(n_clusters=3, random_state=0).fit(X, y)
    axes = plot_membership_functions(ts, X=X)
    assert axes.shape == (1,) and len(axes[0].lines) >= 3
    ax = plot_local_models(ts, X, y)
    assert ax.get_ylabel() == "y"
    mgg = ModifiedGG(n_clusters=3, random_state=0).fit(X, y)
    assert plot_membership_functions(mgg, ranges=[(-10, 10)]).shape == (1,)
    assert plot_local_models(mgg, X, y, show_firing=False) is not None
    with pytest.raises(ValueError, match="one input"):
        plot_local_models(ts, np.hstack([X, X]), y)
    Xc, yc = two_class
    clf = SupervisedFuzzyClassifier(n_clusters=4, random_state=0).fit(Xc, yc)
    fig, axs = plt.subplots(1, 2)
    plot_decision_regions(clf, Xc, yc, ax=axs[0])
    plot_decision_regions(clf, Xc, yc, ax=axs[1], proba_class=1, show_rules=False)
    assert len(axs[0].patches) >= 4  # one ellipse per rule
    plot_membership_functions(clf, X=Xc, feature_names=["a", "b"])
    with pytest.raises(ValueError, match="two input"):
        plot_decision_regions(clf, Xc[:, :1], yc)
    Xi, yi = iris
    disc = FuzzyPartitionDiscretizer(n_sets=3, random_state=0).fit(Xi, yi)
    ax = plot_fuzzy_partition(disc, 2, X=Xi, y=yi, feature_name="petal length")
    assert ax.get_xlabel() == "petal length"
    assert plot_fuzzy_partition(disc, 0) is not None
    Xp, yp = piecewise
    tree = HingingHyperplaneTree(max_depth=2, random_state=0).fit(Xp, yp)
    ax = plot_hinge_tree_surface(tree, Xp, yp)
    assert ax.get_xlabel() == "$x_1$"
    with pytest.raises(ValueError, match="two input"):
        plot_hinge_tree_surface(tree, Xp[:, :1])
    with pytest.raises(TypeError):
        plot_membership_functions(tree)
    plt.close("all")
