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
from fuzzyclust.datasets import make_fuzzy_blobs  # noqa: E402
from fuzzyclust.incomplete import (  # noqa: E402
    STRATEGIES,
    IncompleteFCM,
    make_incomplete_blobs,
    make_mcar,
    partial_distances,
    plot_imputation,
    plot_incomplete_clusters,
)


def center_error(V, V_ref):
    """Largest distance between matched prototypes."""
    D = cdist(V, V_ref)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].max())


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def blobs4d():
    """Three well separated 4-D blobs, complete, with 30 % MCAR holes."""
    return make_incomplete_blobs(
        n_clusters=3, n_samples_per_cluster=150, n_features=4, missing_rate=0.3, random_state=0
    )


@pytest.fixture(scope="module")
def reference(blobs4d):
    _, _, X = blobs4d
    return FCM(n_clusters=3, random_state=0).fit(X)


# ------------------------------------------------------------------ sklearn compatibility
@parametrize_with_checks([IncompleteFCM(n_clusters=2, strategy=s, random_state=0) for s in STRATEGIES])
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ helpers
def test_make_mcar_properties():
    X = np.arange(200.0).reshape(50, 4)
    Xm, mask = make_mcar(X, missing_rate=0.3, random_state=0, return_mask=True)
    assert Xm.shape == X.shape and mask.dtype == bool
    assert np.array_equal(np.isnan(Xm), mask)
    assert 0.2 < mask.mean() < 0.4
    assert (~mask).any(axis=1).all() and (~mask).any(axis=0).all()
    np.testing.assert_array_equal(Xm[~mask], X[~mask])
    assert not np.isnan(X).any()  # the input is untouched
    # a very high rate still leaves one value per row and per column
    Xh, mh = make_mcar(X, missing_rate=0.98, random_state=0, return_mask=True)
    assert (~mh).any(axis=1).all() and (~mh).any(axis=0).all()
    np.testing.assert_array_equal(make_mcar(X, 0.3, random_state=1), make_mcar(X, 0.3, random_state=1))
    assert not np.isnan(make_mcar(X, missing_rate=0.0)).any()
    with pytest.raises(ValueError):
        make_mcar(X, missing_rate=1.0)
    with pytest.raises(ValueError):
        make_mcar(X.ravel())


def test_make_incomplete_blobs_shapes():
    Xm, y, X = make_incomplete_blobs(n_clusters=2, n_samples_per_cluster=30, n_features=3, missing_rate=0.25,
                                     random_state=0)
    assert Xm.shape == X.shape == (60, 3) and y.shape == (60,)
    assert np.isnan(Xm).any() and not np.isnan(X).any()
    np.testing.assert_array_equal(Xm[~np.isnan(Xm)], X[~np.isnan(Xm)])


def test_partial_distances_definition():
    X = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, np.nan], [np.nan, 2.0, 2.0]])
    V = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d2 = partial_distances(X, V)
    assert d2.shape == (3, 2)
    np.testing.assert_allclose(d2[0], [3.0, 0.0])  # complete sample: Euclidean
    np.testing.assert_allclose(d2[1], [0.0, 3.0])  # one observed feature rescaled by n/I_k = 3
    np.testing.assert_allclose(d2[2], [3.0 * 2.0 / 2.0 * 1.0 * 1.0 * 1.0, 3.0 * 8.0 / 2.0])
    with pytest.raises(ValueError):
        partial_distances(np.array([[np.nan, np.nan]]), V[:, :2])


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_recover_clusters_and_fitted_attributes(strategy, blobs4d, reference):
    Xm, y, X = blobs4d
    model = IncompleteFCM(n_clusters=3, strategy=strategy, random_state=0).fit(Xm)
    N = Xm.shape[0]
    assert model.cluster_centers_.shape == (3, 4)
    assert model.membership_.shape == (N, 3) and model.labels_.shape == (N,)
    np.testing.assert_allclose(model.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert model.distances_.shape == (N, 3) and np.isfinite(model.distances_).all()
    assert model.missing_mask_.shape == Xm.shape and model.n_missing_ == int(np.isnan(Xm).sum())
    assert model.X_completed_.shape == Xm.shape and not np.isnan(model.X_completed_).any()
    # observed values are never changed
    obs = ~model.missing_mask_
    np.testing.assert_array_equal(model.X_completed_[obs], Xm[obs])
    assert model.covariances_.shape == (3, 4, 4)
    assert model.n_iter_ == len(model.objective_history_) >= 1
    assert adjusted_rand_score(y, model.labels_) > 0.8
    # prototypes close to those of FCM on the complete data (spread of the blobs is 0.67)
    assert center_error(model.cluster_centers_, reference.cluster_centers_) < 0.4


@pytest.mark.parametrize("rate", [0.2, 0.3, 0.4])
def test_pds_and_ocs_recover_centers_within_tolerance(rate):
    """PDS and OCS stay close to the complete-data prototypes at 20-40 % missingness."""
    Xm, y, X = make_incomplete_blobs(n_clusters=3, n_samples_per_cluster=200, n_features=4, missing_rate=rate,
                                     random_state=3)
    ref = FCM(n_clusters=3, random_state=0).fit(X)
    pds = IncompleteFCM(n_clusters=3, strategy="pds", random_state=0).fit(Xm)
    ocs = IncompleteFCM(n_clusters=3, strategy="ocs", n_init=3, random_state=0).fit(Xm)
    assert center_error(pds.cluster_centers_, ref.cluster_centers_) < 0.25
    assert center_error(ocs.cluster_centers_, ref.cluster_centers_) < 0.5
    # the partial-distance prototypes are the exact weighted means of the observed values
    Um = pds.membership_**2
    I = ~pds.missing_mask_
    X0 = np.where(I, Xm, 0.0)
    # U is one step ahead of V at termination, hence the tolerance
    np.testing.assert_allclose(pds.cluster_centers_, (Um.T @ X0) / (Um.T @ I), atol=1e-3)


def test_ocs_optimality_conditions_and_monotone_objective(blobs4d):
    Xm, _, X = blobs4d
    ocs = IncompleteFCM(n_clusters=3, strategy="ocs", random_state=0).fit(Xm)
    J = ocs.objective_history_
    assert np.all(np.diff(J) <= 1e-8 * abs(J[0]))  # three-block alternating optimization descends
    # at convergence every missing value is the membership-weighted mean of the prototype coordinates
    Um = ocs.membership_**2
    est = (Um @ ocs.cluster_centers_) / Um.sum(axis=1, keepdims=True)
    mask = ocs.missing_mask_
    np.testing.assert_allclose(ocs.X_completed_[mask], est[mask], atol=2e-2)  # U is one step ahead of V, X_M
    # the completed data are clustered with plain FCM distances
    np.testing.assert_allclose(ocs.distances_, cdist(ocs.X_completed_, ocs.cluster_centers_), rtol=1e-6)
    # the imputation is informative: far better than the feature means
    rmse = np.sqrt(np.mean((ocs.X_completed_[mask] - X[mask]) ** 2))
    means = np.nanmean(Xm, axis=0)
    rmse_mean = np.sqrt(np.mean((np.broadcast_to(means, Xm.shape)[mask] - X[mask]) ** 2))
    assert rmse < 0.7 * rmse_mean


def test_nps_fills_with_nearest_prototype(blobs4d):
    Xm, _, _ = blobs4d
    nps = IncompleteFCM(n_clusters=3, strategy="nps", random_state=0).fit(Xm)
    mask = nps.missing_mask_
    nearest = partial_distances(Xm, nps.cluster_centers_).argmin(axis=1)
    filled = nps.cluster_centers_[nearest]
    np.testing.assert_allclose(nps.X_completed_[mask], filled[mask], atol=1e-6)
    # hard completion: the imputed values take only c distinct values per feature
    for j in range(Xm.shape[1]):
        assert len(np.unique(np.round(nps.X_completed_[mask[:, j], j], 6))) <= 3


def test_wds_uses_complete_rows_only_and_classifies_the_rest(blobs4d):
    Xm, y, X = blobs4d
    complete = ~np.isnan(Xm).any(axis=1)
    wds = IncompleteFCM(n_clusters=3, strategy="wds", random_state=0).fit(Xm)
    fcm = FCM(n_clusters=3, random_state=0).fit(Xm[complete])
    np.testing.assert_allclose(wds.cluster_centers_, fcm.cluster_centers_, atol=1e-8)
    np.testing.assert_allclose(wds.membership_[complete], fcm.membership_, atol=1e-6)
    assert wds.n_iter_ == fcm.n_iter_
    # the incomplete rows get memberships from partial distances
    U_inc = wds.predict_proba(Xm[~complete])
    np.testing.assert_allclose(wds.membership_[~complete], U_inc, atol=1e-9)
    assert adjusted_rand_score(y[~complete], wds.labels_[~complete]) > 0.7
    # too few complete rows
    Xfew = Xm.copy()
    Xfew[complete, 0] = np.nan
    Xfew[np.flatnonzero(complete)[:2], 0] = Xm[np.flatnonzero(complete)[:2], 0]
    with pytest.raises(ValueError):
        IncompleteFCM(n_clusters=3, strategy="wds").fit(Xfew)


def test_strategy_differences_in_completion_and_cost(blobs4d):
    """PDS/WDS impute post hoc, OCS/NPS during the run; NPS is a hard version of OCS."""
    Xm, _, X = blobs4d
    fits = {s: IncompleteFCM(n_clusters=3, strategy=s, random_state=0).fit(Xm) for s in STRATEGIES}
    mask = fits["pds"].missing_mask_
    # post-hoc OCS formula for PDS and WDS
    for s in ("pds", "wds"):
        m = fits[s]
        Um = m.membership_**2
        est = (Um @ m.cluster_centers_) / Um.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(m.X_completed_[mask], est[mask])
    # OCS imputations are fuzzy averages and lie between NPS's hard choices and the
    # feature means; they differ from the NPS values
    assert np.abs(fits["ocs"].X_completed_[mask] - fits["nps"].X_completed_[mask]).max() > 1e-3
    # the completion strategies need more iterations than PDS (they move the data)
    assert fits["ocs"].n_iter_ > fits["pds"].n_iter_


def test_no_missing_values_reduces_to_fcm(blobs4d):
    _, _, X = blobs4d
    ref = FCM(n_clusters=3, random_state=0).fit(X)
    for s in STRATEGIES:
        m = IncompleteFCM(n_clusters=3, strategy=s, random_state=0).fit(X)
        np.testing.assert_allclose(m.cluster_centers_, ref.cluster_centers_, atol=1e-8)
        np.testing.assert_allclose(m.membership_, ref.membership_, atol=1e-8)
        assert m.n_missing_ == 0
        np.testing.assert_array_equal(m.X_completed_, X)


def test_fuzziness_init_restarts_and_reproducibility(blobs4d):
    Xm, _, _ = blobs4d
    base = IncompleteFCM(n_clusters=3, random_state=0).fit(Xm)
    again = IncompleteFCM(n_clusters=3, random_state=0).fit(Xm)
    np.testing.assert_array_equal(base.labels_, again.labels_)
    np.testing.assert_allclose(base.membership_, again.membership_)
    soft = IncompleteFCM(n_clusters=3, m=3.0, random_state=0).fit(Xm)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    tight = IncompleteFCM(n_clusters=3, tol=1e-8, random_state=0).fit(Xm)
    assert tight.n_iter_ >= base.n_iter_
    few = IncompleteFCM(n_clusters=3, max_iter=2, random_state=0).fit(Xm)
    assert few.n_iter_ == 2
    multi = IncompleteFCM(n_clusters=3, strategy="ocs", n_init=4, random_state=0).fit(Xm)
    single = IncompleteFCM(n_clusters=3, strategy="ocs", n_init=1, random_state=0).fit(Xm)
    assert multi.objective_ <= single.objective_ + 1e-9
    pp = IncompleteFCM(n_clusters=3, init="kmeans++", random_state=0).fit(Xm)
    assert adjusted_rand_score(pp.labels_, base.labels_) > 0.9
    given = IncompleteFCM(n_clusters=3, init=base.cluster_centers_).fit(Xm)
    np.testing.assert_allclose(given.cluster_centers_, base.cluster_centers_, atol=1e-3)
    # starting OCS from the PDS prototypes is a recommended practice
    warm = IncompleteFCM(n_clusters=3, strategy="ocs", init=base.cluster_centers_).fit(Xm)
    assert warm.n_iter_ <= single.n_iter_


def test_predict_transform_score_on_incomplete_new_data(blobs4d):
    Xm, y, X = blobs4d
    model = IncompleteFCM(n_clusters=3, strategy="pds", random_state=0).fit(Xm)
    np.testing.assert_allclose(model.predict_proba(Xm), model.membership_, atol=1e-6)
    assert (model.predict(Xm) == model.labels_).all()
    np.testing.assert_allclose(model.transform(Xm), model.distances_, rtol=1e-6)
    assert model.score(Xm) == pytest.approx(-model.objective_, rel=1e-3)
    # new samples with other holes
    new = make_mcar(X[:20], missing_rate=0.5, random_state=5)
    U = model.predict_proba(new)
    assert U.shape == (20, 3) and np.isfinite(U).all()
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    assert (model.predict(new) == model.labels_[:20]).mean() > 0.7
    assert model.transform(new).shape == (20, 3)
    assert list(model.get_feature_names_out()) == [f"dist_cluster_{i}" for i in range(3)]
    for s in ("ocs", "nps", "wds"):
        m = IncompleteFCM(n_clusters=3, strategy=s, random_state=0).fit(Xm)
        assert m.predict_proba(new).shape == (20, 3)
        assert (m.predict(Xm) == m.labels_).mean() > 0.9


def test_pandas_input(blobs4d):
    Xm, _, _ = blobs4d
    df = pd.DataFrame(Xm, columns=list("abcd"))
    model = IncompleteFCM(n_clusters=3, random_state=0).fit(df)
    assert list(model.feature_names_in_) == list("abcd")
    ref = IncompleteFCM(n_clusters=3, random_state=0).fit(Xm)
    np.testing.assert_array_equal(model.labels_, ref.labels_)
    out = model.set_output(transform="pandas").transform(df)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == list(model.get_feature_names_out())


def test_parameter_validation(blobs4d):
    Xm, _, _ = blobs4d
    for kw in (dict(strategy="bogus"), dict(m=1.0), dict(n_init=0), dict(n_clusters=0), dict(max_iter=0)):
        with pytest.raises(ValueError):
            IncompleteFCM(**kw).fit(Xm)
    bad_row = Xm.copy()
    bad_row[0] = np.nan
    with pytest.raises(ValueError):
        IncompleteFCM().fit(bad_row)
    bad_col = Xm.copy()
    bad_col[:, 1] = np.nan
    with pytest.raises(ValueError):
        IncompleteFCM().fit(bad_col)


# ------------------------------------------------------------------ plots
def test_plot_helpers():
    X, y = make_fuzzy_blobs(n_clusters=3, random_state=0)
    Xm, mask = make_mcar(X, missing_rate=0.25, random_state=0, return_mask=True)
    model = IncompleteFCM(n_clusters=3, strategy="ocs", random_state=0).fit(Xm)
    ax = plot_incomplete_clusters(
        model.X_completed_, model.labels_, model.missing_mask_, centers=model.cluster_centers_
    )
    assert ax.get_legend() is not None
    ax = plot_incomplete_clusters(model.X_completed_, model.labels_, model.missing_mask_, legend=False)
    assert ax.get_legend() is None
    ax = plot_imputation(X, model.X_completed_, mask, feature_names=["u", "v"])
    assert ax.get_xlabel() == "true value"
    assert any("u (RMSE" in t.get_text() for t in ax.get_legend().get_texts())
    plt.close("all")
