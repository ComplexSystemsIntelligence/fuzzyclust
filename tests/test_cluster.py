import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from fuzzyclust import FCM, GG, GK, KMeans, KMedoids
from fuzzyclust.datasets import make_different_densities, make_ellipses, make_fuzzy_blobs

ALL = [KMeans, KMedoids, FCM, GK, GG]


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


@parametrize_with_checks([cls(n_clusters=2, random_state=0) for cls in ALL])
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("cls", ALL)
def test_recovers_blobs(cls, blobs):
    X, y = blobs
    est = cls(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, est.labels_) > 0.95
    assert est.membership_.shape == (X.shape[0], 3)
    np.testing.assert_allclose(est.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert est.cluster_centers_.shape == (3, 2)
    assert est.distances_.shape == (X.shape[0], 3)
    assert est.n_iter_ == len(est.objective_history_)
    assert np.isfinite(est.objective_)


@pytest.mark.parametrize("cls", [FCM, GK])
def test_objective_monotone(cls, blobs):
    X, _ = blobs
    est = cls(n_clusters=3, random_state=0).fit(X)
    J = est.objective_history_
    assert np.all(np.diff(J) <= 1e-6 * np.abs(J[:-1]) + 1e-9)


def test_predict_matches_fit(blobs):
    X, _ = blobs
    for cls in ALL:
        est = cls(n_clusters=3, random_state=0).fit(X)
        U = est.predict_proba(X)
        # for the converged partition, re-evaluation must reproduce the memberships
        np.testing.assert_allclose(U, est.membership_, atol=1e-3)
        assert (est.predict(X) == est.labels_).all()
        np.testing.assert_allclose(est.transform(X), est.distances_, rtol=1e-6)


def test_fuzziness_exponent_effect(blobs):
    X, _ = blobs
    crisp = FCM(n_clusters=3, m=1.2, random_state=0).fit(X)
    fuzzy = FCM(n_clusters=3, m=3.0, random_state=0).fit(X)
    assert crisp.membership_.max(axis=1).mean() > fuzzy.membership_.max(axis=1).mean()


def test_init_variants(blobs):
    X, _ = blobs
    base = FCM(n_clusters=3, random_state=0).fit(X)
    by_centers = FCM(n_clusters=3, init=base.cluster_centers_).fit(X)
    by_membership = FCM(n_clusters=3, init=base.membership_).fit(X)
    kpp = FCM(n_clusters=3, init="kmeans++", random_state=0).fit(X)
    for est in (by_centers, by_membership, kpp):
        assert adjusted_rand_score(base.labels_, est.labels_) == 1.0
    with pytest.raises(ValueError):
        FCM(n_clusters=3, init=np.zeros((5, 5))).fit(X)
    with pytest.raises(ValueError):
        FCM(n_clusters=3, init="bogus").fit(X)


def test_gk_beats_fcm_on_ellipses():
    X, y = make_ellipses(random_state=0)
    gk = GK(n_clusters=3, random_state=0).fit(X)
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, gk.labels_) > adjusted_rand_score(y, fcm.labels_)
    # volume constraint det(A_i) = rho_i
    for A in gk.norm_matrices_:
        assert np.linalg.det(A) == pytest.approx(1.0, rel=1e-6)
    assert gk.eigenvalues_.shape == (3, 2)


def test_gk_regularisation_and_volumes():
    X, _ = make_ellipses(random_state=1)
    gk = GK(n_clusters=3, gamma=0.5, beta=1e3, rho=[1.0, 2.0, 0.5], random_state=0).fit(X)
    for A, rho in zip(gk.norm_matrices_, [1.0, 2.0, 0.5], strict=False):
        assert np.linalg.det(A) == pytest.approx(rho, rel=1e-6)
    for F in gk.covariances_:
        w = np.linalg.eigvalsh(F)
        assert w.max() / w.min() <= 1e3 * (1 + 1e-6)
    with pytest.raises(ValueError):
        GK(n_clusters=3, gamma=2.0).fit(X)


def test_gg_handles_densities():
    X, y = make_different_densities(random_state=0)
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    gg = GG(n_clusters=3, init=fcm.membership_).fit(X)
    assert adjusted_rand_score(y, gg.labels_) > adjusted_rand_score(y, fcm.labels_)
    assert gg.priors_.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(gg.predict_proba(X * 50.0)))  # far-away points: no overflow


def test_gg_fmle_m1(blobs):
    X, y = blobs
    gg = GG(n_clusters=3, m=1.0, init="kmeans++", random_state=0).fit(X)
    assert adjusted_rand_score(y, gg.labels_) > 0.9


def test_kmedoids_prototypes_are_data_points(blobs):
    X, _ = blobs
    for rule in ("exact", "nearest_mean"):
        km = KMedoids(n_clusters=3, medoid=rule, random_state=0).fit(X)
        np.testing.assert_array_equal(km.cluster_centers_, X[km.medoid_indices_])
    with pytest.raises(ValueError):
        KMedoids(n_clusters=3, medoid="bogus").fit(X)


def test_fcm_norms(blobs):
    X, _ = blobs
    Xs = X * np.array([1.0, 100.0])
    e = FCM(n_clusters=3, norm="euclidean", random_state=0).fit(Xs)
    d = FCM(n_clusters=3, norm="diagonal", random_state=0).fit(Xs)
    mh = FCM(n_clusters=3, norm="mahalanobis", random_state=0).fit(Xs)
    custom = FCM(n_clusters=3, norm=np.eye(2), random_state=0).fit(Xs)
    np.testing.assert_allclose(e.membership_, custom.membership_)
    assert d.norm_matrix_[0, 0] > d.norm_matrix_[1, 1]
    assert mh.norm_matrix_.shape == (2, 2)
    with pytest.raises(ValueError):
        FCM(n_clusters=3, norm="bogus").fit(X)
    with pytest.raises(ValueError):
        FCM(n_clusters=3, norm=np.eye(3)).fit(X)


def test_parameter_validation(blobs):
    X, _ = blobs
    with pytest.raises(ValueError):
        FCM(n_clusters=0).fit(X)
    with pytest.raises(ValueError):
        FCM(n_clusters=3, m=1.0).fit(X)
    with pytest.raises(ValueError):
        FCM(n_clusters=10_000).fit(X)
    with pytest.raises(ValueError):
        FCM(n_clusters=3, max_iter=0).fit(X)


def test_pandas_and_pipeline(blobs):
    X, y = blobs
    df = pd.DataFrame(X, columns=["temperature", "pressure"])
    pipe = make_pipeline(StandardScaler(), FCM(n_clusters=3, random_state=0)).fit(df)
    fcm = pipe[-1]
    assert fcm.n_features_in_ == 2
    direct = FCM(n_clusters=3, random_state=0).set_output(transform="pandas").fit(df)
    out = direct.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["dist_cluster_0", "dist_cluster_1", "dist_cluster_2"]
    assert list(direct.feature_names_in_) == ["temperature", "pressure"]
    assert adjusted_rand_score(y, pipe.predict(df)) > 0.95


def test_score_is_negative_objective(blobs):
    X, _ = blobs
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    assert fcm.score(X) == pytest.approx(-fcm.objective_, rel=1e-3)


def test_verbose_prints(blobs, capsys):
    X, _ = blobs
    FCM(n_clusters=3, random_state=0, verbose=1).fit(X)
    assert "J =" in capsys.readouterr().out
