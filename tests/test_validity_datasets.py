import numpy as np
import pandas as pd
import pytest

from fuzzyclust import FCM, KMeans, plotting
from fuzzyclust import datasets as ds
from fuzzyclust.preprocessing import lagged_matrix, make_regressors, normalize
from fuzzyclust.validity import (
    INDEX_DIRECTION,
    alternative_dunn_index,
    classification_entropy,
    dunn_index,
    modified_partition_coefficient,
    partition_coefficient,
    scan_n_clusters,
    suggest_n_clusters,
    validity_indices,
    xie_beni,
)


@pytest.fixture(scope="module")
def fitted():
    X, y = ds.make_fuzzy_blobs(random_state=0)
    return X, y, FCM(n_clusters=3, random_state=0).fit(X)


def test_partition_indices_bounds(fitted):
    X, _, fcm = fitted
    U = fcm.membership_
    pc = partition_coefficient(X, U)
    assert 1 / 3 <= pc <= 1.0
    assert 0.0 <= classification_entropy(X, U) <= np.log(3)
    assert 0.0 <= modified_partition_coefficient(X, U) <= 1.0
    hard = np.eye(3)[fcm.labels_]
    assert partition_coefficient(X, hard) == pytest.approx(1.0)
    assert classification_entropy(X, hard) == pytest.approx(0.0, abs=1e-8)


def test_all_indices_finite(fitted):
    X, _, fcm = fitted
    res = validity_indices(X, fcm.membership_, fcm.cluster_centers_, m=2.0)
    assert set(res) == set(INDEX_DIRECTION)
    assert all(np.isfinite(v) for v in res.values())
    assert dunn_index(X, fcm.membership_) > 0
    assert alternative_dunn_index(X, fcm.membership_, fcm.cluster_centers_) >= 0
    assert xie_beni(X, fcm.membership_, fcm.cluster_centers_) > 0


def test_scan_and_suggest(fitted):
    X, _, _ = fitted
    table = scan_n_clusters(FCM(random_state=0), X, range(2, 6), indices=["PC", "CE", "XB", "SIL"])
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == [2, 3, 4, 5]
    sugg = suggest_n_clusters(table)
    assert sugg["XB"] == 3 and sugg["SIL"] == 3
    km_table = scan_n_clusters(KMeans(random_state=0), X, [2, 3], indices=["SIL"])
    assert km_table.shape == (2, 1)


def test_bad_membership(fitted):
    X, _, _ = fitted
    with pytest.raises(ValueError):
        partition_coefficient(X, np.ones((X.shape[0], 3)) * 2)
    with pytest.raises(ValueError):
        xie_beni(X, np.ones((5, 3)) / 3, np.zeros((3, 2)))


@pytest.mark.parametrize(
    "maker",
    [
        ds.make_fuzzy_blobs,
        ds.make_ellipses,
        ds.make_different_densities,
        ds.make_rings,
        ds.make_lines,
        ds.make_noisy_blobs,
        ds.make_curved_manifold,
        ds.make_two_moons,
    ],
)
def test_generators_reproducible(maker):
    X1, y1 = maker(random_state=0)
    X2, y2 = maker(random_state=0)
    np.testing.assert_array_equal(X1, X2)
    assert X1.shape[0] == y1.shape[0] and X1.ndim == 2
    X3, _ = maker(random_state=1)
    assert not np.array_equal(X1, X3)


def test_loaders():
    mc = ds.load_motorcycle()
    assert mc.data.shape == (133, 2) and mc.strain.shape == (133,)
    assert ds.load_iris().data.shape == (150, 4)
    assert ds.load_wine().data.shape[1] == 13


def test_normalize_roundtrip():
    X = np.random.RandomState(0).rand(20, 3) * 10
    for method in ("var", "range"):
        Xs, scaler = normalize(X, method)
        np.testing.assert_allclose(scaler.inverse_transform(Xs), X)
    df = pd.DataFrame(X, columns=list("abc"))
    Xs, _ = normalize(df, "range")
    assert isinstance(Xs, pd.DataFrame) and list(Xs.columns) == list("abc")
    with pytest.raises(ValueError):
        normalize(X, "bogus")


def test_regressors():
    y = np.arange(10.0)
    u = np.arange(10.0) * 10
    Phi, target, names = make_regressors(y, u, n_a=2, n_b=2, n_k=1)
    assert names == ["y(k-1)", "y(k-2)", "u(k-1)", "u(k-2)"]
    assert Phi.shape == (8, 4) and target.shape == (8,)
    np.testing.assert_array_equal(Phi[0], [1.0, 0.0, 10.0, 0.0])
    assert target[0] == 2.0
    Phi2, t2, n2 = make_regressors(np.c_[y, -y], None, n_a=1)
    assert n2 == ["y1(k-1)", "y2(k-1)"] and t2.shape == (9, 2)
    L = lagged_matrix(y, 3)
    assert L.shape == (7, 3) and L[0].tolist() == [2.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        make_regressors(y, None, n_a=0)


def test_plotting_smoke(fitted):
    import matplotlib

    matplotlib.use("Agg")
    X, _, fcm = fitted
    ax = plotting.plot_clusters(X, fcm.labels_, fcm.cluster_centers_)
    assert ax is not None
    plotting.plot_membership_contours(fcm, X, resolution=30)
    plotting.plot_covariance_ellipses(fcm.cluster_centers_, fcm.covariances_)
    table = scan_n_clusters(FCM(random_state=0), X, [2, 3], indices=["PC", "XB"])
    plotting.plot_validity_scan(table)
    plotting.plot_objective([fcm], ["FCM"])
    plotting.plot_membership_profiles(fcm.membership_[:50], stacked=True)
    plotting.plot_clusters(X, np.where(fcm.labels_ == 0, -1, fcm.labels_))
    matplotlib.pyplot.close("all")
