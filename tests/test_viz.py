import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks  # noqa: E402

from fuzzyclust import FCM, GK  # noqa: E402
from fuzzyclust.datasets import load_iris, make_fuzzy_blobs  # noqa: E402
from fuzzyclust.viz import (  # noqa: E402
    FuzzySammon,
    FuzzySOM,
    Sammon,
    grid_laplacian,
    make_ellipsoids,
    make_hemisphere,
    make_spiral_3d,
    pca_projection,
    plot_map,
    plot_projection,
    plot_som_lattice,
    plot_stress,
    projection_quality,
    recomputed_membership,
    sammon_stress,
)


@pytest.fixture(scope="module")
def blobs5d():
    return make_fuzzy_blobs(n_clusters=3, n_samples_per_cluster=60, n_features=5, random_state=0)


@pytest.fixture(scope="module")
def iris_fcm():
    X = MinMaxScaler().fit_transform(load_iris().data)
    return X, FCM(n_clusters=3, random_state=0, tol=1e-6).fit(X)


# ------------------------------------------------------------------ sklearn compatibility
@parametrize_with_checks([FuzzySOM(grid_shape=(3, 1), random_state=0)])
def test_fuzzy_som_sklearn_compatible(estimator, check):
    check(estimator)


def test_sammon_sklearn_checks_fail_only_in_transform():
    # Sammon mapping has no out-of-sample extension, so ``transform`` deliberately raises
    # NotImplementedError. Every sklearn check that calls ``transform`` therefore fails with
    # exactly that exception; all other checks (fit, params, dtypes, pandas input, ...) pass.
    results = check_estimator(Sammon(max_iter=15, random_state=0), on_fail=None)
    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) < len(results) // 2
    assert all(isinstance(r["exception"], NotImplementedError) for r in failed)


# ------------------------------------------------------------------ PCA
def test_pca_projection_shapes_and_prototypes(blobs5d):
    X, _ = blobs5d
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    Y, Z, pca = pca_projection(X, 2, centers=fcm.cluster_centers_)
    assert Y.shape == (X.shape[0], 2) and Z.shape == (3, 2)
    assert pca.explained_variance_ratio_.shape == (2,)
    # prototypes are projected with the same affine map as the data
    np.testing.assert_allclose(pca.transform(fcm.cluster_centers_), Z)
    # the fuzzy means of the projected data coincide with the projected FCM centers
    Um = fcm.membership_**2
    np.testing.assert_allclose((Um.T @ Y) / Um.sum(axis=0)[:, None], Z, atol=1e-3)  # FCM tol
    Y1, Z1, _ = pca_projection(X, 2)
    assert Z1 is None
    with pytest.raises(ValueError):
        pca_projection(X, 0)
    with pytest.raises(ValueError):
        pca_projection(X, 2, centers=np.zeros((3, 4)))


# ------------------------------------------------------------------ Sammon
def test_sammon_stress_properties(blobs5d):
    X, _ = blobs5d
    assert sammon_stress(X, X) == 0.0
    assert sammon_stress(X, 2.0 * X) > 0.0  # scale is not preserved
    Y, _, _ = pca_projection(X, 2)
    E = sammon_stress(X, Y)
    assert 0.0 < E < 1.0
    with pytest.raises(ValueError):
        sammon_stress(X, Y[:-1])


def test_sammon_reduces_stress_below_pca(blobs5d):
    X, y = blobs5d
    Y_pca, _, _ = pca_projection(X, 2)
    sam = Sammon(n_components=2, max_iter=100, alpha=0.3, init="pca").fit(X)
    assert sam.embedding_.shape == (X.shape[0], 2)
    assert sam.n_iter_ == sam.stress_history_.size
    assert sam.stress_ == pytest.approx(sammon_stress(X, sam.embedding_), rel=1e-9)
    assert sam.stress_ < sammon_stress(X, Y_pca)
    # the stress decreases over the first iterations
    assert sam.stress_history_[0] > sam.stress_history_[-1]
    np.testing.assert_allclose(sam.embedding_.mean(axis=0), 0.0, atol=1e-8)
    # cluster structure survives the projection
    fcm2 = FCM(n_clusters=3, random_state=0).fit(sam.embedding_)
    assert adjusted_rand_score(y, fcm2.labels_) > 0.95


def test_sammon_reproducible_and_inits(blobs5d):
    X, _ = blobs5d
    a = Sammon(init="random", max_iter=20, random_state=3).fit_transform(X)
    b = Sammon(init="random", max_iter=20, random_state=3).fit_transform(X)
    np.testing.assert_array_equal(a, b)
    c = Sammon(init=a, max_iter=5).fit_transform(X)
    assert c.shape == a.shape
    with pytest.raises(ValueError):
        Sammon(init="bogus").fit(X)
    with pytest.raises(ValueError):
        Sammon(init=np.zeros((3, 2))).fit(X)
    with pytest.raises(ValueError):
        Sammon(n_components=0).fit(X)
    with pytest.raises(ValueError):
        Sammon(alpha=-1.0).fit(X)


def test_sammon_transform_not_implemented(blobs5d):
    X, _ = blobs5d
    sam = Sammon(max_iter=5).fit(X)
    with pytest.raises(NotImplementedError, match="fit_transform"):
        sam.transform(X)


def test_sammon_pandas_input(blobs5d):
    X, _ = blobs5d
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    sam = Sammon(max_iter=5).fit(df)
    assert list(sam.feature_names_in_) == list(df.columns)


# ------------------------------------------------------------------ quality measures
def test_recomputed_membership_and_quality(iris_fcm):
    X, fcm = iris_fcm
    Y, Z, _ = pca_projection(X, 2, fcm.cluster_centers_)
    U_star = recomputed_membership(Y, Z, m=2.0)
    assert U_star.shape == fcm.membership_.shape
    np.testing.assert_allclose(U_star.sum(axis=1), 1.0, atol=1e-9)
    q = projection_quality(fcm.membership_, Y, Z, m=2.0, X=X)
    assert set(q) == {"P", "F", "F_star", "E"}
    assert q["F"] == pytest.approx(np.sum(fcm.membership_**2) / X.shape[0])
    assert q["F_star"] == pytest.approx(np.sum(U_star**2) / X.shape[0])
    assert q["P"] == pytest.approx(np.mean(np.abs(fcm.membership_ - U_star)))
    # Table 3.4 of the MATLAB toolbox manual (PCA on iris, range-scaled, FCM c=3)
    assert q["P"] == pytest.approx(0.0203, abs=2e-3)
    assert q["F"] == pytest.approx(0.7420, abs=2e-3)
    assert q["F_star"] == pytest.approx(0.7850, abs=2e-3)
    assert q["E"] == pytest.approx(0.0117, abs=1e-3)
    # perfect representation: an embedding whose own partition is U
    q0 = projection_quality(U_star, Y, Z, m=2.0)
    assert q0["P"] == pytest.approx(0.0, abs=1e-12) and "E" not in q0
    with pytest.raises(ValueError):
        projection_quality(fcm.membership_, Y, Z[:2])


# ------------------------------------------------------------------ FUZZSAM
def test_fuzzy_sammon_beats_pca_and_sammon_on_iris(iris_fcm):
    X, fcm = iris_fcm
    Y, Z, _ = pca_projection(X, 2, fcm.cluster_centers_)
    P_pca = projection_quality(fcm.membership_, Y, Z, 2.0)["P"]
    fs = FuzzySammon(max_iter=100, alpha=0.4).fit(fcm, X)
    assert fs.embedding_.shape == (150, 2) and fs.embedded_centers_.shape == (3, 2)
    assert fs.stress_history_.size == fs.n_iter_
    assert fs.stress_history_[0] > fs.stress_
    # projected centers are the weighted means of the projected data
    Um = fcm.membership_**2
    np.testing.assert_allclose((Um.T @ fs.embedding_) / Um.sum(axis=0)[:, None], fs.embedded_centers_)
    q = fs.quality(X)
    assert q["P"] < 0.3 * P_pca
    assert abs(q["F_star"] - q["F"]) < 0.01  # Table 3.4: 0.7426 vs 0.7420
    np.testing.assert_allclose(
        fs.recomputed_membership(), recomputed_membership(fs.embedding_, fs.embedded_centers_)
    )


def test_fuzzy_sammon_uses_algorithm_distances():
    # GK clusters are ellipsoids; FUZZSAM maps the GK (Mahalanobis-type) distances, so the
    # GK partition is reproduced in the plane far better than from the PCA picture.
    X, y = make_ellipsoids(n_clusters=3, n_samples_per_cluster=80, n_features=6, random_state=0)
    gk = GK(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, gk.labels_) > 0.9
    Y, Z, _ = pca_projection(X, 2, gk.cluster_centers_)
    P_pca = projection_quality(gk.membership_, Y, Z, 2.0)["P"]
    fs = FuzzySammon(max_iter=100).fit(gk, X)
    assert fs.quality()["P"] < 0.5 * P_pca
    # the preserved distances are those of the estimator
    np.testing.assert_allclose(fs.distances_, gk.transform(X))
    # in the plane the GK clusters look like compact circular groups: recomputed hard
    # labels agree with the GK labels
    assert adjusted_rand_score(gk.labels_, fs.recomputed_membership().argmax(axis=1)) > 0.9


def test_fuzzy_sammon_input_variants(blobs5d):
    X, _ = blobs5d
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    a = FuzzySammon(max_iter=10).fit(fcm, X)
    b = FuzzySammon(max_iter=10, init="random", random_state=0).fit(fcm)  # membership_/distances_
    assert a.m_ == fcm.m
    np.testing.assert_allclose(b.distances_, fcm.distances_)
    with pytest.raises(ValueError, match="init='pca'"):
        FuzzySammon(init="pca").fit(membership=fcm.membership_, distances=fcm.distances_)
    c = FuzzySammon(init="random", max_iter=10, random_state=1).fit(
        membership=fcm.membership_, distances=fcm.distances_
    )
    d = FuzzySammon(init="random", max_iter=10, random_state=1).fit(
        membership=fcm.membership_, distances=fcm.distances_
    )
    np.testing.assert_array_equal(c.embedding_, d.embedding_)
    e = FuzzySammon(init=a.embedding_, max_iter=3).fit(fcm, X)
    assert e.embedding_.shape == a.embedding_.shape
    # Kovács–Abonyi weighting (mu instead of mu^m)
    k = FuzzySammon(weight_exponent=1.0, max_iter=10).fit(fcm, X)
    Uw = fcm.membership_
    np.testing.assert_allclose((Uw.T @ k.embedding_) / Uw.sum(axis=0)[:, None], k.embedded_centers_)
    with pytest.raises(ValueError):
        FuzzySammon().fit()
    with pytest.raises(ValueError):
        FuzzySammon().fit(membership=fcm.membership_, distances=fcm.distances_[:, :2])
    with pytest.raises(ValueError):
        FuzzySammon(m=0.5).fit(fcm, X)


def test_fuzzy_sammon_new_data(blobs5d):
    X, _ = blobs5d
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    Xnew = X[::3] + 0.05
    fs = FuzzySammon(max_iter=10).fit(fcm, Xnew)
    assert fs.embedding_.shape == (Xnew.shape[0], 2)
    np.testing.assert_allclose(fs.membership_, fcm.predict_proba(Xnew))


# ------------------------------------------------------------------ fuzzy SOM
def test_grid_laplacian():
    L, pos = grid_laplacian((3, 3), order=2)
    assert L.shape == (9, 9) and pos.shape == (9, 2)
    np.testing.assert_allclose(L, L.T)
    assert np.all(np.linalg.eigvalsh(L) > -1e-10)
    # a linear (affine) arrangement of the prototypes has zero second-order penalty
    V_lin = pos @ np.array([[1.0, 2.0, 0.5], [-0.3, 0.7, 1.1]]) + 3.0
    assert np.einsum("ij,ik,jk->", L, V_lin, V_lin) == pytest.approx(0.0, abs=1e-10)
    L1, _ = grid_laplacian((3, 3), order=1)
    # first order penalizes any non-constant map: sum of squared edge lengths = 12 unit edges
    assert np.einsum("ij,ik,jk->", L1, pos.astype(float), pos.astype(float)) == pytest.approx(12.0)
    assert np.einsum("ij,ik,jk->", L1, V_lin, V_lin) > 0
    with pytest.raises(ValueError):
        grid_laplacian((3, 3), order=3)
    with pytest.raises(ValueError):
        grid_laplacian((0, 3))


def test_fuzzy_som_zero_regularisation_is_fcm(blobs5d):
    X, y = blobs5d
    som = FuzzySOM(grid_shape=(3, 1), regularization=0.0, init="kmeans++", random_state=0).fit(X)
    fcm = FCM(n_clusters=3, init="kmeans++", random_state=0).fit(X)
    assert adjusted_rand_score(y, som.labels_) > 0.95
    np.testing.assert_allclose(
        np.sort(som.cluster_centers_, axis=0), np.sort(fcm.cluster_centers_, axis=0), atol=1e-3
    )
    assert som.n_clusters_ == 3 and som.grid_positions_.shape == (3, 2)
    assert som.membership_.shape == (X.shape[0], 3)
    assert list(som.get_feature_names_out()) == ["dist_node_0", "dist_node_1", "dist_node_2"]


def test_fuzzy_som_prototype_update_is_exact_minimiser():
    X, _ = make_hemisphere(n_samples=300, random_state=0)
    som = FuzzySOM(grid_shape=(4, 4), regularization=0.7, random_state=0, max_iter=5).fit(X)
    Um = som.membership_**2
    V = som.cluster_centers_
    L = som.regularization_matrix_
    # gradient of J w.r.t. V must vanish for the prototypes computed from the final partition
    som._update_prototypes(X, Um)
    V = som.cluster_centers_
    grad = 2 * (V * Um.sum(axis=0)[:, None] - Um.T @ X) + 2 * 0.7 * L @ V
    np.testing.assert_allclose(grad, 0.0, atol=1e-9)
    # the objective contains the penalty and decreases
    J = som.objective_history_
    assert J.size == som.n_iter_
    assert np.all(np.diff(J) <= 1e-6 * np.abs(J[:-1]) + 1e-9)


def test_fuzzy_som_regularisation_orders_the_map():
    X, _ = make_hemisphere(random_state=0)
    rho = {}
    for th in (0.0, 0.5):
        som = FuzzySOM(grid_shape=(6, 6), regularization=th, random_state=0, tol=1e-6).fit(X)
        rho[th] = spearmanr(pdist(som.grid_positions_), pdist(som.cluster_centers_)).correlation
        assert som.u_matrix().shape == (6, 6)
        assert som.component_planes().shape == (3, 6, 6)
        assert som.hits().shape == (6, 6)
        assert som.hits().sum() == pytest.approx(X.shape[0])
    assert rho[0.5] > 0.95 > rho[0.0]
    # stronger smoothing -> shorter grid edges
    a = FuzzySOM(grid_shape=(6, 6), regularization=0.5, random_state=0).fit(X).neighbor_distance()
    b = FuzzySOM(grid_shape=(6, 6), regularization=5.0, random_state=0).fit(X).neighbor_distance()
    assert b < a


def test_fuzzy_som_1d_chain_and_orders():
    X, t = make_spiral_3d(turns=0.75, random_state=0)
    som = FuzzySOM(grid_shape=(7,), regularization=2.0, random_state=0, tol=1e-6).fit(X)
    # the chain follows the arc: the curve parameter of the best-matching node is monotone
    # in the grid index (Spearman correlation close to one, up to orientation)
    tt = np.array([t[som.labels_ == i].mean() for i in range(7)])
    assert abs(spearmanr(np.arange(7), tt).correlation) > 0.95
    assert som.u_matrix().shape == (7,)
    # on a full turn (paper's first example) FCM (regularization=0) leaves the chain tangled,
    # the regularized version orders it
    X, t = make_spiral_3d(turns=1.0, random_state=0)
    rho = {}
    for th in (0.0, 2.0):
        s = FuzzySOM(grid_shape=(7,), regularization=th, random_state=0, tol=1e-6).fit(X)
        rho[th] = spearmanr(pdist(s.grid_positions_), pdist(s.cluster_centers_)).correlation
    assert rho[2.0] > rho[0.0] + 0.1


def test_fuzzy_som_first_order_penalty_contracts_the_map():
    # first-order differences penalize the edge lengths themselves, so a strong weight collapses
    # the map onto the data center; second-order differences only penalize curvature and keep
    # the extent of the map (this is the argument of Migály et al. for the second order).
    X, _ = make_hemisphere(random_state=0)
    som2 = FuzzySOM(grid_shape=(6, 6), regularization=2.0, difference_order=2, random_state=0).fit(X)
    som1 = FuzzySOM(grid_shape=(6, 6), regularization=2.0, difference_order=1, random_state=0).fit(X)
    assert np.ptp(som1.cluster_centers_, axis=0).max() < 0.1 * np.ptp(som2.cluster_centers_, axis=0).max()
    assert np.ptp(som2.cluster_centers_, axis=0).max() > 1.5


def test_fuzzy_som_predict_and_validation(blobs5d):
    X, _ = blobs5d
    som = FuzzySOM(grid_shape=(2, 2), regularization=0.3, random_state=0).fit(X)
    np.testing.assert_allclose(som.predict_proba(X), som.membership_, atol=1e-3)
    assert som.transform(X).shape == (X.shape[0], 4)
    assert (som.predict(X) == som.labels_).all()
    for kw in (dict(grid_shape=(0, 2)), dict(grid_shape=(2, 2, 2, 2)), dict(regularization=-1.0),
               dict(difference_order=3), dict(m=1.0), dict(grid_shape=(50, 50)), dict(init="bogus")):
        with pytest.raises(ValueError):
            FuzzySOM(**kw).fit(X)
    with pytest.raises(ValueError):
        FuzzySOM(grid_shape=(2, 2), init=np.zeros((3, 3))).fit(X)
    by_centers = FuzzySOM(grid_shape=(2, 2), init=som.cluster_centers_).fit(X)
    assert adjusted_rand_score(by_centers.labels_, som.labels_) == 1.0
    # more grid axes than features is allowed (jittered PCA init)
    FuzzySOM(grid_shape=(2, 2), random_state=0).fit(X[:, :1])


# ------------------------------------------------------------------ plotting
def test_plot_helpers_run(iris_fcm):
    X, fcm = iris_fcm
    Y, Z, _ = pca_projection(X, 2, fcm.cluster_centers_)
    ax = plot_projection(Y, Z, U=fcm.membership_, misclassified=fcm.labels_ != load_iris().target)
    assert ax.get_xlabel() == "$y_1$"
    plot_projection(Y, labels=load_iris().target, contours=False)
    with pytest.raises(ValueError):
        plot_projection(np.zeros((5, 3)))
    sam = Sammon(max_iter=5).fit(X)
    plot_stress([sam.stress_history_, sam.stress_history_], labels=["a", "b"])
    df = pd.DataFrame(X, columns=list("abcd"))
    som = FuzzySOM(grid_shape=(3, 3), random_state=0).fit(df)
    plot_map(som, "umatrix", annotate=True)
    plot_map(som, "component", feature="c")
    plot_map(som, "component", feature=1)
    plot_map(som, "hits", X=df)
    plot_som_lattice(som, X, dims=(0, 2), labels=load_iris().target)
    with pytest.raises(ValueError):
        plot_map(som, "component")
    with pytest.raises(ValueError):
        plot_map(som, "bogus")
    plt.close("all")
