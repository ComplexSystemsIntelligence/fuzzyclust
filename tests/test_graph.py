import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.cluster import FCM, GG  # noqa: E402
from fuzzyclust.datasets import (  # noqa: E402
    make_curved_manifold,
    make_fuzzy_blobs,
    make_rings,
    make_two_moons,
)
from fuzzyclust.graph import (  # noqa: E402
    GeodesicFCM,
    IsomapFCM,
    MSTClustering,
    fuzzy_cluster_similarity,
    geodesic_distances,
    isomap_embedding,
    make_chained_blobs,
    make_density_blobs,
    make_s_curve,
    minimum_spanning_tree,
    neighbors_graph,
    plot_geodesic_path,
    plot_mst,
    plot_neighbors_graph,
    plot_similarity_dendrogram,
)

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


@pytest.fixture(scope="module")
def chained():
    return make_chained_blobs(random_state=0)


@pytest.fixture(scope="module")
def density():
    return make_density_blobs(random_state=0)


@pytest.fixture(scope="module")
def rings():
    return make_rings(random_state=0)


@pytest.fixture(scope="module")
def spiral():
    return make_curved_manifold(n_samples=400, random_state=0)


# ------------------------------------------------------------------ sklearn compatibility
@parametrize_with_checks(
    [
        MSTClustering(n_clusters=2, random_state=0),
        MSTClustering(random_state=0),
        GeodesicFCM(n_clusters=2, random_state=0),
        IsomapFCM(n_clusters=2, random_state=0),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ data generators
def test_generators_shapes():
    X, y = make_chained_blobs(n_samples_per_blob=50, n_chain=10, random_state=0)
    assert X.shape == (110, 2) and np.array_equal(np.unique(y), [0, 1, 2])
    assert (y == 2).sum() == 10
    X, y = make_density_blobs(sizes=(60, 20, 20), random_state=0)
    assert X.shape == (100, 2) and np.bincount(y).tolist() == [60, 20, 20]
    # the diffuse group really is more spread out than the tight ones
    assert X[y == 0].std(axis=0).mean() > 3 * X[y == 1].std(axis=0).mean()
    X, t = make_s_curve(n_samples=120, random_state=0)
    assert X.shape == (120, 3) and t.shape == (120,)
    np.testing.assert_allclose(make_density_blobs(random_state=3)[0], make_density_blobs(random_state=3)[0])
    np.testing.assert_allclose(make_chained_blobs(random_state=3)[0], make_chained_blobs(random_state=3)[0])


# ------------------------------------------------------------------ tree utilities
def test_minimum_spanning_tree_properties(blobs):
    X, _ = blobs
    edges, weights = minimum_spanning_tree(X)
    N = X.shape[0]
    assert edges.shape == (N - 1, 2) and weights.shape == (N - 1,)
    assert np.all(np.diff(weights) >= 0)  # sorted by length
    np.testing.assert_allclose(weights, np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1))
    # a spanning tree: one connected component
    from scipy.sparse import coo_matrix

    A = coo_matrix((np.ones(N - 1), (edges[:, 0], edges[:, 1])), shape=(N, N))
    assert connected_components(A, directed=False)[0] == 1


def test_minimum_spanning_tree_backends_and_edge_cases(blobs):
    X, _ = blobs
    _, w_scipy = minimum_spanning_tree(X, backend="scipy")
    nx = pytest.importorskip("networkx")
    assert nx is not None
    _, w_nx = minimum_spanning_tree(X, backend="networkx")
    np.testing.assert_allclose(w_scipy.sum(), w_nx.sum())
    with pytest.raises(ValueError):
        minimum_spanning_tree(X, backend="bogus")
    # duplicated points are connected by a zero-length edge, not dropped
    Xd = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    edges, weights = minimum_spanning_tree(Xd)
    assert edges.shape == (2, 2) and weights.min() == pytest.approx(0.0, abs=1e-9)
    edges, weights = minimum_spanning_tree(Xd[:1])
    assert edges.shape == (0, 2) and weights.shape == (0,)


def test_fuzzy_cluster_similarity_properties(blobs):
    X, _ = blobs
    U = FCM(n_clusters=3, random_state=0).fit(X).membership_
    S = fuzzy_cluster_similarity(U)
    assert S.shape == (3, 3)
    np.testing.assert_allclose(np.diag(S), 1.0)
    np.testing.assert_allclose(S, S.T)
    assert np.all(S >= 0) and np.all(S <= 1 + 1e-12)
    # identical clusters are fully similar, disjoint crisp clusters not at all
    U2 = np.c_[U[:, 0], U[:, 0]]
    assert fuzzy_cluster_similarity(U2)[0, 1] == pytest.approx(1.0)
    crisp = np.eye(2)[np.repeat([0, 1], 10)]
    assert fuzzy_cluster_similarity(crisp)[0, 1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        fuzzy_cluster_similarity(np.ones(5))


# ------------------------------------------------------------------ MST clustering
def test_mst_recovers_blobs_with_gg_refinement(blobs):
    X, y = blobs
    mst = MSTClustering(random_state=0).fit(X)
    assert mst.n_clusters_ == 3
    assert adjusted_rand_score(y, mst.labels_) > 0.95
    assert adjusted_rand_score(y, mst.tree_labels_) > 0.95
    assert isinstance(mst.refiner_, GG)
    assert mst.membership_.shape == (X.shape[0], 3)
    np.testing.assert_allclose(mst.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert mst.cluster_centers_.shape == (3, 2) and mst.covariances_.shape == (3, 2, 2)
    assert mst.mst_edges_.shape == (X.shape[0] - 1, 2)
    assert mst.cut_edges_.shape == (2, 2) and mst.cut_phases_.shape == (2,)
    assert mst.cut_quality_.shape == (2,) and np.all(mst.cut_quality_ >= 0) and np.all(mst.cut_quality_ <= 2)
    assert mst.cut_quality_.min() > 1.5  # the refinement confirms both separations
    assert mst.hypervolumes_.shape == (3,) and np.all(mst.hypervolumes_ > 0)
    assert mst.cluster_similarity_.shape == (3, 3)
    assert mst.n_iter_ == len(mst.objective_history_)
    # the GG refinement started from the tree partition keeps it
    assert adjusted_rand_score(mst.tree_labels_, mst.labels_) > 0.95


def test_mst_volume_criterion_breaks_the_chain(chained):
    """Zahn's length criteria see one cluster through the chain; the hypervolume criterion splits it."""
    X, y = chained
    blob = y < 2
    # no phase-0 criterion, splitting disabled -> the whole data set is one tree cluster
    one = MSTClustering(threshold=None, inconsistency=None, min_gain=1.0, refine=None).fit(X)
    assert one.n_clusters_ == 1 and one.cut_edges_.shape == (0, 2)
    assert one.cut_quality_.shape == (0,)
    # volume-based splitting alone separates the blobs from each other and from the chain
    split = MSTClustering(threshold=None, inconsistency=None, random_state=0).fit(X)
    assert split.n_clusters_ >= 3
    assert np.all(split.cut_phases_ == 1)
    assert adjusted_rand_score(y[blob], split.labels_[blob]) > 0.95
    # the two blobs end up in different clusters
    assert split.labels_[y == 0].min() != split.labels_[y == 1].min()
    # a larger required gain is more conservative
    conservative = MSTClustering(threshold=None, inconsistency=None, min_gain=0.6, random_state=0).fit(X)
    assert conservative.n_clusters_ <= split.n_clusters_


def test_mst_local_criterion_handles_different_densities(density):
    """A global length threshold cannot separate groups of different density, the local one can."""
    X, y = density
    global_only = MSTClustering(threshold=2.0, inconsistency=None, min_gain=1.0, refine=None).fit(X)
    local = MSTClustering(inconsistency=2.5, n_hops=3, min_gain=1.0, refine=None).fit(X)
    assert adjusted_rand_score(y, local.tree_labels_) > 0.99
    assert local.n_clusters_ == 3
    assert adjusted_rand_score(y, global_only.tree_labels_) < adjusted_rand_score(y, local.tree_labels_)
    # the standard-deviation form of Zahn's criterion with a deep neighborhood
    std_form = MSTClustering(
        inconsistency=None, inconsistency_std=3.0, n_hops=4, min_gain=1.0, refine=None
    ).fit(X)
    assert std_form.n_clusters_ == 3 and adjusted_rand_score(y, std_form.tree_labels_) > 0.99
    # a stricter factor / deeper neighborhood never produces more coarse clusters
    pairs = ((dict(inconsistency=3.0), dict(inconsistency=2.0)), (dict(n_hops=4), dict(n_hops=1)))
    for strict, loose in pairs:
        a = MSTClustering(min_gain=1.0, refine=None, **strict).fit(X)
        b = MSTClustering(min_gain=1.0, refine=None, **loose).fit(X)
        assert a.n_clusters_ <= b.n_clusters_


def test_mst_min_cluster_size_and_max_clusters(density, rings):
    X, _ = density
    small = MSTClustering(min_cluster_size=2, refine=None).fit(X)
    large = MSTClustering(min_cluster_size=15, refine=None).fit(X)
    assert large.n_clusters_ <= small.n_clusters_
    assert np.bincount(large.tree_labels_).min() >= 15
    Xr, _ = rings
    capped = MSTClustering(max_clusters=4, refine=None).fit(Xr)
    assert capped.n_clusters_ <= 4


def test_mst_given_number_of_clusters(rings, blobs):
    X, y = rings
    # the tree partition alone separates the rings perfectly when two clusters are requested
    pure = MSTClustering(n_clusters=2, refine=None).fit(X)
    assert pure.n_clusters_ == 2 and pure.refiner_ is None
    assert adjusted_rand_score(y, pure.labels_) == 1.0
    assert np.all((pure.membership_ == 0) | (pure.membership_ == 1))
    assert pure.n_iter_ == 0
    # Gaussian prototypes cannot describe a ring: the GG refinement of two clusters fails
    gg = MSTClustering(n_clusters=2, refine="gg", random_state=0).fit(X)
    assert adjusted_rand_score(y, gg.tree_labels_) == 1.0
    assert adjusted_rand_score(y, gg.labels_) < 0.5
    # forcing more clusters than the criteria find, and fewer
    Xb, yb = blobs
    for c in (2, 3, 5):
        m = MSTClustering(n_clusters=c, random_state=0).fit(Xb)
        assert m.n_clusters_ == c and len(np.unique(m.tree_labels_)) == c
        assert m.membership_.shape == (Xb.shape[0], c)
    assert adjusted_rand_score(yb, MSTClustering(n_clusters=3, random_state=0).fit(Xb).labels_) > 0.95


def test_mst_refinement_variants(chained):
    X, y = chained
    gg = MSTClustering(random_state=0).fit(X)
    fcm = MSTClustering(refine="fcm", random_state=0).fit(X)
    none = MSTClustering(refine=None, random_state=0).fit(X)
    assert isinstance(gg.refiner_, GG) and isinstance(fcm.refiner_, FCM) and none.refiner_ is None
    np.testing.assert_array_equal(gg.tree_labels_, fcm.tree_labels_)
    np.testing.assert_array_equal(gg.tree_labels_, none.tree_labels_)
    # GG keeps the tree partition (it only polishes it), FCM is pulled away by the chain
    assert adjusted_rand_score(gg.tree_labels_, gg.labels_) > 0.9
    kept_gg = adjusted_rand_score(gg.tree_labels_, gg.labels_)
    assert adjusted_rand_score(fcm.tree_labels_, fcm.labels_) < kept_gg
    # the crisp variant: covariances of the tree clusters, FCM-type memberships for new data
    assert none.covariances_.shape == (none.n_clusters_, 2, 2)
    U = none.predict_proba(X)
    assert U.shape == (X.shape[0], none.n_clusters_)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    # Fukuyama-Sugeno as splitting criterion
    fs = MSTClustering(criterion="fukuyama_sugeno", random_state=0).fit(X)
    assert fs.n_clusters_ >= 2
    # fuzziness and covariance regularization are forwarded to the refinement
    soft = MSTClustering(m=3.0, random_state=0).fit(X)
    assert soft.membership_.max(axis=1).mean() < gg.membership_.max(axis=1).mean()
    reg = MSTClustering(gamma=0.5, random_state=0).fit(X)
    assert reg.refiner_.gamma == 0.5


def test_mst_predict_transform_and_pandas(blobs):
    X, y = blobs
    mst = MSTClustering(random_state=0).fit(X)
    assert (mst.predict(X) == mst.labels_).all()
    np.testing.assert_allclose(mst.predict_proba(X), mst.membership_, atol=1e-6)
    np.testing.assert_allclose(mst.transform(X), mst.distances_, rtol=1e-6)
    assert mst.score(X) == pytest.approx(-mst.objective_, rel=1e-3)
    new = X[:10] + 0.01
    assert mst.predict(new).shape == (10,) and mst.predict_proba(new).shape == (10, 3)
    assert (mst.predict(new) == mst.labels_[:10]).all()
    assert list(mst.get_feature_names_out()) == ["dist_cluster_0", "dist_cluster_1", "dist_cluster_2"]
    df = pd.DataFrame(X, columns=["a", "b"])
    pm = MSTClustering(random_state=0).fit(df)
    assert list(pm.feature_names_in_) == ["a", "b"]
    np.testing.assert_array_equal(pm.labels_, mst.labels_)
    out = pm.set_output(transform="pandas").transform(df)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == list(pm.get_feature_names_out())


def test_mst_reproducible_and_networkx_backend(blobs):
    X, _ = blobs
    a = MSTClustering(random_state=0).fit(X)
    b = MSTClustering(random_state=0).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_allclose(a.membership_, b.membership_)
    pytest.importorskip("networkx")
    nx_model = MSTClustering(backend="networkx", random_state=0).fit(X)
    np.testing.assert_allclose(np.sort(nx_model.mst_weights_), np.sort(a.mst_weights_))
    assert adjusted_rand_score(nx_model.labels_, a.labels_) == 1.0


def test_mst_parameter_validation(blobs):
    X, _ = blobs
    bad = [
        dict(n_clusters=0),
        dict(n_clusters=1000),
        dict(threshold=-1.0),
        dict(inconsistency=0.0),
        dict(inconsistency_std=-2.0),
        dict(n_hops=0),
        dict(criterion="bogus"),
        dict(min_gain=-0.1),
        dict(min_cluster_size=0),
        dict(max_clusters=0),
        dict(refine="bogus"),
        dict(m=1.0),
        dict(backend="bogus"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            MSTClustering(**kw).fit(X)


# ------------------------------------------------------------------ graph utilities
def test_neighbors_graph_properties(blobs):
    X, _ = blobs
    G = neighbors_graph(X, n_neighbors=5)
    N = X.shape[0]
    assert G.shape == (N, N)
    assert abs(G - G.T).max() == 0  # symmetric
    assert G.diagonal().sum() == 0  # no self loops
    degree = np.diff(G.indptr)
    assert degree.min() >= 5  # every sample lists its 5 neighbors (plus those listing it)
    # radius graph
    Gr = neighbors_graph(X, radius=0.5)
    assert Gr.data.max() <= 0.5
    # coincident samples stay connected through a tiny edge
    Xd = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
    Gd = neighbors_graph(Xd, n_neighbors=1)
    assert Gd[0, 1] > 0 and Gd[2, 3] > 0
    with pytest.raises(ValueError):
        neighbors_graph(X, n_neighbors=0)
    with pytest.raises(ValueError):
        neighbors_graph(X, radius=-1.0)


def test_geodesic_distances_on_a_curve_and_components(rings):
    # on a smooth curve the geodesic is the arc length, longer than the chord
    t = np.linspace(0, np.pi, 100)
    arc = np.c_[np.cos(t), np.sin(t)]
    geo = geodesic_distances(arc, n_neighbors=2)
    D = geo.distances
    assert geo.n_components == 1 and geo.bridges.shape == (0, 2)
    np.testing.assert_allclose(np.diag(D), 0.0)
    assert D[0, -1] == pytest.approx(np.pi, rel=0.01)  # arc length of the half circle
    assert D[0, -1] > np.linalg.norm(arc[0] - arc[-1])  # geodesic >= Euclidean
    assert np.all(D >= np.linalg.norm(arc[:, None] - arc[None], axis=2) - 1e-9)
    # the rings fall into two components; bridging keeps them far apart
    X, y = rings
    raw = geodesic_distances(X, n_neighbors=6, connect_components=False)
    assert raw.n_components == 2
    assert np.isinf(raw.distances[y == 0][:, y == 1]).all()
    assert np.isfinite(raw.distances[y == 0][:, y == 0]).all()
    assert np.array_equal(raw.component_labels == raw.component_labels[0], y == y[0])
    con = geodesic_distances(X, n_neighbors=6, connect_components=True)
    assert np.isfinite(con.distances).all() and con.bridges.shape == (1, 2)
    intra = con.distances[y == 0][:, y == 0].max()
    assert con.distances[y == 0][:, y == 1].min() > intra  # penalized bridge
    plain = geodesic_distances(X, n_neighbors=6, bridge_penalty=0.0)
    assert plain.distances[y == 0][:, y == 1].min() < intra
    with pytest.raises(ValueError):
        geodesic_distances(X, bridge_penalty=-1.0)


def test_isomap_embedding_recovers_a_bent_line():
    # a curve in 3-D with a 1-D manifold: the first Isomap coordinate is the arc length
    t = np.linspace(0, 3 * np.pi, 200)
    X = np.c_[np.cos(t), np.sin(t), 0.3 * t]
    D = geodesic_distances(X, n_neighbors=3).distances
    Y, w = isomap_embedding(D, n_components=2)
    assert Y.shape == (200, 2) and w.shape == (200,)
    assert np.all(np.diff(w) <= 1e-9)  # decreasing eigenvalues
    arc = np.sqrt(1 + 0.3**2) * t
    r = abs(np.corrcoef(Y[:, 0], arc)[0, 1])
    assert r > 0.999
    assert w[0] > 50 * abs(w[1])  # essentially one-dimensional
    with pytest.raises(ValueError):
        isomap_embedding(np.array([[0.0, np.inf], [np.inf, 0.0]]))


# ------------------------------------------------------------------ GeodesicFCM
def test_geodesic_fcm_beats_fcm_on_manifolds(spiral, rings):
    X, y = spiral
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    geo = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, fcm.labels_) < 0.2
    assert adjusted_rand_score(y, geo.labels_) > 0.95
    assert geo.n_components_ == 1
    # medoids are data points
    assert np.all(np.isin(geo.medoid_indices_, np.arange(X.shape[0])))
    np.testing.assert_array_equal(geo.cluster_centers_, X[geo.medoid_indices_])
    assert geo.geodesic_distances_.shape == (X.shape[0], X.shape[0])
    assert geo.membership_.shape == (X.shape[0], 2)
    np.testing.assert_allclose(geo.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert geo.n_iter_ == len(geo.objective_history_) and np.isfinite(geo.objective_)
    # rings: two graph components, one medoid each
    Xr, yr = rings
    georings = GeodesicFCM(n_clusters=2, n_neighbors=6, random_state=0).fit(Xr)
    assert georings.n_components_ == 2 and georings.bridge_edges_.shape == (1, 2)
    assert adjusted_rand_score(yr, georings.labels_) == 1.0
    assert adjusted_rand_score(yr, FCM(n_clusters=2, random_state=0).fit(Xr).labels_) < 0.1


def test_geodesic_fcm_number_of_neighbors_effect(spiral):
    """Few neighbors fragment the arms into components, many short-circuit them."""
    X, y = spiral
    comps = [
        GeodesicFCM(n_clusters=2, n_neighbors=k, random_state=0).fit(X).n_components_ for k in (3, 6, 10)
    ]
    assert comps[0] > comps[1] > comps[2] == 1
    # two interleaved moons: a short-circuit through the gap spoils the clustering
    Xm, ym = make_two_moons(noise=0.04, random_state=0)
    good = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(Xm)
    assert good.n_components_ == 2 and adjusted_rand_score(ym, good.labels_) == 1.0
    Xm2, ym2 = make_two_moons(noise=0.08, random_state=0)
    short = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(Xm2)
    assert short.n_components_ == 1 and adjusted_rand_score(ym2, short.labels_) < 0.7


def test_geodesic_fcm_components_and_bridges(rings):
    X, y = rings
    # without bridging the distances across components are infinite and the
    # memberships of the other component vanish
    nocon = GeodesicFCM(n_clusters=2, n_neighbors=6, connect_components=False, random_state=0).fit(X)
    assert np.isinf(nocon.geodesic_distances_).any()
    assert nocon.bridge_edges_.shape == (0, 2)
    assert adjusted_rand_score(y, nocon.labels_) == 1.0
    assert nocon.membership_.max(axis=1).min() > 0.99
    # a plain Euclidean bridge lets the far side of the outer ring switch to the inner medoid
    plain = GeodesicFCM(n_clusters=2, n_neighbors=6, bridge_penalty=0.0, random_state=0).fit(X)
    assert adjusted_rand_score(y, plain.labels_) < 0.9
    # the radius graph gives the same picture as the k-graph
    rad = GeodesicFCM(n_clusters=2, radius=0.6, random_state=0).fit(X)
    assert rad.n_components_ == 2 and adjusted_rand_score(y, rad.labels_) == 1.0


def test_geodesic_fcm_init_fuzziness_restarts(spiral):
    X, y = spiral
    base = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(X)
    rnd = GeodesicFCM(n_clusters=2, n_neighbors=10, init="random", n_init=3, random_state=0).fit(X)
    given = GeodesicFCM(n_clusters=2, n_neighbors=10, init=base.medoid_indices_).fit(X)
    for est in (rnd, given):
        assert adjusted_rand_score(base.labels_, est.labels_) > 0.95
    np.testing.assert_array_equal(given.medoid_indices_, base.medoid_indices_)
    # restarts keep the lowest objective
    single = GeodesicFCM(n_clusters=2, n_neighbors=10, init="random", random_state=0).fit(X)
    assert rnd.objective_ <= single.objective_ + 1e-9
    # larger m -> fuzzier partition
    soft = GeodesicFCM(n_clusters=2, n_neighbors=10, m=3.0, random_state=0).fit(X)
    assert soft.membership_.max(axis=1).mean() < base.membership_.max(axis=1).mean()
    # objective decreases
    J = base.objective_history_
    assert J[-1] <= J[0]
    with pytest.raises(ValueError):
        GeodesicFCM(n_clusters=2, init="bogus").fit(X)
    with pytest.raises(ValueError):
        GeodesicFCM(n_clusters=2, init=np.array([0.5, 1.5])).fit(X)
    with pytest.raises(ValueError):
        GeodesicFCM(n_clusters=2, init=np.array([0, 10_000])).fit(X)


def test_geodesic_fcm_predict_new_data_and_pandas(spiral):
    X, y = spiral
    geo = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(X)
    # training points are reproduced exactly through the nearest-sample proxy
    np.testing.assert_allclose(geo.predict_proba(X.copy()), geo.membership_, atol=1e-9)
    assert (geo.predict(X) == geo.labels_).all()
    np.testing.assert_allclose(geo.transform(X), geo.distances_, rtol=1e-6)
    # points slightly off the manifold inherit the label of their nearest sample
    rng = np.random.RandomState(0)
    new = X[:20] + rng.randn(20, 2) * 1e-3
    assert (geo.predict(new) == geo.labels_[:20]).all()
    assert geo.score(X) == pytest.approx(-geo.objective_, rel=1e-6)
    df = pd.DataFrame(X, columns=["u", "v"])
    pg = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(df)
    assert list(pg.feature_names_in_) == ["u", "v"]
    np.testing.assert_array_equal(pg.labels_, geo.labels_)
    assert isinstance(pg.set_output(transform="pandas").transform(df), pd.DataFrame)


def test_geodesic_fcm_parameter_validation(blobs):
    X, _ = blobs
    bad = (dict(n_neighbors=0), dict(radius=-1.0), dict(m=1.0), dict(bridge_penalty=-1.0), dict(n_init=0))
    for kw in bad:
        with pytest.raises(ValueError):
            GeodesicFCM(n_clusters=2, **kw).fit(X)


# ------------------------------------------------------------------ IsomapFCM
def test_isomap_fcm_on_manifolds(spiral, rings):
    X, y = spiral
    iso = IsomapFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, iso.labels_) > 0.95
    assert iso.embedding_.shape == (X.shape[0], 2) and iso.embedded_centers_.shape == (2, 2)
    assert 0.0 <= iso.residual_variance_ <= 1.0
    np.testing.assert_array_equal(iso.cluster_centers_, X[iso.medoid_indices_])
    assert np.isfinite(iso.geodesic_objective_) and iso.geodesic_objective_ > 0
    np.testing.assert_allclose(iso.membership_.sum(axis=1), 1.0, atol=1e-9)
    # predictions through the nearest-sample proxy
    np.testing.assert_allclose(iso.predict_proba(X.copy()), iso.membership_, atol=1e-9)
    assert (iso.predict(X) == iso.labels_).all()
    assert iso.transform(X[:5]).shape == (5, 2)
    # one-dimensional embedding of the two bridged rings separates them as well
    Xr, yr = rings
    iso1 = IsomapFCM(n_clusters=2, n_components=1, n_neighbors=6, random_state=0).fit(Xr)
    assert iso1.embedding_.shape == (Xr.shape[0], 1)
    assert adjusted_rand_score(yr, iso1.labels_) == 1.0
    # the geodesic objective of Algorithm II is not worse than that of Algorithm I (as in the paper)
    geo = GeodesicFCM(n_clusters=2, n_neighbors=10, random_state=0).fit(X)
    assert geo.objective_ <= iso.geodesic_objective_ * 1.05
    # a fragmented graph (k too small): plain bridges re-join the fragments of an arm,
    # penalized bridges keep them apart and the two clusters cannot collect them
    frag_plain = IsomapFCM(n_clusters=2, n_neighbors=4, bridge_penalty=0.0, random_state=0).fit(X)
    frag_pen = IsomapFCM(n_clusters=2, n_neighbors=4, bridge_penalty=1.0, random_state=0).fit(X)
    assert frag_plain.n_components_ > 2 and frag_plain.bridge_edges_.shape[0] == frag_plain.n_components_ - 1
    assert adjusted_rand_score(y, frag_plain.labels_) > adjusted_rand_score(y, frag_pen.labels_)
    # more embedding dimensions keep more of the geodesic geometry
    iso3 = IsomapFCM(n_clusters=2, n_components=3, n_neighbors=10, random_state=0).fit(X)
    assert iso3.residual_variance_ <= iso.residual_variance_ + 1e-12
    df = pd.DataFrame(X, columns=["u", "v"])
    assert list(IsomapFCM(n_clusters=2, random_state=0).fit(df).feature_names_in_) == ["u", "v"]
    for kw in (dict(n_components=0), dict(n_neighbors=0), dict(m=1.0), dict(radius=0.0)):
        with pytest.raises(ValueError):
            IsomapFCM(n_clusters=2, **kw).fit(X)


# ------------------------------------------------------------------ plots
def test_plot_helpers(chained, spiral):
    X, _ = chained
    mst = MSTClustering(random_state=0).fit(X)
    ax = plot_mst(X, mst)
    assert ax.get_legend() is not None
    ax = plot_mst(X, mst, labels=mst.labels_, show_cuts=False, legend=False)
    assert len(ax.collections) >= 1
    ax = plot_similarity_dendrogram(mst.cluster_similarity_)
    assert ax.get_ylabel().startswith("$1 - S")
    ax = plot_similarity_dendrogram(np.ones((1, 1)))
    assert ax is not None
    Xs, _ = spiral
    geo = GeodesicFCM(n_clusters=2, n_neighbors=5, random_state=0).fit(Xs)
    assert geo.bridge_edges_.shape[0] >= 1
    ax = plot_neighbors_graph(Xs, geo)
    assert ax.get_legend_handles_labels()[1].count("bridge") == 1
    ax = plot_neighbors_graph(Xs, geo.neighbors_graph_, labels=geo.labels_)
    ax = plot_geodesic_path(Xs, geo, 0, 10, ax=ax)
    labels = ax.get_legend_handles_labels()[1]
    assert any(name.startswith("geodesic path") for name in labels)
    ax = plot_geodesic_path(Xs, geo.neighbors_graph_, 0, 10)
    assert ax is not None
    plt.close("all")
