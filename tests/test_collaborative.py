import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import adjusted_rand_score, rand_score  # noqa: E402

from fuzzyclust._utils import labels_to_membership  # noqa: E402
from fuzzyclust.cluster import FCM  # noqa: E402
from fuzzyclust.collaborative import (  # noqa: E402
    CollaborativeFCM,
    align_partitions,
    make_collaborative_sites,
    plot_collaboration_trajectory,
    plot_site_partitions,
)
from fuzzyclust.validity import fuzzy_adjusted_rand_index, fuzzy_rand_index  # noqa: E402

# ``CollaborativeFCM.fit`` takes a *list* of data sites ``[X_1, ..., X_P]`` that share their rows;
# scikit-learn's ``parametrize_with_checks`` feeds a single 2-D array to ``fit`` and compares
# ``predict``/``transform`` outputs with array semantics, which does not fit the per-site lists the
# estimator returns. The checks are therefore not run; the tests below cover the contract instead
# (a single array is accepted as one site and then equals FCM).


@pytest.fixture(scope="module")
def sites():
    return make_collaborative_sites(random_state=0)


@pytest.fixture(scope="module")
def swapped():
    return make_collaborative_sites(swap_fraction=0.2, random_state=1)


# ------------------------------------------------------------------ utilities
def test_site_agreement_uses_validity_indices(sites):
    """The sites are compared with the toolbox-wide fuzzy Rand indices of ``fuzzyclust.validity``."""
    Xs, _ = sites
    m = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(Xs)
    S = m.site_agreement()
    assert S[0, 1] == pytest.approx(fuzzy_rand_index(m.memberships_[0], m.memberships_[1]))
    S_ari = m.site_agreement(index=fuzzy_adjusted_rand_index)
    assert S_ari[0, 1] == pytest.approx(fuzzy_adjusted_rand_index(m.memberships_[0], m.memberships_[1]))
    np.testing.assert_allclose(np.diag(S_ari), 1.0)
    # crisp copies reduce to the classical Rand index
    y1 = np.repeat([0, 1, 2], 10)
    y2 = y1.copy()
    y2[:3] = 1
    U1, U2 = labels_to_membership(y1), labels_to_membership(y2)
    assert fuzzy_rand_index(U1, U2) == pytest.approx(rand_score(y1, y2))


def test_align_partitions():
    U = labels_to_membership(np.repeat([0, 1, 2], 5))
    perm = align_partitions(U, U[:, [2, 0, 1]])
    np.testing.assert_array_equal(U[:, [2, 0, 1]][:, perm], U)


def test_generator():
    Xs, y = make_collaborative_sites(n_clusters=4, n_samples_per_cluster=20, n_sites=3,
                                     n_features_per_site=(2, 3, 1), random_state=0)
    assert [X.shape for X in Xs] == [(80, 2), (80, 3), (80, 1)] and y.shape == (80,)
    Xs2, _ = make_collaborative_sites(n_clusters=4, n_samples_per_cluster=20, n_sites=3,
                                      n_features_per_site=(2, 3, 1), random_state=0)
    for a, b in zip(Xs, Xs2, strict=True):
        np.testing.assert_allclose(a, b)
    # the spreads differ between the sites, and a swap changes the last site only
    Xs, y = make_collaborative_sites(spreads=(0.3, 2.0), random_state=0)
    within = [np.mean([X[y == i].std(axis=0).mean() for i in range(3)]) for X in Xs]
    assert within[0] < within[1]
    Xa, _ = make_collaborative_sites(swap_fraction=0.0, random_state=3)
    Xb, _ = make_collaborative_sites(swap_fraction=0.3, random_state=3)
    np.testing.assert_allclose(Xa[0], Xb[0])
    assert not np.allclose(Xa[1], Xb[1])


# ------------------------------------------------------------------ CollaborativeFCM
def test_alpha_zero_equals_independent_fcm(sites):
    Xs, y = sites
    cfcm = CollaborativeFCM(n_clusters=3, alpha=0.0, random_state=0).fit(Xs)
    assert cfcm.n_sites_ == 2 and cfcm.n_iter_ == 0 and np.all(cfcm.alpha_ == 0)
    for U, U0, V, V0 in zip(cfcm.memberships_, cfcm.independent_memberships_, cfcm.cluster_centers_,
                            cfcm.independent_centers_, strict=True):
        np.testing.assert_allclose(U, U0)
        np.testing.assert_allclose(V, V0)
    assert np.all(cfcm.collaboration_change_ == 0)
    # site 1 alone equals an FCM run (up to the cluster alignment)
    fcm = FCM(n_clusters=3, random_state=0).fit(Xs[0])
    assert adjusted_rand_score(fcm.labels_, cfcm.labels_[0]) == 1.0
    np.testing.assert_allclose(
        np.sort(fcm.cluster_centers_, axis=0), np.sort(cfcm.cluster_centers_[0], axis=0)
    )
    assert cfcm.objective_.shape == (2,) and cfcm.objective_history_.shape == (1, 2)
    # a single array is one site
    one = CollaborativeFCM(n_clusters=3, random_state=0).fit(Xs[0])
    assert one.n_sites_ == 1 and adjusted_rand_score(one.labels_[0], fcm.labels_) == 1.0


def test_collaboration_pulls_partitions_together(sites):
    """The fuzzy Rand index between the sites increases monotonically with alpha."""
    Xs, y = sites
    alphas = (0.0, 0.3, 1.0, 3.0, 10.0)
    models = [CollaborativeFCM(n_clusters=3, alpha=a, random_state=0).fit(Xs) for a in alphas]
    agreement = [m.site_agreement()[0, 1] for m in models]
    distance = [m.partition_distances_[0, 1] for m in models]
    assert np.all(np.diff(agreement) > 0)
    assert np.all(np.diff(distance) < 0)
    assert agreement[-1] > 0.98
    change = np.array([m.collaboration_change_ for m in models])
    assert np.all(np.diff(change, axis=0) >= -1e-9)  # the sites give in more and more
    # the clear site keeps its (correct) partition, the blurred site improves
    assert all(adjusted_rand_score(y, m.labels_[0]) > 0.98 for m in models)
    assert adjusted_rand_score(y, models[-1].labels_[1]) >= adjusted_rand_score(y, models[0].labels_[1])
    m = models[2]
    assert m.n_iter_ > 0 and m.objective_history_.shape == (m.n_iter_, 2)
    for U in m.memberships_:
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(U >= -1e-12)
    np.testing.assert_allclose(m.partition_distances_, m.partition_distances_.T)
    assert m.independent_partition_distances_[0, 1] > m.partition_distances_[0, 1]
    np.testing.assert_allclose(m.consensus_membership_.sum(axis=1), 1.0)


def test_asymmetric_collaboration_matrix(sites):
    Xs, y = sites
    # only site 1 listens to site 2. In Pedrycz (2002) a site whose row of the collaboration
    # matrix is zero minimizes the plain FCM functional in phase 2 as well, so it keeps iterating
    # FCM from its phase-1 partition: it moves only by the residual of the phase-1 stopping rule
    # (max |dU| < tol), not by collaboration -- an exact equality would be the wrong requirement.
    A = np.array([[0.0, 2.0], [0.0, 0.0]])
    m = CollaborativeFCM(n_clusters=3, alpha=A, random_state=0).fit(Xs)
    assert m.collaboration_change_[0] > 0.01
    assert m.collaboration_change_[1] < 1e-3 < m.collaboration_change_[0] / 10
    np.testing.assert_allclose(m.memberships_[1], m.independent_memberships_[1], atol=1e-3)
    # with a tight tolerance the non-listening site is a fixed point of FCM and does not move
    tight = CollaborativeFCM(n_clusters=3, alpha=A, tol=1e-9, max_iter=1000, random_state=0).fit(Xs)
    assert tight.collaboration_change_[1] < 1e-7
    # the diagonal is ignored
    m2 = CollaborativeFCM(n_clusters=3, alpha=A + 5.0 * np.eye(2), random_state=0).fit(Xs)
    np.testing.assert_allclose(m2.memberships_[0], m.memberships_[0])
    # three sites, one of them deviating
    Xs3, y3 = make_collaborative_sites(n_sites=3, spreads=(0.5, 0.5, 1.5), random_state=2)
    m3 = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(Xs3)
    assert m3.n_sites_ == 3 and m3.site_agreement().shape == (3, 3)
    S = m3.site_agreement()
    assert np.allclose(np.diag(S), 1.0) and S[0, 2] > CollaborativeFCM(
        n_clusters=3, alpha=0.0, random_state=0
    ).fit(Xs3).site_agreement()[0, 2]


def test_disagreeing_site_is_corrected(swapped):
    """A site whose data contradict the others on some objects is pulled toward the majority."""
    Xs, y = swapped
    indep = CollaborativeFCM(n_clusters=3, alpha=0.0, random_state=0).fit(Xs)
    coll = CollaborativeFCM(n_clusters=3, alpha=3.0, random_state=0).fit(Xs)
    assert adjusted_rand_score(y, coll.labels_[1]) > adjusted_rand_score(y, indep.labels_[1]) + 0.1
    assert adjusted_rand_score(y, coll.labels_[0]) > 0.85


def test_predict_transform_reproducible_pandas(sites):
    Xs, y = sites
    m = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(Xs)
    Us = m.predict_proba(Xs)
    for U, M in zip(Us, m.memberships_, strict=True):
        np.testing.assert_allclose(U, M, atol=1e-4)
    labels = m.predict([X[:10] for X in Xs])
    assert len(labels) == 2 and labels[0].shape == (10,)
    D = m.transform(Xs)
    assert D[0].shape == (300, 3) and D[1].shape == (300, 3)
    again = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(Xs)
    for a, b in zip(m.memberships_, again.memberships_, strict=True):
        np.testing.assert_allclose(a, b)
    multi = CollaborativeFCM(n_clusters=3, alpha=1.0, n_init=3, init="kmeans++", random_state=0).fit(Xs)
    assert adjusted_rand_score(multi.labels_[0], m.labels_[0]) > 0.95
    dfs = [pd.DataFrame(X, columns=[f"s{p}f{j}" for j in range(X.shape[1])]) for p, X in enumerate(Xs)]
    pm = CollaborativeFCM(n_clusters=3, alpha=1.0, random_state=0).fit(dfs)
    for a, b in zip(pm.memberships_, m.memberships_, strict=True):
        np.testing.assert_allclose(a, b)


def test_parameter_validation(sites):
    Xs, y = sites
    bad = (dict(n_clusters=0), dict(n_clusters=10_000), dict(m=3.0), dict(max_iter=0), dict(alpha=-1.0),
           dict(alpha=np.ones((3, 3))))
    for kw in bad:
        with pytest.raises(ValueError):
            CollaborativeFCM(**kw).fit(Xs)
    with pytest.raises(ValueError):
        CollaborativeFCM().fit([Xs[0], Xs[1][:-1]])  # different numbers of rows
    with pytest.raises(ValueError):
        CollaborativeFCM().fit([])
    m = CollaborativeFCM(n_clusters=3, random_state=0).fit(Xs)
    with pytest.raises(ValueError):
        m.predict_proba([Xs[0]])  # wrong number of sites
    with pytest.raises(ValueError):
        m.predict_proba([Xs[0], Xs[1][:, :1]])  # wrong number of features


# ------------------------------------------------------------------ plots
def test_plot_helpers(sites):
    Xs, y = sites
    alphas = (0.0, 1.0, 5.0)
    models = [CollaborativeFCM(n_clusters=3, alpha=a, random_state=0).fit(Xs) for a in alphas]
    axes = plot_site_partitions(Xs, models[1])
    assert axes.shape == (2,) and axes[0].get_title().startswith("site 1")
    axes = plot_site_partitions(Xs, models[1], independent=True)
    assert "independent" in axes[1].get_title()
    ax = plot_collaboration_trajectory(alphas, models)
    assert len(ax.lines) == 1 and ax.get_xlabel().startswith("collaboration")
    ax = plot_collaboration_trajectory(alphas, models, metric="distance")
    assert "distance" in ax.get_ylabel()
    with pytest.raises(ValueError):
        plot_collaboration_trajectory(alphas, models, metric="bogus")
    plt.close("all")
