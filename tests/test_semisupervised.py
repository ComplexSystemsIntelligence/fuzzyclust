import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.cluster import FCM  # noqa: E402
from fuzzyclust.datasets import make_fuzzy_blobs  # noqa: E402
from fuzzyclust.semisupervised import (  # noqa: E402
    PairwiseConstrainedFCM,
    SemiSupervisedFCM,
    check_pairwise_constraints,
    constraint_satisfaction,
    make_pairwise_constraints,
    make_partial_labels,
    make_partially_labeled_pair,
    plot_label_propagation,
    plot_pairwise_constraints,
)

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def pair():
    return make_partially_labeled_pair(random_state=0)


@pytest.fixture(scope="module")
def blobs():
    return make_fuzzy_blobs(n_clusters=3, random_state=0)


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    """``SemiSupervisedFCM`` reads ``y`` as partial class labels (``-1`` = unlabelled).

    scikit-learn's generic checks pass an arbitrary integer ``y`` with more distinct values
    than ``n_clusters``; the estimator then raises an informative ``ValueError`` ("every class
    needs a cluster"), which is the documented behavior. ``fit(X)`` without ``y`` passes all
    checks (it reduces to FCM).
    """
    out = {}
    if type(estimator).__name__ == "SemiSupervisedFCM":
        reason = "y is read as partial labels; the check passes more classes than clusters."
        for check in (
            "check_fit_score_takes_y",
            "check_estimators_overwrite_params",
            "check_dont_overwrite_parameters",
            "check_estimators_fit_returns_self",
            "check_readonly_memmap_input",
            "check_n_features_in_after_fitting",
            "check_positive_only_tag_during_fit",
            "check_dtype_object",
            "check_f_contiguous_array_estimator",
            "check_transformer_data_not_an_array",
            "check_transformer_general",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_fit2d_1feature",
            "check_dict_unchanged",
            "check_fit2d_predict1d",
        ):
            out[check] = reason
    return out


@parametrize_with_checks(
    [
        SemiSupervisedFCM(n_clusters=2, random_state=0),
        PairwiseConstrainedFCM(n_clusters=2, random_state=0),
        PairwiseConstrainedFCM(n_clusters=3, agglomeration=5.0, random_state=0),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ data generators
def test_generators():
    X, y, yp = make_partially_labeled_pair(n_samples=200, label_fraction=0.1, random_state=0)
    assert X.shape == (200, 2) and np.array_equal(np.unique(y), [0, 1])
    assert (yp >= 0).sum() == 20 and np.all(yp[yp >= 0] == y[yp >= 0])
    np.testing.assert_allclose(X, make_partially_labeled_pair(n_samples=200, random_state=0)[0])
    # partial labels: exact count, one per class first, reproducible
    yp = make_partial_labels(np.repeat([0, 1, 2], 40), n_labeled=5, random_state=1)
    assert (yp >= 0).sum() == 5 and np.array_equal(np.unique(yp[yp >= 0]), [0, 1, 2])
    yp2 = make_partial_labels(np.repeat([0, 1, 2], 40), n_labeled=5, random_state=1)
    np.testing.assert_array_equal(yp, yp2)
    assert (make_partial_labels(np.repeat([0, 1], 10), fraction=0.0) == -1).all()
    # pairwise constraints consistent with the labels
    y = np.repeat([0, 1, 2], 20)
    ml, cl = make_pairwise_constraints(y, 15, 25, random_state=0)
    assert ml.shape == (15, 2) and cl.shape == (25, 2)
    assert (y[ml[:, 0]] == y[ml[:, 1]]).all() and (y[cl[:, 0]] != y[cl[:, 1]]).all()
    assert (ml[:, 0] < ml[:, 1]).all()  # unordered pairs stored sorted, no duplicates
    assert len({tuple(p) for p in ml}) == 15
    with pytest.raises(ValueError):
        make_pairwise_constraints(np.array([0, 0, 1]), n_must_link=5)


def test_constraint_utilities():
    M, C = check_pairwise_constraints([(3, 1), (0, 2)], None, n_samples=5)
    np.testing.assert_array_equal(M, [[1, 3], [0, 2]])
    assert C.shape == (0, 2)
    with pytest.raises(ValueError):
        check_pairwise_constraints([(0, 7)], None, n_samples=5)
    with pytest.raises(ValueError):
        check_pairwise_constraints([(2, 2)], None, n_samples=5)
    with pytest.raises(ValueError):
        check_pairwise_constraints([(0, 1)], [(1, 0)], n_samples=5)
    ml, cl = constraint_satisfaction([0, 0, 1, 1], [(0, 1), (0, 2)], [(0, 3), (2, 3)])
    assert ml == pytest.approx(0.5) and cl == pytest.approx(0.5)
    assert np.isnan(constraint_satisfaction([0, 1], None, None)[0])


# ------------------------------------------------------------------ SemiSupervisedFCM
def test_ssfcm_alpha_zero_is_fcm(pair):
    X, y, yp = pair
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    ss = SemiSupervisedFCM(n_clusters=2, alpha=0.0, init="random", random_state=0).fit(X, yp)
    np.testing.assert_allclose(ss.membership_, fcm.membership_, atol=1e-8)
    np.testing.assert_allclose(ss.cluster_centers_, fcm.cluster_centers_, atol=1e-8)
    # no labels at all: plain FCM as well (the "labels" start is random then), all clusters unsupervised
    none = SemiSupervisedFCM(n_clusters=2, alpha=3.0, random_state=0).fit(X)
    np.testing.assert_allclose(none.membership_, fcm.membership_, atol=1e-8)
    assert (none.label_mapping_ == -1).all() and none.classes_.size == 0 and not none.labelled_mask_.any()


def test_ssfcm_supervision_moves_partition_towards_labels(pair):
    """The unsupervised optimum cuts both bars; the labels rotate it toward the classes."""
    X, y, yp = pair
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    ss = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, yp)
    assert adjusted_rand_score(y, fcm.labels_) < 0.5
    assert adjusted_rand_score(y, ss.labels_) > adjusted_rand_score(y, fcm.labels_) + 0.1
    # labeled samples follow their labels, their membership moved toward f
    lab = yp >= 0
    assert (ss.label_mapping_[ss.labels_][lab] == yp[lab]).all()
    assert ss.score_labels(X, yp) > 0.7  # predictions for "new" data carry no labels
    f = ss.label_membership_[lab]
    assert np.abs(ss.membership_[lab] - f).mean() < np.abs(fcm.membership_[lab] - f).mean()
    # bookkeeping
    assert np.array_equal(ss.classes_, [0, 1]) and np.array_equal(ss.label_mapping_, [0, 1])
    assert ss.labelled_mask_.sum() == lab.sum() and ss.alpha_ == 1.0
    assert ss.label_membership_.shape == (X.shape[0], 2)
    np.testing.assert_allclose(ss.membership_.sum(axis=1), 1.0, atol=1e-9)
    assert ss.covariances_.shape == (2, 2, 2)
    assert ss.n_iter_ == len(ss.objective_history_) and np.all(np.diff(ss.objective_history_) <= 1e-6)
    # more labels pull further
    _, _, yp20 = make_partially_labeled_pair(label_fraction=0.3, random_state=0)
    ss20 = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, yp20)
    assert adjusted_rand_score(y, ss20.labels_) > adjusted_rand_score(y, ss.labels_)


def test_ssfcm_alpha_study_and_auto(pair):
    X, y, yp = pair
    lab = yp >= 0
    f = SemiSupervisedFCM(n_clusters=2, random_state=0).fit(X, yp).label_membership_[lab]
    dev = []
    for a in (0.2, 1.0, 5.0):
        ss = SemiSupervisedFCM(n_clusters=2, alpha=a, random_state=0).fit(X, yp)
        dev.append(np.abs(ss.membership_[lab] - f).mean())
    assert dev[0] > dev[1] > dev[2]  # larger alpha -> labeled memberships closer to the labels
    auto = SemiSupervisedFCM(n_clusters=2, alpha="auto", random_state=0).fit(X, yp)
    assert auto.alpha_ == pytest.approx(X.shape[0] / lab.sum())
    # the weight alpha/(1+alpha) on the label for a labeled sample (paper's convex combination)
    ss = SemiSupervisedFCM(n_clusters=2, alpha=3.0, random_state=0).fit(X, yp)
    inv = 1.0 / (ss.distances_**2 + 1e-10)
    fcm_u = inv / inv.sum(axis=1, keepdims=True)
    expected = (fcm_u[lab] + 3.0 * f) / 4.0
    np.testing.assert_allclose(ss.membership_[lab], expected, atol=1e-6)


def test_ssfcm_fuzzy_labels_init_and_surplus_clusters(pair):
    X, y, yp = pair
    # a label membership matrix with zero rows for unlabelled samples gives the same result
    F = np.zeros((X.shape[0], 2))
    F[yp == 0, 0] = 1.0
    F[yp == 1, 1] = 1.0
    a = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, yp)
    b = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, F)
    np.testing.assert_allclose(a.membership_, b.membership_, atol=1e-8)
    # soft labels are renormalized
    c = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, 2.0 * F)
    np.testing.assert_allclose(c.label_membership_, F)
    # a random start can tie cluster j to the wrong bar and end in a worse local optimum;
    # the "labels" start (default) and restarts avoid it
    rnd = SemiSupervisedFCM(n_clusters=2, alpha=1.0, init="random", random_state=0).fit(X, yp)
    assert rnd.objective_ > a.objective_
    assert adjusted_rand_score(y, rnd.labels_) < adjusted_rand_score(y, a.labels_)
    multi = SemiSupervisedFCM(n_clusters=2, alpha=1.0, init="random", n_init=5, random_state=0).fit(X, yp)
    assert multi.objective_ == pytest.approx(a.objective_, rel=1e-4)
    # more clusters than classes: the surplus cluster is unsupervised
    three = SemiSupervisedFCM(n_clusters=3, alpha=1.0, random_state=0).fit(X, yp)
    assert np.array_equal(three.label_mapping_, [0, 1, -1])
    assert three.predict_classes(X).min() >= -1 and three.label_membership_[:, 2].sum() == 0
    # class labels need not be 0..c-1
    shifted = np.where(yp >= 0, yp + 5, -1)
    s = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, shifted)
    assert np.array_equal(s.classes_, [5, 6]) and set(np.unique(s.predict_classes(X))) <= {5, 6}


def test_ssfcm_predict_new_data_and_pandas(pair):
    X, y, yp = pair
    ss = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, yp)
    # new data carries no labels: FCM memberships to the supervised prototypes
    U = ss.predict_proba(X)
    inv = 1.0 / (ss.transform(X) ** 2 + 1e-10)
    np.testing.assert_allclose(U, inv / inv.sum(axis=1, keepdims=True), atol=1e-8)
    assert (ss.predict(X[:5]) == U[:5].argmax(axis=1)).all()
    assert ss.predict_classes(X[:5]).shape == (5,)
    assert ss.score(X) == pytest.approx(-float(np.sum(U**2 * ss.transform(X) ** 2)), rel=1e-6)
    df = pd.DataFrame(X, columns=["a", "b"])
    ps = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(df, pd.Series(yp))
    assert list(ps.feature_names_in_) == ["a", "b"]
    np.testing.assert_allclose(ps.membership_, ss.membership_)
    assert isinstance(ps.set_output(transform="pandas").transform(df), pd.DataFrame)


def test_ssfcm_parameter_validation(pair):
    X, y, yp = pair
    for kw in (dict(alpha=-1.0), dict(alpha="bogus"), dict(m=3.0), dict(init="bogus")):
        with pytest.raises(ValueError):
            SemiSupervisedFCM(n_clusters=2, **kw).fit(X, yp)
    with pytest.raises(ValueError):  # three classes, two clusters
        SemiSupervisedFCM(n_clusters=2).fit(X, np.where(yp >= 0, np.arange(X.shape[0]) % 3, -1))
    with pytest.raises(ValueError):  # wrong length
        SemiSupervisedFCM(n_clusters=2).fit(X, yp[:-1])
    with pytest.raises(ValueError):  # wrong label matrix shape
        SemiSupervisedFCM(n_clusters=2).fit(X, np.zeros((X.shape[0], 3)))
    with pytest.raises(ValueError):
        SemiSupervisedFCM(n_clusters=2).fit(X, -np.ones((X.shape[0], 2)))


# ------------------------------------------------------------------ PairwiseConstrainedFCM
def test_pcca_without_constraints_is_fcm(blobs):
    X, y = blobs
    fcm = FCM(n_clusters=3, random_state=0).fit(X)
    pcca = PairwiseConstrainedFCM(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(pcca.membership_, fcm.membership_, atol=1e-6)
    assert pcca.n_clusters_ == 3 and pcca.must_link_.shape == (0, 2) and pcca.cannot_link_.shape == (0, 2)
    assert np.isnan(pcca.must_link_satisfied_) and pcca.alpha_ == 0.0
    assert pcca.cardinalities_.shape == (3,) and pcca.covariances_.shape == (3, 2, 2)


def test_pcca_constraints_satisfied_in_proportion_to_the_penalty(pair):
    """The bars cannot be separated by FCM; the pairs pull the partition toward the classes."""
    X, y, _ = pair
    ml, cl = make_pairwise_constraints(y, 30, 30, random_state=0)
    sat, aris = [], []
    for alpha in (0.0, 1.0, 5.0, 10.0):
        p = PairwiseConstrainedFCM(n_clusters=2, alpha=alpha, random_state=0)
        p.fit(X, must_link=ml, cannot_link=cl)
        sat.append(0.5 * (p.must_link_satisfied_ + p.cannot_link_satisfied_))
        aris.append(adjusted_rand_score(y, p.labels_))
        np.testing.assert_allclose(p.membership_.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(p.membership_ >= 0)
        assert p.alpha_ == alpha
    assert sat[0] < 0.8 and sat[-1] > 0.95
    assert all(s2 >= s1 - 0.05 for s1, s2 in zip(sat[:-1], sat[1:], strict=True))
    assert aris[0] < 0.5 and aris[-1] > 0.9
    # satisfaction attributes agree with the utility
    p = PairwiseConstrainedFCM(n_clusters=2, alpha=5.0, random_state=0).fit(X, must_link=ml, cannot_link=cl)
    assert (p.must_link_satisfied_, p.cannot_link_satisfied_) == constraint_satisfaction(p.labels_, ml, cl)
    np.testing.assert_array_equal(p.must_link_, np.sort(ml, axis=1))
    # automatic penalty: J / number of constraints at the last iteration
    auto = PairwiseConstrainedFCM(n_clusters=2, alpha="auto", random_state=0)
    auto.fit(X, must_link=ml, cannot_link=cl)
    assert auto.alpha_ > 0 and 0.5 * (auto.must_link_satisfied_ + auto.cannot_link_satisfied_) > 0.8
    # must-link only and cannot-link only are accepted
    only = PairwiseConstrainedFCM(n_clusters=2, alpha=5.0, random_state=0).fit(X, must_link=ml)
    assert np.isnan(only.cannot_link_satisfied_) and only.must_link_satisfied_ > 0.8


def test_pcca_competitive_agglomeration_reduces_clusters(blobs):
    X, y = blobs
    ca = PairwiseConstrainedFCM(n_clusters=8, agglomeration=5.0, random_state=0).fit(X)
    assert ca.n_clusters_ == 3 and adjusted_rand_score(y, ca.labels_) > 0.95
    assert ca.membership_.shape == (X.shape[0], 3) and ca.cluster_centers_.shape == (3, 2)
    assert ca.cardinalities_.shape == (3,) and ca.cardinalities_.min() >= ca.min_cardinality
    assert list(ca.get_feature_names_out()) == ["dist_cluster_0", "dist_cluster_1", "dist_cluster_2"]
    assert ca.predict_proba(X).shape == (X.shape[0], 3) and ca.transform(X).shape == (X.shape[0], 3)
    # a weak agglomeration keeps more clusters, a higher cardinality threshold removes more
    weak = PairwiseConstrainedFCM(n_clusters=8, agglomeration=1.0, random_state=0).fit(X)
    assert weak.n_clusters_ >= ca.n_clusters_
    strict = PairwiseConstrainedFCM(n_clusters=8, agglomeration=1.0, min_cardinality=5.0, random_state=0)
    strict.fit(X)
    assert strict.n_clusters_ <= weak.n_clusters_
    # a slower decay keeps the bias alive longer and agglomerates at least as much
    slow = PairwiseConstrainedFCM(n_clusters=8, agglomeration=2.0, agglomeration_decay=50.0, random_state=0)
    fast = PairwiseConstrainedFCM(n_clusters=8, agglomeration=2.0, agglomeration_decay=10.0, random_state=0)
    slow.fit(X)
    fast.fit(X)
    assert slow.n_clusters_ <= fast.n_clusters_
    # constraints and agglomeration together
    ml, cl = make_pairwise_constraints(y, 20, 20, random_state=0)
    both = PairwiseConstrainedFCM(n_clusters=6, alpha=2.0, agglomeration=5.0, random_state=0).fit(
        X, must_link=ml, cannot_link=cl
    )
    assert both.n_clusters_ <= 6 and both.must_link_satisfied_ > 0.9


def test_pcca_predict_reproducible_pandas_and_validation(blobs):
    X, y = blobs
    ml, cl = make_pairwise_constraints(y, 10, 10, random_state=0)
    a = PairwiseConstrainedFCM(n_clusters=3, alpha=2.0, random_state=0).fit(X, must_link=ml, cannot_link=cl)
    b = PairwiseConstrainedFCM(n_clusters=3, alpha=2.0, random_state=0).fit(X, must_link=ml, cannot_link=cl)
    np.testing.assert_allclose(a.membership_, b.membership_)
    # new samples: FCM memberships to the fitted prototypes (no constraints)
    U = a.predict_proba(X[:10])
    inv = 1.0 / (a.transform(X[:10]) ** 2 + 1e-10)
    np.testing.assert_allclose(U, inv / inv.sum(axis=1, keepdims=True))
    assert (a.predict(X[:10]) == U.argmax(axis=1)).all()
    assert a.score(X) < 0
    multi = PairwiseConstrainedFCM(n_clusters=3, alpha=2.0, n_init=3, random_state=0).fit(X, must_link=ml)
    assert multi.objective_ <= a.objective_ + 1e-6 or multi.objective_ == pytest.approx(a.objective_, rel=0.2)
    df = pd.DataFrame(X, columns=["a", "b"])
    pd_model = PairwiseConstrainedFCM(n_clusters=3, alpha=2.0, random_state=0)
    pd_model.fit(df, must_link=ml, cannot_link=cl)
    assert list(pd_model.feature_names_in_) == ["a", "b"]
    np.testing.assert_allclose(pd_model.membership_, a.membership_)
    bad = (
        dict(alpha=-1.0),
        dict(alpha="bogus"),
        dict(m=3.0),
        dict(agglomeration=-1.0),
        dict(agglomeration_decay=0.0),
        dict(min_cardinality=-1.0),
        dict(n_init=0),
    )
    for kw in bad:
        with pytest.raises(ValueError):
            PairwiseConstrainedFCM(n_clusters=3, **kw).fit(X)
    with pytest.raises(ValueError):
        PairwiseConstrainedFCM(n_clusters=3).fit(X, must_link=[(0, 1)], cannot_link=[(1, 0)])
    with pytest.raises(ValueError):
        PairwiseConstrainedFCM(n_clusters=3).fit(X, must_link=[(0, 10_000)])


# ------------------------------------------------------------------ plots
def test_plot_helpers(pair):
    X, y, yp = pair
    ss = SemiSupervisedFCM(n_clusters=2, alpha=1.0, random_state=0).fit(X, yp)
    ax = plot_label_propagation(X, yp, ss)
    names = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(n.startswith("labeled") for n in names) and any(n.startswith("cluster") for n in names)
    ax = plot_label_propagation(X, yp, labels=ss.labels_, legend=False)
    assert ax is not None
    ml, cl = make_pairwise_constraints(y, 10, 10, random_state=0)
    p = PairwiseConstrainedFCM(n_clusters=2, alpha=5.0, random_state=0).fit(X, must_link=ml, cannot_link=cl)
    ax = plot_pairwise_constraints(X, ml, cl, labels=p.labels_, centers=p.cluster_centers_)
    assert len(ax.collections) >= 2
    ax = plot_pairwise_constraints(X, ml, None, violated_only=True, legend=False)
    assert ax is not None
    plt.close("all")
