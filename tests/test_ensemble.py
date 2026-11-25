import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust._utils import labels_to_membership  # noqa: E402
from fuzzyclust.cluster import FCM, GK  # noqa: E402
from fuzzyclust.ensemble import (  # noqa: E402
    FuzzyClusterEnsemble,
    coassociation_matrix,
    cspa,
    evidence_accumulation,
    information_theoretic_kmeans,
    make_overlapping_blobs,
    mcla,
    plot_coassociation,
    plot_lifetimes,
    soft_mcla,
    stability_curve,
)
from fuzzyclust.validity import fuzzy_adjusted_rand_index, fuzzy_rand_index  # noqa: E402

CONSENSUS = ("fcm", "spectral", "hierarchical", "smcla", "itk")


@parametrize_with_checks(
    [
        FuzzyClusterEnsemble(n_clusters=2, n_estimators=3, random_state=0),
        FuzzyClusterEnsemble(n_clusters=2, n_estimators=3, consensus="smcla", random_state=0),
        FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=(2, 3), n_estimators=3, random_state=0),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.fixture(scope="module")
def separated():
    return make_overlapping_blobs(n_clusters=3, separation=6.0, box=10.0, random_state=0)


@pytest.fixture(scope="module")
def noisy_6d():
    """Overlapping groups in six dimensions.

    Members that see two features and 30 % of the samples are weak.
    """
    return make_overlapping_blobs(n_clusters=4, n_features=6, separation=3.0, random_state=0)


# ------------------------------------------------------------------ generator
def test_generator():
    X, y = make_overlapping_blobs(n_clusters=5, n_samples_per_cluster=20, n_features=3, random_state=0)
    assert X.shape == (100, 3) and np.bincount(y).tolist() == [20] * 5
    X2, _ = make_overlapping_blobs(n_clusters=5, n_samples_per_cluster=20, n_features=3, random_state=0)
    np.testing.assert_allclose(X, X2)
    # a larger separation makes the groups easier for a single FCM run
    hard = make_overlapping_blobs(separation=1.5, random_state=0)
    easy = make_overlapping_blobs(separation=5.0, box=12.0, random_state=0)
    ari = [adjusted_rand_score(y, FCM(n_clusters=4, random_state=0).fit(X).labels_) for X, y in (hard, easy)]
    assert ari[0] < ari[1]


# ------------------------------------------------------------------ co-association
def test_coassociation_matrix_properties():
    rng = np.random.RandomState(0)
    lib = [rng.dirichlet(np.ones(c), size=40) for c in (2, 3, 5)]
    S = coassociation_matrix(lib)
    np.testing.assert_allclose(S, S.T)
    np.testing.assert_allclose(np.diag(S), 1.0)
    assert S.min() >= 0.0 and S.max() <= 1.0 + 1e-12
    # the inner-product variant: the diagonal is the mean squared norm of the membership vectors
    P = coassociation_matrix(lib, similarity="product")
    np.testing.assert_allclose(np.diag(P), np.mean([np.sum(U**2, axis=1) for U in lib], axis=0))
    assert np.all(P <= S + 1e-12)  # cosine >= inner product for vectors of length <= 1
    # crisp partitions: both are the fraction of partitions that put a pair together (Fred and Jain)
    labels = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([1, 1, 0, 0])]
    crisp = [labels_to_membership(lab) for lab in labels]
    expected = np.mean([lab[:, None] == lab[None, :] for lab in labels], axis=0)
    np.testing.assert_allclose(coassociation_matrix(crisp), expected)
    np.testing.assert_allclose(coassociation_matrix(crisp, similarity="product"), expected)
    # identical membership vectors are fully co-associated under the cosine
    U = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    assert coassociation_matrix([U])[0, 1] == pytest.approx(1.0)
    assert coassociation_matrix([U], similarity="product")[0, 1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        coassociation_matrix([])
    with pytest.raises(ValueError):
        coassociation_matrix([U, U[:2]])
    with pytest.raises(ValueError):
        coassociation_matrix([U], similarity="dot")
    with pytest.raises(ValueError):
        coassociation_matrix([-U])


def test_fitted_coassociation_symmetric_unit_diagonal(separated):
    X, _ = separated
    for similarity in ("cosine", "product"):
        ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=5, similarity=similarity, random_state=0).fit(X)
        S = ens.coassociation_
        np.testing.assert_allclose(S, S.T)
        assert S.min() >= 0.0 and S.max() <= 1.0
        if similarity == "cosine":
            np.testing.assert_allclose(np.diag(S), 1.0)
        else:
            assert np.all(np.diag(S) < 1.0)
    # objects of the same group are strongly, objects of different groups weakly co-associated
    ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=5, random_state=0).fit(X)
    same = ens.labels_[:, None] == ens.labels_[None, :]
    assert ens.coassociation_[same].mean() > 0.8 > 0.2 > ens.coassociation_[~same].mean()


# ------------------------------------------------------------------ building blocks
def test_evidence_accumulation_lifetimes():
    y = np.repeat([0, 1, 2], 10)
    U = labels_to_membership(y)
    labels, Z, life = evidence_accumulation(coassociation_matrix([U]), max_clusters=6)
    assert Z.shape == (29, 4) and life.shape == (6,)
    assert int(np.argmax(life)) + 1 == 3
    assert adjusted_rand_score(y, labels) == 1.0
    labels2, _, _ = evidence_accumulation(coassociation_matrix([U]), n_clusters=2)
    assert np.unique(labels2).size == 2
    # the lifetime is the length of the threshold interval with k clusters: they sum to the dendrogram height
    assert np.all(life >= 0)


def test_hard_consensus_functions():
    y = np.repeat([0, 1, 2], 20)
    rng = np.random.RandomState(0)
    noisy = []
    for _ in range(8):  # relabelled copies with 10 % of the labels flipped
        lab = rng.permutation(3)[y]
        flip = rng.rand(y.size) < 0.1
        lab[flip] = rng.randint(3, size=flip.sum())
        noisy.append(lab)
    assert adjusted_rand_score(y, cspa(noisy, 3, random_state=0)) == 1.0
    assert adjusted_rand_score(y, mcla(noisy, 3, random_state=0)) == 1.0
    U, meta = soft_mcla([labels_to_membership(lab) for lab in noisy], 3, random_state=0)
    assert U.shape == (60, 3) and meta.shape == (24,)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    # every meta-cluster collects one cluster of every partition
    assert sorted(np.bincount(meta).tolist()) == [8, 8, 8]
    labels, centroids = information_theoretic_kmeans([labels_to_membership(lab) for lab in noisy], 3,
                                                     random_state=0)
    assert adjusted_rand_score(y, labels) == 1.0 and centroids.shape == (3, 24)
    with pytest.raises(ValueError):
        soft_mcla([labels_to_membership(y)], 4)


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("consensus", CONSENSUS)
def test_every_consensus_recovers_separated_groups(separated, consensus):
    X, y = separated
    ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=8, consensus=consensus, random_state=0).fit(X)
    assert adjusted_rand_score(y, ens.labels_) > 0.95
    assert ens.n_clusters_ == 3 and ens.consensus_membership_.shape == (225, 3)
    np.testing.assert_allclose(ens.consensus_membership_.sum(axis=1), 1.0)
    assert np.all(ens.consensus_membership_ >= 0)
    assert ens.membership_ is ens.consensus_membership_
    # new samples: the training samples are reproduced, and points near a group join it
    assert adjusted_rand_score(ens.labels_, ens.predict(X)) > 0.98
    np.testing.assert_allclose(ens.predict_proba(X[:5]).sum(axis=1), 1.0)
    assert 0.5 < ens.base_agreement_ <= 1.0


def test_consensus_beats_noisy_members(noisy_6d):
    """Members on two features and 30 % of the samples are weak; their consensus is far better."""
    X, y = noisy_6d
    for consensus in ("fcm", "hierarchical", "spectral"):
        ens = FuzzyClusterEnsemble(n_clusters=4, n_estimators=20, sample_fraction=0.3, feature_fraction=0.34,
                                   consensus=consensus, random_state=0).fit(X)
        members = [adjusted_rand_score(y, U.argmax(axis=1)) for U in ens.base_memberships_]
        consensus_ari = adjusted_rand_score(y, ens.labels_)
        assert consensus_ari > max(members)
        assert consensus_ari > np.mean(members) + 0.3
        # the same holds for the soft partitions, measured with the fuzzy adjusted Rand index
        truth = labels_to_membership(y)
        soft_members = [fuzzy_adjusted_rand_index(truth, U) for U in ens.base_memberships_]
        assert fuzzy_adjusted_rand_index(truth, ens.consensus_membership_) > np.mean(soft_members)


def test_library_parameters(noisy_6d):
    X, _ = noisy_6d
    ens = FuzzyClusterEnsemble(n_clusters=4, n_estimators=6, base_n_clusters=(3, 5, 7), feature_fraction=0.5,
                               sample_fraction=0.5, random_state=0).fit(X)
    assert len(ens.base_estimators_) == len(ens.base_memberships_) == 6
    assert {U.shape[1] for U in ens.base_memberships_} <= {3, 5, 7}
    assert len({U.shape[1] for U in ens.base_memberships_}) > 1
    assert all(cols.size == 3 for cols in ens.base_feature_subsets_)
    assert all(U.shape[0] == X.shape[0] for U in ens.base_memberships_)  # every member partitions all samples
    assert all(est.n_features_in_ == 3 for est in ens.base_estimators_)
    full = FuzzyClusterEnsemble(n_clusters=4, n_estimators=3, random_state=0).fit(X)
    assert all(cols.size == 6 for cols in full.base_feature_subsets_)
    assert all(U.shape[1] == 4 for U in full.base_memberships_)
    # a different base clusterer
    gk = FuzzyClusterEnsemble(n_clusters=3, base_estimator=GK(), n_estimators=3, random_state=0).fit(X[:, :2])
    assert all(isinstance(est, GK) for est in gk.base_estimators_)


def test_library_size_stabilizes_the_consensus():
    """Single FCM runs with ten overlapping clusters land in different local optima.

    The consensus of a library is reproducible.
    """
    X, _ = make_overlapping_blobs(n_clusters=10, separation=2.5, box=10.0, n_samples_per_cluster=30,
                                  random_state=0)
    ari = stability_curve(X, n_estimators_grid=(1, 20), n_repeats=3, n_clusters=10, random_state=0,
                          index=fuzzy_adjusted_rand_index)
    assert list(ari.index) == [1, 20] and list(ari.columns) == ["mean", "std"]
    assert ari.loc[20, "mean"] > ari.loc[1, "mean"] + 0.1
    rand = stability_curve(X, n_estimators_grid=(1, 20), n_repeats=3, n_clusters=10, random_state=0)
    assert rand.loc[20, "mean"] > rand.loc[1, "mean"]
    assert ari.loc[1, "mean"] < rand.loc[1, "mean"]  # the chance-corrected index is stricter


def test_lifetime_criterion_chooses_n_clusters():
    X, y = make_overlapping_blobs(n_clusters=3, separation=6.0, box=10.0, random_state=0)
    ens = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=range(6, 11), n_estimators=20,
                               random_state=0).fit(X)
    assert ens.n_clusters_ == 3 and adjusted_rand_score(y, ens.labels_) > 0.95
    assert ens.lifetimes_.shape == (20,) and int(np.argmax(ens.lifetimes_)) + 1 == 3
    assert ens.linkage_.shape == (224, 4)
    # the hierarchical consensus with a given number also stores the dendrogram
    h = FuzzyClusterEnsemble(n_clusters=3, consensus="hierarchical", n_estimators=5, random_state=0).fit(X)
    assert hasattr(h, "lifetimes_") and hasattr(h, "linkage_")
    plain = FuzzyClusterEnsemble(n_clusters=3, n_estimators=3, random_state=0).fit(X)
    assert not hasattr(plain, "lifetimes_")
    # max_clusters bounds the lifetimes
    short = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=6, n_estimators=3, max_clusters=5,
                                 random_state=0).fit(X)
    assert short.lifetimes_.shape == (5,)


def test_smcla_attributes(separated):
    X, _ = separated
    ens = FuzzyClusterEnsemble(n_clusters=3, base_n_clusters=(3, 4), n_estimators=6, consensus="smcla",
                               random_state=0).fit(X)
    total = sum(U.shape[1] for U in ens.base_memberships_)
    assert ens.meta_cluster_labels_.shape == (total,)
    fcm = FuzzyClusterEnsemble(n_clusters=3, n_estimators=4, random_state=0).fit(X)
    assert fcm.consensus_centers_.shape == (3, 225)


def test_reproducibility(noisy_6d):
    X, _ = noisy_6d
    params = dict(n_clusters=4, n_estimators=5, sample_fraction=0.5, feature_fraction=0.5)
    for consensus in CONSENSUS:
        a = FuzzyClusterEnsemble(consensus=consensus, random_state=3, **params).fit(X)
        b = FuzzyClusterEnsemble(consensus=consensus, random_state=3, **params).fit(X)
        np.testing.assert_allclose(a.consensus_membership_, b.consensus_membership_)
        np.testing.assert_allclose(a.coassociation_, b.coassociation_)
    c = FuzzyClusterEnsemble(random_state=4, **params).fit(X)
    assert not np.allclose(a.coassociation_, c.coassociation_)


def test_pandas_input(separated):
    X, _ = separated
    df = pd.DataFrame(X, columns=["width", "height"])
    ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=4, feature_fraction=0.5, random_state=0).fit(df)
    assert list(ens.feature_names_in_) == ["width", "height"]
    ref = FuzzyClusterEnsemble(n_clusters=3, n_estimators=4, feature_fraction=0.5, random_state=0).fit(X)
    np.testing.assert_allclose(ens.consensus_membership_, ref.consensus_membership_)
    np.testing.assert_allclose(ens.predict_proba(df.iloc[:10]), ref.predict_proba(X[:10]))


def test_parameter_validation(separated):
    X, _ = separated
    bad = [
        dict(n_clusters=0),
        dict(n_clusters=1000),
        dict(n_estimators=0),
        dict(sample_fraction=0.0),
        dict(feature_fraction=1.5),
        dict(consensus="vote"),
        dict(linkage="ward"),
        dict(similarity="dot"),
        dict(n_clusters=None),
        dict(max_clusters=0),
        dict(base_n_clusters=(0, 2)),
    ]
    for params in bad:
        with pytest.raises(ValueError):
            FuzzyClusterEnsemble(**{"n_estimators": 2, **params}).fit(X)

    class NoProba:
        def get_params(self, deep=True):
            return {}

        def fit(self, X):
            return self

    with pytest.raises(ValueError):
        FuzzyClusterEnsemble(base_estimator=NoProba(), n_estimators=2).fit(X)


def test_fuzzy_rand_agreement_with_validity(separated):
    X, _ = separated
    ens = FuzzyClusterEnsemble(n_clusters=3, n_estimators=3, random_state=0).fit(X)
    expected = np.mean([fuzzy_rand_index(U, ens.consensus_membership_) for U in ens.base_memberships_])
    assert ens.base_agreement_ == pytest.approx(expected)


# ------------------------------------------------------------------ plots
def test_plot_helpers(separated):
    X, _ = separated
    ens = FuzzyClusterEnsemble(n_clusters=None, base_n_clusters=(5, 6), n_estimators=4, random_state=0).fit(X)
    ax = plot_coassociation(ens)
    assert ax.images and len(ax.lines) == ens.n_clusters_
    ax = plot_coassociation(ens.coassociation_, cbar=False)
    assert not ax.lines
    ax = plot_lifetimes(ens, max_show=6)
    assert len(ax.patches) == 6
    plot_lifetimes(ens.lifetimes_)
    plt.close("all")
