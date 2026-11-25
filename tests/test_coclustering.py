import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from fuzzyclust.coclustering import (  # noqa: E402
    FCCM,
    FCoDoK,
    make_block_matrix,
    plot_coclusters,
    plot_item_memberships,
)

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def blocks():
    return make_block_matrix(random_state=0)


@pytest.fixture(scope="module")
def unequal_blocks():
    return make_block_matrix(
        n_blocks=4, n_objects_per_block=(60, 30, 30, 20), n_items_per_block=(10, 30, 15, 20),
        n_noise_items=15, random_state=1,
    )


# ------------------------------------------------------------------ sklearn compatibility
def _expected_failures(estimator):
    """Co-clustering works on a non-negative object--item matrix.

    ``check_clustering`` feeds Gaussian blobs with negative entries, which the estimators
    reject with the informative "Negative values in data" ``ValueError`` (the ``positive_only``
    tag is set accordingly).
    """
    reason = "co-clustering needs a non-negative object-item matrix; blobs with negative values are rejected."
    return {"check_clustering": reason, "check_clustering(readonly_memmap=True)": reason}


@parametrize_with_checks(
    [FCCM(n_clusters=2, random_state=0), FCoDoK(n_clusters=2, random_state=0)],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# ------------------------------------------------------------------ data generator
def test_make_block_matrix():
    R, rows, cols = make_block_matrix(n_blocks=3, n_objects_per_block=10, n_items_per_block=4, random_state=0)
    assert R.shape == (30, 12) and np.all(R >= 0) and np.all(R == np.round(R))
    assert np.bincount(rows).tolist() == [10, 10, 10] and np.bincount(cols).tolist() == [4, 4, 4]
    # inside the blocks the counts are larger than outside
    same = rows[:, None] == cols[None, :]
    assert R[same].mean() > 3 * R[~same].mean()
    # noise items, unequal blocks, no shuffling keeps the block order
    R, rows, cols = make_block_matrix(
        n_blocks=2, n_objects_per_block=(5, 8), n_items_per_block=(3, 2), n_noise_items=4, shuffle=False,
        random_state=0,
    )
    assert R.shape == (13, 9) and (cols == -1).sum() == 4
    np.testing.assert_array_equal(rows, np.repeat([0, 1], [5, 8]))
    np.testing.assert_array_equal(cols, [0, 0, 0, 1, 1, -1, -1, -1, -1])
    np.testing.assert_allclose(make_block_matrix(random_state=2)[0], make_block_matrix(random_state=2)[0])


# ------------------------------------------------------------------ FCCM
def test_fccm_recovers_blocks(blocks):
    R, rows, cols = blocks
    fccm = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    assert adjusted_rand_score(rows, fccm.labels_) == 1.0
    assert adjusted_rand_score(cols, fccm.item_labels_) == 1.0
    assert fccm.membership_.shape == (120, 3) and fccm.item_membership_.shape == (60, 3)
    np.testing.assert_allclose(fccm.membership_.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(fccm.item_membership_.sum(axis=0), 1.0, atol=1e-9)  # per co-cluster
    assert np.all(fccm.membership_ >= 0) and np.all(fccm.item_membership_ >= 0)
    assert fccm.aggregation_ > 0 and fccm.n_iter_ == len(fccm.objective_history_)
    # the regularized objective is maximized: non-decreasing along the iterations
    assert np.all(np.diff(fccm.objective_history_) >= -1e-6)
    # the item memberships of a co-cluster are concentrated on its own items
    for i in range(3):
        own = cols == np.bincount(rows[fccm.labels_ == i]).argmax()
        assert fccm.item_membership_[own, i].sum() > 0.9


def test_fccm_temperature_parameters(blocks):
    """lambda_u controls the fuzziness of the objects, lambda_w that of the items."""
    R, rows, cols = blocks
    pcs = []
    for lu in (0.3, 1.0, 3.0):
        m = FCCM(n_clusters=3, lambda_u=lu, lambda_w=5.0, random_state=0).fit(R)
        pcs.append(np.mean(np.sum(m.membership_**2, axis=1)))
    assert pcs[0] > pcs[1] > pcs[2]
    ents = []
    for lw in (5.0, 20.0):
        m = FCCM(n_clusters=3, lambda_u=0.3, lambda_w=lw, random_state=0).fit(R)
        W = m.item_membership_
        ents.append(-np.sum(W * np.log(W + 1e-12)))
    assert ents[0] < ents[1]
    # far too large temperatures: everything uniform (the degenerate fuzzy limit)
    flat = FCCM(n_clusters=3, lambda_u=100.0, lambda_w=1000.0, random_state=0).fit(R)
    assert np.allclose(flat.membership_, 1.0 / 3.0, atol=1e-2)


def test_fccm_init_restarts_predict_pandas(blocks, unequal_blocks):
    R, rows, cols = blocks
    base = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    kpp = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, init="kmeans++", random_state=0).fit(R)
    given = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, init=base.membership_).fit(R)
    multi = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, n_init=3, random_state=0).fit(R)
    for m in (kpp, given, multi):
        assert adjusted_rand_score(rows, m.labels_) == 1.0
    assert multi.objective_ >= base.objective_ - 1e-6
    # reproducible
    again = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    np.testing.assert_allclose(again.membership_, base.membership_)
    # new rows: the training rows are reproduced, a pure block row is assigned to its block
    np.testing.assert_allclose(base.predict_proba(R), base.membership_, atol=1e-9)
    assert (base.predict(R) == base.labels_).all()
    new = np.zeros((3, R.shape[1]))
    for i in range(3):
        new[i, cols == i] = 3.0
    assert adjusted_rand_score(base.predict(new), np.arange(3)) == 1.0
    assert base.transform(R).shape == (120, 3) and base.score(R) > 0
    assert list(base.get_feature_names_out()) == ["cocluster_0", "cocluster_1", "cocluster_2"]
    df = pd.DataFrame(R, columns=[f"t{j}" for j in range(R.shape[1])])
    pm = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(df)
    assert list(pm.feature_names_in_)[:2] == ["t0", "t1"]
    np.testing.assert_allclose(pm.membership_, base.membership_)
    out = pm.set_output(transform="pandas").transform(df)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == list(pm.get_feature_names_out())
    # unequal blocks and noise items
    R2, rows2, cols2 = unequal_blocks
    m = FCCM(n_clusters=4, lambda_u=1.0, lambda_w=20.0, n_init=3, random_state=0).fit(R2)
    assert adjusted_rand_score(rows2, m.labels_) > 0.95
    real = cols2 >= 0
    assert adjusted_rand_score(cols2[real], m.item_labels_[real]) > 0.95
    # noise items get small memberships in every co-cluster
    W = m.item_membership_ / m.item_membership_.max(axis=0)
    assert W[~real].max() < W[real].max(axis=1).mean()


def test_coclustering_parameter_validation(blocks):
    R, _, _ = blocks
    for cls in (FCCM, FCoDoK):
        for kw in (dict(n_clusters=0), dict(n_clusters=1000), dict(lambda_u=0.0), dict(lambda_w=-1.0),
                   dict(max_iter=0), dict(n_init=0), dict(init="bogus"), dict(init=np.ones((5, 5)))):
            with pytest.raises(ValueError):
                cls(**kw).fit(R)
        with pytest.raises(ValueError, match="Negative"):
            cls(n_clusters=2).fit(R - 1.0)


# ------------------------------------------------------------------ FCoDoK
def test_fcodok_recovers_blocks_and_differs_from_fccm(blocks):
    R, rows, cols = blocks
    codok = FCoDoK(n_clusters=3, lambda_u=2.0, lambda_w=50.0, random_state=0).fit(R)
    assert adjusted_rand_score(rows, codok.labels_) == 1.0
    assert adjusted_rand_score(cols, codok.item_labels_) == 1.0
    np.testing.assert_allclose(codok.membership_.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(codok.item_membership_.sum(axis=0), 1.0, atol=1e-9)
    assert np.all(codok.membership_ >= 0) and np.all(codok.item_membership_ >= 0)
    # the quadratic regularizer clips: items outside a block get exactly zero membership
    assert (codok.item_membership_ == 0).sum() > 0
    fccm = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    assert (fccm.item_membership_ == 0).sum() == 0  # soft-max never reaches zero
    # fuzzier objects with a larger lambda_u
    soft = FCoDoK(n_clusters=3, lambda_u=5.0, lambda_w=50.0, random_state=0).fit(R)
    assert soft.membership_.max(axis=1).mean() < codok.membership_.max(axis=1).mean()
    # predictions, restarts, pandas
    np.testing.assert_allclose(codok.predict_proba(R), codok.membership_, atol=1e-9)
    multi = FCoDoK(n_clusters=3, lambda_u=2.0, lambda_w=50.0, n_init=3, random_state=0).fit(R)
    assert multi.objective_ >= codok.objective_ - 1e-6
    df = pd.DataFrame(R)
    assert FCoDoK(n_clusters=3, lambda_u=2.0, lambda_w=50.0, random_state=0).fit(df).n_features_in_ == 60


# ------------------------------------------------------------------ plots
def test_plot_helpers(blocks):
    R, rows, cols = blocks
    fccm = FCCM(n_clusters=3, lambda_u=1.0, lambda_w=20.0, random_state=0).fit(R)
    ax = plot_coclusters(R, fccm)
    assert len(ax.images) == 1 and len(ax.lines) == 3
    ax = plot_coclusters(R, row_labels=rows, col_labels=cols, cbar=False, show_blocks=False)
    assert len(ax.lines) == 0
    ax = plot_coclusters(R)
    assert ax is not None
    ax = plot_item_memberships(fccm, top=20)
    assert len(ax.patches) == 60
    ax = plot_item_memberships(fccm, normalize=False, item_names=[f"w{j}" for j in range(60)])
    assert ax.get_ylabel() == "item membership"
    plt.close("all")
