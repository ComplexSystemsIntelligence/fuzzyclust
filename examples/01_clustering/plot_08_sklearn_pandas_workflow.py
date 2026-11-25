r"""
Working with pandas and scikit-learn: pipelines, DataFrames, and model selection
================================================================================

Every estimator of :mod:`fuzzyclust` follows the scikit-learn conventions, so
it works with the usual tooling: :class:`~sklearn.pipeline.Pipeline`,
``set_output(transform="pandas")``, :func:`~sklearn.base.clone`,
:class:`~sklearn.model_selection.GridSearchCV`, pickling, and so on. Feature
names of a :class:`pandas.DataFrame` are stored in ``feature_names_in_``,
and the distance matrix returned by ``transform`` can itself be a DataFrame.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fuzzyclust import FCM, GK
from fuzzyclust.datasets import load_iris
from fuzzyclust.validity import validity_indices

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# %%
# A pipeline of scaling and GK. ``set_output`` makes ``transform`` return a
# DataFrame whose columns are named after the clusters.
pipe = Pipeline([("scale", StandardScaler()), ("gk", GK(n_clusters=3, random_state=0))])
pipe.set_output(transform="pandas")
pipe.fit(df)
distances = pipe.transform(df)
print(distances.head().round(3))
print("feature names seen by GK:", list(pipe["gk"].feature_names_in_))

# %%
# The memberships as a DataFrame, with the species attached and averaged per
# species:
U = pd.DataFrame(pipe.predict_proba(df), columns=[f"cluster {i + 1}" for i in range(3)])
U["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(U.groupby("species", observed=True).mean().round(3))

# %%
# Model selection with ``GridSearchCV``
# -------------------------------------
# Clustering has no ground truth, but any validity index can serve as the
# scoring function. Here we select :math:`c` and :math:`m` of FCM by the
# Xie–Beni index (lower is better, hence the negative sign), computed with
# the estimator's ``predict_proba`` on held-out folds.
def neg_xie_beni(estimator, X):
    fcm = estimator[-1]
    Xs = estimator[:-1].transform(X)
    U = fcm.predict_proba(Xs)
    return -validity_indices(Xs, U, fcm.cluster_centers_, m=fcm.m, indices=["XB"])["XB"]


grid = GridSearchCV(
    Pipeline([("scale", StandardScaler()), ("fcm", FCM(random_state=0))]),
    param_grid={"fcm__n_clusters": [2, 3, 4, 5], "fcm__m": [1.5, 2.0, 2.5]},
    scoring=neg_xie_beni,
    cv=3,
)
grid.fit(df)
print("best parameters:", grid.best_params_)
res = pd.DataFrame(grid.cv_results_)
pivot = res.pivot(index="param_fcm__n_clusters", columns="param_fcm__m", values="mean_test_score")
print((-pivot).round(3))  # Xie–Beni values

# %%
fig, ax = plt.subplots(figsize=(5.5, 4))
for m in pivot.columns:
    ax.plot(pivot.index, -pivot[m], marker="o", label=f"m = {m}")
ax.set_xlabel("number of clusters $c$")
ax.set_ylabel("Xie–Beni index (held-out)")
ax.set_xticks(pivot.index)
ax.legend(frameon=False)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# The XB index favors two clusters on iris (setosa versus the rest), a
# well-known property of the data set: versicolor and virginica overlap in
# feature space. Validity indices measure geometric separation, not
# agreement with external labels.
#
# Finally, the fitted pipeline can be saved with ``pickle`` or ``joblib``
# like any other scikit-learn object:
import io
import pickle

buf = io.BytesIO()
pickle.dump(pipe, buf)
restored = pickle.loads(buf.getvalue())
print("pickle round trip ok:", np.allclose(restored.predict_proba(df), pipe.predict_proba(df)))

# %%
# Parameter study: the scoring index and the folds decide the "best" model
# -------------------------------------------------------------------------
# Without ground truth, the scoring function *is* the model-selection
# criterion, so the grid search is repeated with several validity indices
# as scorers and with three and five folds. Two details matter in practice.
# First, the iris rows are sorted by species, so the folds must be shuffled
# (``KFold(shuffle=True)``): an unshuffled fold made of one species gives a
# single predicted cluster and an undefined silhouette. Second, the indices
# that measure fuzziness (PC) or compactness relative to separation (XB, S)
# agree on two clusters, whereas the fuzzy hypervolume, which estimates one
# covariance per cluster from 30--50 held-out samples, changes its verdict
# with the number of folds.

# [snippet:scoring_scan]
from sklearn.model_selection import KFold


def index_scorer(index, sign):
    def score(estimator, X):
        fcm = estimator[-1]
        Xs = estimator[:-1].transform(X)
        U = fcm.predict_proba(Xs)
        return sign * validity_indices(Xs, U, fcm.cluster_centers_, m=fcm.m, indices=[index])[index]

    return score


scoring = {"XB": index_scorer("XB", -1), "S": index_scorer("S", -1), "SIL": index_scorer("SIL", +1),
           "PC": index_scorer("PC", +1), "FHV": index_scorer("FHV", -1)}
best = {}
for n_folds in (3, 5):
    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("fcm", FCM(random_state=0))]),
        param_grid={"fcm__n_clusters": [2, 3, 4, 5], "fcm__m": [1.5, 2.0, 2.5]},
        scoring=scoring, refit=False, cv=KFold(n_folds, shuffle=True, random_state=0),
    ).fit(df)
    res = pd.DataFrame(search.cv_results_)
    for name in scoring:
        i = res[f"mean_test_{name}"].idxmax()
        best[(name, f"{n_folds} folds")] = f"c = {res.loc[i, 'param_fcm__n_clusters']}, m = {res.loc[i, 'param_fcm__m']}"
# [/snippet:scoring_scan]
print("best (c, m) per scoring index and number of folds:")
print(pd.Series(best).unstack())

# %%
# Take-away
# ---------
# Because the estimators follow the scikit-learn contract, scaling, fitting,
# membership prediction, and persistence compose without glue code, and
# DataFrames keep feature and cluster names attached to the numbers. The
# same contract makes model selection with ``GridSearchCV`` possible, but
# the user has to supply the criterion: a validity index evaluated with
# ``predict_proba`` on held-out folds. The outcome depends on that choice
# (the geometric indices pick the two-group structure of iris, while the
# hypervolume needs more data than small folds provide) and on sensible
# folds (shuffle sorted data). Treat the search as a consistent way to rank
# candidates, not as a replacement for inspecting the partitions.
