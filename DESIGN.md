# fuzzyclust — design contract

This document is the single source of truth for notation, API conventions and
code organisation. Every module, docstring, example and book chapter follows it.

## 1. Unified mathematical notation (used in docstrings, Sphinx and the book)

| Symbol | Meaning | Code name |
|---|---|---|
| $N$ | number of samples (observations) | `n_samples` |
| $n$ | number of features (dimension of the data space) | `n_features` |
| $c$ | number of clusters | `n_clusters` |
| $\mathbf{X} = [\mathbf{x}_1,\dots,\mathbf{x}_N]^T \in \mathbb{R}^{N\times n}$ | data matrix, rows are samples $\mathbf{x}_k$ | `X` |
| $\mathbf{x}_k$, $k=1,\dots,N$ | $k$-th sample (row vector) | `X[k]` |
| $\mathbf{v}_i$, $i=1,\dots,c$ | prototype (centre) of cluster $i$ | `cluster_centers_[i]` |
| $\mathbf{V} = [\mathbf{v}_1,\dots,\mathbf{v}_c]^T \in \mathbb{R}^{c\times n}$ | matrix of prototypes | `cluster_centers_` |
| $\mu_{ik} \in [0,1]$ | membership of sample $k$ in cluster $i$ | `membership_[k, i]` |
| $\mathbf{U} = [\mu_{ik}] \in \mathbb{R}^{c\times N}$ | fuzzy partition matrix (math convention) | `membership_` is $\mathbf{U}^T$, shape `(n_samples, n_clusters)` |
| $m > 1$ | fuzziness exponent (weighting exponent) | `m` |
| $D^2_{ik\mathbf{A}_i} = (\mathbf{x}_k-\mathbf{v}_i)^T \mathbf{A}_i (\mathbf{x}_k-\mathbf{v}_i)$ | squared inner-product distance | `transform(X)` returns $D_{ik}$ (not squared), shape `(n_samples, n_clusters)` |
| $\mathbf{A}_i$ | norm-inducing matrix of cluster $i$ | `norm_matrices_[i]` |
| $\mathbf{F}_i = \frac{\sum_k \mu_{ik}^m (\mathbf{x}_k-\mathbf{v}_i)(\mathbf{x}_k-\mathbf{v}_i)^T}{\sum_k \mu_{ik}^m}$ | fuzzy covariance matrix | `covariances_[i]` |
| $\alpha_i = \frac{1}{N}\sum_k \mu_{ik}$ | prior probability of cluster $i$ (GG) | `priors_[i]` |
| $J(\mathbf{X};\mathbf{U},\mathbf{V}) = \sum_{i=1}^{c}\sum_{k=1}^{N} \mu_{ik}^m D^2_{ik}$ | objective (c-means functional) | `objective_`, `objective_history_` |
| $\epsilon$ | termination tolerance on $\max_{ik}|\mu_{ik}^{(l)}-\mu_{ik}^{(l-1)}|$ | `tol` |
| $l$ | iteration counter | `n_iter_` |
| $\rho_i$ | cluster volume (GK) | `rho` |
| $\gamma$ | covariance regularisation weight (GK, Babuška et al. 2002) | `gamma` |
| $\beta$ | condition-number bound (GK) | `beta` |
| $\mathbf{y}_k \in \mathbb{R}^q$ | low-dimensional projection of $\mathbf{x}_k$ ($q=2$ typically) | `embedding_` |
| $\mathbf{z}_i$ | projected cluster centre | `embedded_centers_` |

Convention: **index $i$ (and $j$) always runs over clusters, index $k$ over samples**.
The partition matrix in formulas is $c\times N$ (Babuška/Bezdek convention);
in code it is stored transposed, `(n_samples, n_clusters)`, so that it aligns
with scikit-learn's `predict_proba` layout. Docstrings state this once in
`BaseFuzzyClusterer` and refer to it.

Regression/identification extensions reuse the same symbols:
$\mathbf{z}_k = [\mathbf{x}_k^T\, y_k]^T$ for the joint input–output sample,
$\boldsymbol{\theta}_i$ for local linear model parameters, $\beta_i(\mathbf{x})$
for normalised rule firing strengths, $\hat y$ for predictions.

## 2. API contract (scikit-learn compatible)

All clustering estimators inherit from `fuzzyclust.base.BaseFuzzyClusterer`
(which mixes in `sklearn.base.BaseEstimator`, `ClusterMixin`, `TransformerMixin`).

### Constructor parameters (common, same names everywhere)
```
n_clusters=3, m=2.0, tol=1e-4, max_iter=300, init="random", n_init=1, random_state=None, verbose=0
```
* `n_init`: number of random restarts; the run with the lowest final objective `J` is kept
  (alternating optimisation finds local optima — show this in examples where it matters, e.g. GK on crossing lines).
* `init`: `"random"` (random partition, MATLAB-toolbox compatible),
  `"kmeans++"`, or an ndarray — either initial centres `(c, n)` or an initial
  membership matrix `(N, c)`.
* Algorithm specific parameters come after these, e.g. GK: `rho=None, gamma=0.0, beta=1e15`.
* No parameter validation or data processing in `__init__` (sklearn rule).

### Fitted attributes (trailing underscore)
```
cluster_centers_      (c, n)
membership_           (N, c)      fuzzy partition U^T
labels_               (N,)        argmax of membership_
distances_            (N, c)      D_ik at the final iteration
covariances_          (c, n, n)   where defined (GK, GG, ...)
norm_matrices_        (c, n, n)   A_i where defined (GK)
priors_               (c,)        alpha_i where defined (GG)
objective_            float       final J
objective_history_    (n_iter_,)  J per iteration
n_iter_               int
n_features_in_, feature_names_in_  (set by sklearn validation; DataFrames preserved)
```

### Methods
```
fit(X, y=None)              -> self
predict(X)                  -> hard labels (N,)
predict_proba(X)            -> memberships for new data (N, c)   [MATLAB clusteval]
transform(X)                -> distances D_ik (N, c)              [TransformerMixin; supports set_output(transform="pandas")]
fit_predict(X), fit_transform(X)
score(X)                    -> -J(X) (higher is better, sklearn convention)
get_feature_names_out()     -> ["dist_cluster_0", ...]
```
Each estimator implements two private hooks only:
```
_init_state(X, rng)         -> set initial centres / memberships
_update_prototypes(X, Um)   -> update centres (+ covariances etc.) from U^m
_distances(X)               -> squared distances D^2_ik (N, c)  for current prototypes
```
The generic alternating-optimisation loop lives in the base class:
`U -> V (prototypes) -> D^2 -> U` until `max|U_new - U_old| < tol` or `max_iter`.

### Input handling
* `X` may be `numpy.ndarray`, `pandas.DataFrame`, or anything `sklearn.utils.validation.validate_data` accepts.
* Never mutate inputs. Feature names are preserved via `feature_names_in_`.
* Scaling is **not** done inside estimators: use `sklearn.preprocessing.StandardScaler` / `MinMaxScaler`
  (helper `fuzzyclust.preprocessing.normalize` replicates MATLAB `clust_normalize`).

### Randomness
* Always go through `rng = sklearn.utils.check_random_state(self.random_state)`.
* Examples set `random_state=0` so figures are reproducible.

### Functions (non-estimator utilities)
* Validity indices: `fuzzyclust.validity.<index>(X, U, V, m=2.0, ...)` with `U` of shape `(N, c)`;
  `validity_indices(X, U, V, m)` returns a `dict`; `scan_n_clusters(estimator, X, n_clusters_range)` returns a `pandas.DataFrame`,
  `suggest_n_clusters(table)` the optimum of every index. Available: PC, CE, MPC, SC, S, XB, FS, DI, ADI, SIL, FHV, PD, APD, PBMF.
* Plotting: `fuzzyclust.plotting.plot_*` functions take `ax=None`, return the `Axes`, never call `plt.show()`.
* Dataset generators: `fuzzyclust.datasets.make_*` return `(X, y)` (or `Bunch`) and accept `random_state`;
  loaders `load_*` return `sklearn.utils.Bunch` with `.data`, `.target`, `.feature_names`, `.DESCR`.

## 3. Package layout
```
src/fuzzyclust/
  __init__.py          public re-exports
  base.py              BaseFuzzyClusterer, generic AO loop
  _utils.py            membership normalisation, fuzzy covariance, hardening, pairwise helpers
  cluster/             KMeans, KMedoids, FCM, GK, GG
  validity.py          PC, CE, MPC, SC, S, XB, FS, DI, ADI, silhouette wrapper, scan_n_clusters
  datasets.py          synthetic generators + bundled small datasets (motorcycle)
  preprocessing.py     normalize / denormalize (MATLAB compat), lagged regressors helpers
  plotting.py          scatter + membership contour maps, ellipses, scans
  viz/                 PCA, Sammon, FuzzySammon (FUZZSAM), projection quality, FuzzySOM
  models/              TS model identification (ModifiedGG), FuzzyCRegression, SupervisedFuzzyClassifier,
                       supervised fuzzy partitioning for decision trees, HingingHyperplaneTree
  timeseries/          segmentation, FNN model-order selection, state-space reconstruction, mixed-input structure selection
  graph/               MST-based hybrid clustering, geodesic-distance FCM
  extensions/          PCM, PFCM, noise clustering, kernel FCM, fuzzy c-varieties / c-shells
examples/              sphinx-gallery scripts: plot_<section>_<nn>_<topic>.py (ONE source for docs, notebooks and book listings)
tests/                 pytest, one file per module; sklearn check_estimator for all estimators
docs/                  Sphinx (pydata theme, numpydoc, sphinx-gallery, myst)
book/                  LaTeX book; figures and listings pulled from examples/ output
```

## 4. Docstring template (numpydoc)
```
"""One-line summary.

Longer description with the idea of the method in plain words.

Parameters
----------
n_clusters : int, default=3
    Number of clusters :math:`c`.
...

Attributes
----------
cluster_centers_ : ndarray of shape (n_clusters, n_features)
    Prototypes :math:`\mathbf{v}_i`.
...

Notes
-----
Objective and update equations in the unified notation, e.g.

.. math::
    J(\mathbf{X};\mathbf{U},\mathbf{V}) = \sum_{i=1}^{c}\sum_{k=1}^{N} \mu_{ik}^m D^2_{ik}

References
----------
.. [1] Author, "Title", Journal, year.

Examples
--------
>>> from fuzzyclust.cluster import FCM
>>> from fuzzyclust.datasets import make_fuzzy_blobs
>>> X, _ = make_fuzzy_blobs(random_state=0)
>>> fcm = FCM(n_clusters=3, random_state=0).fit(X)
>>> fcm.membership_.shape
(300, 3)
"""
```

## 5. Example script template (sphinx-gallery)
Every example is a runnable `.py` with a reST docstring header; it must
(a) generate its own data with `random_state=0`, (b) explain the idea of the
method and *why this data set exposes the property of interest*, (c) end with a
cell that varies the method's main parameter(s) and shows how the result changes
(figure or printed table), followed by a short **Take-away** paragraph, (d) produce
figures with `matplotlib` only through the helpers in `fuzzyclust.plotting` (fixed
palette: cluster *i* always has the same colour), (e) run in < 30 s, (f) have a
title on every axes. File names: `examples/<section>/plot_<nn>_<topic>.py`.

```python
"""
Fuzzy c-means: the effect of the fuzziness exponent m
=====================================================

Short explanation of the method (2–5 sentences) and what the figure shows.
"""
# %%
# Section heading rendered as text
import numpy as np
...
```

## 6. Code style
* Python >= 3.10, type hints on public signatures, `numpy` docstrings, `ruff` clean, no prints (use `verbose`).
* Vectorised numpy; loops only over clusters, never over samples when avoidable.
* `float` eps: `_utils.EPS = 1e-10` added to squared distances before the power (MATLAB compatible).

## 7. LaTeX notation macros (book/preamble.tex)
`\X \xk \vi \V \U \mik \A \Ai \Fi \Dik \Dsq \J \R \T \norm{} \abs{} \argmin \argmax \diag \tr`
and `\code{}`, `\py{}` for inline code. Algorithm boxes: `algobox`; summaries: `summarybox`;
API parameter boxes: `apibox`; practical hints: `hintbox`; chapter roadmap: `roadmap`.
Every chapter: roadmap → problem & idea (with one TikZ concept figure) → theory →
algorithm box → summary box → using it in fuzzyclust (apibox + snippets) → examples
(figures from the gallery via `\exfig`, parameter effects discussed) → further reading.
