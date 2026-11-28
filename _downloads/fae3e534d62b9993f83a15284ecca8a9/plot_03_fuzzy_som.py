r"""
Fuzzy self-organizing map: ordered prototypes by regularized FCM
================================================================

Fuzzy c-means places its prototypes wherever the data are dense, in no
particular order. The fuzzy self-organizing map of Abonyi, Migály, and
Szeifert arranges the prototypes on the nodes of a low-dimensional grid and
adds a smoothness penalty to the FCM functional,

.. math::
    J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^m\|\mathbf{x}_k-\mathbf{v}_i\|^2
        + \vartheta\,\mathrm{tr}(\mathbf{V}^T\mathbf{L}\mathbf{V}),

where :math:`\mathbf{L}=\mathbf{G}^T\mathbf{G}` is built from second-order
finite differences of the prototypes along the grid axes. The membership
update is unchanged; the prototype update becomes a linear system that
couples neighboring nodes. The result is a topology-preserving map like
Kohonen's SOM, with the advantages of a fuzzy partition and of an objective
function that is minimized exactly. This example reproduces the two
geometric case studies of the paper, shows the effect of :math:`\vartheta`
and of the order of the difference operator, and draws the U-matrix and
component planes of a map of the wine data.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from fuzzyclust.datasets import load_wine
from fuzzyclust.viz import FuzzySOM, make_hemisphere, make_spiral_3d, plot_map, plot_som_lattice

# %%
# A chain of prototypes along a curve
# -----------------------------------
# The data are 300 noisy points on one turn of a helix in 3-D. Seven
# prototypes on a one-dimensional grid are initialized on the first principal
# axis. Plain FCM (:math:`\vartheta=0`) finds seven groups along the curve but
# connects them in a tangled order; with the penalty, the chain follows the
# curve.
X, t = make_spiral_3d(n_samples=300, turns=1.0, noise=0.2, random_state=0)

# [snippet:chain]
fcm_chain = FuzzySOM(grid_shape=(7,), regularization=0.0, random_state=0).fit(X)
som_chain = FuzzySOM(grid_shape=(7,), regularization=2.0, random_state=0).fit(X)
# [/snippet:chain]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for ax, model, name in zip(axes, (fcm_chain, som_chain), ("FCM (regularization = 0)", "fuzzy SOM (regularization = 2)"),
                           strict=True):
    plot_som_lattice(model, X, dims=(0, 1), ax=ax)
    ax.set_title(f"{name}: J = {model.objective_:.1f}")
plt.tight_layout()

# %%
# A 6 x 6 map folded onto a hemisphere
# ------------------------------------
# The data are 900 points on the upper half of the unit sphere (noise 0.1).
# The lattice of prototypes, drawn in the :math:`(x_1, x_2)` plane, shows
# whether the map is ordered: with :math:`\vartheta=0` the grid edges cross;
# with :math:`\vartheta=0.5` the lattice is a regular net (paper, Figs. 4
# and 6). The Spearman correlation between grid distances and prototype
# distances summarizes the ordering in one number.
Xh, uv = make_hemisphere(n_samples=900, noise=0.1, random_state=0)

# [snippet:hemisphere]
maps = {th: FuzzySOM(grid_shape=(6, 6), regularization=th, random_state=0).fit(Xh) for th in (0.0, 0.5, 5.0)}
# [/snippet:hemisphere]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for ax, (th, model) in zip(axes, maps.items(), strict=True):
    rho = spearmanr(pdist(model.grid_positions_), pdist(model.cluster_centers_)).correlation
    plot_som_lattice(model, Xh, dims=(0, 1), ax=ax)
    ax.set_title(rf"$\vartheta$ = {th}:  J = {model.objective_:.2f},  ordering $\rho$ = {rho:.2f}")
plt.tight_layout()

# %%
# The regularization costs little in the data term but produces an ordered
# map; a very large :math:`\vartheta` flattens the map into a nearly planar
# grid that no longer follows the curvature of the hemisphere (the
# prototypes' third coordinate shrinks). The component planes make this
# visible: each plane shows one coordinate of the prototypes over the grid.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for j, ax in enumerate(axes):
    plot_map(maps[0.5], kind="component", feature=j, ax=ax)
    ax.set_title(rf"$x_{j + 1}$ over the grid ($\vartheta$ = 0.5)")
plt.tight_layout()

# %%
# Second-order versus first-order differences
# -------------------------------------------
# The "smoothly distributed FCM" of Pascual-Marqui et al. penalizes the
# first-order differences (the lengths of the grid edges). This attracts all
# prototypes toward the center of the data. With the second-order operator,
# only the curvature of the map is penalized, and a regular planar grid
# costs nothing. The extent of the map shrinks with :math:`\vartheta` in the
# first case and stays constant in the second.
# [snippet:order]
thetas = (0.0, 0.25, 0.5, 1.0, 2.0)
extent = {order: [np.ptp(FuzzySOM(grid_shape=(6, 6), regularization=th, difference_order=order,
                                  random_state=0).fit(Xh).cluster_centers_, axis=0).max()
                  for th in thetas] for order in (1, 2)}
# [/snippet:order]
fig, ax = plt.subplots(figsize=(5.5, 4))
for order, style in ((2, "o-"), (1, "s--")):
    ax.plot(thetas, extent[order], style, lw=1.8, label=f"difference order {order}")
ax.set_xlabel(r"regularization weight $\vartheta$")
ax.set_ylabel("largest extent of the prototypes")
ax.legend(frameon=False)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# U-matrix and component planes of real data
# ------------------------------------------
# A 6 x 6 map is fitted to the standardized wine data (13 attributes). In
# high dimensions with many prototypes, the memberships for :math:`m=2`
# become almost uniform, so a smaller exponent (:math:`m=1.3`) is used to
# give the map contrast. The U-matrix shows the mean distance of every
# prototype to its grid neighbors: dark ridges separate groups of similar
# prototypes, that is, clusters. The fuzzy hit histogram shows where the
# data fall, and the component planes of two attributes reveal which
# variables separate the groups.
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
# [snippet:wine_som]
som_w = FuzzySOM(grid_shape=(6, 6), m=1.3, regularization=0.5, random_state=0).fit(Xw)
# [/snippet:wine_som]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
plot_map(som_w, kind="umatrix", ax=axes[0])
plot_map(som_w, kind="hits", ax=axes[1])
plot_map(som_w, kind="component", feature=wine.feature_names.index("flavanoids"), ax=axes[2])
axes[2].set_title("flavanoids")
plot_map(som_w, kind="component", feature=wine.feature_names.index("color_intensity"), ax=axes[3])
axes[3].set_title("color intensity")
plt.tight_layout()

# %%
# The hit histograms of the three cultivars show that each occupies its own
# region of the map, separated by the U-matrix ridges, and the planes show
# the chemistry behind the split: high flavanoid content on one side, high
# color intensity on another.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for cls, ax in enumerate(axes):
    plot_map(som_w, kind="hits", X=Xw[wine.target == cls], ax=ax)
    h = som_w.hits(Xw[wine.target == cls])
    i, j = np.unravel_index(h.argmax(), h.shape)
    ax.set_title(f"hits of cultivar {cls + 1} (peak at node ({i}, {j}))")
plt.tight_layout()

# %%
# Parameter study: grid size and fuzziness exponent on the hemisphere
# --------------------------------------------------------------------
# Besides :math:`\vartheta`, a map has two structural parameters. The grid
# size fixes the number of prototypes. More nodes lower the data term of
# :math:`J` and, with :math:`\vartheta=0.5`, keep the map ordered (Spearman
# :math:`\rho` between grid and prototype distances close to 1), whereas the
# unregularized lattices of the same size are less ordered (:math:`\rho`
# between 0.79 and 0.95). The exponent :math:`m` acts in the familiar way,
# with one consequence that matters for maps: as :math:`m` grows, the
# memberships of the 36 prototypes approach :math:`1/c`, all prototypes are
# pulled toward the center of the data, and the map shrinks (the range of
# the third coordinate collapses). This is why the wine map above was
# fitted with :math:`m=1.3`.

# [snippet:grid_m_scan]
print("grid    theta   objective J   data term   ordering rho   range of x3")
for size in (3, 4, 6, 8):
    for th in (0.0, 0.5):
        som = FuzzySOM(grid_shape=(size, size), regularization=th, random_state=0).fit(Xh)
        d2 = ((Xh[:, None, :] - som.cluster_centers_[None]) ** 2).sum(axis=-1)
        data_term = (som.membership_ ** som.m * d2).sum()
        rho = spearmanr(pdist(som.grid_positions_), pdist(som.cluster_centers_)).correlation
        print(f"{size}x{size}   {th:5.1f}   {som.objective_:11.2f}   {data_term:9.2f}   {rho:12.2f}"
              f"   {np.ptp(som.cluster_centers_[:, 2]):11.2f}")

print("\nexponent m   ordering rho   partition coefficient   range of x3   (6 x 6 grid, theta = 0.5)")
for m in (1.3, 1.6, 2.0, 2.5):
    som = FuzzySOM(grid_shape=(6, 6), m=m, regularization=0.5, random_state=0).fit(Xh)
    rho = spearmanr(pdist(som.grid_positions_), pdist(som.cluster_centers_)).correlation
    pc = np.mean(som.membership_ ** 2) * som.cluster_centers_.shape[0]
    print(f"{m:10.1f}   {rho:12.2f}   {pc:21.3f}   {np.ptp(som.cluster_centers_[:, 2]):11.2f}")
# [/snippet:grid_m_scan]

# %%
# Take-away
# ---------
# The fuzzy SOM is FCM with one extra term, a smoothness penalty that
# couples the prototypes along a grid and turns an unordered set of cluster
# centers into a topology-preserving map; the functional is still minimized
# exactly by alternating optimization. A moderate :math:`\vartheta` buys the
# ordering for a small increase in the data term (under ten percent here),
# whereas a large one flattens the map. The second-order operator leaves a
# regular planar grid unpenalized, while the first-order penalty of smoothly
# distributed FCM contracts it. The grid size sets the resolution. With many
# prototypes in many dimensions the exponent must be kept small; otherwise
# the memberships flatten and the map collapses to the center of the data.
# The U-matrix, hit histograms, and component planes are then read as for a
# Kohonen map, with a fuzzy partition attached.
