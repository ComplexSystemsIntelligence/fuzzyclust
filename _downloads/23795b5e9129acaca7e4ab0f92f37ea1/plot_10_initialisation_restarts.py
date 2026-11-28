r"""
Local optima and restarts: initializing Gustafson--Kessel on crossing lines
===========================================================================

Alternating optimization decreases the objective :math:`J` monotonically, but
it stops at the first stationary point it reaches, and which one that is
depends on the initial partition. Data made of several crossing line segments
are a classical trap for the Gustafson--Kessel algorithm: an elongated
prototype may lock onto the "wrong" pieces of two lines, and the iteration
cannot escape, because every piece is locally consistent with a thin
ellipsoid. This example fits GK from many random initializations, shows the
spread of the final objectives and partitions, and compares the two devices
offered by :mod:`fuzzyclust`: random restarts (``n_init``) and k-means++
seeding (``init="kmeans++"``). Restarts help; on these data, the "clever"
seeding does not.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust import GK
from fuzzyclust.datasets import make_lines
from fuzzyclust.plotting import PALETTE, plot_clusters, plot_membership_contours, plot_objective

X, y_true = make_lines(n_lines=3, n_samples_per_line=80, noise=0.15, random_state=0)
plot_clusters(X, y_true).set_title("three noisy line segments (generating labels)")

# %%
# Twenty random initializations
# -----------------------------
# Every run uses the default initialization (random prototypes inside the
# bounding box of the data) with a different seed.
# [snippet:restarts]
seeds = range(20)
runs = [GK(n_clusters=3, random_state=s).fit(X) for s in seeds]
J = np.array([r.objective_ for r in runs])
ari = np.array([adjusted_rand_score(y_true, r.labels_) for r in runs])
# [/snippet:restarts]
for s, j, a in zip(seeds, J, ari, strict=True):
    print(f"seed {s:2d}: J = {j:8.3f}   ARI = {a:.2f}   iterations = {runs[s].n_iter_}")
print(f"\ndistinct local optima (J rounded): {np.unique(J.round(1)).size}")

# %%
# The runs end in only two distinct objective values. The best and the
# worst partitions are shown side by side: the best one recovers the three
# lines; the worst one keeps the diagonal line as one cluster but cuts the
# two upper lines into left and right halves and pairs the halves.
best, worst = runs[int(J.argmin())], runs[int(J.argmax())]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].hist(J, bins=25, color=PALETTE[0])
axes[0].set_xlabel("final objective $J$")
axes[0].set_ylabel("number of runs")
axes[0].set_title("20 random initializations of GK")
axes[0].grid(True, color="0.9")
for ax, model, name in zip(axes[1:], (best, worst), ("best", "worst"), strict=True):
    plot_membership_contours(model, X, ax=ax, labels=False)
    ax.set_title(f"{name} run: J = {model.objective_:.1f}, ARI = {adjusted_rand_score(y_true, model.labels_):.2f}")
plt.tight_layout()

# %%
# The objective histories of a few runs show that all of them converge
# monotonically: a poor result is a converged result, not an unfinished one.
fig, ax = plt.subplots(figsize=(6, 4))
plot_objective(runs[:6], labels=[f"seed {s}" for s in range(6)], ax=ax)
ax.set_title("GK objective per iteration for six seeds")
plt.tight_layout()

# %%
# Restarts versus k-means++ seeding
# ---------------------------------
# ``n_init`` runs the whole optimization several times and keeps the run with
# the smallest final :math:`J`; ``init="kmeans++"`` spreads the initial
# prototypes over the data (every new seed is drawn with probability
# proportional to its squared distance from the seeds chosen so far) instead
# of drawing them uniformly in the bounding box. Over 20 outer seeds we count
# how often each strategy ends in the best optimum found above.
J_best = J.min()
strategies = {
    "random, n_init=1": dict(init="random", n_init=1),
    "random, n_init=5": dict(init="random", n_init=5),
    "random, n_init=10": dict(init="random", n_init=10),
    "k-means++, n_init=1": dict(init="kmeans++", n_init=1),
}
results = {}
for name, kw in strategies.items():
    Js = np.array([GK(n_clusters=3, random_state=s, **kw).fit(X).objective_ for s in range(20)])
    results[name] = Js
    print(f"{name:22s}: best optimum reached in {np.mean(Js <= J_best * 1.001):.0%} of the runs, "
          f"median J = {np.median(Js):.2f}")

fig, ax = plt.subplots(figsize=(7, 4))
positions = np.arange(len(results))
for j, Js in enumerate(results.values()):
    ax.scatter(np.full(Js.size, j) + np.random.RandomState(0).uniform(-0.15, 0.15, Js.size), Js,
               color=PALETTE[j], s=30, alpha=0.8)
ax.axhline(J_best, color="0.5", ls="--", lw=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(list(results), rotation=15)
ax.set_ylabel("final objective $J$ (20 outer seeds)")
ax.set_title("restarts reduce the risk of a poor local optimum, seeding does not")
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study: the number of restarts
# ---------------------------------------
# ``n_init`` is the only parameter of the restart device, and its cost is
# linear. For ten outer seeds and increasing ``n_init`` we record the success
# rate (the best optimum reached within 0.1%), the mean agreement with the
# generating labels, and the approximate number of GK iterations spent
# (iterations of the kept run times ``n_init``). The restarts of one fit are
# drawn from the same random stream, so the first restart of every fit is
# the single run with the same seed, and the success rate can only grow with
# ``n_init``. Here about one start in three succeeds, and the success rate
# rises steadily while the cost rises linearly.

# [snippet:n_init_scan]
print("n_init   best optimum reached   mean ARI   ~iterations spent")
for n_init in (1, 2, 3, 5, 8):
    fits = [GK(n_clusters=3, random_state=s, n_init=n_init).fit(X) for s in range(10)]
    Js = np.array([f.objective_ for f in fits])
    aris = [adjusted_rand_score(y_true, f.labels_) for f in fits]
    print(f"{n_init:6d}   {np.mean(Js <= J_best * 1.001):20.0%}   {np.mean(aris):8.2f}   "
          f"{np.mean([f.n_iter_ for f in fits]) * n_init:21.0f}")
# [/snippet:n_init_scan]

# %%
# **Take-away.** The alternating optimization of every algorithm in the
# toolbox is a local method. On well-separated compact clusters almost every
# start reaches the same optimum; on elongated, crossing, or differently
# sized clusters the outcome depends on the start. ``n_init`` is the
# simplest insurance; it multiplies the run time by the number of restarts.
# k-means++ seeding is designed for compact spherical groups: it pushes the
# seeds to the extremes of the data. On line-shaped clusters these are the
# *ends* of different lines, exactly the configuration from which GK slides
# into the poor optimum, so here it fails in every run. Seeding heuristics
# do not replace restarts, and for the Gath--Geva algorithm the recommended
# practice is to pass the partition of a converged FCM or GK run as
# ``init``.
