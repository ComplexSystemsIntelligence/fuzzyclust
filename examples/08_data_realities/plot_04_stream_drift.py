r"""
Clustering a drifting stream chunk by chunk with ``partial_fit``
=================================================================

Single-pass and online FCM need only one chunk in memory, so they can
consume a stream: ``partial_fit`` processes one chunk and keeps a
summary of everything seen so far (:math:`c` weighted prototypes for
spFCM, all chunk prototypes for oFCM). Both summaries treat the stream as
*stationary*: an observation from the first chunk counts as much as one
from the last. When the clusters drift, the summary lags behind. This
example sends a stream of 40 chunks through both algorithms while the
three cluster centers move along straight lines, and measures the
distance of the model's prototypes from the current true centers. A
forgetting factor :math:`\lambda` (``forgetting_factor``, an addition to
the published spFCM) multiplies the carried weights before every chunk,
so that old data fade geometrically; the parameter study shows that the
right :math:`\lambda` balances lag against noise and depends on how fast
the stream drifts.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from fuzzyclust.large import OnlineFCM, SinglePassFCM, make_drifting_stream
from fuzzyclust.plotting import PALETTE


def tracking_error(V, C):
    """Mean distance between matched prototypes and true centers."""
    D = cdist(V, C)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].mean())


stream = list(make_drifting_stream(n_chunks=40, chunk_size=500, n_clusters=3, drift=0.12, random_state=0))
true_centers = np.array([C for _, _, C in stream])

# %%
# Feeding the stream
# -------------------
# Every model sees the chunks in order and is queried after each one. The
# published spFCM (:math:`\lambda=1`) and oFCM summarize the whole past;
# spFCM with :math:`\lambda=0.5` keeps an effective memory of about two
# chunks. The chunk prototypes of oFCM (``chunk_centers_``) are a by-product:
# FCM on each chunk alone, the most recent but also the noisiest estimate.

# [snippet:stream]
models = {
    "spFCM, $\\lambda=1$ (paper)": SinglePassFCM(n_clusters=3, chunk_size=500, random_state=0),
    "spFCM, $\\lambda=0.5$": SinglePassFCM(n_clusters=3, chunk_size=500, forgetting_factor=0.5, random_state=0),
    "oFCM (merged)": OnlineFCM(n_clusters=3, chunk_size=500, random_state=0),
}
errors = {name: [] for name in models}
errors["oFCM, last chunk only"] = []
for Xc, _, C in stream:
    for name, model in models.items():
        model.partial_fit(Xc)
        errors[name].append(tracking_error(model.cluster_centers_, C))
    errors["oFCM, last chunk only"].append(tracking_error(models["oFCM (merged)"].chunk_centers_[-1], C))
# [/snippet:stream]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
first, last = stream[0][0], stream[-1][0]
for ax, name in zip(axes, models, strict=True):
    ax.scatter(*first.T, s=3, color="0.8", label="first chunk")
    ax.scatter(*last.T, s=3, color="0.55", label="last chunk")
    for i in range(3):
        ax.plot(true_centers[:, i, 0], true_centers[:, i, 1], color="black", ls=":", lw=1.2)
        traj = models[name].chunk_centers_ if "spFCM" in name else None
        if traj is not None:
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=PALETTE[i], lw=1.6)
    ax.scatter(*true_centers[-1].T, marker="+", s=160, color="black", zorder=5, label="true centers now")
    ax.scatter(*models[name].cluster_centers_.T, marker="X", s=110, color="white", edgecolor="black",
               zorder=6, label="model prototypes")
    ax.set_title(name)
axes[0].legend(frameon=False, fontsize=7, loc="best")
plt.tight_layout()

# %%
# Tracking error along the stream
# --------------------------------
# The published spFCM and oFCM fall behind linearly: their prototypes sit
# near the time average of the centers, which trails the current centers
# by half the distance traveled. With forgetting, spFCM settles at a
# constant small lag; the single-chunk FCM has no lag but the full
# sampling noise of 500 samples.
fig, ax = plt.subplots(figsize=(8, 4.2))
styles = ["o-", "s-", "^-", "x--"]
for j, (name, err) in enumerate(errors.items()):
    ax.plot(np.arange(1, len(err) + 1), err, styles[j], ms=3, color=PALETTE[j], label=name)
ax.set_xlabel("chunk")
ax.set_ylabel("mean distance to the current true centers")
ax.set_title("Tracking error on a drifting stream")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Parameter study: forgetting factor against drift speed
# -------------------------------------------------------
# The forgetting factor trades lag for noise. Small chunks (200 samples)
# make the noise visible. Without drift, a long memory (:math:`\lambda`
# close to one) is best, because every observation helps average out the
# noise; the faster the drift, the smaller the best :math:`\lambda`. The
# error is averaged over the second half of the stream, after the
# start-up.

# [snippet:scan]
lambdas = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
drifts = (0.0, 0.02, 0.05, 0.15)
table = np.zeros((len(drifts), len(lambdas)))
for a, drift in enumerate(drifts):
    chunks = list(make_drifting_stream(n_chunks=40, chunk_size=200, n_clusters=3, drift=drift, random_state=0))
    for b, lam in enumerate(lambdas):
        model = SinglePassFCM(n_clusters=3, chunk_size=200, forgetting_factor=lam, random_state=0)
        errs = []
        for Xc, _, C in chunks:
            errs.append(tracking_error(model.partial_fit(Xc).cluster_centers_, C))
        table[a, b] = np.mean(errs[20:])
# [/snippet:scan]

fig, ax = plt.subplots(figsize=(8, 4.2))
for a, drift in enumerate(drifts):
    ax.semilogy(lambdas, table[a], "o-", color=PALETTE[a], label=f"drift {drift} per chunk")
    b = int(np.argmin(table[a]))
    ax.scatter(lambdas[b], table[a, b], s=120, facecolors="none", edgecolors=PALETTE[a], linewidths=1.5)
ax.set_xlabel(r"forgetting factor $\lambda$")
ax.set_ylabel("mean tracking error (chunks 21-40)")
ax.set_title(r"The best $\lambda$ (circled) falls as the drift grows")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9", which="both")
plt.tight_layout()

for a, drift in enumerate(drifts):
    print(f"drift {drift:4.2f}: " + " ".join(f"{e:.3f}" for e in table[a]) +
          f"  best lambda = {lambdas[int(np.argmin(table[a]))]}")

# %%
# **Take-away.** ``partial_fit`` turns spFCM and oFCM into stream
# clusterers with constant memory, but as published both assume that the
# stream does not change: their prototypes describe the whole past and
# trail drifting clusters by half the distance traveled. A forgetting
# factor below one gives spFCM a finite memory of about
# :math:`1/(1-\lambda)` chunks and lets it follow the drift. Choose
# :math:`\lambda` near one for stationary data, around 0.5 for slow
# drift, and smaller values (or larger chunks) when the clusters move
# quickly; the chunk prototypes of oFCM show what a memoryless model sees.
