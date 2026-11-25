r"""
MIMO Takagi–Sugeno model with local MIMO ARX consequents
========================================================

Clustering-based identification is usually applied to multi-input
single-output systems, so a process with :math:`n_y` outputs needs
:math:`n_y` separate rule bases whose interaction remains hidden. Abonyi and
Szeifert instead proposed one rule base whose rules carry *local MIMO ARX
models*,

.. math::
    R_i:\ \text{If } \mathbf{x}(k) \text{ is } A_i \text{ then }
    \mathbf{y}^i(k+1)=\sum_{j=1}^{n_a}\mathbf{A}^i_j\mathbf{y}(k-j+1)
    +\sum_{j=1}^{n_b}\mathbf{B}^i_j\mathbf{u}(k-j-n_d+1)+\mathbf{c}^i,

where the antecedent acts on a *scheduling vector* :math:`\mathbf{x}(k)`,
typically a few of the lagged outputs. The clustering algorithm is the
modified Gath–Geva algorithm with a multivariate Gaussian output
distribution: each cluster has a Gaussian on the scheduling variables, a
local MIMO ARX model, and the covariance matrix of its errors. The model is a
linear parameter-varying (LPV) system whose matrices are interpolated by the
firing strengths. This example identifies a two-input two-output process
whose input gains depend on the operating point.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuzzyclust.models import (
    TakagiSugeno,
    make_mimo_plant,
    plot_membership_functions,
    simulate_free_run,
)
from fuzzyclust.plotting import PALETTE
from fuzzyclust.preprocessing import make_regressors

# %%
# Process and regressors
# ----------------------
# The process has two states, both measured, and gains :math:`g_1(y_1)` and
# :math:`g_2(y_2)` that shrink as the outputs move away from zero. Random
# step sequences excite it; 600 samples are used for identification and a
# separate realization for validation. ``make_regressors`` arranges the
# regression vector as :math:`[\mathbf{y}(k-1)^T\ \mathbf{u}(k-1)^T]`.

# [snippet:data]
u_id, y_id = make_mimo_plant(n_samples=600, hold=10, random_state=0)
u_val, y_val = make_mimo_plant(n_samples=300, hold=10, random_state=7)
Phi_id, Y_id, names = make_regressors(y_id, u_id, n_a=1, n_b=1, n_k=1)
Phi_val, Y_val, _ = make_regressors(y_val, u_val, n_a=1, n_b=1, n_k=1)
print("regressors:", names)
# [/snippet:data]

# %%
# Identification with the outputs as scheduling variables
# -------------------------------------------------------
# ``antecedent_features=[0, 1]`` restricts the antecedents to the two lagged
# outputs (the operating point); the consequents use all four regressors.

# [snippet:fit]
ts = TakagiSugeno(n_clusters=3, antecedent_features=[0, 1], n_init=5, restart_criterion="error",
                  random_state=0).fit(Phi_id, Y_id)
print(ts.rule_base(feature_names=names, output_names=["y1", "y2"]).round(3).to_string(index=False))
y_sim = simulate_free_run(ts, u_val, y_val[:1], n_a=1, n_b=1, n_k=1)
# [/snippet:fit]
rmse = np.sqrt(np.mean((y_sim[1:] - y_val[1:]) ** 2, axis=0))
print(f"free-run RMSE on validation data: y1 {rmse[0]:.4f}, y2 {rmse[1]:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for o in range(2):
    axes[o].plot(y_val[:, o], "k-", lw=1.2, label="process")
    axes[o].plot(y_sim[:, o], color=PALETTE[o], lw=1.2, ls="--", label="free-run simulation")
    axes[o].set_ylabel(f"$y_{o + 1}(k)$")
    axes[o].set_title(f"Output {o + 1}: free-run RMSE {rmse[o]:.3f}")
    axes[o].legend(frameon=False, fontsize=8, loc="upper right")
    axes[o].grid(True, color="0.9")
axes[2].step(np.arange(u_val.shape[0]), u_val[:, 0], color=PALETTE[2], lw=1, label="$u_1$")
axes[2].step(np.arange(u_val.shape[0]), u_val[:, 1], color=PALETTE[3], lw=1, label="$u_2$")
axes[2].set_ylabel("inputs")
axes[2].set_xlabel("sample $k$")
axes[2].set_title("Validation input: random steps")
axes[2].legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()

# %%
# The MIMO rule base is simulated as one model: both outputs are fed back
# jointly, so the cross-couplings (:math:`y_1` enters the equation of
# :math:`y_2` and vice versa) are represented inside every local model.
#
# Operating regimes and local gains
# ---------------------------------
# The membership functions on the two scheduling variables show how the
# clustering has split the operating range. The local input gains
# :math:`[\mathbf{B}^i_1]_{11}` and :math:`[\mathbf{B}^i_1]_{22}` of the rules
# are compared with the gain curves of the process evaluated at the rule
# centers.
fig, axes = plt.subplots(2, 1, figsize=(7, 4.6))
plot_membership_functions(ts, X=Phi_id, feature_names=names, axes=axes)
axes[0].set_title("Antecedent sets on the scheduling variables $y_1(k-1)$, $y_2(k-1)$")
plt.tight_layout()

gains = pd.DataFrame(
    {
        "rule": np.arange(1, 4),
        "center y1": ts.centers_[:, 0].round(2),
        "center y2": ts.centers_[:, 1].round(2),
        "gain u1 -> y1": ts.coef_[:, 0, 2].round(3),
        "true g1(center)": (0.2 + 0.4 / (1 + ts.centers_[:, 0] ** 2)).round(3),
        "gain u2 -> y2": ts.coef_[:, 1, 3].round(3),
        "true g2(center)": (0.1 + 0.5 * np.exp(-ts.centers_[:, 1] ** 2)).round(3),
    }
)
print(gains.to_string(index=False))

# %%
# The rule at large outputs matches the true gains closely, and the rule near
# the origin has the largest :math:`u_2\to y_2` gain, as it should. The
# membership-weighted fit over a wide operating region flattens the
# :math:`u_1\to y_1` gains of the two rules near the origin, which come out
# almost equal although the true gains differ. Weighted least
# squares per rule provides this local interpretation, and the interpolated
# matrices :math:`\mathbf{A}(\mathbf{x})` and :math:`\mathbf{B}(\mathbf{x})`
# of the LPV view express it.
#
# Parameter study: number of rules and choice of scheduling variables
# -------------------------------------------------------------------
# One rule is a linear MIMO ARX model. Adding rules lets the model follow the
# operating-point-dependent gains. Scheduling on the lagged inputs instead of
# the outputs is tried as well, as is scheduling on all regressors. The
# criterion is the free-run RMSE on the validation data.

# [snippet:scan]
rows = []
for sched_name, sched in (("outputs", [0, 1]), ("inputs", [2, 3]), ("all", None)):
    for c in (1, 2, 3, 4, 6):
        model = TakagiSugeno(n_clusters=c, antecedent_features=sched, n_init=5, restart_criterion="error",
                             random_state=0).fit(Phi_id, Y_id)
        sim = simulate_free_run(model, u_val, y_val[:1], n_a=1, n_b=1, n_k=1)
        err = np.sqrt(np.mean((sim[1:] - y_val[1:]) ** 2, axis=0))
        rows.append({"scheduling": sched_name, "rules": c, "RMSE y1": err[0], "RMSE y2": err[1],
                     "RMSE total": np.sqrt(np.mean((sim[1:] - y_val[1:]) ** 2))})
scan = pd.DataFrame(rows)
# [/snippet:scan]
print(scan.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4))
for i, (sched_name, sub) in enumerate(scan.groupby("scheduling", sort=False)):
    ax.plot(sub["rules"], sub["RMSE total"], "o-", color=PALETTE[i], label=f"scheduling on {sched_name}")
ax.set_xlabel("number of rules $c$")
ax.set_ylabel("free-run RMSE (both outputs)")
ax.set_title("Free-run error versus number of rules and scheduling variables")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# **Take-away.** A single rule base with local MIMO ARX consequents models a
# multivariable process compactly: the antecedents act on a low-dimensional
# scheduling vector, the consequents on the full regressor vector, and the
# whole model is simulated as one system. One rule is the linear MIMO ARX
# model. Adding rules lowers the free-run error on the whole, though not
# monotonically: the scan has plateaus (:math:`c=3` and :math:`c=4` are
# equally good for output scheduling) because the clustering, not the
# prediction error, places the rules. The choice of scheduling variables is
# a structural decision. Here the nonlinearity lies in the output-dependent
# gains, so scheduling on the outputs (or on all regressors) works, whereas
# scheduling on the inputs brings little because the inputs do not define
# the operating point of this process.
