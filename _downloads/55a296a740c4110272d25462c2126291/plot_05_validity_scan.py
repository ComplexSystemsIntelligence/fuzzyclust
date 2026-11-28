r"""
Choosing the number of clusters with validity indices
======================================================

A clustering algorithm always delivers a partition for the requested
:math:`c`, whether or not it is meaningful. Cluster validity indices
quantify how well a partition fits the data; running the algorithm for
:math:`c = 2,\dots,c_{\max}` and locating the extremum of the indices is the
classical recipe for selecting the number of clusters. No single index is
reliable on its own, so :mod:`fuzzyclust.validity` offers a family of them:

======  ===============================  ========================  =========
Index   Name                             Measures                  Optimum
======  ===============================  ========================  =========
PC      partition coefficient            fuzziness only            max
CE      classification entropy           fuzziness only            min
MPC     modified partition coefficient   fuzziness, rescaled       max
SC      partition index                  compactness/separation    min
S       separation index                 compactness/separation    min
XB      Xie–Beni                         compactness/separation    min
FS      Fukuyama–Sugeno                  compactness − separation  min
DI      Dunn's index                     crisp, O(N²)              max
ADI     alternative Dunn index           crisp, cheap bound        max
SIL     silhouette                       crisp                     max
======  ===============================  ========================  =========
"""

# %%
import matplotlib.pyplot as plt

from fuzzyclust import FCM, GK
from fuzzyclust.datasets import make_fuzzy_blobs
from fuzzyclust.plotting import plot_clusters, plot_validity_scan
from fuzzyclust.validity import scan_n_clusters, suggest_n_clusters

X, y_true = make_fuzzy_blobs(n_clusters=5, n_samples_per_cluster=80, spread=0.55, box=12.0, random_state=2)
plot_clusters(X, y_true).set_title("data generated from 5 components")

# %%
# ``scan_n_clusters`` refits the estimator for every :math:`c` and returns a
# ``DataFrame`` of index values; ``suggest_n_clusters`` reads off the optimum
# of every column, taking into account whether the index is maximized or
# minimized.
table = scan_n_clusters(FCM(m=2.0, random_state=0), X, n_clusters_range=range(2, 10))
print(table.round(3))
print("\nsuggested number of clusters per index:")
print(suggest_n_clusters(table))

# %%
# Plotting the (min–max normalized) indices against :math:`c` shows the
# consensus: the separation-based indices (XB, S, DI, ADI, and SIL) agree on
# :math:`c=5`. PC and CE measure fuzziness only and drift with :math:`c`
# and :math:`m`; here they peak at :math:`c=5`, but the table below shows
# that for :math:`m\ge2.5` they point to :math:`c=2`.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_validity_scan(table[["PC", "CE", "MPC"]], ax=axes[0])
axes[0].set_title("partition-based indices")
plot_validity_scan(table[["SC", "S", "XB", "DI", "ADI", "SIL"]], ax=axes[1])
axes[1].set_title("geometry-based indices")
plt.tight_layout()

# %%
# Validity indices are algorithm-agnostic: the same scan works for GK (or any
# other estimator of the toolbox). Because GK adapts the cluster shapes, its
# compactness/separation indices are usually slightly better than those of
# FCM for the same :math:`c`.
table_gk = scan_n_clusters(GK(random_state=0), X, range(2, 10), indices=["XB", "SIL"])
comparison = table[["XB", "SIL"]].join(table_gk, lsuffix="_FCM", rsuffix="_GK")
print(comparison.round(3))

# %%
# In practice, compute several indices, prefer the value of :math:`c` on
# which the geometry-based indices agree, and inspect the resulting partition
# visually (e.g., with the projections of :mod:`fuzzyclust.viz`) before
# trusting it. An index extremum is a hint, not a proof.

# %%
# Parameter study: the verdict and the fuzziness exponent
# -------------------------------------------------------
# Every index is evaluated on the partition that FCM delivers, and that
# partition depends on :math:`m`. The scan is therefore repeated for several
# exponents, and the suggested number of clusters is tabulated per index.
# XB, DI, SIL, and MPC keep pointing at :math:`c=5` for every exponent
# because the prototypes barely move with :math:`m`; S, FS, and ADI deviate
# only for the smallest or the largest exponents. The indices that only measure
# fuzziness (PC, CE) behave differently: for a soft partition
# (:math:`m\ge2.5`) they flip to :math:`c=2`, since fewer clusters simply
# means less fuzziness to share. The Xie--Beni curves show that the minimum
# deepens as :math:`m` grows but stays at :math:`c=5`.

# [snippet:m_scan]
import pandas as pd

m_values = (1.3, 1.6, 2.0, 2.5, 3.0)
tables = {m: scan_n_clusters(FCM(m=m, random_state=0), X, n_clusters_range=range(2, 10)) for m in m_values}
verdict = pd.DataFrame({f"m = {m}": suggest_n_clusters(t) for m, t in tables.items()})
# [/snippet:m_scan]
print("suggested number of clusters per index and exponent:")
print(verdict.loc[["PC", "CE", "MPC", "SC", "S", "XB", "FS", "DI", "ADI", "SIL"]])

fig, ax = plt.subplots(figsize=(6.5, 4))
for m, t in tables.items():
    ax.plot(t.index, t["XB"], "o-", label=f"m = {m}")
ax.axvline(5, color="0.5", ls=":")
ax.set_xlabel("number of clusters $c$")
ax.set_ylabel("Xie–Beni index (lower is better)")
ax.set_title("the Xie–Beni minimum stays at $c=5$ for every exponent")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="0.9")
plt.tight_layout()

# %%
# Take-away
# ---------
# A validity scan turns the choice of :math:`c` into a comparison of
# partitions, and the indices that combine compactness with separation
# (XB, S, DI, SIL, and the volume-based FHV/PD) are the ones to trust: on
# these five well-separated groups they agree with each other and with the
# GK variant of the scan, and XB, DI, and SIL also across the range of
# exponents tested (S drops to three clusters at :math:`m=3`). The
# fuzziness-only indices PC and CE change their verdict with :math:`m`
# (:math:`c=5` for :math:`m\le2`, :math:`c=2` for larger exponents); SC favors many small
# clusters on this data set. No index replaces a look at the partition (the
# projections of :mod:`fuzzyclust.viz` are the next step), but a consensus
# of the geometry-based indices is a strong hint.
