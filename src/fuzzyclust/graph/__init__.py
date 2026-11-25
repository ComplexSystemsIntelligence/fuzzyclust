# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Graph-based fuzzy clustering.

Prototype-based algorithms describe a cluster by a simple geometrical
object and therefore fail on arbitrarily shaped groups. The methods of this
sub-package start from a graph over the samples instead:

* :class:`MSTClustering` -- hybrid minimal spanning tree / Gath--Geva
  clustering: cut inconsistent tree edges (Zahn's criteria), split further
  by the fuzzy hypervolume criterion, refine with GG
  (Vathy-Fogarassy, Feil, Abonyi 2005);
* :class:`GeodesicFCM` -- fuzzy c-medoids with geodesic (shortest-path)
  distances on a k-nearest-neighbor graph (Feil, Abonyi, Algorithm II);
* :class:`IsomapFCM` -- fuzzy c-means on the Isomap projection with
  re-projected centers (Feil, Abonyi, Algorithm I);
* :func:`minimum_spanning_tree`, :func:`neighbors_graph`,
  :func:`geodesic_distances`, :func:`isomap_embedding`,
  :func:`fuzzy_cluster_similarity` -- the graph utilities behind them;
* :func:`plot_mst`, :func:`plot_neighbors_graph`, :func:`plot_geodesic_path`,
  :func:`plot_similarity_dendrogram` -- matplotlib helpers;
* :func:`make_chained_blobs`, :func:`make_density_blobs`, :func:`make_s_curve`
  -- data generators.

Everything is built on :mod:`scipy.sparse.csgraph`; ``networkx`` is an
optional backend of :func:`minimum_spanning_tree` only
(``pip install fuzzyclust[graph]``).
"""

from ._data import make_chained_blobs, make_density_blobs, make_s_curve
from ._plot import plot_geodesic_path, plot_mst, plot_neighbors_graph, plot_similarity_dendrogram
from .geodesic import GeodesicFCM, IsomapFCM, geodesic_distances, isomap_embedding, neighbors_graph
from .mst import MSTClustering, fuzzy_cluster_similarity, minimum_spanning_tree

__all__ = [
    "MSTClustering",
    "GeodesicFCM",
    "IsomapFCM",
    "minimum_spanning_tree",
    "fuzzy_cluster_similarity",
    "neighbors_graph",
    "geodesic_distances",
    "isomap_embedding",
    "plot_mst",
    "plot_neighbors_graph",
    "plot_geodesic_path",
    "plot_similarity_dendrogram",
    "make_chained_blobs",
    "make_density_blobs",
    "make_s_curve",
]
