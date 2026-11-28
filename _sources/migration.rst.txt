Migrating from the MATLAB toolbox
=================================

The MATLAB *Fuzzy Clustering and Data Analysis Toolbox* passed ``data`` and
``param`` structures to functions and returned a ``result`` structure.
``fuzzyclust`` turns every algorithm into a scikit-learn estimator.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - MATLAB
     - fuzzyclust
   * - ``Kmeans(data,param)`` / ``Kmedoid``
     - :class:`fuzzyclust.cluster.KMeans` / :class:`~fuzzyclust.cluster.KMedoids`
   * - ``FCMclust(data,param)``
     - :class:`fuzzyclust.cluster.FCM`
   * - ``GKclust(data,param)``
     - :class:`fuzzyclust.cluster.GK` (``rho``, ``gamma``, ``beta``)
   * - ``GGclust(data,param)``
     - :class:`fuzzyclust.cluster.GG`
   * - ``clusteval(new,result,param)``
     - ``est.predict_proba(X_new)``, ``est.transform(X_new)``, :func:`fuzzyclust.plotting.plot_membership_contours`
   * - ``validity(result,data,param)``
     - :func:`fuzzyclust.validity.validity_indices`, :func:`~fuzzyclust.validity.scan_n_clusters`
   * - ``clust_normalize`` / ``clust_denormalize``
     - :func:`fuzzyclust.preprocessing.normalize` and ``scaler.inverse_transform``
   * - ``PCA`` / ``Sammon`` / ``FuzSam`` / ``projeval``
     - :func:`fuzzyclust.viz.pca_projection`, :class:`~fuzzyclust.viz.Sammon`, :class:`~fuzzyclust.viz.FuzzySammon`, :func:`~fuzzyclust.viz.projection_quality`
   * - ``param.c, param.m, param.e``
     - ``n_clusters``, ``m``, ``tol`` (plus ``init``, ``n_init``, ``max_iter``, ``random_state``)
   * - ``result.data.f``, ``result.data.d``
     - ``membership_`` (n_samples × n_clusters), ``distances_``
   * - ``result.cluster.v / P / M / V / D``
     - ``cluster_centers_``, ``covariances_``, ``norm_matrices_``, ``eigenvectors_``, ``eigenvalues_``
   * - ``result.iter``, ``result.cost``
     - ``n_iter_``, ``objective_history_`` / ``objective_``

Behavioral differences (details in Appendix B of the book):

* the MATLAB functions tested the optional fields with ``exist('param.m')==1``,
  which is false for a field of a structure, so ``m``, ``e``, ``ro``, ``gamma``
  and ``beta`` always took their default values; ``fuzzyclust`` honors every
  constructor argument;
* the Gustafson–Kessel covariance regularization follows the published formula
  of Babuška, van der Veen and Kaymak;
* the Gath–Geva distance is evaluated in the log domain, so distant samples do
  not overflow;
* random restarts (``n_init``) and k-means++ seeding are available, and an
  iteration cap (``max_iter``) is added;
* plotting is separated from fitting.
