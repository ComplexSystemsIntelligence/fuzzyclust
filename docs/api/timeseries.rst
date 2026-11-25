``fuzzyclust.timeseries``
=========================

.. currentmodule:: fuzzyclust.timeseries

.. automodule:: fuzzyclust.timeseries
   :no-members:

Fuzzy segmentation, model order selection, state-space reconstruction and prediction, structure selection with mixed inputs.

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   FuzzySegmentation
   ChaoticPredictor
   MixedInputStructureSelection

Functions
---------

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst

   bottom_up_segmentation
   pca_segment_cost
   false_nearest_neighbors
   fnn_model_order
   select_model_order
   cluster_based_threshold
   delay_embedding
   select_delay
   embedding_scan
   select_embedding_dimension
   correlation_dimension
   select_structure_mixed
   fuzzy_variance
   plot_segmentation
   plot_fnn_table
   plot_embedding_scan
   plot_phase_space
   plot_structure_ranking
   make_piecewise_timeseries
   make_mackey_glass
   make_lorenz
   make_rossler
   make_nonlinear_io_system
   make_mixed_input_system
