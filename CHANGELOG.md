# Changelog

## Unreleased

* EPUB edition of the book, generated from the LaTeX source (`book/epub/make_epub.py`); published next to the PDF on the documentation site.

## 1.0 (2026-11-25) — first public release

Versions 0.1 and 0.2 were internal development versions and were not published. Changes since 0.2:

* Book: language and technical review of every chapter in three rounds (copy-edit, check of every number against the output of the example scripts and of every API statement against the code, proofreading of the typeset pages with every figure compared with its caption); every reference checked against the publisher's record, DOIs added, new references where a claim needed the right source; numbered algorithm boxes; new design (Libertinus, Raleway, IBM Plex Mono; a TikZ cover drawn from a Gustafson–Kessel run of the package); the example appendix is now a catalog with figure references; one-step build with `book/build.sh`.
* Documentation: the same visual identity (logo, banner, colors, fonts); the landing page covers all sub-packages; the prose of all 61 gallery examples edited and checked against their output.
* Code: SPDX license header in every source file; docstrings corrected where the review found inaccuracies (k-medoids, MST cut cost, categorical noise, merging similarity, robust fuzzy c-medoids, attribute weights, PCCA penalty, separation index); American spelling in docstrings and user-visible strings (`rule_base()` columns now read "center"); `plot_membership_contours` draws only levels below the maximum, so plateaus of membership one no longer show rounding noise; clearer legends and labels in a few plotting helpers. Apart from these column labels, the public API of 0.2 is unchanged.

## 0.2 (development version)

* New sub-packages: `relational` (RFCM, NERF c-means, fuzzy c-medoids, fuzzy k-modes/k-prototypes, DTW/autocorrelation/Gower dissimilarities), `incomplete` (WDS/PDS/OCS/NPS), `large` (wFCM, rseFCM, spFCM, oFCM, `partial_fit`), `image` (FCM_S, FCM_S1/S2, EnFCM, FGFCM, FLICM), `semisupervised` (SSFCM, PCCA), `coclustering` (FCCM, Fuzzy CoDoK), `collaborative` (collaborative FCM), `ensemble` (soft consensus, evidence accumulation).
* Core: entropy/quadratic-regularized FCM with deterministic annealing, polynomial fuzzifier, subtractive and mountain clustering, compatible cluster merging; extensions: volume prototypes with adaptive merging, SCAD and weighted FCM.
* Validity: Kwon and PCAES indices, fuzzy Rand / adjusted Rand / Jaccard (Hüllermeier et al., Campello), stability-based selection of the number of clusters.
* Book: three new chapters (relational/categorical/mixed data; incomplete, very large and image data; partial supervision, co-clustering, collaboration and ensembles); chapters 3, 4 and 6 extended.

## 0.1 (development version)

* Core algorithms: `KMeans`, `KMedoids`, `FCM`, `GK`, `GG` on a shared alternating-optimization base class with scikit-learn API, pandas support, random restarts.
* Validity: PC, CE, MPC, SC, S, XB, FS, DI, ADI, silhouette, FHV, PD, APD, PBMF; `scan_n_clusters`, `suggest_n_clusters`.
* Visualization: PCA, Sammon, fuzzy Sammon mapping (FUZZSAM), projection quality, fuzzy self-organizing map.
* Extensions: PCM, PFCM, noise clustering, kernel FCM, fuzzy c-varieties, fuzzy c-shells.
* Graph: MST-based hybrid clustering with Gath–Geva refinement, geodesic-distance fuzzy c-medoids, Isomap-FCM.
* Models: modified Gath–Geva Takagi–Sugeno identification (SISO/MIMO), fuzzy c-regression, supervised fuzzy classifier with feature selection, supervised fuzzy partitions for decision trees, hinging-hyperplane regression tree.
* Time series: fuzzy segmentation with PPCA prototypes, bottom-up baseline, clustering-based false-nearest-neighbour model order selection, chaotic state-space reconstruction and prediction, structure selection with mixed inputs.
* 38 gallery examples, > 1500 tests, LaTeX book.
