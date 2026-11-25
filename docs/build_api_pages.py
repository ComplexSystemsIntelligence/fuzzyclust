"""Generate docs/api/*.rst (autosummary stubs) from the public API of every sub-package."""

from __future__ import annotations

import importlib
import inspect
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent / "api"
ROOT.mkdir(exist_ok=True)

SECTIONS = [
    ("fuzzyclust.cluster", "Core partitioning algorithms", "Hard and fuzzy prototype-based clustering: K-means, K-medoids, FCM, Gustafson--Kessel, Gath--Geva."),
    ("fuzzyclust.validity", "Cluster validity", "Validity indices and the selection of the number of clusters."),
    ("fuzzyclust.extensions", "Possibilistic, noise, kernel and shape-prototype clustering", "PCM, PFCM, noise clustering, kernel FCM, fuzzy c-varieties and c-shells."),
    ("fuzzyclust.graph", "Graph-based clustering", "Minimal spanning tree based hybrid clustering and geodesic-distance fuzzy c-medoids."),
    ("fuzzyclust.viz", "Visualization", "PCA, Sammon and fuzzy Sammon projections, projection quality, fuzzy self-organizing map."),
    ("fuzzyclust.models", "Model identification", "Takagi--Sugeno identification by modified Gath--Geva clustering, fuzzy c-regression, supervised fuzzy classifiers, fuzzy partitions for trees, hinging-hyperplane trees."),
    ("fuzzyclust.timeseries", "Time series", "Fuzzy segmentation, model order selection, state-space reconstruction and prediction, structure selection with mixed inputs."),
    ("fuzzyclust.relational", "Relational, categorical and mixed data", "Relational FCM, NERF c-means, fuzzy c-medoids, fuzzy k-modes and k-prototypes, DTW / autocorrelation / Gower dissimilarities."),
    ("fuzzyclust.incomplete", "Incomplete data", "Fuzzy c-means with missing values: whole-data, partial-distance, optimal-completion and nearest-prototype strategies."),
    ("fuzzyclust.large", "Very large data and streams", "Weighted, random-sample-extension, single-pass and online FCM."),
    ("fuzzyclust.image", "Image segmentation", "Spatially constrained FCM: FCM_S, FCM_S1/S2, EnFCM, FGFCM, FLICM."),
    ("fuzzyclust.semisupervised", "Partial supervision", "Semi-supervised FCM and pairwise-constrained fuzzy clustering."),
    ("fuzzyclust.coclustering", "Co-clustering", "Fuzzy co-clustering of objects and items (FCCM, Fuzzy CoDoK)."),
    ("fuzzyclust.collaborative", "Collaborative clustering", "Collaborative FCM across data sites."),
    ("fuzzyclust.ensemble", "Cluster ensembles", "Soft consensus clustering and evidence accumulation."),
    ("fuzzyclust.datasets", "Data sets", "Synthetic generators and bundled example data."),
    ("fuzzyclust.preprocessing", "Preprocessing", "Scaling and regressor construction."),
    ("fuzzyclust.plotting", "Plotting helpers", "Scatter, contour, ellipse and validity plots shared by the examples."),
    ("fuzzyclust.base", "Base classes", "The alternating-optimisation template every clusterer builds on."),
]


def kind(obj):
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "data"


def main():
    index = ["API reference", "=" * 13, "", "The public API of ``fuzzyclust`` by sub-package. Every estimator follows the scikit-learn conventions (``fit`` / ``predict`` / ``predict_proba`` / ``transform``); the notation of the docstrings is the one of the :doc:`book <../book>`.", "", ".. toctree::", "   :maxdepth: 1", ""]
    for modname, title, blurb in SECTIONS:
        mod = importlib.import_module(modname)
        names = list(getattr(mod, "__all__", []))
        classes = [n for n in names if kind(getattr(mod, n)) == "class"]
        funcs = [n for n in names if kind(getattr(mod, n)) == "function"]
        data = [n for n in names if kind(getattr(mod, n)) == "data"]
        slug = modname.split(".")[-1]
        lines = [f"``{modname}``", "=" * (len(modname) + 4), "", f".. currentmodule:: {modname}", "", f".. automodule:: {modname}", "   :no-members:", "", blurb, ""]
        for heading, items, tmpl in [("Classes", classes, "autosummary/class.rst"), ("Functions", funcs, "autosummary/function.rst"), ("Constants", data, None)]:
            if not items:
                continue
            lines += [heading, "-" * len(heading), "", ".. autosummary::", "   :toctree: generated/"]
            if tmpl:
                lines.append(f"   :template: {tmpl}")
            lines.append("")
            lines += [f"   {n}" for n in items]
            lines.append("")
        (ROOT / f"{slug}.rst").write_text("\n".join(lines))
        index.append(f"   {slug}")
    (ROOT / "index.rst").write_text("\n".join(index) + "\n")
    print("api pages written")


if __name__ == "__main__":
    main()
