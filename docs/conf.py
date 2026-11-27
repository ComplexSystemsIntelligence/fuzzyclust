"""Sphinx configuration of the fuzzyclust documentation."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

import fuzzyclust  # noqa: E402

project = "fuzzyclust"
author = "János Abonyi"
copyright = f"{date.today().year}, {author}"
release = fuzzyclust.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# --- autodoc / autosummary / numpydoc -------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False, "show-inheritance": True}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# --- sphinx-gallery ---------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "within_subsection_order": "FileNameSortKey",
    "subsection_order": "ExplicitOrder",
    "download_all_examples": True,
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib",),
    "reference_url": {"fuzzyclust": None},
    "backreferences_dir": os.path.join("api", "generated"),
    "doc_module": ("fuzzyclust",),
    "inspect_global_variables": False,
    "binder": {
        "org": "ComplexSystemsIntelligence",
        "repo": "fuzzyclust",
        "branch": "main",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "../binder/requirements.txt",
        "notebooks_dir": "notebooks",
        "use_jupyter_lab": True,
    },
    "promote_jupyter_magic": True,
    "first_notebook_cell": "%pip install -q fuzzyclust",
    "default_thumb_file": os.path.join(os.path.dirname(__file__), "_static", "logo.png"),
}
from sphinx_gallery.sorting import ExplicitOrder  # noqa: E402

sphinx_gallery_conf["subsection_order"] = ExplicitOrder(
    [
        "../examples/01_clustering",
        "../examples/02_visualization",
        "../examples/06_extensions",
        "../examples/05_graph",
        "../examples/03_models",
        "../examples/04_timeseries",
        "../examples/07_relational",
        "../examples/08_data_realities",
        "../examples/09_beyond_unsupervised",
    ]
)

# --- HTML -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"
html_title = "fuzzyclust"
html_theme_options = {
    "github_url": "https://github.com/ComplexSystemsIntelligence/fuzzyclust",
    "logo": {"text": "fuzzyclust"},
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {"github_user": "ComplexSystemsIntelligence", "github_repo": "fuzzyclust", "github_version": "main", "doc_path": "docs"}
mathjax3_config = {"tex": {"macros": {"bm": ["\\boldsymbol{#1}", 1]}}}
