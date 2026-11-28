Installation
============

``fuzzyclust`` requires Python 3.10 or newer and depends on NumPy, SciPy,
scikit-learn (≥ 1.6), pandas and matplotlib.

.. code-block:: bash

   pip install fuzzyclust

Optional extras
---------------

.. code-block:: bash

   pip install "fuzzyclust[graph]"   # networkx backend for the graph module (SciPy is used by default)
   pip install "fuzzyclust[test]"    # pytest and friends
   pip install "fuzzyclust[docs]"    # build this documentation
   pip install "fuzzyclust[dev]"     # everything, plus ruff, build and twine

From source
-----------

.. code-block:: bash

   git clone https://github.com/ComplexSystemsIntelligence/fuzzyclust.git
   cd fuzzyclust
   pip install -e ".[dev]"
   pytest                      # a few minutes, > 3700 tests

Building the documentation and the book
---------------------------------------

.. code-block:: bash

   # documentation (executes every gallery example)
   sphinx-build -b html docs docs/_build/html

   # book (needs TeX Live with latexmk, biber and the raleway, plex and
   # libertinus font packages)
   book/build.sh
