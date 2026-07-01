====
Data
====

.. currentmodule:: easyidp.data

Dependencies
============

For downloading demo datasets, install the optional data backend:

.. code-block:: bash

    pip install "easyidp[data]"

Or 

.. code-block:: bash

    uv sync --all-extras

Quick Start
===========

The data module is an optional shortcut for official EasyIDP demo datasets.
It keeps examples readable by turning long demo-dataset paths into short
``pathlib.Path`` attributes such as ``lotus.shp`` and
``lotus.metashape.project``. Most EasyIDP APIs still accept ordinary file paths
directly, so ``idp.data`` is not required for normal workflows.

.. code-block:: python

    >>> import easyidp as idp
    >>> lotus = idp.data.Lotus()
    >>> lotus
    <easyidp.data.dataset.Lotus object at 0x...>
    Dataset for the lotus plot in Tanashi, Tokyo.
    Size: ...
    Status: ...

    >>> lotus.download()  # use when data is missing
    >>> lotus.shp
    PosixPath('.../2017_tanashi_lotus/plots.shp')

    >>> roi = idp.ROI(lotus.shp)
    >>> ms = idp.Metashape(lotus.metashape.project)

By default, EasyIDP downloads demo datasets from Google Drive. If you are in
mainland China, use the ModelScope mirror instead:

.. code-block:: python

    lotus.download(mirror="modelscope")

Configuration
=============

The default data directory uses the operating system's application data path,
as returned by ``easyidp.config.default_data_dir()``. Change it through the
``data_dir`` key before constructing dataset objects:

.. code-block:: python

    import easyidp as idp

    idp.config.set(data_dir="/path/to/easyidp.data")
    lotus = idp.data.Lotus()

.. note::

    Changing ``data_dir`` through ``idp.config.set(data_dir=...)`` does
    **not** migrate or move already-cached datasets to the new location.
    New ``Dataset`` objects constructed after the change will use the new
    path, but existing objects retain the root they were created with.


Classes
=======

.. autosummary::
    :toctree: autodoc
    :template: autosummary/data_dataset_class.rst

    Lotus
    ForestBirds
    TestData

Functions
=========

.. autofunction:: list_datasets

For manifest parsing, downloader internals, and contributor-facing helpers,
see :doc:`Data Advanced API <autodoc/easyidp.data.advanced>`.
