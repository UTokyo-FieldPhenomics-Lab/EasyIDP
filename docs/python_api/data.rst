====
Data
====

.. currentmodule:: easyidp.data

Purpose
=======

The data module is an optional shortcut for official EasyIDP demo datasets. It is not required for normal EasyIDP workflows. Most EasyIDP APIs accept ordinary file paths directly, so users can pass their own ``.shp``, ``.tif``, Pix4D, Metashape, or point-cloud paths without constructing an ``idp.data`` object.

The main purpose of this module is to keep examples readable:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    roi = idp.ROI(lotus.shp)
    ms = idp.Metashape(lotus.metashape.project)

Construction is lightweight. It does not download or extract data. Call ``download()`` explicitly when needed:

.. code-block:: python

    lotus = idp.data.Lotus()
    if not lotus.is_ready():
        lotus.download()

Configuration
=============

The default data directory comes from ``idp.config``:

.. code-block:: python

    import easyidp as idp

    idp.config.update(data_dir="/path/to/easyidp.data")
    lotus = idp.data.Lotus()

OpenXLab downloads use anonymous public dataset CDN URLs. They do not require the OpenXLab SDK, login, Access Key, or Secret Key:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    lotus.download(mirror="openxlab")

For Google Drive downloads, install the smaller optional backend instead:

.. code-block:: bash

    pip install "easyidp[gdrive]"

Datasets
========

.. autosummary::
    :toctree: autodoc

    Lotus
    ForestBirds
    TestData

Functions
=========

.. autosummary::
    :toctree: autodoc

    list_datasets
