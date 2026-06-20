====
Data
====

.. currentmodule:: easyidp.data

Purpose
=======

The data module is an optional shortcut for easy downloading official EasyIDP demo datasets. It is not required for normal EasyIDP workflows. Most EasyIDP APIs accept ordinary file paths directly, so users can pass their own ``.shp``, ``.tif``, Pix4D, Metashape, or point-cloud paths without constructing an ``idp.data`` object.

The main purpose of this module is to keep example path readable:

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

Dependencies
============

For downloading demo datasets, need to manually install the optional data backend:

.. code-block:: bash

    pip install "easyidp[data]"

When developing EasyIDP from the source tree, install all development groups
and package extras together:

.. code-block:: bash

    uv sync --all-groups --all-extras


Configuration
=============

The default data directory is default os app storage path:

For windows, it is ``%APPDATA%/easyidp.data``. 
For Linux, it is ``~/.local/share/easyidp.data``. 
For MacOS, it is ``~/Library/Application Support/easyidp.data``.

But users can change the default data directory by updating the ``data_dir`` key in the global configuration. The example below shows how to set the data directory to ``/path/to/easyidp.data``.

.. code-block:: python

    import easyidp as idp

    idp.config.update(data_dir="/path/to/easyidp.data")
    lotus = idp.data.Lotus()

Mirrors
=======

By default, easyidp try to download dataset from Shared Google Drive by `gdown` package. For users in China mainland, please use OpenXLab mirror for better downloading experience. At current stage, easyidp uses anonymous public dataset CDN URLs. They do not require the OpenXLab SDK, login, Access Key, or Secret Key:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    lotus.download(mirror="openxlab")


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
