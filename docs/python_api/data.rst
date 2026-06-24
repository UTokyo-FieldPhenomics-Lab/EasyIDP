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

    idp.config.set(data_dir="/path/to/easyidp.data")
    lotus = idp.data.Lotus()

REPL Representation
===================

Dataset construction is still lightweight but now prints status information
when the object is displayed in a REPL.  The ``repr`` output includes the
dataset title, size, download status, and the cache directory path:

.. code-block:: python

    >>> import easyidp as idp
    >>> fb = idp.data.ForestBirds()
    >>> fb
    <easyidp.data.dataset.ForestBirds object at 0x...>
    Official EasyIDP forest birds demo dataset from Florida.
    Size: 2.12 GB
    Status: not downloaded. call .download() to save at
        /home/user/.local/share/easyidp.data/2022_florida_forestbirds
    You can change the download location with:
    idp.config.set(data_dir="/path/to/easyidp.data")

When the dataset is fully available on disk, the ``Status`` line changes to:

.. code-block:: text

    Status: available at
        /home/user/.local/share/easyidp.data/2022_florida_forestbirds

.. note::

    Changing ``data_dir`` through ``idp.config.set(data_dir=...)`` does
    **not** migrate or move already-cached datasets to the new location.
    New ``Dataset`` objects constructed after the change will use the new
    path, but existing objects retain the root they were created with.

Mirrors
=======

By default, EasyIDP tries to download datasets from Shared Google Drive by the
``gdown`` package. For users in mainland China, use the ModelScope mirror for
better download reliability. EasyIDP uses the public ModelScope dataset SDK and
does not require a ModelScope token or interactive login for the official public
demo datasets:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    lotus.download(mirror="modelscope")


Classes
=======

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

Advanced API
============

``easyidp.data`` builds short demo-data attributes from JSON manifest keys.
Dotted keys such as ``metashape.project`` and ``metashape.outputs.dom`` are
expanded into runtime namespaces so users can write ``lotus.metashape.project``
or ``lotus.metashape.outputs.dom``.

The same module also contains explicit downloader helpers for Google Drive,
ModelScope mirrors, downloaded-file verification, and zip extraction.

The recursive namespace object is implemented as
``easyidp.data.dataset._PathNamespace``. The objects below are intended for
advanced users and contributors who need to understand manifest parsing,
runtime path expansion, dataset validation, and downloader internals. They are
not exported from ``easyidp.data`` unless shown in the public sections above.

Classes
-------

.. autosummary::

    dataset.Dataset
    dataset._PathNamespace

Functions
---------

.. autosummary::

    dataset._load_manifest
    dataset._validate_attr_name
    dataset._validate_manifest
    dataset._insert_path
    downloader.download_dataset
    downloader.safe_extract_zip
    downloader._select_mirror
    downloader._download_gdrive
    downloader._download_modelscope
    downloader._fetch_modelscope_file_info
    downloader._verify_downloaded_file
    downloader._file_sha256
    downloader._result
