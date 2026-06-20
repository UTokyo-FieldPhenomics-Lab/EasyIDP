======
Config
======

.. currentmodule:: easyidp.config

Purpose
=======

The config module stores package-wide EasyIDP preferences in a small JSON file.
It is the single public entry point for settings such as the demo dataset root,
logger level, and startup banner display.

Most users should access it through ``easyidp.config``:

.. code-block:: python

    import easyidp as idp

    idp.config.set(data_dir="/path/to/easyidp.data")
    data_dir = idp.config.get("data_dir")

Settings
========

``data_dir``
    Root folder for EasyIDP demo datasets.

``log_level``
    Logger level used by EasyIDP, such as ``"INFO"`` or ``"DEBUG"``.

``show_banner``
    Whether EasyIDP shows the startup banner during import.

Classes
=======

.. autosummary::
    :toctree: autodoc

    EasyIDPConfig

Functions
=========

.. autosummary::
    :toctree: autodoc

    get
    set
    reset

Advanced API
============

These helpers are mainly useful for contributors and advanced users who need
to inspect EasyIDP's platform-specific default locations.

Functions
---------

.. autosummary::

    default_config_path
    default_data_dir
