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

Functions
=========

.. autofunction:: get

.. autofunction:: set

.. autofunction:: reset

Advanced API
============

Most users should only call the module-level ``get()``, ``set()``, and
``reset()`` functions above. Internally, ``easyidp.config`` creates one
module-level configuration object and exposes bound methods from it:

.. code-block:: python

    config = EasyIDPConfig()
    get = config.get
    set = config.set
    reset = config.reset

This means ``idp.config.set(...)`` is the public shortcut for the singleton
configuration object's ``set(...)`` method, not a request for users to
instantiate ``EasyIDPConfig`` themselves.

The contributor-facing configuration object and path helpers are documented on
the hidden :doc:`Config Advanced API <autodoc/easyidp.config.advanced>` page.
