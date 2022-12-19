=======
GeoTiff
=======

.. currentmodule:: easyidp.geotiff

Class
=====

A summary of class ``GeoTiff``

.. autosummary::
    :toctree: autodoc

    GeoTiff
    

Functions
=========

This module (``easyidp.geotiff``) also contains the following standard-alone functions for preocessing GeoTiff file directly.

.. caution::
    
    The :class:`easyidp.GeoTiff <easyidp.geotiff.GeoTiff>` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need them.

.. autosummary::
    :toctree: autodoc

    get_header
    get_imarray
    geo2pixel
    pixel2geo
    tifffile_crop
    point_query
    save_geotiff