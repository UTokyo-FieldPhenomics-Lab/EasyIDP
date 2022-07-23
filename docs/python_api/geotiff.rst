=======
GeoTiff
=======


Class
=====

A summary of class ``GeoTiff``

.. autosummary::

    easyidp.GeoTiff.read_geotiff
    easyidp.GeoTiff.has_data
    easyidp.GeoTiff.point_query
    .. easyidp.GeoTiff.crop
    easyidp.GeoTiff.crop_polygon
    easyidp.GeoTiff.save_geotiff
    easyidp.GeoTiff.math_polygon
    .. easyidp.GeoTiff.create_grid

.. autoclass:: easyidp.GeoTiff
    :members:
    :noindex:


Functions
=========

.. caution::
    
    The ``easyidp.GeoTiff`` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need to.

.. autosummary::
    :toctree: autodoc

    easyidp.geotiff.get_header
    easyidp.geotiff.get_imarray
    easyidp.geotiff.geo2pixel
    easyidp.geotiff.pixel2geo
    easyidp.geotiff.tifffile_crop