===========
API Summary
===========

The EasyIDP package have the following modules:

- :doc:`Point cloud Module <./pointcloud>` : read, write, check and crop opertation.
- :doc:`GeoTiff Module <./geotiff>` : read, write, crop, and statistics opertation.
- :doc:`ROI Module <./roi>` : read region of interest from shp and txt file.
  
  - :doc:`shp Submodule <./shp>`: read shape (\*.shp) file.
  - :doc:`jsonfile Submodule <./json>`: read and write json file.


For each module, consisted by several base functions (e.g. ``easyidp.geotiff.*`` ) and an advanced wrapper class for them (e.g. ``easyidp.GeoTiff``). In the most cases, please use the upper case class wrapper rather than the lowercase base functions unless you really need them.

For example, this function can be used to read geotiff meta infomation:

.. code-block:: python

    >>> import easyidp as idp
    >>> header = idp.geotiff.get_header("one_geotiff_file.tif")

But it is more recommended use the advanced wrapper in most application cases:

.. code-block:: python

    >>> geo = idp.GeoTiff("one_geotiff_files.tif")
    >>> header = geo.header

Although it may seem like more code, advanced wrappers have more convenient functions to use without caring about specific data structure details. Most of our example cases are using the advanced class wrapper.