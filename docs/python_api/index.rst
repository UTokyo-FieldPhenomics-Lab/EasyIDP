===========
API Summary
===========

The EasyIDP package have the following modules:

- :doc:`Data Module <./data>` : Download and provide path to example files.
- :doc:`Point cloud Module <./pointcloud>` : read, write, check and crop operation.
- :doc:`GeoTiff Module <./geotiff>` : read, write, crop, and statistics operation.
  
  - :doc:`cvtools Submodule <./cvtools>` : processing ndarray images.

- :doc:`ROI Module <./roi>` : read region of interest from shp and txt file.
  
  - :doc:`shp Submodule <./shp>`: read shape (\*.shp) file.
  - :doc:`jsonfile Submodule <./json>`: read and write json file.

- :doc:`reconstruct Module <./reconstruct>` : process 3D reconstruction software project.

  - :doc:`Pix4D Submodule <./pix4d>`: handle Pix4D projects.
  - :doc:`Metashape Submodule <./metashape>`: handle Metashape project.


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

.. caution:: 

    The :class:`easyidp.Container` object, its child objects, and objects contains this object, like :class:`easyidp.ROI <easyidp.roi.ROI>` , ``ProjectPool`` , :class:`easyidp.Recons <easyidp.reconstruct.Recons>` , ``Pix4D`` , ``Metashape``, can not be saved by pickle. 
    
    Please check this link for more details `What can be pickled and unpickled <https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>`_ .
