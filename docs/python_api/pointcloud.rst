===========
Point Cloud
===========


Class
=====

A summary of class ``PointCloud``

.. autosummary::

    easyidp.PointCloud.has_colors
    easyidp.PointCloud.has_points
    easyidp.PointCloud.has_normals
    easyidp.PointCloud.crop_point_cloud


.. autoclass:: easyidp.PointCloud
    :members:
    :noindex:


Functions
=========

.. caution::
    
    The ``easyidp.PointCloud`` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need to.

.. autosummary::
    :toctree: autodoc

    easyidp.pointcloud.write_ply
    easyidp.pointcloud.read_las