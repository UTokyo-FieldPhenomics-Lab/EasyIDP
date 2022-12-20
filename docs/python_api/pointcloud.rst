===========
Point Cloud
===========

.. currentmodule:: easyidp.pointcloud

Class
=====

A summary of class ``easyidp.pointcloud.PointCloud``, can be simple accessed by ``easyidp.PointCloud``

.. autosummary::
    :toctree: autodoc

    PointCloud


Functions
=========

This module (``easyidp.pointcloud``) also contains the following standard-alone functions for preocessing PointCloud file (ply, las, laz) directly.

.. caution::
    
    The :class:`easyidp.PointCloud <easyidp.pointcloud.PointCloud>` class is an advanced wrapper around the following functions, which is generally sufficient for most simple application cases, please don't use the following functions unless you really need to.

.. autosummary::
    :toctree: autodoc

    read_las
    read_laz
    read_ply
    write_las
    write_laz
    write_ply
    