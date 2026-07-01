===========
Point Cloud
===========

.. currentmodule:: easyidp.pointcloud

Class
=====

A summary of class ``easyidp.pointcloud.PointCloud``, can be simply accessed by ``easyidp.PointCloud``.

.. autosummary::
    :toctree: autodoc
    :template: autosummary/pointcloud_class.rst

    PointCloud


Functions
=========

These functions are the recommended entry points for reading and writing
point cloud files. Use these instead of the legacy per-format functions.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Function
      - Purpose
    * - :func:`read_point_cloud`
      - Read a point-cloud file into a new :class:`PointCloud`.
    * - :func:`write_point_cloud`
      - Write a :class:`PointCloud` through explicit format or suffix dispatch.

.. autofunction:: read_point_cloud

.. autofunction:: write_point_cloud

Old Compatibility API
=====================

.. caution::

    The following per-format functions emit ``FutureWarning`` and may be
    deprecated in a future v3.0 release.  Prefer :func:`read_point_cloud` or
    :func:`write_point_cloud` (or use :class:`PointCloud` directly).

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Function
      - Replacement
    * - :func:`read_las`
      - Use :func:`read_point_cloud`.
    * - :func:`read_laz`
      - Use :func:`read_point_cloud`.
    * - :func:`read_ply`
      - Use :func:`read_point_cloud`.
    * - :func:`write_las`
      - Use :func:`write_point_cloud`.
    * - :func:`write_laz`
      - Use :func:`write_point_cloud`.
    * - :func:`write_ply`
      - Use :func:`write_point_cloud`.

.. autofunction:: read_las

.. autofunction:: read_laz

.. autofunction:: read_ply

.. autofunction:: write_las

.. autofunction:: write_laz

.. autofunction:: write_ply


Advanced API
============

Advanced APIs are documented as compact module pages instead of one page
per helper function:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Page
      - Scope
    * - :doc:`compat <autodoc/easyidp.pointcloud.compat>`
      - Legacy wrappers and ``PointCloudCompatMixin`` implementation notes.
    * - :doc:`geometry <autodoc/easyidp.pointcloud.geometry>`
      - KDTree-backed polygon and multipolygon point selection helpers.
    * - :doc:`io <autodoc/easyidp.pointcloud.io>`
      - Public IO dispatchers and low-level LAS/PLY backend helpers.

Internal ``PointCloud`` helper methods and compatibility implementation
details are maintained from these module-level advanced pages, not from the
main :class:`PointCloud` class page.
