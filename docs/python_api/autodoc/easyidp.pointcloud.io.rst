:orphan:

easyidp.pointcloud.io
=====================

.. automodule:: easyidp.pointcloud.io
   :no-members:

Dispatcher Functions
--------------------

These functions are re-exported as :func:`easyidp.pointcloud.read_point_cloud`
and :func:`easyidp.pointcloud.write_point_cloud`. Prefer the re-exported
module-level names in user code.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - :func:`read_point_cloud`
     - Read by explicit format or file suffix.
   * - :func:`write_point_cloud`
     - Write by explicit format or file suffix.

.. autofunction:: read_point_cloud
   :no-index:

.. autofunction:: write_point_cloud
   :no-index:

LAS/LAZ Backend
---------------

``easyidp.pointcloud.io.las`` provides low-level array IO for LAS and LAZ
files. It returns and accepts raw NumPy arrays rather than ``PointCloud``
objects.

.. currentmodule:: easyidp.pointcloud.io.las

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - :func:`read`
     - Read LAS/LAZ data into raw arrays.
   * - :func:`write`
     - Write raw arrays to LAS/LAZ.

.. autofunction:: read

.. autofunction:: write

PLY Backend
-----------

``easyidp.pointcloud.io.ply`` provides low-level array IO for PLY files. It
returns and accepts raw NumPy arrays rather than ``PointCloud`` objects.

.. currentmodule:: easyidp.pointcloud.io.ply

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - :func:`read`
     - Read PLY data into raw arrays.
   * - :func:`write`
     - Write raw arrays to PLY.

.. autofunction:: read

.. autofunction:: write
