:orphan:

easyidp.pointcloud.geometry
===========================

.. automodule:: easyidp.pointcloud.geometry
   :no-members:

KDTree Geometry Helpers
-----------------------

These helpers are internal extension points for point-cloud crop queries.
They operate on absolute XY coordinates and return point indices. Ordinary
users should call :meth:`easyidp.pointcloud.PointCloud.crop` or
:meth:`easyidp.ROI.crop` instead.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - :func:`query_indices_by_polygon`
     - Return point indices inside one polygon.
   * - :func:`query_indices_by_multipolygon`
     - Return point indices inside any polygon of a MultiPolygon.

.. autofunction:: query_indices_by_polygon

.. autofunction:: query_indices_by_multipolygon
